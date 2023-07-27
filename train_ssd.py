import argparse
import datetime
import logging
import os

import coloredlogs
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.img_utils import batch_tensor_to_images, non_maximum_suppress
from data.data_utils import collate_ground_truth_boxes
from data.voc_dataset import VOCDataset, VOC_CLASSES
from ssd.anchor import AnchorGenerator
from ssd.box_utils import point_form
from ssd.encoder import TargetEncoder
from ssd.loss import MultiBoxLoss
from ssd.model import SingleShotDetector

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    generator = AnchorGenerator()
    anchor_boxes_by_layer = generator.anchor_boxes
    prior_boxes = []
    for layer in anchor_boxes_by_layer:
        prior_boxes.append(anchor_boxes_by_layer[layer].view(-1, 4))
    prior_boxes = torch.cat(prior_boxes, dim=0)
    prior_boxes = point_form(prior_boxes, clip=True)
    prior_boxes = prior_boxes.to(device)
    encoder = TargetEncoder(prior_boxes, iou_threshold=0.5)

    dataset = VOCDataset("trainval")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              collate_fn=collate_ground_truth_boxes)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            collate_fn=collate_ground_truth_boxes)

    multibox_loss = MultiBoxLoss()
    model = SingleShotDetector().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    summary_writer = None
    start_time = datetime.datetime.now()
    if args.tensorboard:
        summary_writer = SummaryWriter(
            log_dir=f"{args.log_dir}/ssd_{start_time.strftime('%Y-%m-%dT%H-%M-%S')}")

    global_step = 0
    epoch = 0
    logger.info(f"load checkpoint is set to {args.load_checkpoint}")
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        logger.info(f"restored checkpoint to epoch={epoch} and global_step={global_step}")

    class_names = {i + 1: VOC_CLASSES[i] for i in range(len(VOC_CLASSES))}
    class_names[0] = "background"

    for _ in range(args.num_epochs):
        for batch_imgs, batch_gt in tqdm(train_loader):
            batch_imgs = batch_imgs.to(device)
            batch_gt = batch_gt.to(device)

            (
                batch_matched_gts,
                batch_matched_labels,
                batch_loc_targets,
                batch_cls_targets
            ) = encoder.batch_encode(batch_gt)

            optimizer.zero_grad()
            batch_loc_preds, batch_cls_preds = model(batch_imgs)
            loss = multibox_loss(batch_loc_preds, batch_cls_preds, batch_loc_targets, batch_cls_targets)

            if args.tensorboard and global_step % args.log_interval == 0:
                summary_writer.add_scalar("loss/train", loss, global_step)

            loss.backward()
            optimizer.step()
            global_step += 1

            if args.tensorboard and global_step % args.visualize_interval == 0:
                batch_box_preds = encoder.batch_decode_localization(batch_loc_preds)
                batch_score_preds, batch_label_preds = encoder.batch_decode_classification(batch_cls_preds)
                (
                    batch_nms_box_preds,
                    batch_nms_score_preds,
                    batch_nms_label_preds,
                ) = non_maximum_suppress(batch_box_preds, batch_score_preds, batch_label_preds, iou_threshold=0.5, score_threshold=0.3)
                images = batch_tensor_to_images(batch_imgs,
                                                batch_nms_box_preds.detach(),
                                                batch_nms_label_preds.detach(),
                                                class_names,
                                                color=(0, 255, 0))
                summary_writer.add_images("detections", images,
                                          global_step=global_step,
                                          dataformats='NHWC')

                images = batch_tensor_to_images(batch_imgs, batch_gt[:, :, :4], batch_gt[:, :, 4], class_names)
                summary_writer.add_images("ground_truths", images,
                                          global_step=global_step,
                                          dataformats='NHWC')

        epoch += 1
        if epoch % args.checkpoint_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.checkpoint_dir}/ssd_{start_time.strftime('%Y-%m-%dT%H-%M-%S')}_epoch_{epoch}.pth")

    if args.tensorboard:
        summary_writer.close()


args = argparse.ArgumentParser(description="Single Shot Detector Training With PyTorch")
# Training settings
args.add_argument("--batch-size", default=16, type=int, help="Batch size for training")
args.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="Learning rate")
args.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
args.add_argument("--weight-decay", default=5e-4, type=float, help="Weight decay for SGD")
args.add_argument("--num-epochs", default=100, type=int, help="Number of epochs to train for")
args.add_argument("--num-workers", default=4, type=int, help="Number of workers used in dataloading")
# Checkpoints settings
args.add_argument("--load-checkpoint", default=None, type=str, help="Path to model checkpoint")
args.add_argument("--checkpoint-dir", default="checkpoints", type=str, help="Directory for saving checkpoint models")
args.add_argument("--checkpoint-interval", default=10, type=int, help="Number of epochs between saving checkpoints")
# Tensorboard settings
args.add_argument("--tensorboard", action='store_true', help="Use tensorboard for loss visualization")
args.add_argument("--log-dir", default="logs", type=str, help="Directory for tensorboard logs")
args.add_argument("--log-interval", default=100, type=int, help="Number of steps between logging loss")
args.add_argument("--visualize-interval", default=100, type=int, help="Number of steps between visualizing images")


if __name__ == "__main__":
    args = args.parse_args()
    train(args)
