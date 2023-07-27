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

from common.img_utils import batch_tensor_to_images, batch_tensor_to_heatmaps, non_maximum_suppress
from centernet.model import CenterNet
from centernet.encoder import KeypointHeatmapEncoder
from centernet.loss import FocalLoss
from data.data_utils import collate_ground_truth_boxes
from data.voc_dataset import VOCDataset, VOC_CLASSES

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pdb

def train(args):
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

    encoder = KeypointHeatmapEncoder((37,37)).to(device)
    model = CenterNet().to(device)
    focal_loss = FocalLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=[0.9, 0.999])

    summary_writer = None
    start_time = datetime.datetime.now()
    if args.tensorboard:
        summary_writer = SummaryWriter(
            log_dir=f"{args.log_dir}/centernet_{start_time.strftime('%Y-%m-%dT%H-%M-%S')}")

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
                batch_center_mask,
                batch_center_cls_heatmap,
                batch_center_reg_heatmap,
            )= encoder.batch_encode(batch_gt)

            optimizer.zero_grad()
            batch_cls_preds, batch_reg_preds = model(batch_imgs)
            loss = focal_loss(batch_center_mask, batch_center_cls_heatmap, batch_center_reg_heatmap,
                            batch_cls_preds, batch_reg_preds)

            if args.tensorboard and global_step % args.log_interval == 0:
                summary_writer.add_scalar("loss/train", loss, global_step)

            loss.backward()
            optimizer.step()
            global_step += 1

            if args.tensorboard and global_step % args.visualize_interval == 0:
                (
                    batch_boxes,
                    batch_box_scores,
                    batch_box_labels,
                ) = encoder.batch_decode(batch_cls_preds, batch_reg_preds)
                (
                    batch_nms_box_preds,
                    batch_nms_score_preds,
                    batch_nms_label_preds,
                ) = non_maximum_suppress(batch_boxes, batch_box_scores, batch_box_labels,
                                         iou_threshold=0.5, score_threshold=0.3)

                heatmaps = batch_tensor_to_heatmaps(batch_cls_preds)
                summary_writer.add_images("heatmaps", heatmaps, global_step=global_step, dataformats='NHWC')

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
            }, f"{args.checkpoint_dir}/centernet_{start_time.strftime('%Y-%m-%dT%H-%M-%S')}_epoch_{epoch}.pth")

    if args.tensorboard:
        summary_writer.close()


args = argparse.ArgumentParser(description="CenterNet Training With PyTorch")
# Training settings
args.add_argument("--batch-size", default=8, type=int, help="Batch size for training")
args.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="Learning rate")
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


if __name__ == '__main__':
    args = args.parse_args()
    train(args)
