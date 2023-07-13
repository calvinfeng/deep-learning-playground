import argparse
import datetime
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.data_utils import collate_ground_truth_boxes
from data.voc_dataset import VOCDataset
from ssd.anchor import AnchorGenerator
from ssd.box_utils import point_form
from ssd.encoder import TargetEncoder
from ssd.loss import MultiBoxLoss
from ssd.model import SingleShotDetector

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
    train_loader = DataLoader(VOCDataset("train"),
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              collate_fn=collate_ground_truth_boxes)
    val_loader = DataLoader(VOCDataset("val"),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            collate_fn=collate_ground_truth_boxes)
    multibox_loss = MultiBoxLoss()
    ssd = SingleShotDetector().to(device)
    optimizer = optim.SGD(ssd.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    summary_writer = None
    if args.save_log:
        current_time = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        summary_writer = SummaryWriter(log_dir=f"{args.log_dir}/ssd_{current_time}")

    global_step = 0
    epoch = 0
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)  # Load a checkpoint after 10 epochs
        ssd.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        epoch = checkpoint['epoch']

    for _ in range(args.num_epochs):
        for batch_img, batch_gt in tqdm(train_loader):
            batch_img = batch_img.to(device)
            batch_gt = batch_gt.to(device)

            (batch_matched_gts,
            batch_matched_labels,
            batch_loc_targets,
            batch_cls_targets) = encoder.encode_batch(batch_gt[:, :, :4], batch_gt[:, :, 4])

            optimizer.zero_grad()
            loc_preds, cls_preds = ssd(batch_img)
            loss = multibox_loss(loc_preds, cls_preds, batch_loc_targets, batch_cls_targets)
            if summary_writer:
                summary_writer.add_scalar("loss/train", loss, global_step)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % 10 == 0:
                break

        epoch += 1
        if epoch % args.checkpoint_interval == 0:
            os.makedirs(args.checkpoint_folder, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': ssd.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.checkpoint_folder}/ssd_epoch_{epoch}.pth")

    if args.save_log:
        summary_writer.close()


args = argparse.ArgumentParser(description="Single Shot Detector Training With PyTorch")
args.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
args.add_argument("--lr", "--learning_rate", default=1e-3, type=float, help="Learning rate")
args.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
args.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay for SGD")
args.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train for")
args.add_argument("--num_workers", default=1, type=int, help="Number of workers used in dataloading")

args.add_argument("--load_checkpoint", default=None, type=str, help="Path to model checkpoint")
args.add_argument("--checkpoint_folder", default="checkpoints", type=str, help="Directory for saving checkpoint models")
args.add_argument("--checkpoint_interval", default=10, type=int, help="Number of epochs between saving checkpoints")

args.add_argument("--save_log", action='store_true', help="Use tensorboard for loss visualization")
args.add_argument("--log_dir", default="logs", type=str, help="Directory for tensorboard logs")


if __name__ == "__main__":
    args = args.parse_args()
    train(args)
