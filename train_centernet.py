import argparse
import logging

import coloredlogs
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from centernet.model import CenterNet
from centernet.encoder import KeypointHeatmapEncoder
from centernet.loss import FocalLoss
from data.data_utils import collate_ground_truth_boxes
from data.voc_dataset import VOCDataset, VOC_CLASSES

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        loss.backward()
        optimizer.step()


args = argparse.ArgumentParser(description="CenterNet Training With PyTorch")
# Training settings
args.add_argument("--batch-size", default=8, type=int, help="Batch size for training")
args.add_argument("--num-workers", default=1, type=int, help="Number of workers used in dataloading")
args.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="Learning rate")


if __name__ == '__main__':
    args = args.parse_args()
    train(args)
