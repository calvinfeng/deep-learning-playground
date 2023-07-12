import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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
    train_dataset = VOCDataset("train")
    train_loader = DataLoader(train_dataset,
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

    for batch_img, batch_gt in train_loader:
        batch_img = batch_img.to(device)
        batch_gt = batch_gt.to(device)

        (batch_matched_gts,
         batch_matched_labels,
         batch_loc_targets,
         batch_cls_targets) = encoder.encode_batch(batch_gt[:, :, :4], batch_gt[:, :, 4])

        optimizer.zero_grad()
        loc_preds, cls_preds = ssd(batch_img)
        loss = multibox_loss(loc_preds, cls_preds, batch_loc_targets, batch_cls_targets)
        print(loss.item())
        loss.backward()
        optimizer.step()


args = argparse.ArgumentParser(description="Single Shot Detector Training With PyTorch")
args.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
args.add_argument("--lr", "--learning_rate", default=1e-3, type=float, help="Learning rate")
args.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
args.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay for SGD")
args.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train for")
args.add_argument("--num_workers", default=1, type=int, help="Number of workers used in dataloading")
args.add_argument("--checkpoint_folder", default="ssd_checkpoints", type=str, help="Directory for saving checkpoint models")
args.add_argument("--checkpoint_interval", default=10, type=int, help="Number of epochs between saving checkpoints")
args.add_argument("--tensorboard", default=True, type=bool, help="Use tensorboard for loss visualization")
args.add_argument("--log_dir", default="logs", type=str, help="Directory for tensorboard logs")


if __name__ == "__main__":
    args = args.parse_args()
    train(args)
