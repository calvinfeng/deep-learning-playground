import pdb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ssd.model import SingleShotDetector
from ssd.anchor import AnchorGenerator, TargetEncoder
from data import VOCDataset
from data.data_utils import collate_ground_truth_boxes
from ssd.box_utils import point_form
from ssd.loss import MultiBoxLoss

BATCH_SIZE = 4


def data_loader():
    generator = AnchorGenerator()
    anchor_boxes_by_layer = generator.anchor_boxes
    priors_boxes = []
    for layer in anchor_boxes_by_layer:
        priors_boxes.append(anchor_boxes_by_layer[layer].view(-1, 4))
    priors_boxes = torch.cat(priors_boxes, dim=0)
    priors_boxes = point_form(priors_boxes, clip=True)

    encoder = TargetEncoder(priors_boxes, iou_threshold=0.5)

    train_dataset = VOCDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                              collate_fn=collate_ground_truth_boxes)
    for batch_imgs, batch_gts in train_loader:
        batch_matched_gts, batch_matched_labels, batch_loc_targets, batch_cls_targets = encoder.encode_batch(batch_gts[:, :, :4], batch_gts[:, :, 4])
        print("Batch Matched Ground Truths", batch_matched_gts.shape)
        print("Batch Matched Labels", batch_matched_labels.shape)
        print("Batch Location Targets", batch_loc_targets.shape)
        print("Batch Classification Targets", batch_cls_targets.shape)

    val_dataset = VOCDataset("val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
                            collate_fn=collate_ground_truth_boxes)

    for x, y in val_loader:
        print(x.shape, y.shape)


def main():
    generator = AnchorGenerator()
    anchor_boxes_by_layer = generator.anchor_boxes
    priors_boxes = []
    for layer in anchor_boxes_by_layer:
        priors_boxes.append(anchor_boxes_by_layer[layer].view(-1, 4))
    priors_boxes = torch.cat(priors_boxes, dim=0)
    priors_boxes = point_form(priors_boxes, clip=True)

    encoder = TargetEncoder(priors_boxes, iou_threshold=0.5)

    train_dataset = VOCDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                              collate_fn=collate_ground_truth_boxes)

    loss = MultiBoxLoss()
    ssd = SingleShotDetector()
    for batch_img, batch_gt in train_loader:
        loc_preds, cls_preds = ssd(batch_img)
        (batch_matched_gts,
         batch_matched_labels,
         batch_loc_targets,
         batch_cls_targets) = encoder.encode_batch(batch_gt[:, :, :4], batch_gt[:, :, 4])
        loss = loss(loc_preds, cls_preds, batch_loc_targets, batch_cls_targets)
        print(loss)
        break


if __name__ == "__main__":
    main()
