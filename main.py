import pdb
import torch
import torch.nn.functional as F

from ssd.model import SingleShotDetector
from ssd.anchor import AnchorGenerator, TargetEncoder
from data import VOCDataset
from ssd.box_utils import point_form


def main():
    dataset = VOCDataset()

    img_tensor, bboxes = dataset[102]
    gt_boxes = bboxes[:, :4]
    gt_labels = bboxes[:, 4]

    generator = AnchorGenerator()
    anchor_boxes_by_layer = generator.anchor_boxes
    priors_boxes = []
    for layer in anchor_boxes_by_layer:
        priors_boxes.append(anchor_boxes_by_layer[layer].view(-1, 4))
    priors_boxes = torch.cat(priors_boxes, dim=0)
    priors_boxes = point_form(priors_boxes, clip=True)

    encoder = TargetEncoder(priors_boxes, iou_threshold=0.5)
    matched_gts, loc_targets, cls_targets = encoder.encode(gt_boxes, gt_labels)

    x = torch.randn((1, 3, 300, 300))
    ssd = SingleShotDetector()
    loc_preds_by_layer, cls_preds_by_layer = ssd(x)

    loc_preds = []
    print("Location Predictions")
    for key, value in loc_preds_by_layer.items():
        print("\t", key, value.shape)
        loc_preds.append(value.view(-1, 4)) # Flatten to (N, 4) where N is the number of priors.
    loc_preds = torch.cat(loc_preds, dim=0)

    cls_preds = []
    print("Classification Predictions")
    for key, value in cls_preds_by_layer.items():
        print("\t", key, value.shape)
        cls_preds.append(value.view(-1, 21)) # Flatten to (N, 21) where N is the number of priors.
    cls_preds = torch.cat(cls_preds, dim=0)

    loss = F.cross_entropy(cls_preds, cls_targets, reduction="sum")
    print(loss)

    encoder.decode(loc_targets)

if __name__ == "__main__":
    main()