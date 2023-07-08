import pdb
import torch

from ssd.model import SingleShotDetector
from ssd.anchor import AnchorGenerator, TargetEncoder
from data import VOCDataset
from ssd.box_utils import point_form


def run_ssd():
    x = torch.randn((1, 3, 300, 300))
    ssd = SingleShotDetector()
    loc_preds, cls_preds = ssd(x)
    print("Location Predictions")
    for key, value in loc_preds.items():
        print("\t", key, value.shape)
    print("Classification Predictions")
    for key, value in cls_preds.items():
        print("\t", key, value.shape)


def main():
    dataset = VOCDataset()
    encoder = TargetEncoder()
    generator = AnchorGenerator()

    anchor_boxes = generator.anchor_boxes
    priors = []
    for layer in anchor_boxes:
        layer_priors = point_form(anchor_boxes[layer].view(-1, 4), clip=True)
        priors.append(layer_priors)
    priors = torch.cat(priors, dim=0)

    img_tensor, bboxes = dataset[102]
    gt_boxes = bboxes[:, :4]
    gt_labels = bboxes[:, 4]

    matched_priors, matched_labels = encoder._match(gt_boxes, gt_labels, priors, iou_threshold=0.5)


if __name__ == "__main__":
    main()