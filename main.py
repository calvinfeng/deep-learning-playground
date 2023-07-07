import torch
from ssd.model import SingleShotDetector
from ssd.anchor import AnchorGenerator
from ssd.box_utils import point_form_np


def main():
    generator = AnchorGenerator()
    anchor_boxes = generator.anchor_boxes
    for layer, anchors in anchor_boxes.items():
        print(layer)
        priors = point_form_np(anchors.reshape(-1, 4), clip=True)
        print(priors.shape)
        print(priors[:10, :])

    x = torch.randn((1, 3, 300, 300))
    ssd = SingleShotDetector()
    loc_preds, cls_preds = ssd(x)
    print("Location Predictions")
    for key, value in loc_preds.items():
        print("\t", key, value.shape)
    print("Classification Predictions")
    for key, value in cls_preds.items():
        print("\t", key, value.shape)

if __name__ == "__main__":
    main()