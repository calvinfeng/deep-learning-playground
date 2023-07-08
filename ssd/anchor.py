import numpy as np
import torch
from typing import Dict, Tuple, List
from ssd.config import (
    DEFAULT_ASPECT_RATIOS,
    DEFAULT_FEATURE_MAP_DIMS,
    DEFAULT_FEATURE_MAP_NAMES,
    DEFAULT_NUM_BOXES,
)
from ssd.box_utils import point_form, intersect, jaccard
import pdb


class AnchorGenerator:
    """Generate anchor boxes for each feature map location.

    Reference: https://arxiv.org/pdf/1512.02325.pdf
    """
    def __init__(self, image_size=(300, 300),
                       feature_map_names=DEFAULT_FEATURE_MAP_NAMES,
                       feature_map_dims=DEFAULT_FEATURE_MAP_DIMS,
                       aspect_ratios=DEFAULT_ASPECT_RATIOS,
                       min_scale=0.2,
                       max_scale=0.9):
        self.image_size: Tuple[int, int] = image_size
        self.feature_map_names: List[str] = feature_map_names
        self.feature_map_dims: Dict[str, Tuple] = feature_map_dims
        self.aspect_ratios: Dict[str, Tuple] = aspect_ratios

        # Compute scales for each feature map based on equation provided in paper
        K = len(feature_map_dims)
        self.scales = [None] * K
        for k in range(K):
            self.scales[k] = min_scale + (max_scale - min_scale) * k / (K - 1)

    @property
    def anchor_boxes(self):
        """Return anchor boxes for each feature map.

        Returns:
            Dict[str, torch.Tensor]: feature map name -> anchor boxes
        """
        anchor_boxes_by_layer = dict()
        for k, fm_name in enumerate(self.feature_map_names):
            # Scale is associated with k-th feature map
            s_k = self.scales[k]
            fm_dim = self.feature_map_dims[fm_name]

            num_aspect_ratios = len(self.aspect_ratios[fm_name])
            num_boxes = num_aspect_ratios
            if 1 in self.aspect_ratios[fm_name]:
                num_boxes += 1

            fm_anchors = torch.zeros((fm_dim[0], fm_dim[1], num_boxes, 4))
            for i in range(fm_dim[0]):
                for j in range(fm_dim[1]):
                    cy = (i + 0.5) / fm_dim[0]
                    cx = (j + 0.5) / fm_dim[1]

                    box_index = 0
                    for ar in self.aspect_ratios[fm_name]:
                        # Anchor box has [cx, cy, w, h] format.
                        # Special case for ratio == 1
                        if ar == 1:
                            fm_anchors[i, j, box_index, 0:4] = torch.Tensor([cx, cy, s_k, s_k])
                            try:
                                s_k_prime = np.sqrt(s_k * self.scales[k + 1])
                                extra_box_tensor = torch.Tensor([cx, cy, s_k_prime, s_k_prime])
                            except IndexError:
                                s_k_prime = 1.0 # Cover the whole image.
                                extra_box_tensor = torch.Tensor([cx, cy, s_k_prime, s_k_prime])
                            fm_anchors[i, j, box_index+1, 0:4] = extra_box_tensor

                            box_index += 2
                        else:
                            fm_anchors[i, j, box_index, 0:4] = torch.Tensor([cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)])
                            box_index += 1

            anchor_boxes_by_layer[fm_name] = fm_anchors
        return anchor_boxes_by_layer


class TargetEncoder:
    """Encode ground truth boxes and labels for training by matching them with anchor boxes.
    """
    def __init__(self, image_size=(300, 300),
                       num_classes=20,
                       feature_map_names=DEFAULT_FEATURE_MAP_NAMES,
                       feature_map_dims=DEFAULT_FEATURE_MAP_DIMS,
                       feature_map_num_boxes=DEFAULT_NUM_BOXES,
                       ):
        self.image_size: Tuple[int, int] = image_size
        self.num_classes = num_classes + 1 # Reserve 1 for background.
        self.feature_map_names: List[str] = feature_map_names
        self.feature_map_dims: Dict[str, Tuple] = feature_map_dims
        self.feature_map_num_boxes: Dict[str, int] = feature_map_num_boxes

    def _match(self, gt_boxes, gt_labels, prior_boxes, iou_threshold=0.5):
        # We have N ground truth boxes and M prior boxes.
        # IoU returns a tensor of shape (N, M) with IoU for each gt_boxes[i], prior_boxes[j] pair.
        iou = jaccard(gt_boxes, prior_boxes)

        # For each ground truth box, find the prior box with the highest IoU.
        # This is a tensor of shape (N,) and (N,).
        best_prior_iou_for_gt, best_prior_idx_for_gt = iou.max(1, keepdim=False)

        # For each prior box, find the ground truth box with the highest IoU.
        # This is a tensor of shape (M,) and (M,).
        best_gt_iou_for_prior, best_gt_idx_for_prior = iou.max(0, keepdim=False)

        """
        Multiple prior boxes can match to the same ground truth box because we are max(dim=1)
        For example, priors[5] and priors[6] are matched to gts[1]
        best_gt_idx_for_prior   0 1 2 3 4 5 6 7 8 9 ...
                              [ . . . . . 1 1 . . . ... ]

        Similarly, from ground truth box perspective, multiple ground truth boxes can match to the
        same prior box because we max(dim=0)
        For example, gts[1] and gts[2] are matched to priors[5]
        best_prior_idx_for_gt   0 1 2 3 4 5 ...
                              [ . 5 5 . . . ...]

        We will encounter a discrepancy. We have to make sure that only multiple prior boxes match
        to the same ground truth box but not multiple ground truth boxes match to the same prior
        box.
        """
        # Is this logic even necessary?
        # best_gt_idx_for_prior_copy = best_gt_idx_for_prior.clone()
        # for i in range(best_prior_idx_for_gt.size(0)):
        #     # Find the best prior index for the current ground truth box.
        #     j = best_prior_idx_for_gt[i]
        #     # Set best ground truth index for the selected prior to current ground truth box.
        #     best_gt_idx_for_prior_copy[j] = i

        # This is a tensor of shape (M, 4). We have assigned a ground truth box for each
        # prior box.
        matched_gt_box_for_prior = gt_boxes[best_gt_idx_for_prior]
        # We have assigned a label for each prior box, and add 1 to offset background label.
        matched_gt_label_for_prior = gt_labels[best_gt_idx_for_prior] + 1

        # Set the IoU score of the best ground truth box for each prior box to iou_threshold. This
        # will ensure that at least 1 best ground truth box will be selected for each prior box if
        # everything is below the IoU threshold. That means, at minimum, we get N prior boxes.
        best_gt_iou_for_prior.index_fill_(0, best_prior_idx_for_gt, iou_threshold)
        matched_gt_label_for_prior[best_gt_iou_for_prior < iou_threshold] = 0 # Background label.

        return matched_gt_box_for_prior, matched_gt_label_for_prior

    # Figure out batching strategy later.
    def encode(self, anchor_boxes_by_layer, gt_boxes, gt_labels, iou_threshold=0.5):
        """Encode ground truth boxes and labels for training.

        Args:
            anchor_boxes_by_layer: Anchor boxes for each feature map layer.
            gt_boxes: Ground truth bounding boxes in [cx, cy, w, h] format.
            gt_labels: Ground truth labels for each bounding box.

        Returns:
            loc_targets: Location targets for each anchor box.
            cls_targets: Class targets for each anchor box.
        """
        priors = []
        for layer in anchor_boxes_by_layer:
            priors.append(point_form(anchor_boxes_by_layer[layer].view(-1, 4), clip=True))
        priors = torch.cat(priors, dim=0)

        # Priors are anchor boxes in [x_min, y_min, x_min, y_min] format with relative coordinate.
        # Ground truth are bounding boxes in [x_min, y_min, x_max, y_max] format with relative coordinate.
        # Match each ground truth box to the anchor box with the best jaccard overlap
        matched_priors, matched_labels = self._match(gt_boxes, gt_labels, priors, iou_threshold)
