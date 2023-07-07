import numpy as np
from typing import Dict, Tuple, List
from ssd.box_utils import point_form_np
from ssd.config import (
    DEFAULT_ASPECT_RATIOS,
    DEFAULT_FEATURE_MAP_DIMS,
    DEFAULT_FEATURE_MAP_NAMES,
    DEFAULT_NUM_BOXES,
)


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
        anchor_boxes_by_layer = dict()
        for k, fm_name in enumerate(self.feature_map_names):
            fm_dim = self.feature_map_dims[fm_name]
            fm_anchors = [[None] * fm_dim[1] for _ in range(fm_dim[0])]
            for i in range(fm_dim[0]):
                for j in range(fm_dim[1]):
                    cy = (i + 0.5) / fm_dim[0]
                    cx = (j + 0.5) / fm_dim[1]
                    anchors_ij = []
                    for ar in self.aspect_ratios[fm_name]:
                        # Anchor box has [cx, cy, w, h] format.
                        s_k = self.scales[k]
                        # Special case for ratio == 1
                        if ar == 1:
                            anchors_ij += [cx, cy, s_k, s_k]
                            try:
                                s_k_prime = np.sqrt(s_k * self.scales[k + 1])
                                anchors_ij += [cx, cy, s_k_prime, s_k_prime]
                            except IndexError:
                                s_k_prime = 1.0 # Cover the whole image.
                                anchors_ij += [cx, cy, s_k_prime, s_k_prime]
                        else:
                            anchors_ij += [cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)]
                    fm_anchors[i][j] = np.array(anchors_ij)
            anchor_boxes_by_layer[fm_name] = np.array(fm_anchors)
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

    # Figure out batching strategy later.
    def encode(self, anchor_boxes_by_layer, gt_boxes, gt_labels):
        """Encode ground truth boxes and labels for training.

        Args:
            anchor_boxes_by_layer: Anchor boxes for each feature map layer.
            gt_boxes: Ground truth bounding boxes in [cx, cy, w, h] format.
            gt_labels: Ground truth labels for each bounding box.

        Returns:
            loc_targets: Location targets for each anchor box.
            cls_targets: Class targets for each anchor box.
        """
