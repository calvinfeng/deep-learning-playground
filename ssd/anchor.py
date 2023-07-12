import numpy as np
import torch
import functools

from typing import Dict, Tuple, List
from ssd.config import (
    DEFAULT_ASPECT_RATIOS,
    DEFAULT_FEATURE_MAP_DIMS,
    DEFAULT_FEATURE_MAP_NAMES,
)


class AnchorGenerator:
    """Generate anchor boxes for each feature map location. Anchor boxes follow conventions from
    the original SSD paper: https://arxiv.org/abs/1512.02325.
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

    # TODO: Run it on GPU and see if it is faster. I only generate anchor boxes once so I don't
    # it matters that much.
    @functools.cached_property
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
                        # Anchor box has center size format, [x, y, width, height].
                        # Special case for ratio == 1, we produce two anchor boxes.
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
