
"""
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
self.feature_map_num_boxes = {
    'Conv4_3': 4,
    'Conv7': 6,
    'Conv8_2': 6,
    'Conv9_2': 6,
    'Conv10_2': 4,
    'Conv11_2': 4
}
Location Predictions
	 Conv4_3 torch.Size([1, 37, 37, 16])
	 Conv7 torch.Size([1, 18, 18, 24])
	 Conv8_2 torch.Size([1, 9, 9, 24])
	 Conv9_2 torch.Size([1, 5, 5, 24])
	 Conv10_2 torch.Size([1, 3, 3, 16])
	 Conv11_2 torch.Size([1, 1, 1, 16])
Classification Predictions
	 Conv4_3 torch.Size([1, 37, 37, 80])
	 Conv7 torch.Size([1, 18, 18, 120])
	 Conv8_2 torch.Size([1, 9, 9, 120])
	 Conv9_2 torch.Size([1, 5, 5, 120])
	 Conv10_2 torch.Size([1, 3, 3, 80])
	 Conv11_2 torch.Size([1, 1, 1, 80])
"""
import numpy as np
from typing import Dict, Tuple, List

DEFAULT_ASPECT_RATIOS = {
    "Conv4_3": (1, 2, 1.0/2),
    "Conv7": (1, 2, 3, 1.0/2, 1.0/3),
    "Conv8_2": (1, 2, 3, 1.0/2, 1.0/3),
    "Conv9_2": (1, 2, 3, 1.0/2, 1.0/3),
    "Conv10_2": (1, 2, 1.0/2),
    "Conv11_2": (1, 2, 1.0/2),
}

DEFAULT_FEATURE_MAP_DIMS = {
    "Conv4_3": (37, 37),
    "Conv7": (18, 18),
    "Conv8_2": (9, 9),
    "Conv9_2": (5, 5),
    "Conv10_2": (3, 3),
    "Conv11_2": (1, 1),
}

class AnchorGenerator:
    """Generate anchor boxes for each feature map location.

    Reference: https://arxiv.org/pdf/1512.02325.pdf
    """
    def __init__(self, image_size=(300, 300),
                       feature_map_names=["Conv4_3", "Conv7", "Conv8_2", "Conv9_2", "Conv10_2", "Conv11_2"],
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


if __name__ == "__main__":
    generator = AnchorGenerator()
    anchor_boxes = generator.anchor_boxes
    for layer, anchors in anchor_boxes.items():
        print(layer, anchors.shape)
