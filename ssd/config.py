"""
Default configuration for SSD300 training on Pascal VOC dataset
Reference: https://arxiv.org/pdf/1512.02325.pdf

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
"""

# Expected feature map names. Model should name feature maps as follows:
DEFAULT_FEATURE_MAP_NAMES = [
    "Conv4_3",
    "Conv7",
    "Conv8_2",
    "Conv9_2",
    "Conv10_2",
    "Conv11_2",
]

# Number of anchor boxes per feature map location [i][j]
DEFAULT_NUM_BOXES = {
    'Conv4_3': 4,
    'Conv7': 6,
    'Conv8_2': 6,
    'Conv9_2': 6,
    'Conv10_2': 4,
    'Conv11_2': 4
}

# Aspect ratios for anchor boxes at each feature map location [i][j]
DEFAULT_ASPECT_RATIOS = {
    "Conv4_3": (1, 2, 1.0/2),
    "Conv7": (1, 2, 3, 1.0/2, 1.0/3),
    "Conv8_2": (1, 2, 3, 1.0/2, 1.0/3),
    "Conv9_2": (1, 2, 3, 1.0/2, 1.0/3),
    "Conv10_2": (1, 2, 1.0/2),
    "Conv11_2": (1, 2, 1.0/2),
}

# Feature map dimensions after convolution and pooling.
DEFAULT_FEATURE_MAP_DIMS = {
    "Conv4_3": (37, 37),
    "Conv7": (18, 18),
    "Conv8_2": (9, 9),
    "Conv9_2": (5, 5),
    "Conv10_2": (3, 3),
    "Conv11_2": (1, 1),
}
