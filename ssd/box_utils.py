import torch
import numpy as np
import pdb

def point_form(prior_boxes, clip=False):
    """Convert prior_boxes (x, y, width, height) to (xmin, ymin, xmax, ymax). Point form may extend
    beyond 1. Optionally this can be clipped by passing clip=True.
    """
    x_min = prior_boxes[:, 0] - prior_boxes[:, 2] / 2
    y_min = prior_boxes[:, 1] - prior_boxes[:, 3] / 2
    x_max = prior_boxes[:, 0] + prior_boxes[:, 2] / 2
    y_max = prior_boxes[:, 1] + prior_boxes[:, 3] / 2
    if clip:
        return torch.stack([x_min, y_min, x_max, y_max], axis=1).clip(0, 1)
    return torch.stack([x_min, y_min, x_max, y_max], axis=1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A, 4].
      box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)

    box_a_min_xy = box_a[:, :2].unsqueeze(1).expand(A, B, 2)
    box_a_max_xy = box_a[:, 2:].unsqueeze(1).expand(A, B, 2)
    box_b_min_xy = box_b[:, :2].unsqueeze(0).expand(A, B, 2)
    box_b_max_xy = box_b[:, 2:].unsqueeze(0).expand(A, B, 2)

    max_xy = torch.min(box_a_max_xy, box_b_max_xy)
    min_xy = torch.max(box_a_min_xy, box_b_min_xy)
    inter_wh = torch.clamp((max_xy - min_xy), min=0)
    return inter_wh[:, :, 0] * inter_wh[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    # Broadcast along the 1-dimension
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    # Broadcast along the 0-dimension
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
