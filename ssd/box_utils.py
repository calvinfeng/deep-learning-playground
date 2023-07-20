import torch
import torchvision.ops as ops
import pdb


def batch_point_form(batch_center_size_boxes, clip=False):
    """Convert center_size_boxes (cx, cy, width, height) to (xmin, ymin, xmax, ymax). Point form may
    extend beyond 1. Optionally this can be clipped by passing clip=True.

    Args:
        batch_center_size_boxes (tensor): Shape(B, N, 4) boxes in center-size form.
        clip (bool, optional): Defaults to False.
    """
    x_min = batch_center_size_boxes[:, :, 0] - batch_center_size_boxes[:, :, 2] / 2
    y_min = batch_center_size_boxes[:, :, 1] - batch_center_size_boxes[:, :, 3] / 2
    x_max = batch_center_size_boxes[:, :, 0] + batch_center_size_boxes[:, :, 2] / 2
    y_max = batch_center_size_boxes[:, :, 1] + batch_center_size_boxes[:, :, 3] / 2

    if clip:
        return torch.stack([x_min, y_min, x_max, y_max], axis=2).clip(0, 1)
    return torch.stack([x_min, y_min, x_max, y_max], axis=2)


def point_form(center_size_boxes, clip=False):
    """Convert center_size_boxes (cx, cy, width, height) to (xmin, ymin, xmax, ymax). Point form may
    extend beyond 1. Optionally this can be clipped by passing clip=True.

    Args:
        center_size_boxes (tensor): Shape(N, 4) boxes in center-size form.
        clip (bool, optional): Defaults to False.
    """
    x_min = center_size_boxes[:, 0] - center_size_boxes[:, 2] / 2
    y_min = center_size_boxes[:, 1] - center_size_boxes[:, 3] / 2
    x_max = center_size_boxes[:, 0] + center_size_boxes[:, 2] / 2
    y_max = center_size_boxes[:, 1] + center_size_boxes[:, 3] / 2

    if clip:
        return torch.stack([x_min, y_min, x_max, y_max], axis=1).clip(0, 1)
    return torch.stack([x_min, y_min, x_max, y_max], axis=1)

def batch_center_size_form(batch_point_form_boxes, clip=False):
    """Convert point_form_boxes (xmin, ymin, xmax, ymax) to (cx, cy, width, height). Center and size
    may extend beyond 1. Optionally this can be clipped by passing clip=True.

    Args:
        batch_point_form_boxes (tensor): Shape(B, N, 4) boxes in point form (xmin, ymin, xmax, ymax)
        clip (bool, optional): Defaults to False.
    """
    x = (batch_point_form_boxes[:, :, 0] + batch_point_form_boxes[:, :, 2]) / 2
    y = (batch_point_form_boxes[:, :, 1] + batch_point_form_boxes[:, :, 3]) / 2
    width = batch_point_form_boxes[:, :, 2] - batch_point_form_boxes[:, :, 0]
    height = batch_point_form_boxes[:, :, 3] - batch_point_form_boxes[:, :, 1]

    if clip:
        return torch.stack([x, y, width, height], axis=2).clip(0, 1)
    return torch.stack([x, y, width, height], axis=2)


def center_size_form(point_form_boxes, clip=False):
    """Convert point_form_boxes (xmin, ymin, xmax, ymax) to (cx, cy, width, height). Center and size
    may extend beyond 1. Optionally this can be clipped by passing clip=True.

    Args:
        point_form_boxes (tensor): Shape(N, 4) boxes in point form (xmin, ymin, xmax, ymax)
    """
    x = (point_form_boxes[:, 0] + point_form_boxes[:, 2]) / 2
    y = (point_form_boxes[:, 1] + point_form_boxes[:, 3]) / 2
    width = point_form_boxes[:, 2] - point_form_boxes[:, 0]
    height = point_form_boxes[:, 3] - point_form_boxes[:, 1]

    if clip:
        return torch.stack([x, y, width, height], axis=1).clip(0, 1)
    return torch.stack([x, y, width, height], axis=1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a (tensor): Shape(A, 4) bounding boxes.
      box_b (tensor): Shape(B, 4) bounding boxes.
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
        box_a: (tensor) Ground truth bounding boxes in point form, Shape: [num_objects, 4]
        box_b: (tensor) Prior boxes in point form, Shape: [num_priors, 4]
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


def non_maximum_suppress(batch_boxes, batch_scores, batch_labels, iou_threshold=0.5,
                                                                  score_threshold=0.5,
                                                                  max_num_boxes=50):
    """Perform non-maximum suppression on a batch of boxes.

    Args:
        batch_boxes (tensor): Shape(B, N, 4) boxes in point-form [min_x, min_y, max_x, max_y].
        batch_scores (tensor): Shape(B, N) score for each box.
        batch_labels (tensor): Shape(B, N) label for each box.
        iou_threshold (float, optional): Defaults to 0.5.
        max_num_boxes (int, optional): Maximum number of boxes to keep. Defaults to 50.
    Returns:
        batch_boxes (tensor): Shape(B, N, 4) boxes in point-form [min_x, min_y, max_x, max_y].
        batch_scores (tensor): Shape(B, N) score for each box.
        batch_labels (tensor): Shape(B, N) label for each box.
    """
    batch_size = batch_boxes.size(0)
    nms_boxes_list = []
    nms_scores_list = []
    nms_labels_list = []

    for i in range(batch_size):
        boxes = batch_boxes[i]
        scores = batch_scores[i]
        labels = batch_labels[i]

        selected_indices = ops.nms(boxes, scores, iou_threshold)
        nms_boxes = boxes[selected_indices]
        nms_scores = scores[selected_indices]
        nms_labels = labels[selected_indices]

        nms_boxes = nms_boxes[nms_scores > score_threshold]
        nms_labels = nms_labels[nms_scores > score_threshold]
        nms_scores = nms_scores[nms_scores > score_threshold]

        # If the number of boxes is less than max_num_boxes, pad with zeros.
        # Otherwise, we just truncate the extra ones.
        if nms_boxes.size(0) > max_num_boxes:
            nms_boxes = nms_boxes[:max_num_boxes]
            nms_scores = nms_scores[:max_num_boxes]
            nms_labels = nms_labels[:max_num_boxes]
        elif nms_boxes.size(0) < max_num_boxes:
            nms_boxes = torch.cat([nms_boxes, torch.zeros(max_num_boxes - nms_boxes.size(0), 4).to(nms_boxes)], dim=0)
            nms_scores = torch.cat([nms_scores, torch.zeros(max_num_boxes - nms_scores.size(0)).to(nms_scores)], dim=0)
            nms_labels = torch.cat([nms_labels, torch.zeros(max_num_boxes - nms_labels.size(0)).to(nms_labels)], dim=0)
        nms_boxes_list.append(nms_boxes)
        nms_scores_list.append(nms_scores)
        nms_labels_list.append(nms_labels)

    return (
        torch.stack(nms_boxes_list, dim=0),
        torch.stack(nms_scores_list, dim=0),
        torch.stack(nms_labels_list, dim=0),
    )
