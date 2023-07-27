import cv2
import numpy as np
import torch
import torchvision.ops as ops
import pdb


def batch_tensor_to_images(batch_img_tensor, batch_bbox_tensor, batch_label_tensor,
                           class_names=None,
                           draw_bbox=True,
                           color=(255, 255, 0),
                           thickness=2):
    """Convert batch tensor of images to numpy array of images.

    Args:
        batch_img_tensor (tensor): Tensor of shape (B, C, H, W) for batch of images.
        batch_bbox_tensor (tensor): Tensor of shape (B, N, 4) for batch of bounding boxes.
        batch_label_tensor (tensor): Tensor of shape (B, N) for batch of labels.
        class_names (dict, optional): _description_. Defaults to None.
        draw_bbox (bool, optional): _description_. Defaults to True.
        color (tuple, optional): _description_. Defaults to (255, 255, 0).
        thickness (int, optional): _description_. Defaults to 2.

    Returns:
        ndarray: Array of shape (B, H, W, C) for batch of images.
    """
    imgs = batch_img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy() * 255
    imgs = imgs.astype(np.uint8)

    if draw_bbox:
        batch_bboxes = batch_bbox_tensor.cpu().numpy()
        batch_labels = batch_label_tensor.cpu().numpy().astype(np.uint8)
        for i in range(imgs.shape[0]):
            pos_mask = batch_labels[i] > 0
            bboxes = batch_bboxes[i][pos_mask]
            labels = batch_labels[i][pos_mask]
            imgs[i] = draw_bounding_boxes_np(imgs[i], bboxes, labels, class_names,
                                             color=color,
                                             thickness=thickness)
    return imgs


def batch_tensor_to_heatmaps(batch_heatmap_tensor):
    """Convert batch tensor of heatmaps to numpy array of heatmaps.

    Args:
        batch_heatmap_tensor (tensor): Tensor of shape (B, C, H, W) for batch of heatmaps.

    Returns:
        ndarray: Array of shape (B, H, W, C) for batch of heatmaps.
    """
    batch_heatmap = batch_heatmap_tensor.permute(0, 2, 3, 1).detach().contiguous().cpu().numpy()
    batch_heatmap = batch_heatmap.astype(np.float32)
    batch_heatmap = batch_heatmap.sum(axis=-1, keepdims=True)
    return batch_heatmap


def tensor_to_image(img_tensor, bbox_tensor, label_tensor,
                    class_names=None,
                    draw_bbox=True,
                    color=(255, 255, 0),
                    thickness=2):
    img = img_tensor.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    img = img.astype(np.uint8)
    if draw_bbox:
        pos_mask = label_tensor > 0
        bboxes = bbox_tensor[pos_mask].cpu().numpy()
        labels = label_tensor[pos_mask].cpu().numpy().astype(np.uint8)
        img = draw_bounding_boxes_np(img, bboxes, labels, class_names,
                                     color=color,
                                     thickness=thickness)
    return img


def draw_bounding_boxes_np(img, bboxes, labels, class_names, color=(255, 255, 0), thickness=2):
    height, width, _ = img.shape
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox
        u_min, v_min = int(x_min * width), int(y_min * height)
        u_max, v_max = int(x_max * width), int(y_max * height)
        img = cv2.rectangle(img, (u_min, v_min), (u_max, v_max),
                               color=color,
                               thickness=thickness)
        img = cv2.putText(img, class_names[label], (u_min, v_min - 5),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.5,
                             color=color,
                             thickness=2)
    return img


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

        # Select for non-background prior boxes.
        pos_mask = labels > 0
        boxes = boxes[pos_mask]
        scores = scores[pos_mask]
        labels = labels[pos_mask]

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
