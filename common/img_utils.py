import cv2
import numpy as np
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
            imgs[i] = draw_bounding_boxes_np(imgs[i], batch_bboxes[i], batch_labels[i], class_names,
                                             color=color,
                                             thickness=thickness)
    return imgs


def tensor_to_image(img_tensor, bbox_tensor, label_tensor,
                    class_names=None,
                    draw_bbox=True,
                    color=(255, 255, 0),
                    thickness=2):
    img = img_tensor.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    img = img.astype(np.uint8)

    if draw_bbox:
        bboxes = bbox_tensor.cpu().numpy()
        labels = label_tensor.cpu().numpy().astype(np.uint8)
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
