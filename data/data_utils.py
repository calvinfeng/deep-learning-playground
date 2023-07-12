import torch


def collate_ground_truth_boxes(batch):
    """Collate images with different number of objects into batch tensors.

    Args:
        batch (List[Tuple]): List of tuples of (image, ground_truth_boxes)
    """
    images_tuple, gt_boxes_tuple = zip(*batch)
    batch_images = torch.stack(images_tuple, dim=0)

    max_num_boxes = max([len(gt_boxes) for gt_boxes in gt_boxes_tuple])

    # Prepare a tensor filled with zeros, fill it with ground truth boxes
    batch_gt_boxes = torch.zeros(len(gt_boxes_tuple), max_num_boxes, 5)
    for i, gt_boxes in enumerate(gt_boxes_tuple):
        t = torch.Tensor(gt_boxes)
        batch_gt_boxes[i, :t.shape[0], :] = t

    return batch_images, batch_gt_boxes
