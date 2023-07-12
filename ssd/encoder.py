import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List

from ssd.box_utils import point_form, center_size_form, jaccard


class TargetEncoder:
    """Encode ground truth boxes and labels for training by matching them with anchor boxes.
    """
    def __init__(self, prior_boxes, image_size=(300, 300),
                                    num_classes=21,
                                    iou_threshold=0.50,
                                    offset_variances=(0.1, 0.1, 0.2, 0.2)):
        """Initialize target encoder.

        Args:
            prior_boxes (tensor): Prior boxes in point form [xmin, ymin, xmax, ymax] of shape (num_priors, 4).
            image_size (tuple, optional): Defaults to (300, 300).
            num_classes (int, optional): Defaults to 20.
            iou_threshold (float, optional): Defaults to 0.50.
            offset_variances (tuple, optional): Defaults to (0.1, 0.1, 0.2, 0.2).
        """
        self.prior_boxes_point_form = prior_boxes
        self.image_size: Tuple[int, int] = image_size
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.offset_variances = offset_variances

    def _match(self, gt_boxes, gt_labels):
        """Match ground truth boxes and labels with prior boxes.

        Args:
            gt_boxes (tensor): Ground truth bounding boxes in point form, [x_min, y_min, x_max, y_max].
            gt_labels (tensor): Ground truth labels for each bounding box.
            prior_boxes (tensor): (num_priors, 4) in point form.
        """
        # We have N ground truth boxes and M prior boxes.
        # IoU returns a tensor of shape (N, M) with IoU for each gt_boxes[i], prior_boxes[j] pair.
        iou = jaccard(gt_boxes, self.prior_boxes_point_form)

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

        Since multiple ground truth boxes can match to the same prior box, we need to make sure that
        each ground truth box is matched to at least one prior box that isn't shared by another
        ground truth.
        """
        for i in range(best_prior_idx_for_gt.size(0)):
            # Find the best prior index for the current ground truth box.
            j = best_prior_idx_for_gt[i]
            if best_gt_idx_for_prior[j] != i:
                # This is an indication that prior -> ground truth and ground truth -> prior are not
                # the same. This means that multiple ground truth boxes are matched to the same
                # prior. Set best ground truth index for the selected prior to current ground truth
                # box. We want to break it apart the best we can, ensuring that all ground truths
                # are used in training.
                best_gt_idx_for_prior[j] = i

        # This is a tensor of shape (M, 4). We have assigned a ground truth box for each
        # prior box.
        matched_gt_box_for_prior = gt_boxes[best_gt_idx_for_prior]
        # We have assigned a label for each prior box.
        matched_gt_label_for_prior = gt_labels[best_gt_idx_for_prior]

        # Set the IoU score of the best ground truth box for each prior box to iou_threshold. This
        # will ensure that at least 1 best ground truth box will be selected for each prior box if
        # everything is below the IoU threshold. That means, at minimum, we get N prior boxes.
        best_gt_iou_for_prior.index_fill_(0, best_prior_idx_for_gt, self.iou_threshold)
        matched_gt_label_for_prior[best_gt_iou_for_prior < self.iou_threshold] = 0 # Background label.

        return matched_gt_box_for_prior, matched_gt_label_for_prior

    def _encode_loc(self, matched_gts):
        """Encode matched ground truth boxes to each prior box into localization offsets.

        P is the number of prior boxes.

        Args:
            matched_gts (tensor): (P, 4) in point form.
            prior_boxes (tensor): (P, 4) in point form.
        """
        # Convert to center form.
        matched_gts = center_size_form(matched_gts)
        prior_boxes = center_size_form(self.prior_boxes_point_form)

        # Encode the variance.
        variance = torch.Tensor(self.offset_variances).to(matched_gts.device)
        delta_xy = (matched_gts[:, :2] - prior_boxes[:, :2]) / (prior_boxes[:, 2:] * variance[:2])
        delta_wh = torch.log(matched_gts[:, 2:] / prior_boxes[:, 2:]) / variance[2:]
        delta = torch.cat([delta_xy, delta_wh], dim=1)
        return delta

    def encode(self, gt_boxes, gt_labels):
        """Encode ground truth boxes and labels for training.

        Args:
            gt_boxes (tensor): Ground truth bounding boxes in point form, [x_min, y_min, x_max, y_max].
            gt_labels (tensor): Ground truth labels for each bounding box.

        P is the number of prior boxes.
        C is the number of classes excluding background.

        Returns:
            matched_gts (tensor): (P, 4) Ground truth boxes matched to each prior box.
            loc_targets (tensor): (P, 4) Location targets for each anchor box.
            cls_targets (tensor): (P, C+1) Class targets for each anchor box.
        """
        # Priors are anchor boxes in [x_min, y_min, x_min, y_min] format with relative coordinate.
        # Ground truth are bounding boxes in [x_min, y_min, x_max, y_max] format with relative coordinate.
        # Match each ground truth box to the anchor box with the best jaccard overlap
        matched_gts, matched_labels = self._match(gt_boxes, gt_labels)

        # Localization is defined by the offset between the matched prior box and the ground truth
        # box, i.e. delta(cx, cy, w, h). We use the offset variance to normalize the offset.
        loc_targets = self._encode_loc(matched_gts)

        # Convert matched labels to one-hot encoding.
        cls_targets = F.one_hot(matched_labels.to(torch.int64), num_classes=self.num_classes).float()

        return matched_gts, matched_labels, loc_targets, cls_targets

    def encode_batch(self, batch_gt_boxes, batch_gt_labels):
        """Encode batch ground truth boxes and labels for training.

        Args:
            batch_gt_boxes (tensor): Tensor of shape (B, N, 4) in point form, [x_min, y_min, x_max, y_max].
            batch_gt_labels (tensor): Tensor of shape (B, N) contains label indices.

        B is the batch size.
        N is the number of objects in each sample.

        Returns:
            batch_matched_gts: (B, P, 4) Batch ground truth boxes matched to each prior box.
            batch_loc_targets: Batch location targets for each anchor box.
            batch_cls_targets: Batch class targets for each anchor box.
        """
        batch_size = batch_gt_boxes.size(0)
        batch_matched_gts = []
        batch_matched_labels = []
        batch_loc_targets = []
        batch_cls_targets = []
        for i in range(batch_size):
            gt_boxes = batch_gt_boxes[i]
            gt_labels = batch_gt_labels[i]
            non_background = gt_labels > 0 # Remove padded values.
            matched_gts, matched_labels, loc_targets, cls_targets = self.encode(gt_boxes[non_background], gt_labels[non_background])
            batch_matched_gts.append(matched_gts)
            batch_matched_labels.append(matched_labels)
            batch_loc_targets.append(loc_targets)
            batch_cls_targets.append(cls_targets)
        batch_matched_gts = torch.stack(batch_matched_gts, dim=0)
        batch_matched_labels = torch.stack(batch_matched_labels, dim=0)
        batch_loc_targets = torch.stack(batch_loc_targets, dim=0)
        batch_cls_targets = torch.stack(batch_cls_targets, dim=0)
        return batch_matched_gts, batch_matched_labels, batch_loc_targets, batch_cls_targets

    def decode(self, loc):
        """Decode localization offsets back to relative coordinates.

        Args:
            loc (tensor): (P, 4) Localization offsets for each prior box.

        Returns:
            boxes (tensor): (P, 4) bounding boxes in point form (xmin, ymin, xmax, ymax).
        """
        # Offsets are computed in center size form.
        priors = center_size_form(self.prior_boxes_point_form)
        delta_xy = loc[:, :2]
        delta_wh = loc[:, 2:]
        # Decode the center size form using offsets and variances.
        variance = torch.Tensor(self.offset_variances).to(loc.device)
        boxes = torch.cat([
            delta_xy * variance[:2] * priors[:, 2:] + priors[:, :2],
            torch.exp(delta_wh * variance[2:]) * priors[:, 2:]
        ], dim=1)

        # Convert boxes to point form.
        return point_form(boxes)
