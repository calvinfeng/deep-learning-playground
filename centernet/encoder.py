import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

import pdb


class KeypointHeatmapEncoder:
    def __init__(self, heatmap_shape: Tuple[int, int], num_classes: int=21, std: float=0.05):
        self.heatmap_shape = heatmap_shape
        self.num_classes = num_classes
        self.std = std
        # Precompute indices grid with numpy
        H, W = self.heatmap_shape
        u_indices = np.arange(W, dtype=np.float32)
        v_indices = np.arange(H, dtype=np.float32)
        u_indices, v_indices = np.meshgrid(u_indices, v_indices)
        self.u_indices = torch.from_numpy(u_indices)
        self.v_indices = torch.from_numpy(v_indices)

    def batch_encode(self, batch_gt):
        batch_size = batch_gt.size(0)
        batch_gt_boxes, batch_gt_labels = batch_gt[:, :, :4], batch_gt[:, :, 4]
        batch_center_mask, batch_center_cls_heatmap, batch_center_reg_heatmap = [], [], []
        for i in range(batch_size):
            gt_boxes = batch_gt_boxes[i]
            gt_labels = batch_gt_labels[i]
            non_background = gt_labels > 0 # Remove padded values.
            center_mask, center_cls_heatmap, center_reg_heatmap = self.encode(gt_boxes[non_background],
                                                                              gt_labels[non_background])
            batch_center_mask.append(center_mask)
            batch_center_cls_heatmap.append(center_cls_heatmap)
            batch_center_reg_heatmap.append(center_reg_heatmap)
        batch_center_mask = torch.stack(batch_center_mask, dim=0)
        batch_center_cls_heatmap = torch.stack(batch_center_cls_heatmap, dim=0)
        batch_center_reg_heatmap = torch.stack(batch_center_reg_heatmap, dim=0)
        return batch_center_mask, batch_center_cls_heatmap, batch_center_reg_heatmap

    def encode(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        H, W = self.heatmap_shape
        center_cls_heatmap = torch.zeros((self.num_classes, H, W))
        center_reg_heatmap = torch.zeros((4, H, W))
        center_mask = torch.zeros((H, W))

        # Convert point-form to center-size form.
        widths = (gt_boxes[:, 2] - gt_boxes[:, 0]) * W
        heights = (gt_boxes[:, 3] - gt_boxes[:, 1]) * H
        center_u = (gt_boxes[:, 0] + gt_boxes[:, 2]) * W / 2 # Size w.r.t feature map
        center_v = (gt_boxes[:, 1] + gt_boxes[:, 3]) * H / 2 # Size w.r.t feature map
        x_stds = widths * self.std
        y_stds = heights * self.std
        (
            center_mask,
            center_cls_heatmap,
            center_reg_heatmap,
        ) = self._populate_heatmaps(
            torch.stack([center_u, center_v, widths, heights], dim=1),
            gt_labels,
            x_stds,
            y_stds,
            center_mask,
            center_cls_heatmap,
            center_reg_heatmap,
        )
        return center_mask, center_cls_heatmap, center_reg_heatmap

    def _populate_heatmaps(
        self,
        boxes_uv: torch.Tensor,
        labels: torch.Tensor,
        x_stds: torch.Tensor,
        y_stds: torch.Tensor,
        center_mask: torch.Tensor,
        center_cls_heatmap: torch.Tensor,
        center_reg_heatmap: torch.Tensor,
    ):
        """Populate classification and regression keypoint heatmaps and mask.

        Args:
            boxes_uv (torch.Tensor): Shape(N, 4) List of boxes in feature map uv coordinate
            x_stds (torch.Tensor): Shape(N,) Gaussian standard deivation for x-axis which is u
            y_stds (torch.Tensor): Shape(N,) Gaussian standard deviation for y-axis which is v
            center_mask (torch.Tensor): Shape(H, W) Classification & regression loss mask
            center_cls_heatmap (torch.Tensor): Shape(num_classes, H, W) Classification keypoint heatmap
            center_reg_heatmap (torch.Tensor): Shape(4, H, W) Regression keypoint heatmap

        Returns:
            torch.Tensor: center_cls_heatmap
            torch.Tensor: center_reg_heatmap
            torch.Tensor: box_dimension_heatmap
            torch.Tensor: center_mask
        """
        H, W = self.heatmap_shape
        for box, label, x_std, y_std in zip(boxes_uv, labels, x_stds, y_stds):
            # In image coordinate, i.e. pixels
            u, v, width, height = box

            x_std = torch.max(torch.tensor([x_std, torch.finfo(torch.float32).eps]))
            y_std = torch.max(torch.tensor([y_std, torch.finfo(torch.float32).eps]))

            uf, vf = torch.floor(u), torch.floor(v)

            # Get the gaussian
            gaussian_xterm = torch.pow(self.u_indices - uf, 2) / ((2 * torch.pow(x_std, 2)))
            gaussian_yterm = torch.pow(self.v_indices - vf, 2) / ((2 * torch.pow(y_std, 2)))
            gaussian = torch.exp(-1 * (gaussian_xterm + gaussian_yterm))

            center_cls_heatmap[label, :, :] = torch.max(center_cls_heatmap[label, :, :], gaussian)

            if 0 <= uf < W and 0 <= vf < H:
                center_mask[int(vf), int(uf)] = 1.0
                # We don't regress relative height/width, but absolute height/width offset
                # w.r.t feature map dimensions. The offsets are generally < 1 because it's the
                # offset between true pixel value and floored pixel value.
                # However, height and width are relatives to keep the range between 0 and 1.
                center_reg_heatmap[0, int(vf), int(uf)] = u - uf # Offset from floored value
                center_reg_heatmap[1, int(vf), int(uf)] = v - vf # Offset from floored value
                center_reg_heatmap[2, int(vf), int(uf)] = width / W # Regress relative width
                center_reg_heatmap[3, int(vf), int(uf)] = height / H # Regress relative height

        return center_mask, center_cls_heatmap, center_reg_heatmap

    def batch_decode(self, batch_center_cls_heatmap, batch_center_reg_heatmap, num_detections=20):
        batch_size = batch_center_cls_heatmap.size(0)
        sig_heatmap = F.sigmoid(batch_center_cls_heatmap)
        maxpool_heatmap = F.max_pool2d(sig_heatmap, kernel_size=5, stride=1, padding=2)
        peak_mask = sig_heatmap == maxpool_heatmap
        peaks = maxpool_heatmap * peak_mask

        batch_keypoint_scores, batch_keypoint_labels = torch.max(peaks, dim=1)
        uv_indices = torch.stack([self.u_indices, self.v_indices], dim=-1)
        batch_uv_indices = uv_indices.expand(batch_size, -1, -1, -1)

        batch_labels = batch_keypoint_labels.view(batch_size, -1)
        batch_scores = batch_keypoint_scores.view(batch_size, -1)

        batch_center_offset = batch_center_reg_heatmap[:, :2, :, :].permute(0, 2, 3, 1).view(batch_size, -1, 2)
        batch_center_wh = batch_center_reg_heatmap[:, 2:, :, :].permute(0, 2, 3, 1).view(batch_size, -1, 2)

        H, W = self.heatmap_shape
        batch_uv_indices = batch_uv_indices.view(batch_size, -1, 2)

        batch_x_min = (batch_uv_indices[:, :, 0] + batch_center_offset[:, :, 0] - batch_center_wh[:, :, 0] * W / 2.0) / W
        batch_y_min = (batch_uv_indices[:, :, 1] + batch_center_offset[:, :, 1] - batch_center_wh[:, :, 1] * H / 2.0) / H
        batch_x_max = (batch_uv_indices[:, :, 0] + batch_center_offset[:, :, 0] + batch_center_wh[:, :, 0] * W / 2.0) / W
        batch_y_max = (batch_uv_indices[:, :, 1] + batch_center_offset[:, :, 1] + batch_center_wh[:, :, 1] * H / 2.0) / H

        batch_decoded_boxes = torch.stack([batch_x_min, batch_y_min, batch_x_max, batch_y_max], axis=2)
        indices_sorted_by_score = torch.argsort(batch_scores, descending=True, dim=1)

        batch_top_scores = batch_scores.gather(1, indices_sorted_by_score[:, :num_detections])
        batch_top_labels = batch_labels.gather(1, indices_sorted_by_score[:, :num_detections])
        # Indices need to be expanded to match the shape of batch_decoded_boxes
        batch_top_boxes = batch_decoded_boxes.gather(1, indices_sorted_by_score[:, :num_detections].unsqueeze(-1).expand(-1, -1, 4))

        return batch_top_boxes, batch_top_scores, batch_top_labels

    def decode(self, center_cls_heatmap, center_reg_heatmap, num_detections=20):
        """Decode keypoint heatmaps into detections.

        Args:
            center_cls_heatmap (torch.Tensor): (num_classes, H, W) Keypoint heatmap for classification.
            center_reg_heatmap (torch.Tensor): (4, H, W) Keypoint heatmap for regression.
            num_detections (int, optional): Number of detections to return. Defaults to 20.

        Returns:
            torch.Tensor: top_boxes
            torch.Tensor: top_scores
            torch.Tensor: top_labels
        """
        sig_heatmap = F.sigmoid(center_cls_heatmap)
        maxpool_heatmap = F.max_pool2d(sig_heatmap, kernel_size=5, stride=1, padding=2)
        peak_mask = sig_heatmap == maxpool_heatmap
        peaks = maxpool_heatmap * peak_mask

        keypoint_scores, keypoint_labels = torch.max(peaks, dim=0)
        uv_indices = torch.stack([self.u_indices, self.v_indices], dim=-1)

        labels = keypoint_labels.view(-1)
        scores = keypoint_scores.view(-1)

        # Regression heatmap is channel first. We need to permute before reshaping.
        center_offset = center_reg_heatmap[:2, :, :].permute(1, 2, 0).view(-1, 2)
        center_wh = center_reg_heatmap[2:, :, :].permute(1, 2, 0).view(-1, 2)

        H, W = self.heatmap_shape
        uv_indices = uv_indices.view(-1, 2)

        x_min = (uv_indices[:, 0] + center_offset[:, 0] - center_wh[:, 0] * W / 2.0) / W
        y_min = (uv_indices[:, 1] + center_offset[:, 1] - center_wh[:, 1] * H / 2.0) / H
        x_max = (uv_indices[:, 0] + center_offset[:, 0] + center_wh[:, 0] * W / 2.0) / W
        y_max = (uv_indices[:, 1] + center_offset[:, 1] + center_wh[:, 1] * H / 2.0) / H

        decoded_boxes = torch.stack([x_min, y_min, x_max, y_max], axis=1)

        indices_sorted_by_score = torch.argsort(scores, descending=True)

        top_scores = scores[indices_sorted_by_score[:num_detections]]
        top_labels = labels[indices_sorted_by_score[:num_detections]]
        top_boxes = decoded_boxes[indices_sorted_by_score[:num_detections]]

        return top_boxes, top_scores, top_labels
