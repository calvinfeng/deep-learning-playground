import torch
from typing import Tuple
import numpy as np
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
        ) = self.populate_heatmaps(
            torch.stack([center_u, center_v, widths, heights], dim=1),
            gt_labels,
            x_stds,
            y_stds,
            center_mask,
            center_cls_heatmap,
            center_reg_heatmap,
        )
        return center_mask, center_cls_heatmap, center_reg_heatmap

    def populate_heatmaps(
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
            boxes_uv (torch.Tensor): List of boxes in feature map uv coordinate
            x_stds (torch.Tensor): Gaussian standard deivation for x-axis which is u
            y_stds (torch.Tensor): Gaussian standard deviation for y-axis which is v
            center_mask (torch.Tensor): Classification & regression loss mask
            center_cls_heatmap (torch.Tensor): Classification keypoint heatmap
            center_reg_heatmap (torch.Tensor): Regression keypoint heatmap

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
                center_reg_heatmap[0, int(vf), int(uf)] = u - uf
                center_reg_heatmap[1, int(vf), int(uf)] = v - vf
                center_reg_heatmap[2, int(vf), int(uf)] = width
                center_reg_heatmap[3, int(vf), int(uf)] = height

        return center_mask, center_cls_heatmap, center_reg_heatmap
