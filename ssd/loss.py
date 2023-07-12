import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes + 1
        self.threshold = 0.5
        self.neg_pos_ratio = 3
        self.variance = [0.1, 0.2]

    def forward(self, loc_preds, cls_preds, loc_targets, cls_targets):
        """
        """
        loc_loss = F.smooth_l1_loss(loc_preds, loc_targets, reduction='none')

        return 0
