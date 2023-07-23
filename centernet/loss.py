import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FocalLoss(nn.Module):
    def __init__(self, num_classes=21, alpha=2.0, beta=4.0):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, center_mask, center_cls_targets, center_reg_targets,
                      center_cls_preds, center_reg_preds):
        cls_center_mask = center_mask.unsqueeze(1).repeat(1, self.num_classes, 1, 1)
        cls_loss = self._pixel_wise_focal_loss(center_cls_targets, center_cls_preds, cls_center_mask)

        reg_center_mask = center_mask.unsqueeze(1).repeat(1, 4, 1, 1)
        reg_loss = self._squared_loss(center_reg_targets, center_reg_preds, reg_center_mask)
        return cls_loss + reg_loss

    def _pixel_wise_focal_loss(self, y_true, y_pred, mask):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        mask = mask.view(-1)

        y_pred_sigmoid = F.sigmoid(y_pred)
        y_pred_log_sigmoid = F.logsigmoid(y_pred)

        # Positive loss - when ground truth values are 1's.
        positive_loss = mask * torch.pow(1.0 - y_pred_sigmoid, self.alpha) * y_pred_log_sigmoid

        # Negative loss - when ground truth values are 0's, which may end up with log(0).
        # We need to apply a clip.
        neg_y_pred_sigmoid = torch.clamp(1.0 - y_pred_sigmoid, min=1e-8, max=1.0)
        neg_y_pred_log_sigmoid = torch.log(neg_y_pred_sigmoid)
        negative_loss = (1.0 - mask) * torch.pow(1 - y_true, self.beta) * \
            torch.pow(y_pred_sigmoid, self.alpha) * neg_y_pred_log_sigmoid

        cls_loss = positive_loss + negative_loss
        cls_loss = -1 * cls_loss.sum() / cls_loss.numel()
        return cls_loss

    def _squared_loss(self, y_true, y_pred, mask):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        mask = mask.view(-1)
        reg_loss = mask * torch.pow(y_true - y_pred, 2)
        reg_loss = reg_loss.sum() / reg_loss.numel()
        return reg_loss
