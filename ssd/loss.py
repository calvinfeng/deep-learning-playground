import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes=21):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = 0.5
        self.neg_pos_ratio = 3
        self.variance = [0.1, 0.2]

    def forward(self, loc_preds, cls_preds, loc_targets, cls_targets):
        """
        """
        not_background = cls_targets[:, :, 0] != 1
        is_background = cls_targets[:, :, 0] == 1

        # Encoder already takes care of encoding offsets between default boxes and ground truth boxes.
        # This smooth L1 loss will satisfy equation (2) on SSD paper.
        loc_loss = F.smooth_l1_loss(loc_preds[not_background], loc_targets[not_background],
                                    reduction='sum', size_average=False)

        # Perform hard negative mining:
        # First find number of non-background targets, aka positive, aka number of matched default boxes
        # as mentioned in the paper for each image in the batch.
        num_pos = not_background.sum(dim=1).unsqueeze(1) # (B, 1)
        max_neg = self.neg_pos_ratio * num_pos # 3 neg to 1 pos.

        B = cls_preds.size(0) # Batch size
        P = cls_preds.size(1) # Number of priors or default boxes

        # Find the loss for all negative examples, and sort them in descending order.
        # Loss is minimized when cls_preds predict 1 for background class, i.e. we want cls_preds to
        # predict 1 for background class for all negative examples. The hard negatives are the ones
        # that predict << 1 for background class.
        neg_conf_loss = -torch.log(F.softmax(cls_preds, dim=2)[:, :, 0])
        neg_conf_loss[not_background] = 0 # Ignore positive examples.
        _, neg_indices = torch.sort(neg_conf_loss, dim=1, descending=True)

        """PyTorch Trick Warning:
        We want to keep only the top max_neg. We will create a mask with values
        [   0      1     2   , ... max_neg, max_neg + 1, ...]
            [True, True, True, ..., False, False, False],
            [True, True, True, ..., False, False, False],
            ...
        ]
        Then we will re-arrange the mask based on the neg indices with respect to the original
        cls_preds tensor. Then we can directly apply the mask to the cls_preds tensor to keep
        the top max_neg negative examples.
        """
        neg_keep = torch.arange(P).unsqueeze(0).expand(B, P) < max_neg
        neg_keep = neg_keep.scatter(1, neg_indices, neg_keep)

        pdb.set_trace()
        return 0
