import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ssd.config import DEFAULT_NUM_BOXES, DEFAULT_FEATURE_MAP_NAMES


class SingleShotDetector(nn.Module):
    def __init__(self, num_classes=21,
                       feature_map_names=DEFAULT_FEATURE_MAP_NAMES,
                       feature_map_num_boxes=DEFAULT_NUM_BOXES):
        super(SingleShotDetector, self).__init__()
        self.num_classes = num_classes
        self.base = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        # Change stride from 2 to 1 to reduce down-sampling.
        self.base[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # Define auxiliary layers
        self.aux_convs = nn.ModuleDict({
            "Conv6": nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            "Conv7": nn.Conv2d(1024, 1024, kernel_size=1),
            "Conv8_1": nn.Conv2d(1024, 256, kernel_size=1),
            "Conv8_2":  nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            "Conv9_1": nn.Conv2d(512, 128, kernel_size=1),
            "Conv9_2": nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            "Conv10_1": nn.Conv2d(256, 128, kernel_size=1),
            "Conv10_2": nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            "Conv11_1": nn.Conv2d(256, 128, kernel_size=1),
            "Conv11_2": nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        })
        # This is also known as k in the paper. It is the number of boxes per feature map location.
        self.feature_map_num_boxes = feature_map_num_boxes
        # Use a list to ensure ordering of feature map names as we collect them for loss calculation.
        self.feature_map_names = feature_map_names

        # Each location has 4 offsets and num_classes.
        self.loc_conv = nn.ModuleDict({
            "Conv4_3": nn.Conv2d(512, self.feature_map_num_boxes['Conv4_3'] * 4, kernel_size=3, padding=1),
            "Conv7": nn.Conv2d(1024, self.feature_map_num_boxes['Conv7'] * 4, kernel_size=3, padding=1),
            "Conv8_2": nn.Conv2d(512, self.feature_map_num_boxes['Conv8_2'] * 4, kernel_size=3, padding=1),
            "Conv9_2": nn.Conv2d(256, self.feature_map_num_boxes['Conv9_2'] * 4, kernel_size=3, padding=1),
            "Conv10_2": nn.Conv2d(256, self.feature_map_num_boxes['Conv10_2'] * 4, kernel_size=3, padding=1),
            "Conv11_2": nn.Conv2d(256, self.feature_map_num_boxes['Conv11_2'] * 4, kernel_size=3, padding=1)
        })
        self.cl_conv = nn.ModuleDict({
            "Conv4_3": nn.Conv2d(512, self.feature_map_num_boxes['Conv4_3'] * self.num_classes, kernel_size=3, padding=1),
            "Conv7": nn.Conv2d(1024, self.feature_map_num_boxes['Conv7'] * self.num_classes, kernel_size=3, padding=1),
            "Conv8_2": nn.Conv2d(512, self.feature_map_num_boxes['Conv8_2'] * self.num_classes, kernel_size=3, padding=1),
            "Conv9_2": nn.Conv2d(256, self.feature_map_num_boxes['Conv9_2'] * self.num_classes, kernel_size=3, padding=1),
            "Conv10_2": nn.Conv2d(256, self.feature_map_num_boxes['Conv10_2'] * self.num_classes, kernel_size=3, padding=1),
            "Conv11_2": nn.Conv2d(256, self.feature_map_num_boxes['Conv11_2'] * self.num_classes, kernel_size=3, padding=1)
        })

    def forward(self, x):
        # Base network
        feature_maps = dict()
        for i in range(23):
            x = self.base[i](x)
        feature_maps["Conv4_3"] = x
        for i in range(23, len(self.base)):
            x = self.base[i](x)

        # Extra feature layers
        for layer_name, layer in self.aux_convs.items():
            x = F.relu(layer(x), inplace=True)
            if layer_name in self.feature_map_num_boxes:
                feature_maps[layer_name] = x

        # Predict localization and class scores
        # After permutation, the tensors need to be contiguous for view() to work.
        loc_preds, cls_preds = [], []
        for fm_name in self.feature_map_names:
            loc_pred = self.loc_conv[fm_name](feature_maps[fm_name])
            cls_pred = self.cl_conv[fm_name](feature_maps[fm_name])
            # PyTorch is channel-first by convention. We need to reshape it to channel-last.
            # Then flatten them to (B, N, 4) and (B, N, num_classes) respectively.
            loc_preds.append(loc_pred.permute(0, 2, 3, 1).contiguous().view(loc_pred.size(0), -1, 4))
            cls_preds.append(cls_pred.permute(0, 2, 3, 1).contiguous().view(cls_pred.size(0), -1, self.num_classes))

        # Optionally reshape. This is dependent on how I implement the loss.
        # loc_preds = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        # cls_preds = torch.cat([o.view(o.size(0), -1) for o in cls_preds], 1)
        return torch.cat(loc_preds, dim=1), torch.cat(cls_preds, dim=1)
