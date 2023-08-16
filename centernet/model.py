import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, rates):
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.non_dilated_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=rate, padding=rate),
                nn.ReLU(inplace=True),
            )
            for rate in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.conv_1x1 = nn.Conv2d(in_channels=256*(len(rates) + 2), out_channels=256, kernel_size=1)

    def forward(self, x):
        branches = [self.non_dilated_conv(x)]
        for _, dilated_conv in enumerate(self.dilated_convs):
            branches.append(dilated_conv(x))

        # Pool the entire spatial extent of the input down to a single point (1, 1).
        # This effectively calculates the average of each feature channel across the entire spatial
        # extent of the input, capturing the global context of the input.
        pool = self.global_pool(x)
        h, w = x.size(2), x.size(3)
        branches.append(F.interpolate(pool, size=(h, w), mode='bilinear', align_corners=True))

        # The final 1x1 conv layer compresses the channel dimension back to 256.
        x = torch.cat(branches, dim=1)
        x = self.conv_1x1(x)
        return x


class CenterNet(nn.Module):
    """To keep it a fair comparison to SSD, I will use VGG16 as the backbone."""
    def __init__(self, num_classes=21, use_dropout=False):
        super(CenterNet, self).__init__()
        self.num_classes = num_classes
        self.base = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
        # Change stride from 2 to 1 to reduce down-sampling. In case I want to use the full VGG.
        self.base[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.feature_extract = nn.Sequential(
            self.base[:23],
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
            AtrousSpatialPyramidPooling(rates=[1, 2, 4, 6, 8, 12, 18]),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout2d(p=0.5)
        # Classify each feature map location into one of the object classes or background.
        self.center_cls_branch = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1),
        )
        # Regress each feature map location into (dx, dy, w, h) for potential object.
        self.center_reg_branch = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1),
        )

    def forward(self, x):
        # The full VGG architecture has a down-sampling rate of 32. Similar to InceptionV2 Mixed_3b
        # layer, I capped the down-sampling rate to 8. I will use VGG up to Conv3_3.
        feature_map = self.feature_extract(x)
        if self.use_dropout:
            feature_map = self.dropout(feature_map)
        center_cls = self.center_cls_branch(feature_map)
        center_reg = self.center_reg_branch(feature_map)
        return center_cls, center_reg


if __name__ == '__main__':
    model = CenterNet()
    x = torch.randn(1, 3, 300, 300)
    cls, reg = model(x)
    print(cls.shape)
    print(reg.shape)
