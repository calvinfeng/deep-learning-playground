from typing import List
import torch.nn as nn

from swin.transformer import StageModule
from centernet.model import AtrousSpatialPyramidPooling


class SwinTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim,
        layers,
        heads: List[int],
        channels=3,
        head_dim=32,
        window_size=7,
        downscaling_factors=(2, 2, 2, 1),
        relative_pos_embedding=True
    ):
        super().__init__()
        self.stage1 = StageModule(
            in_channels=channels,
            hidden_dim=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dim=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dim=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class TransformerCenterNet(nn.Module):
    def __init__(self, num_classes=21, use_dropout=False):
        super(TransformerCenterNet, self).__init__()
        self.num_classes = num_classes
        self.feature_extract = nn.Sequential(
            SwinTransformerBackbone(
                hidden_dim=96,
                layers=[2, 2, 6, 2],
                heads=[3, 6, 12, 24],
                channels=3,
                head_dim=32,
                window_size=7,
                downscaling_factors=(2, 2, 2, 1),
                relative_pos_embedding=True),
            nn.Conv2d(in_channels=96*8, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
            AtrousSpatialPyramidPooling(rates=[6, 12, 18, 24]),
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
