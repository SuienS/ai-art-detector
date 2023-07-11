import torch
from torch import nn
from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionBlock(nn.Module):
    """
    Implements Squeeze-and-Excitation Networks
    https://doi.org/10.1109/CVPR.2018.00745
    """

    def __init__(self, tot_channels, reduction):
        super(AttentionBlock, self).__init__()

        # Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # SE FCN
        self.fcn = nn.Sequential(
            nn.Linear(tot_channels, tot_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(tot_channels // reduction, tot_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.cat(x, dim=1)
        n_batches, n_channels, _, _ = x.size()
        y = self.avgpool(x).view(n_batches, n_channels)
        y = self.fcn(y).view(n_batches, n_channels, 1, 1)

        return x * y.expand_as(x)  # Per channel multiplication


class ArtVisionModel(nn.Module):

    def __init__(self, n_classes, n_art_img_features=9):
        super(ArtVisionModel, self).__init__()

        # Pretrained ConvNeXt
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

        self.features_low_CNB = self.convnext.features[:4]

        # Features - Low level
        self.features_low = self.convnext.features[4]

        # Features - Mid level
        self.features_mid_CNB = self.convnext.features[5]

        # Features - High level
        self.features_mid = self.convnext.features[6]

        # Features - Top level
        self.features_high_CNB = self.convnext.features[7]

        self.low_feat_pooling = self.convnext.features[6]

        # Attention block
        self.attention_block = AttentionBlock(3072, 16)

        # Classifier portion
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.LayerNorm([3072, 1, 1]),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(
                in_features=3072,
                out_features=512,
                bias=True
            ),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=512,
                out_features=n_classes,
                bias=True
            )
        )

    def forward(self, img):
        # Features
        x = self.features_low_CNB(img)

        x_low = self.features_low(x)

        x = self.features_mid_CNB(x_low)

        x_mid = self.features_mid(x)

        x_high = self.features_high_CNB(x_mid)

        x_low = self.low_feat_pooling(x_low)

        x = self.attention_block((x_high, x_mid, x_low))

        # Classifier
        x = self.classifier(x)

        return x
