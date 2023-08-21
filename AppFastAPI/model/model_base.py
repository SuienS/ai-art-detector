import torch
from torch import nn
from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ArtVisionModelBase(nn.Module):
    """
    Base ConvNeXt Model
    """

    def __init__(self, n_classes):
        super(ArtVisionModelBase, self).__init__()

        # Pretrained ConvNeXt
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

        # Features
        self.features = self.convnext.features

        # Avg pooling layer
        self.avgpool = self.convnext.avgpool

        # Classifier portion
        self.classifier = nn.Sequential(
            self.convnext.classifier[0],
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=1024,
                out_features=256,
                bias=True
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=256,
                out_features=n_classes,
                bias=True
            )
        )

    def forward(self, x):
        # Features
        x = self.features(x)

        # Pooling
        x = self.avgpool(x)

        # Classifier
        x = self.classifier(x)

        return x
