import torch
from torch import nn
from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ArtVisionModel(nn.Module):

    def __init__(self, n_classes, n_art_img_features=9):
        super(ArtVisionModel, self).__init__()

        # Pretrained ConvNeXt
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

        # Features
        self.features = self.convnext.features

        # CNBlock pooling layer
        self.cnb_pooling = nn.Sequential(
            self.convnext.avgpool,
            self.convnext.classifier[0],
            nn.Flatten()
        )

        # Colour features
        self.col_features = nn.Sequential(
            nn.Linear(
                in_features=n_art_img_features,
                out_features=64,
                bias=True
            ),
            nn.Flatten()
        )

        # Classifier portion
        self.classifier = nn.Sequential(
            #             self.convnext.classifier[0],
            #             nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=1024 + 64,
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

    def forward(self, img, img_feat):
        # Features
        img = self.features(img)

        # Pooling
        img = self.cnb_pooling(img)

        # Colour features
        img_feat = self.col_features(img_feat)

        x = torch.cat((img, img_feat), dim=1)

        # Classifier
        x = self.classifier(x)

        return x

