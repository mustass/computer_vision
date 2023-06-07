import torch
import torch.nn as nn
from torchvision import models
from torch import nn
import torch
from omegaconf import DictConfig


class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ResNet, self).__init__()
        self.params = cfg.model.params
        # First conv layers needs to output the desired number of features.
        conv_layers = [
            nn.Conv2d(
                self.params.n_in,
                self.params.n_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        ]
        for i in range(self.params.num_res_blocks):
            conv_layers.append(ResNetBlock(self.params.n_features))
        self.res_blocks = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * self.params.n_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, self.params.num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.res_blocks(x)
        # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class ResNet50ImgNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ResNet50ImgNet, self).__init__()
        self.params = cfg.model.params
        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, self.params.num_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        out = self.classifier(representations)
        return out


class ResNet18ImgNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ResNet18ImgNet, self).__init__()
        self.params = cfg.model.params
        # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, self.params.num_classes)

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        out = self.classifier(representations)
        return out

class ResNet18ImgNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ResNet18ImgNet, self).__init__()
        self.params = cfg.model.params
       # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, self.params.num_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        if self.training:
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        else:
            representations = self.feature_extractor(x).flatten(1)
        out = self.classifier(representations)
        return out