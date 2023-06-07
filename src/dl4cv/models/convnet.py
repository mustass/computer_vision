import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ConvNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ConvNet, self).__init__()
        self.params = cfg.model.params

        self.dropout=nn.Dropout(0.25)
        self.conv1 = nn.Conv2d(3, 32, 3,padding=1) # (B,3,224,224) -> (B,64,224,224)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # (B,64,224,224) -> (B,64,112,112)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # (B,64,112,112) -> (B,128,112,112)
        self.bn2 = nn.BatchNorm2d(64)
        # MaxPool # (B,128,112,112) -> (B,128,56,56)
        self.conv3 = nn.Conv2d(64,128,3, padding=1) # (B,128,56,56) -> (B,256,56,56)
        self.bn3 = nn.BatchNorm2d(128)
        # MaxPool # (B,256,56,56) -> (B,256,28,28)
        self.conv4 = nn.Conv2d(128,256,3, padding=1) # (B,256,28,28) -> (B,512,28,28)
        self.bn4 = nn.BatchNorm2d(256)
        # MaxPool # (B,512,28,28) -> (B,512,14,14)
        self.conv5 = nn.Conv2d(256,256,3, padding=1) # (B,512,14,14) -> (B,512,14,14)
        self.bn5 = nn.BatchNorm2d(256)
        # MaxPool # (B,512,14,14) -> (B,512,7,7)
        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, self.params.num_classes)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.dropout(x)
        x = self.pool(self.bn5(F.relu(self.conv5(x))))
        x = self.dropout(x)
        x = x.view(-1, 256 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x