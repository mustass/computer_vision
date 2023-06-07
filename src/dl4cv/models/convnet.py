import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ConvNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ConvNet, self).__init__()
        self.params = cfg.model.params

        self.conv1 = nn.Conv2d(3, 8, 5)  # (B,3,224,224) -> (B,8,220,220)
        self.pool = nn.MaxPool2d(2, 2)  # (B,8,220,220) -> (B,8,110,110)
        self.conv2 = nn.Conv2d(8, 16, 5)  # (B,8,110,110) -> (B,16,106,106)
        # MaxPool # (B,16,106,106) -> (B,16,53,53)
        self.conv3 = nn.Conv2d(16, 32, 5)  # (B,16,53,53) -> (B,32,49,49)
        self.fc1 = nn.Linear(32 * 24 * 24, 1000)
        self.fc2 = nn.Linear(1000, self.params.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
