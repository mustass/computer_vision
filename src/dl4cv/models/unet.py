from torch import nn
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

class UNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.downsample0 = nn.Conv2d(64,64,2,stride=2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.downsample1 = nn.Conv2d(64,64,2,stride=2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.downsample2 = nn.Conv2d(64,64,2,stride=2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.downsample3 = nn.Conv2d(64,64,2,stride=2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(64,64,2,2)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64,64,2,2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64,64,2,2)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64,64,2,2)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        ae0 = F.relu(self.enc_conv0(x))
        e0 = self.downsample0(ae0)
        ae1 = F.relu(self.enc_conv1(e0))
        e1 = self.downsample1(ae1)
        ae2 = F.relu(self.enc_conv2(e1))
        e2 = self.downsample2(ae2)
        ae3 = F.relu(self.enc_conv3(e2))
        e3 = self.downsample3(ae3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        ad0 = torch.cat([self.upsample0(b), ae3], dim=1)
        d0 = F.relu(self.dec_conv0(ad0))
        ad1 = torch.cat([self.upsample1(d0), ae2], dim=1)
        d1 = F.relu(self.dec_conv1(ad1))
        ad2 = torch.cat([self.upsample2(d1), ae1], dim=1)
        d2 = F.relu(self.dec_conv2(ad2))
        ad3 = torch.cat([self.upsample3(d2), ae0], dim=1)
        d3 = self.dec_conv3(ad3)  # no activation
        return d3