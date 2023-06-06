from pathlib import Path

import pandas as pd
import torch
import torchvision as T
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import random_split
from dl4cv.utils.technical_utils import load_obj


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train = T.datasets.CIFAR10(
            root=str(Path(get_original_cwd()) / "data"),
            download=True,
            train=True,
            transform=T.transforms.ToTensor(),
        )
        self.test = T.datasets.CIFAR10(
            root=str(Path(get_original_cwd()) / "data"),
            download=True,
            train=False,
            transform=T.transforms.ToTensor(),
        )

class HotDogNotHotDog(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_trnsfrms = transforms.Compose([ load_obj(aug.class_name)(**aug.params) if aug.params else load_obj(aug.class_name)() for aug in self.cfg.augmentation.train])
        self.test_trnsfrms = transforms.Compose([ load_obj(aug.class_name)(**aug.params) if aug.params else load_obj(aug.class_name)() for aug in self.cfg.augmentation.test])
        
        self.train = T.datasets.ImageFolder(
            root=self.cfg.datamodule.params.train_dir,
            transform=self.train_trnsfrms,
        )

        self.test = T.datasets.ImageFolder(
            root=self.cfg.datamodule.params.test_dir,
            transform=self.test_trnsfrms,
        )

        self.splits = random_split(
                self.test, self.cfg.datamodule.params.split
            )
        
        self.val = self.splits[0]
        self.test = self.splits[1]
