from typing import Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torch
from dl4cv.utils.technical_utils import load_obj
from torchvision.datasets import MNIST
from torchvision import transforms
import albumentations as A
from pathlib import Path
from hydra.utils import get_original_cwd

class CVDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None, inference: Optional[bool] = False):
        self.inference = inference
        self.dataset = load_obj(self.cfg.datamodule.params.dataset)(self.cfg)

        self.train = self.dataset.train
        self.val = self.dataset.val
        self.test = self.dataset.test

    def train_dataloader(self):
        assert not self.inference, "In inference mode, there is no train_dataloader."
        return DataLoader(
            self.train,
            # ççcollate_fn=FloatCollator,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        assert not self.inference, "In inference mode, there is no val_dataloader."
        return DataLoader(
            self.val,
            # collate_fn=FloatCollator,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self):
        assert not self.inference, "In inference mode, there is no test_dataloader."
        return DataLoader(
            self.test,
            # collate_fn=FloatCollator,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )


class SegmentDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None, inference: Optional[bool] = False):
        self.inference = inference
        self.train, self.val, self.test = load_obj(self.cfg.datamodule.params.builder)(
            self.cfg
        )

    def train_dataloader(self):
        assert not self.inference, "In inference mode, there is no train_dataloader."
        return DataLoader(
            self.train,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        assert not self.inference, "In inference mode, there is no val_dataloader."
        return DataLoader(
            self.val,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self):
        assert not self.inference, "In inference mode, there is no test_dataloader."
        return DataLoader(
            self.test,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )

class MNISTDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.datamodule.params.batch_size
        self.num_workers = self.cfg.datamodule.params.num_workers
        self.data_dir = str(Path(get_original_cwd()) / self.cfg.datamodule.params.data_dir) 

        self.transform = transforms.Compose(
            [   transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        # self.transform = A.Compose(
        #     [
        #         load_obj(aug.class_name)(**aug.params)
        #         if aug.params
        #         else load_obj(aug.class_name)()
        #         for aug in self.cfg.augmentation.train
        #     ]
        # )
    
    
    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
