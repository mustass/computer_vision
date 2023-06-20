from typing import Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torch
from dl4cv.utils.technical_utils import load_obj
from dl4cv.datasets.taco_data import taco_train_collate_fn, taco_val_test_collate_fn


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
        self.train, self.val, self.test = load_obj(self.cfg.datamodule.params.builder)(
            self.cfg
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )


class ObjectDetectDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None, inference: Optional[bool] = False):
        self.train, self.val, self.test, self.inference = load_obj(self.cfg.datamodule.params.builder)(
            self.cfg
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            collate_fn=taco_train_collate_fn,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            collate_fn=taco_val_test_collate_fn,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            collate_fn=taco_val_test_collate_fn,
            batch_size=self.cfg.datamodule.params.batch_size,
            num_workers=self.cfg.datamodule.params.num_workers,
            pin_memory=self.cfg.datamodule.params.pin_memory,
            drop_last=True,
        )
