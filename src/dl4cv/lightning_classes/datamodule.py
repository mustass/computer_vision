from typing import Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torch
from dl4cv.utils.technical_utils import load_obj


class CVDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg

    def setup(self, stage: Optional[str] = None, inference: Optional[bool] = False):
        self.inference = inference
        self.dataset = load_obj(self.config.datamodule.params.dataset)(self.config)
        
        self.splits = random_split(
                self.dataset.train, self.config.datamodule.params.split
            )
        
        self.train = self.splits[0]
        self.val = self.splits[1]
        self.test = self.dataset.test

    def train_dataloader(self):
        assert not self.inference, "In inference mode, there is no train_dataloader."
        return DataLoader(
            self.train,
            #ççcollate_fn=FloatCollator,
            batch_size=self.config.datamodule.params.batch_size,
            num_workers=self.config.datamodule.params.num_workers,
            pin_memory=self.config.datamodule.params.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        assert not self.inference, "In inference mode, there is no val_dataloader."
        return DataLoader(
            self.val,
            #collate_fn=FloatCollator,
            batch_size=self.config.datamodule.params.batch_size,
            num_workers=self.config.datamodule.params.num_workers,
            pin_memory=self.config.datamodule.params.pin_memory,
        )

    def test_dataloader(self):
        assert not self.inference, "In inference mode, there is no test_dataloader."
        return DataLoader(
            self.test,
            #collate_fn=FloatCollator,
            batch_size=self.config.datamodule.params.batch_size,
            num_workers=self.config.datamodule.params.num_workers,
            pin_memory=self.config.datamodule.params.pin_memory,
        )

