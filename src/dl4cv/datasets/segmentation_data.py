from pathlib import Path

import torch
import torchvision as T
import albumentations as A
import numpy as np
from PIL import Image
import cv2
from omegaconf import DictConfig
from torchvision import transforms
from dl4cv.utils.technical_utils import load_obj

class PH2(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, train=True, indices=None):
        super().__init__()
        self.cfg = cfg
        self.ToTensor = transforms.ToTensor()
        self.train = train
        self.train_trnsfrms = A.Compose(
            [
                load_obj(aug.class_name)(**aug.params)
                if aug.params
                else load_obj(aug.class_name)()
                for aug in self.cfg.augmentation.train
            ]
        )
        self.test_trnsfrms = A.Compose(
            [
                load_obj(aug.class_name)(**aug.params)
                if aug.params
                else load_obj(aug.class_name)()
                for aug in self.cfg.augmentation.test
            ]
        )

        self.sample_dirs = [
            x for x in Path(self.cfg.datamodule.params.path).iterdir() if x.is_dir()
        ]
        self.sample_dirs = [self.sample_dirs[i] for i in indices]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        image_path = (
            f"{sample_dir}/{sample_dir.name}_Dermoscopic_Image/{sample_dir.name}.bmp"
        )
        target = f"{sample_dir}/{sample_dir.name}_lesion/{sample_dir.name}_lesion.bmp"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
        if self.train:
            transformed = self.train_trnsfrms(image=img, mask=mask)
            img =  self.ToTensor(transformed["image"])
            mask = self.ToTensor(transformed["mask"]).squeeze()
        else:
            transformed = self.test_trnsfrms(image=img, mask=mask)
            img =  self.ToTensor(transformed["image"])
            mask = self.ToTensor(transformed["mask"]).squeeze()
        return img, mask


class DRIVE(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, train=True, indices=None):
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.ToTensor = transforms.ToTensor()
        self.train_trnsfrms = A.Compose(
            [
                load_obj(aug.class_name)(**aug.params)
                if aug.params
                else load_obj(aug.class_name)()
                for aug in self.cfg.augmentation.train
            ]
        )
        self.test_trnsfrms = A.Compose(
            [
                load_obj(aug.class_name)(**aug.params)
                if aug.params
                else load_obj(aug.class_name)()
                for aug in self.cfg.augmentation.test
            ]
        )

        path = Path(self.cfg.datamodule.params.path) / "training"

        self.sample_images = [x for x in (path / "images").iterdir()]
        self.sample_images = [self.sample_images[i] for i in indices]
        self.sample_masks = [x for x in (path / "1st_manual").iterdir()]
        self.sample_masks = [self.sample_masks[i] for i in indices]

    def __len__(self):
        assert len(self.sample_images) == len(self.sample_masks)
        return len(self.sample_images)

    def __getitem__(self, idx):
        image_path = self.sample_images[idx]
        target = self.sample_masks[idx]
        img = Image.open(str(image_path))
        mask = Image.open(str(target))
        img = np.array(img)
        mask = np.array(mask) / 255
        if self.train:
            transformed = self.train_trnsfrms(image=img, mask=mask)
            img = self.ToTensor(transformed["image"])
            mask = self.ToTensor(transformed["mask"]).squeeze()
        else:
            transformed = self.test_trnsfrms(image=img, mask=mask)
            img = self.ToTensor(transformed["image"])
            mask = self.ToTensor(transformed["mask"]).squeeze()
        return img, mask


def build_drive(cfg: DictConfig):
    indices = np.random.permutation(np.arange(20))
    train = DRIVE(cfg, indices=indices[: cfg.datamodule.params.split[0]])
    val = DRIVE(
        cfg,
        train=False,
        indices=indices[
            cfg.datamodule.params.split[0] : cfg.datamodule.params.split[0]
            + cfg.datamodule.params.split[1]
        ],
    )
    test = DRIVE(
        cfg,
        train=False,
        indices=indices[
            cfg.datamodule.params.split[0] + cfg.datamodule.params.split[1] :
        ],
    )
    return train, val, test


def build_ph2(cfg: DictConfig):
    indices = np.random.permutation(np.arange(200))
    train = PH2(cfg, indices=indices[: cfg.datamodule.params.split[0]])
    val = PH2(
        cfg,
        train=False,
        indices=indices[
            cfg.datamodule.params.split[0] : cfg.datamodule.params.split[0]
            + cfg.datamodule.params.split[1]
        ],
    )
    test = PH2(
        cfg,
        train=False,
        indices=indices[
            cfg.datamodule.params.split[0] + cfg.datamodule.params.split[1] :
        ],
    )
    return train, val, test
