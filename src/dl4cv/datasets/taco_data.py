import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ExifTags
import cv2
from omegaconf import DictConfig

from dl4cv.utils.technical_utils import load_obj
from dl4cv.utils.object_detect_utils import get_iou, fix_orientation

from pathlib import Path
import json
import pickle


class TACO(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.BACKGROUND_LABEL = -1

        self.img_size = self.cfg.datamodule.params.img_size

        self.transform = T.Compose(
            [
                load_obj(aug.class_name)(**aug.params)
                if aug.params
                else load_obj(aug.class_name)()
                for aug in self.cfg.augmentation.train
            ]
        )

        if split == "train":
            self.path = Path(self.cfg.datamodule.train.params.path)
            self.num_to_return = self.cfg.datamodule.train.params.num_to_return
        elif split == "val":
            self.path = Path(self.cfg.datamodule.val.params.path)
            self.num_to_return = self.cfg.datamodule.val.params.num_to_return
        elif split == "test":
            self.path = Path(self.cfg.datamodule.test.params.path)
            self.num_to_return = self.cfg.datamodule.test.params.num_to_return
        else:
            raise ValueError(f"Split {split} not supported.")

        self.annotations = self._load_annotations()
        self.regions = self._load_regions()

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        self.orientation = orientation
        
    def _load_annotations(self):
        file = self.path / f'{self.split}_annotations.json'
        with open(file, "r") as f:
                annotations= json.load(f)
        return annotations
    
    def _load_regions(self):
        file = self.path / f'ss_regions_{self.split}.pkl'
        with open(file, "rb") as f:
            regions= pickle.load(f)
        return regions


    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        regions = self.regions[idx]
        images, labels = self._get_regions(regions)
        images = torch.cat(images,0)
        labels = torch.cat(labels,0)
        
        return images, labels
    
    def _get_regions(self, regions):
        image = self._get_image(regions)

        trash_regions = [key for key in regions['regions'].keys() if regions["regions"][key]["label"] != self.BACKGROUND_LABEL ]
        out_regions = trash_regions

        background_regions = [ key for key in regions['regions'].keys() if regions["regions"][key]["label"] == self.BACKGROUND_LABEL]
        
        if len(trash_regions) > int(0.5 * self.num_to_return):
            out_regions = np.random.choice(trash_regions, int(0.5 * self.num_to_return), replace=False)
        out_regions = np.concatenate([out_regions, np.random.choice(background_regions, int(self.num_to_return - len(out_regions)), replace=False)])
        
        assert len(out_regions) == self.num_to_return
        
        regions = [regions["regions"][region] for region in out_regions]

        images = []
        labels = []

        for region in regions:
            x1 = region["coordinates"]["x1"]
            y1 = region["coordinates"]["y1"]
            x2 = region["coordinates"]["x2"]
            y2 = region["coordinates"]["y2"]
            cropped_image = image[y1:y2, x1:x2]
            
            try:
                transformed_image = self.transform(cropped_image)
            except Exception as excpt:
                print(f"Exception: {excpt}")
                print(f"Original Image: {image.shape}")
                print(f"Cropped Image: {cropped_image.shape}")
                print(f"Region: {region['coordinates']}")
                raise excpt

            images.append(transformed_image.unsqueeze(0))
            labels.append(torch.from_numpy(self._encode_labels(region["label"])).unsqueeze(0))
        
        return images, labels

    def _get_image(self, regions):
        image_path = regions["filename"]
        img = fix_orientation(image_path, self.orientation)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]),interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) # RGB image out

    def _encode_labels(self, label):
        encoded_label = np.zeros(29)
        if label != self.BACKGROUND_LABEL:
            encoded_label[label] = 1
        else:
            encoded_label[-1] = 1
        return encoded_label
        

def build_taco(cfg: DictConfig):

    train = TACO(cfg, split="train")
    val = TACO(cfg, split="val")
    test = TACO(cfg, split="test")

    return train, val, test