from PIL import Image, ExifTags
from pycocotools.coco import COCO
import pylab
from pathlib import Path
import torch
import torchvision as T
import albumentations as A
import numpy as np
from PIL import Image
import cv2
from omegaconf import DictConfig
from torchvision import transforms
import json

class Taco(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, train=True, indices=None,coco_obj=None):
        super().__init__()
        self.cfg = cfg
        self.dataset_path = Path(cfg.datamodule.params.path)        
        self.coco = coco_obj
        self.index = indices

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path = self.coco.loadImgs(self.index[idx])[0]['file_name']
        img = cv2.imread(str(self.dataset_path/img_path))
        an_ids = self.coco.getAnnIds(imgIds=self.coco.loadImgs(self.index[idx])[0]['id'],iscrowd=None)
        print(len(an_ids))
        anns_sel = self.coco.loadAnns(an_ids)
        print(anns_sel[0])
        return img, anns_sel[0]['bbox']

def build_taco(cfg: DictConfig, category_name = "Bottle"):
    
    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    
    # Loads dataset as a coco object
    coco = COCO(cfg.datamodule.params.annotations_path)


    # Get image ids
    imgIds = []
    catIds = coco.getCatIds(catNms=[category_name])
    if catIds:
        # Get all images containing an instance of the chosen category
        imgIds = coco.getImgIds(catIds=catIds)
    else:
        # Get all images containing an instance of the chosen super category
        catIds = coco.getCatIds(supNms=[category_name])
        for catId in catIds:
            imgIds += (coco.getImgIds(catIds=catId))
        imgIds = list(set(imgIds))

    dataset = Taco(cfg,coco_obj=coco,indices=imgIds)
    
    len_dataset = len(dataset)

    # Split dataset into train and test
    train_size = int(0.8 * len_dataset)
    val_size = int(0.1 * len_dataset)
    test_size = len_dataset - train_size-val_size
    train_dataset,val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,val_size, test_size])

    return train_dataset,val_dataset,test_dataset
