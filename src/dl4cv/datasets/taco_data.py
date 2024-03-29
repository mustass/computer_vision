import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ExifTags
import cv2
from omegaconf import DictConfig

from dl4cv.utils.technical_utils import load_obj
from dl4cv.utils.object_detect_utils import fix_orientation

from pathlib import Path
import json
import pickle


class TACO(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split="train", inference=False):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.train = split == "train"
        self.inference = inference

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
        file = self.path / f"{self.split}_annotations.json"
        with open(file, "r") as f:
            annotations = json.load(f)
        return annotations

    def _load_regions(self):
        file = self.path / f"ss_regions_{self.split}.pkl"
        with open(file, "rb") as f:
            regions = pickle.load(f)
        return regions

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        regions = self.regions[idx]
        if not self.inference:
            images, labels, regions_selected, gt_regions = self._get_regions_train(regions)
        else:
            images, labels, regions_selected, gt_regions = self._get_regions_pipeline_test(regions)
        images = torch.cat(images, 0)
        labels = torch.cat(labels, 0)
        return {
            "images": images,
            "labels": labels,
            "filename": regions["filename"],
            "image_id": regions["image_id"],
            "regions": regions_selected,
            "gt_regions": gt_regions,
        }

    def _get_regions_train(self, regions):
        image = self._get_image(regions)

        gt_regions = regions["gt_values"]

        trash_regions = [
            region
            for region in regions["regions"].values()
            if region["label"] != self.BACKGROUND_LABEL
            and self._sanitize_regions(region)
        ]

        background_regions = [
            region
            for region in regions["regions"].values()
            if region["label"] == self.BACKGROUND_LABEL
            and self._sanitize_regions(region)
        ]

        out_regions = trash_regions
        if self.train:
            out_regions = trash_regions + gt_regions

        if len(out_regions) > int(0.5 * self.num_to_return):
            out_regions_indecies = np.random.choice(
                np.arange(len(out_regions)),
                int(0.5 * self.num_to_return),
                replace=False,
            )
            out_regions = [out_regions[i] for i in out_regions_indecies]
        
        background_regions_indecies = np.random.choice(
                np.arange(len(background_regions)),
                int(self.num_to_return - len(out_regions)),
                replace=False,
            )

        out_background = [background_regions[i] for i in background_regions_indecies]

        regions = out_regions + out_background

        assert len(regions) == self.num_to_return

        images = []
        labels = []

        for region in regions:
            x1 = max(0,region["coordinates"]["x1"])
            y1 = max(0,region["coordinates"]["y1"])
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
            labels.append(
                torch.from_numpy(self._encode_labels(region["label"])).unsqueeze(0)
            )

        return images, labels, regions, gt_regions

    def _get_regions_pipeline_test(self, regions):
        image = self._get_image(regions)

        gt_regions = regions["gt_values"]

        regions_keys = list(regions["regions"].keys())

        regions_out = [regions["regions"][key] for key in regions_keys[:self.num_to_return]]

        if len(regions_out) < self.num_to_return:
            print(f'Len region keys: {len(regions_keys)}')
            print(f'Oversampling!!!')
            regions_keys = np.random.choice(
                regions_keys,
                self.num_to_return,
                replace=True,
            )
            regions_out = [regions["regions"][key] for key in regions_keys]

        images = []
        labels = []

        for region in regions_out:
            x1 = max(0,region["coordinates"]["x1"])
            y1 = max(0,region["coordinates"]["y1"])
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
            labels.append(
                torch.from_numpy(self._encode_labels(region["label"])).unsqueeze(0)
            )

        return images, labels, regions_out, gt_regions

    def _sanitize_regions(self, region, train=True):
        x1 = max(0,region["coordinates"]["x1"])
        y1 = max(0,region["coordinates"]["y1"])
        x2 = region["coordinates"]["x2"]
        y2 = region["coordinates"]["y2"]

        valid_y = y2 < self.img_size[1] and y1 < self.img_size[1] and y1 > 0 and y2 > 0
        valid_x = x2 < self.img_size[0] and x1 < self.img_size[0] and x1 > 0 and x2 > 0

        valid_iou = region["iou"] > 0.7 or region["iou"] < 0.3

        return (valid_x and valid_y) and valid_iou

    def _get_image(self, regions):
        image_path = regions["filename"]
        img = fix_orientation(image_path, self.orientation)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(
            img, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_AREA
        )
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)  # RGB image out

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
    inference = TACO(cfg, split="test", inference=True)
    return train, val, test,inference


def taco_train_collate_fn(batch):
    out_images = []
    out_labels = []

    for data_point in batch:
        out_images.append(data_point["images"])
        out_labels.append(data_point["labels"])

    return torch.cat(out_images, 0), torch.cat(out_labels, 0)


def taco_val_test_collate_fn(batch):
    out_images = []
    out_labels = []
    out_image_ids = []
    out_regions_selected = []
    out_regions_gt = []
    out_labels_gt = []
    for data_point in batch:
        out_images.append(data_point["images"])
        out_labels.append(data_point["labels"])
        out_image_ids.append(data_point["image_id"])
        for region in data_point["regions"]:
            _region = torch.tensor(
                [
                    region["coordinates"]["x1"],
                    region["coordinates"]["x2"],
                    region["coordinates"]["y1"],
                    region["coordinates"]["y2"],
                ]
            )
            _region = _region.unsqueeze(0)
            out_regions_selected.append(_region)

        for region in data_point["gt_regions"]:
            _region = torch.tensor(
                [
                    region["coordinates"]["x1"],
                    region["coordinates"]["x2"],
                    region["coordinates"]["y1"],
                    region["coordinates"]["y2"],
                ]
            )
            _region = _region.unsqueeze(0)
            out_regions_gt.append(_region)
            out_labels_gt.append(region["label"])

    return (
        torch.cat(out_images, 0),
        torch.cat(out_labels, 0),
        torch.tensor(out_image_ids),
        torch.cat(out_regions_selected, 0),
        torch.cat(out_regions_gt, 0),
        torch.tensor(out_labels_gt),
    )
