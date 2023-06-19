import argparse
import glob
from hydra import compose, initialize
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from dl4cv.utils.utils import set_seed
from dl4cv.lightning_classes.plmodel import LitODModel
from dl4cv.datasets import build_taco 
from dl4cv.datasets.taco_data import taco_val_test_collate_fn
from tqdm import tqdm
import os


def main(cfg: DictConfig, batch_size: int = 1, strategy="max") -> None:
    """
    Run pytorch-lightning model inference
    Args:
        cfg: hydra config
    Returns:
        None
    """
    set_seed(cfg.training.seed)

    device = torch.device(cfg.inference.device)

    model_names = glob.glob(f"outputs/{cfg.inference.run_name}/saved_models/*.ckpt")
    _, _, test_set = build_taco(cfg)
    # Dirty trick to get the ground truth boxes
    test_set.train = True
    loader = torch.utils.data.DataLoader(
        test_set,
        collate_fn=taco_val_test_collate_fn,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )
    lit_model = LitODModel.load_from_checkpoint(checkpoint_path=model_names[0], cfg=cfg)
    lit_model.to(device)
    predictions = []
    for batch in tqdm(loader):
        predictions.append(
            lit_model.nms_on_image(
                batch,
            )
        )




    if not os.path.exists(
        f"outputs/{cfg.inference.run_name}/saliency_maps_sigma_{cfg.inference.saliency_params.sigma}"
    ):
        os.makedirs(
            f"outputs/{cfg.inference.run_name}/saliency_maps_sigma_{cfg.inference.saliency_params.sigma}"
        )
    for i, saliency_map in enumerate(saliency_maps):
        np.save(
            f"outputs/{cfg.inference.run_name}/saliency_maps_sigma_{cfg.inference.saliency_params.sigma}/saliency_map_{i}.npy",
            saliency_map.cpu().numpy(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mAP for object detection")
    parser.add_argument(
        "--run_name", help="folder_name", type=str, default="Resnet50_Whale"
    )
    parser.add_argument("--device", help="device", type=str, default="cpu")
    args = parser.parse_args()

    initialize(config_path="../configs")
    inference_cfg = compose(config_name="config_taco_training")
    inference_cfg["inference"]["run_name"] = args.run_name
    inference_cfg["inference"]["device"] = args.device
    path = f"outputs/{inference_cfg.inference.run_name}/.hydra/config.yaml"

    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)

    cfg_yaml["inference"] = inference_cfg["inference"]
    cfg = OmegaConf.create(cfg_yaml)
    main(cfg, batch_size=args.batch_size, strategy=args.strategy)
