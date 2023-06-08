import argparse
import glob
from hydra import compose, initialize
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from dl4cv.utils.utils import set_seed
from dl4cv.lightning_classes.plmodel import LitCVModel
from dl4cv.datasets import HotDogNotHotDog
from tqdm import tqdm
import os


def saliency_map(cfg: DictConfig, batch_size: int = 1, strategy="max") -> None:
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
    test_set = HotDogNotHotDog(cfg).test
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
    )
    lit_model = LitCVModel.load_from_checkpoint(checkpoint_path=model_names[0], cfg=cfg)
    lit_model.to(device)
    saliency_maps = []
    for batch in tqdm(loader):
        saliency_maps.append(
            lit_model.saliency_step(
                batch,
                0,
                cfg.inference.saliency_params.sigma,
                cfg.inference.saliency_params.num_samples,
                strategy,
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
    parser = argparse.ArgumentParser(description="Salience Map for HotDog Not HotDog")
    parser.add_argument(
        "--run_name", help="folder_name", type=str, default="2020_06_21_04_53_55"
    )
    parser.add_argument("--device", help="device", type=str, default="cpu")
    parser.add_argument("--batch_size", help="batch_size", type=int, default=1)
    parser.add_argument("--sigma", help="sigma", type=float, default=0.1)
    parser.add_argument("--strategy", help="strategy", type=str, default="max")
    parser.add_argument("--num_samples", help="num_samples", type=int, default=20)
    args = parser.parse_args()

    initialize(config_path="../configs")
    inference_cfg = compose(config_name="config_hotdog_training")
    inference_cfg["inference"]["run_name"] = args.run_name
    inference_cfg["inference"]["device"] = args.device
    inference_cfg["inference"]["saliency_params"]["num_samples"] = args.num_samples
    inference_cfg["inference"]["saliency_params"]["sigma"] = args.sigma
    path = f"outputs/{inference_cfg.inference.run_name}/.hydra/config.yaml"

    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)

    cfg_yaml["inference"] = inference_cfg["inference"]
    cfg = OmegaConf.create(cfg_yaml)
    saliency_map(cfg, batch_size=args.batch_size, strategy=args.strategy)
