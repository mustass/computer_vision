from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.distributions import Normal
from torch.autograd import Variable


from dl4cv.utils.technical_utils import load_obj


class LitCVModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.model = load_obj(cfg.model.class_name)(cfg=cfg)
        self.loss = load_obj(cfg.loss.class_name)()

        self.metrics = torch.nn.ModuleDict(
            {
                self.cfg.metric.metric.metric_name: load_obj(
                    self.cfg.metric.metric.class_name
                )(**cfg.metric.metric.params)
            }
        )
        self.save_hyperparameters()

    def forward(self, x, *args, **kwargs):
        out = self.model(x)
        return out

    def saliency_step(self, batch, batch_idx, sigma, num_samples, strategy):
        input, _ = batch
        input = input.squeeze()
        self.model.eval()
        sigma = sigma / (input.max() - input.min())
        _samples = torch.randn(num_samples, *input.shape)
        _samples = sigma * _samples + input
        saliency_map = []
        for _sample in _samples:
            _sample = _sample.clone()
            _sample = _sample.to(self.device)
            _sample = _sample.unsqueeze(0)
            _sample.requires_grad_().retain_grad()
            _output = self.model(_sample)
            _output.backward()
            if strategy == "max":
                _saliency_map = _sample.grad.abs().squeeze().max(0)[0]
            elif strategy == "mean":
                _saliency_map = _sample.grad.abs().squeeze().mean(0)[0]
            else:
                raise NotImplementedError
            saliency_map.append(_saliency_map)
        saliency_map = torch.stack(saliency_map).mean(0)
        saliency_map = saliency_map / saliency_map.max()
        return saliency_map

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )

        scheduler = load_obj(self.cfg.scheduler.class_name)(
            optimizer, **self.cfg.scheduler.params
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": self.cfg.scheduler.step,
                    "monitor": self.cfg.scheduler.monitor,
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        input, target = batch
        target = target.float()
        predicted = self.model(input).squeeze()
        loss = self.loss(predicted, target)
        self.log(
            "train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](predicted, target)
            self.log(
                f"train_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        target = target.float()
        predicted = self.model(input).squeeze()
        loss = self.loss(predicted, target)

        self.log(
            "val_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](predicted, target)
            self.log(
                f"val_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        input, target = batch

        target = target.float()

        predicted = self.model(input).squeeze()
        loss = self.loss(predicted, target)

        self.log(
            "test_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics:
            score = self.metrics[metric](predicted, target)
            self.log(
                f"test_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss
