import pytorch_lightning as pl
import torch
from omegaconf import DictConfig


from dl4cv.utils.technical_utils import load_obj
from dl4cv.utils.segmentation_utils import plot_results

class LitSegModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.model = load_obj(cfg.model.class_name)(cfg=cfg)
        self.loss = load_obj(cfg.loss.class_name)()
        self.do_plots = cfg.training.save_plots

        self.metrics = torch.nn.ModuleDict(
            {
                self.cfg.metric.metric.metric_name: load_obj(
                    self.cfg.metric.metric.class_name
                )(**cfg.metric.metric.params)
            }
        )

        if "other_metrics" in self.cfg.metric.keys():
            for metric in self.cfg.metric.other_metrics:
                self.metrics.update(
                    {metric.metric_name: load_obj(metric.class_name)(**metric.params)}
                )

        self.save_hyperparameters()

    def forward(self, x, *args, **kwargs):
        out = self.model(x)
        return out

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
        target = target.int()
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
        target = target.int()
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

        if self.do_plots:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.WandbLogger):
                    wandb_logger = logger
            plot_results(
                predicted,
                target,
                input,
                self.current_epoch,
                batch_idx,
                self.cfg.datamodule.params.batch_size,
                self.cfg.callbacks.model_checkpoint.params.dirpath,
                wandb_logger= wandb_logger
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
        target = target.int()
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

        if self.do_plots:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.WandbLogger):
                    wandb_logger = logger
            plot_results(
                predicted,
                target,
                input,
                self.current_epoch,
                batch_idx,
                self.cfg.datamodule.params.batch_size,
                self.cfg.callbacks.model_checkpoint.params.dirpath,
                test=True,
                wandb_logger=wandb_logger
            )

        return loss
