import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as L
from dl4cv.utils.technical_utils import load_obj


class GAN(L.LightningModule):
    def __init__(
        self,
        cfg,
        channels=1,
        width=28,
        height=28,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.cfg = cfg
        # networks
        data_shape = (channels, width, height)
        self.discriminator = load_obj(cfg.model.discriminator.class_name)(cfg=cfg)
        self.generator = load_obj(cfg.model.generator.class_name)(cfg=cfg)
        self.adversarial_loss = load_obj(cfg.loss.class_name)()
        
        # self.metrics = torch.nn.ModuleDict(
        #     {
        #         self.cfg.metric.metric.metric_name: load_obj(
        #             self.cfg.metric.metric.class_name
        #         )(**cfg.metric.metric.params)
        #     }
        # )

        # if "other_metrics" in self.cfg.metric.keys():
        #     for metric in self.cfg.metric.other_metrics:
        #         self.metrics.update(
        #             {metric.metric_name: load_obj(metric.class_name)(**metric.params)}
        #         )
        
        
        self.validation_z = torch.randn(8, 100)

        self.example_input_array = torch.zeros(2, 100)
    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch):
        imgs, _ = batch
        #print(f'imgs requires grad: {imgs.requires_grad}')
        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], 100)
        z = z.type_as(imgs)
        #print(f'z requires grad: {z.requires_grad}')
        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            if isinstance(logger, L.loggers.TensorBoardLogger):
                tensorboard_logger = logger
            elif isinstance(logger, L.loggers.WandbLogger):
                wandb_logger = logger
        
        tensorboard_logger.experiment.add_image("generated_images", grid, 0)
        wandb_logger.experiment.log({"generated_images": [grid]})
        
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop


        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        #print(f'valid requires grad: {valid.requires_grad}')
        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log(
            "generator_Loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        #print(f'g_loss requires grad: {g_loss.requires_grad}')
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        #print(f'valid requires grad: {valid.requires_grad}')
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        #print(f'real_loss requires grad: {real_loss.requires_grad}')
        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        #print(f'fake requires grad: {fake.requires_grad}')
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
        #print(f'fake_loss requires grad: {fake_loss.requires_grad}')
        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log(
            "discriminator_Loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
        
        #for metric in self.metrics:
        #    score = self.metrics[metric](imgs, self(z))
        #    self.log(
        #        f"train_{metric}",
        #        score,
        #        on_step=True,
        #        on_epoch=True,
        #        prog_bar=True,
        #        logger=True,
        #    )
    def configure_optimizers(self):

        opt_d = load_obj(self.cfg.optimizer.discriminator.class_name)(
            self.discriminator.parameters(), **self.cfg.optimizer.discriminator.params
        )

        opt_g = load_obj(self.cfg.optimizer.generator.class_name)(
            self.generator.parameters(), **self.cfg.optimizer.generator.params
        )

        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            if isinstance(logger, L.loggers.TensorBoardLogger):
                tensorboard_logger = logger
            elif isinstance(logger, L.loggers.WandbLogger):
                wandb_logger = logger
        
        tensorboard_logger.experiment.add_image("generated_images", grid, self.current_epoch)
        wandb_logger.log_image({"generated_images": [grid]})