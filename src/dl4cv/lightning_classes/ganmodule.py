import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
import torchvision

from dl4cv.utils.technical_utils import load_obj

class LitGANModel(pl.LightningModule):
    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.cfg = cfg
        self.discriminator = load_obj(cfg.model.discriminator.class_name)(cfg=cfg)
        self.generator = load_obj(cfg.model.generator.class_name)(cfg=cfg)
        self.adversarial_loss = load_obj(cfg.loss.class_name)()

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

        self.validation_z = torch.randn(8, self.generator.latent_dim) 

        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, x, *args, **kwargs):
        return self.generator(x)

    def configure_optimizers(self):
        optimizer_d = load_obj(self.cfg.optimizer.discriminator.class_name)(
            self.discriminator.parameters(), **self.cfg.optimizer.discriminator.params
        )

        optimizer_g = load_obj(self.cfg.optimizer.generator.class_name)(
            self.generator.parameters(), **self.cfg.optimizer.generator.params
        )


        scheduler_d = load_obj(self.cfg.scheduler.discriminator.class_name)(
            optimizer_d, **self.cfg.scheduler.discriminator.params
        )

        scheduler_g = load_obj(self.cfg.scheduler.generator.class_name)(
            optimizer_g, **self.cfg.scheduler.generator.params
        )

        return (
            [optimizer_d, optimizer_g],
            [
                {
                    "scheduler": scheduler_d,
                    "interval": self.cfg.scheduler.discriminator.step,
                    "monitor": self.cfg.scheduler.discriminator.monitor
                },
                {
                    "scheduler": scheduler_g ,
                    "interval": self.cfg.scheduler.generator.step,
                    "monitor": self.cfg.scheduler.generator.monitor
                }
            ],
        )

    def training_step(self, batch):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.generator.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)

        for logger in self.loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    tensorboard_logger = logger
                elif isinstance(logger, pl.loggers.WandbLogger):
                    wandb_logger = logger
        
        #tensorboard_logger.experiment.add_image("generated_images", grid, 0)
        #wandb_logger.experiment.log({"generated_images": [grid]})
        
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log(
            "generator_Loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        scheduler_g.step()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log(
            "discriminator_Loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        scheduler_d.step()
        self.untoggle_optimizer(optimizer_d)

        for metric in self.metrics:
            score = self.metrics[metric](imgs, self(z))
            self.log(
                f"train_{metric}",
                score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tensorboard_logger = logger
            elif isinstance(logger, pl.loggers.WandbLogger):
                wandb_logger = logger
        
        tensorboard_logger.experiment.add_image("generated_images", grid, self.current_epoch)
        wandb_logger.log_image({"generated_images": [grid]})
        