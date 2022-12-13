import argparse
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from torkit3d_baselines.nerf.dataset import NeRFSyntheticDataset
from torkit3d_baselines.nerf.model import NeRF


def concat_collate_fn(data_list, scalar_keys=None):
    data_batch = defaultdict(list)
    for batch_ind, data_dict in enumerate(data_list):
        for key in data_dict:
            data_batch[key].append(data_dict[key])
    for k, v in data_batch.items():
        if scalar_keys is not None and k in scalar_keys:
            data_batch[k] = np.hstack(v)
        else:
            data_batch[k] = np.concatenate(v)
    data_batch = {k: torch.from_numpy(v) for k, v in data_batch.items()}
    return data_batch


class LitNeRF(pl.LightningModule):
    def __init__(
        self,
        near,
        far,
        n_segments=64,
        n_importance=-1,
        image_size=None,
        chunk_size=8192,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # NeRF hparams
        self.near = near
        self.far = far
        self.n_segments = n_segments
        self.n_importance = n_importance
        pe_dim = self.hparams.get("pe_dim", 10)
        pe_dim_viewdir = self.hparams.get("pe_dim_viewdir", 4)

        # Inference hparams
        self.image_size = image_size
        self.chunk_size = chunk_size

        # Create models
        self.nerf = NeRF(pe_dim=pe_dim, pe_dim_viewdir=pe_dim_viewdir)
        if self.n_importance > 0:
            self.nerf_fine = NeRF(pe_dim=pe_dim, pe_dim_viewdir=pe_dim_viewdir)

        # Metrics
        metrics = torchmetrics.MetricCollection(
            dict(psnr=torchmetrics.PeakSignalNoiseRatio())
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def configure_optimizers(self):
        lr = self.hparams.get("lr", 5e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        *args,
        **kwargs
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, *args, **kwargs)

        # exponential LR scheduler
        init_lr = optimizer.defaults["lr"]
        decay_rate = 0.1
        lr = init_lr * (decay_rate ** (self.global_step / 250000))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        # self.log("lr", lr)

    def forward(self, rays: torch.Tensor, near, far):
        preds = self.nerf.render_rays(
            rays[..., 0:3],
            rays[..., 3:6],
            near,
            far,
            n_segments=self.n_segments,
            perturb=self.training,
        )
        if self.n_importance > 0:
            preds_fine = self.nerf_fine.render_rays_stratified(
                rays[..., 0:3],
                rays[..., 3:6],
                preds["bins"],
                preds["weights"],
                n_importance=self.n_importance,
            )
            # Update keys
            preds = {"coarse/" + k: v for k, v in preds.items()}
            preds.update(preds_fine)
        return preds

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch["rays"], self.near, self.far)
        loss = F.mse_loss(preds["rgb"], batch["rays_rgb"])
        if "coarse/rgb" in preds:
            loss_coarse = F.mse_loss(preds["coarse/rgb"], batch["rays_rgb"])
            psnr_coarse = torchmetrics.functional.peak_signal_noise_ratio(
                preds["coarse/rgb"], batch["rays_rgb"]
            )
            loss = loss + loss_coarse
        with torch.no_grad():
            self.train_metrics(preds["rgb"], batch["rays_rgb"])
            self.log("train/loss", loss, prog_bar=True)
            self.log_dict(self.train_metrics, prog_bar=True)
            if "coarse/rgb" in preds:
                self.log("train/loss_coarse", loss_coarse)
                self.log("train/psnr_coarse", psnr_coarse)
        return loss

    def on_validation_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred_rgb = []
        batch_rays = torch.split(batch["rays"], self.chunk_size)
        batch_rays_rgb = torch.split(batch["rays_rgb"], self.chunk_size)
        for rays, rays_rgb in zip(batch_rays, batch_rays_rgb):
            preds = self.forward(rays, self.near, self.far)
            self.val_metrics.update(preds["rgb"], rays_rgb)
            pred_rgb.append(preds["rgb"])

        if batch_idx < 4:
            return torch.cat(pred_rgb).reshape(*self.image_size, 3)

    @torch.no_grad()
    def validation_epoch_end(self, outputs) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

        # Summary images
        self.logger.experiment.add_images(
            "pred_rgb", torch.stack(outputs), self.global_step, dataformats="NHWC"
        )

        # Clear cache
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="nerf")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image_size", type=int, default=800)
    parser.add_argument("--n_segments", type=int, default=64)
    parser.add_argument("--n_importance", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=int(2e5))
    parser.add_argument("--val_check_interval", type=int, default=int(1e4))
    parser.add_argument("--val_subset", type=int, default=10)
    args = parser.parse_args()

    near, far = 2.0, 6.0  # nerf_synthetic blender
    image_size = (args.image_size, args.image_size)
    num_workers = 1
    pl.seed_everything(args.seed)

    model = LitNeRF(
        near=near,
        far=far,
        n_segments=args.n_segments,
        n_importance=args.n_importance,
        image_size=image_size,
    )
    print(model)

    train_dataset = NeRFSyntheticDataset(
        "data/nerf_synthetic/lego", "train", image_size=image_size
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=concat_collate_fn,
        persistent_workers=True,
    )

    val_dataset = NeRFSyntheticDataset(
        "data/nerf_synthetic/lego", "val", image_size=image_size, n_rays=None
    )
    # Use a subset for validation
    val_dataset = Subset(val_dataset, list(range(args.val_subset)))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=concat_collate_fn,
        persistent_workers=True,
    )

    logger = TensorBoardLogger(
        save_dir="logs", name=args.exp_name, default_hp_metric=False
    )
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,  # comment for debug
        check_val_every_n_epoch=None,
        val_check_interval=args.val_check_interval,
        logger=logger,
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
