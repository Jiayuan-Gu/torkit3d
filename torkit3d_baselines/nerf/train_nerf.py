import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from torkit3d.utils.misc import get_latest_checkpoint
from torkit3d_baselines.nerf.dataset import NeRFSyntheticDataset, SpiralPosesDataset
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
        **kwargs,
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
        **kwargs,
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

    def render_rays_batch(self, batch_rays):
        pred_rgb = []
        for rays in torch.split(batch_rays, self.chunk_size):
            preds = self.forward(rays, self.near, self.far)
            pred_rgb.append(preds["rgb"])
        pred_rgb = torch.cat(pred_rgb).reshape(*self.image_size, 3)
        return pred_rgb

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred_rgb = self.render_rays_batch(batch["rays"])

        # Update metrics
        psnr = torchmetrics.functional.peak_signal_noise_ratio(
            pred_rgb.reshape(-1, 3), batch["rays_rgb"]
        )
        self.log("psnr", psnr)

        # Save if output directory is provided
        output_dir = self.hparams.get("test_output_dir", None)
        if output_dir is not None:
            save_image(
                pred_rgb.permute(2, 0, 1), Path(output_dir) / f"{batch_idx:04d}.png"
            )

    @torch.no_grad()
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        pred_rgb = self.render_rays_batch(batch["rays"])
        pred_rgb = pred_rgb.mul(255).clamp_(0, 255)
        return pred_rgb.to(device="cpu", dtype=torch.uint8).numpy()

    # -------------------------------------------------------------------------- #
    # Data
    # -------------------------------------------------------------------------- #
    def train_dataloader(self):
        train_dataset = NeRFSyntheticDataset(
            "data/nerf_synthetic/lego", "train", image_size=self.image_size
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            collate_fn=concat_collate_fn,
            persistent_workers=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataset = NeRFSyntheticDataset(
            "data/nerf_synthetic/lego", "val", image_size=self.image_size, n_rays=None
        )
        # Use a subset for validation
        val_dataset = Subset(val_dataset, list(range(10)))
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=concat_collate_fn,
            persistent_workers=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataset = NeRFSyntheticDataset(
            "data/nerf_synthetic/lego", "test", image_size=self.image_size, n_rays=None
        )
        # test_dataset = Subset(test_dataset, list(range(10)))
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=concat_collate_fn,
        )
        return test_dataloader

    def predict_dataloader(self):
        test_dataset = SpiralPosesDataset(
            self.image_size,
            fov=0.6911112070083618,
            n_azimuths=60,
            elevations=np.deg2rad([30]),
            radius=4.0,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=concat_collate_fn,
        )
        return test_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="lego")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image_size", type=int, default=800)
    parser.add_argument("--n_segments", type=int, default=64)
    parser.add_argument("--n_importance", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=int(2e5))
    parser.add_argument("--val_check_interval", type=int, default=int(1e4))
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--test-output-dir", type=str)
    args = parser.parse_args()

    near, far = 2.0, 6.0  # nerf_synthetic blender
    image_size = (args.image_size, args.image_size)
    pl.seed_everything(args.seed)

    model = LitNeRF(
        near=near,
        far=far,
        n_segments=args.n_segments,
        n_importance=args.n_importance,
        image_size=image_size,
    )
    print(model)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.exp_name,
        version=args.version,
        default_hp_metric=False,
    )

    if args.test or args.predict:
        if args.ckpt_path is None:
            ckpt_dir = logger.log_dir + "/checkpoints"
            ckpt_path = get_latest_checkpoint(ckpt_dir)
        else:
            ckpt_path = args.ckpt_path

        test_output_dir = args.test_output_dir
        if test_output_dir == "@":
            _ckpt_path = Path(ckpt_path)
            test_output_dir = _ckpt_path.parent / "../test_images_{}".format(
                _ckpt_path.stem
            )
            test_output_dir.mkdir(exist_ok=True)

        # Load checkpoints and override hparams
        model = LitNeRF.load_from_checkpoint(
            ckpt_path,
            near=near,
            far=far,
            n_segments=args.n_segments,
            n_importance=args.n_importance,
            image_size=image_size,
            test_output_dir=test_output_dir,
        )

        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
        if args.test:
            trainer.test(model)
        if args.predict and test_output_dir is not None:
            results = trainer.predict(model)
            import imageio  # fmt: skip
            video_path = Path(test_output_dir) / "spiral_rgb.mp4"
            imageio.mimwrite(video_path, results, fps=15, quality=8)
    else:
        trainer = pl.Trainer(
            max_steps=args.max_steps,
            accelerator="gpu",
            devices=1,
            num_sanity_val_steps=0,  # comment for debug
            check_val_every_n_epoch=None,
            val_check_interval=args.val_check_interval,
            logger=logger,
        )
        if args.ckpt_path is not None:
            ckpt_path = args.ckpt_path
        elif trainer.checkpoint_callback is not None:
            if trainer.checkpoint_callback.dirpath is None:
                ckpt_dir = logger.log_dir + "/checkpoints"
            else:
                ckpt_dir = trainer.checkpoint_callback.dirpath
            ckpt_path = get_latest_checkpoint(ckpt_dir)
        else:
            ckpt_path = None

        trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
