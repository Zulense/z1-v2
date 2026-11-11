import torch, os, sys
import pytorch_lightning as pl
from torch import nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from pathlib import Path

# Add the parent directory (z1-v2)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from .autoencoder import LeanVAE
from modules.lpips import LPIPS
from modules.gan_loss import AdverserialLoss

class AutoEncoderEngine(pl.LightningModule):

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.video_data = data

        self.autoencoder = LeanVAE(args=args)
        self.automatic_optimization = False
        self.kl_weight = args.kl_weight
        self.disc_iter_start = args.disc_iter_start

        self.perceptual_weight = args.perceptual_weight
        self.l1_wight = args.l1_weight

        self.automatic_optimization = False
        self.grad_clip_val = args.grad_clip_val

        if not hasattr(args, "grad_clip_val_disc"):
            args.grad_clip_val_disc = 1.0

        self.grad_clip_val_disc = args.grad_clip_val_disc

        self.apply(self._init_weights)
        self.perceptual_model = LPIPS().eval()
        self.perceptual_model.requires_grad_(False)
        self.gan_loss = AdverserialLoss(disc_weight=args.disc_weight)
        self.save_hyperparameters()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, optimizer_idx=None, x_recon=None, log_image=False):
        if log_image:
            return self.autoencoder(x, log_image)

        if optimizer_idx == 0:
            assert x.ndim == 5, f"make sure shape of {x.shape} if 5 dim."
            B, C, T, H, W = x.shape
            x, x_recon, x_dwt, x_dwt_rec, posterior = self.autoencoder(x)

            recon_loss = F.l1_loss(x_recon, x) * self.l1_wight
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight

            g_loss = 0.0
            if self.global_step >= self.disc_iter_start:
                g_loss = self.gan_loss(x, x_recon, optimizer_idx=0)
                self.log(name="train/g_loss",
                         value=g_loss,
                         prog_bar=True,
                         logger=True,
                         on_step=True,
                         on_epoch=True)

            recon_loss_low = (
                F.l1_loss(x_dwt_rec[0][:, :3], x_dwt[0][:, :3])
                + F.l1_loss(x_dwt_rec[1][:, :3], x_dwt[1][:, :3])
            ) * self.l1_wight * 0.05
            recon_loss_high = (
                F.l1_loss(x_dwt_rec[0][:, :3], x_dwt[0][:, 3:])
                + F.l1_loss(x_dwt_rec[1][:, 3:], x_dwt[1][:, 3:])
            ) * self.l1_wight * 0.1

            k = 4
            valid_start_indices = torch.tensor([x for x in range(T - k + 1) if x % 4 == 1])
            start_idx = valid_start_indices[torch.randint(0, len(valid_start_indices), (B,))]
            frame_idx = start_idx.unsqueeze(1) + torch.arange(k)
            frame_idx = torch.cat((torch.zeros((B, 1), dtype=torch.int), frame_idx), dim=1).to(self.device)

            frames_idx_selected = frame_idx.reshape(-1, 1, k+1, 1, 1).repeat(1, C, 1, H, W)
            frames = torch.gather(x, 2, frames_idx_selected)
            frames_recon = torch.gather(x_recon, 2, frames_idx_selected)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous().view(-1, 3, H, W)
            perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

            self.log("train/recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/kl_loss", kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/rec_loss_low", recon_loss_low, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/rec_loss_high", recon_loss_high, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            return perceptual_loss + recon_loss + recon_loss_low + recon_loss_high + kl_loss + g_loss, x_recon

        if optimizer_idx == 1:
            disc_loss = self.gan_loss(inputs=x, recon=x_recon, optimizer_idx=1)
            self.log("train/disc_loss", disc_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return disc_loss

        return perceptual_loss, recon_loss, kl_loss


    def training_step(self, batch, batch_idx):

        x = batch[0]['video']
        cur_global_step = self.global_step

        sch1, sch2 = self.lr_schedulers()
        opt1, opt2 = self.optimizers()

        cur_global_step = self.global_step

        self.toggle_optimizer(optimizer=opt1)
        loss_generator, x_recon = self.forward(x, optimizer_idx=0)
        opt1.zero_grad()
        self.manual_backward(loss_generator)
        if self.grad_clip_val is not None:
            self.clip_gradients(opt1, gradient_clip_val=self.grad_clip_val)
        opt1.step()
        sch1.step(cur_global_step)
        self.untoggle_optimizer(opt1)

        if cur_global_step > self.disc_iter_start:
            self.toggle_optimizer(opt2)
            loss_disc = self.forward(x, optimizer_idx=1, x_recon=x_recon)

            opt2.zero_grad()
            self.manual_backward(loss_disc)

            if self.grad_clip_val_disc is not None:
                self.clip_gradients(opt2, gradient_clip_val=self.grad_clip_val_disc)
            opt2.step()
            sch2.step(cur_global_step)
            self.untoggle_optimizer(opt2)

    def validation_step(self, batch, batch_idx):
        x = batch['video']
        perceptual_loss, recon_loss, kl_loss = self.forward(x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/kl_loss', kl_loss, prog_bar=True)


    def train_dataloader(self):
        dataloaders = self.video_data._dataloader(train=True)
        return dataloaders

    def val_dataloader(self):
        return self.video_data._dataloader(False)[0]

    def configure_optimizers(self):

        opt_ae = torch.optim.Adam(params=self.autoencoder.parameters(),
                                  lr=self.args.lr,
                                  bias=(0.5, 0.9))
        opt_disc = torch.optim.Adam(params=self.gan_loss.get_trainable_parameters(),
                                    lr=self.args.lr_min,
                                    betas=(0.5, 0.9))

        lr_min = self.args.lr_min
        train_iters = self.args.max_step - self.disc_iter_start
        warmup_steps = self.args.warmup_steps
        warmup_lr_init = self.args.warmup_lr_init

        sch_ae = CosineLRScheduler(optimizer=opt_ae,
                                   lr_min=lr_min,
                                   t_initial=train_iters,
                                   warmup_lr_init=warmup_lr_init,
                                   warmup_t=warmup_steps,
                                   cycle_mul=1.,
                                   cycle_limit=1,
                                   t_in_epochs=True)
        sch_disc = CosineLRScheduler(
            optimizer=opt_ae,
            lr_min=lr_min,
            t_initial=train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=self.args.disc_wrmup_steps,
            cycle_mul=1.,
            cycle_limit=1,
            t_in_epochs=True
        )

        return [opt_ae, opt_disc], [{"scheduler": sch_ae, "interval": "step"}, 
                                    {"scheduler": sch_disc, "interval": "step"}]



    def log_videos(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        x = batch['video']
        x, x_rec = self(x, log_image=True)
        log["inputs"] = x
        log["recon"] = x_rec
        return log

    



        





