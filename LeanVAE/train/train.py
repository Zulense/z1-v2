import torch, os, sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

# Add the parent directory (z1-v2)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset.dataset import VideoData
from models.play import AutoEncoderEngine
from callback import VideoLogger
from args import add_model_specific_args

def main(args):
    pl.seed_everything(1234, workers=True)

    # Module-specific arguments
    data = VideoData(args=args)
    model = AutoEncoderEngine(args, data)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                                     save_top_k=3,
                                     mode='min',
                                     filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                                     save_top_k=-1,
                                     filename='{epoch}-{step}-{recon_loss:.2f}'))
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    print(f"<----- Log the reconstructed videos.... ---->")
    callbacks.append(VideoLogger(batch_frequency=1500,
                                 max_videos=4,
                                 clamp=True))

    precision = "32-true"
    if args.bf16:
        precision = "bf16-mixed"
    elif args.fp16:
        precision = "16-mixed"

    strategy = "ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"

    wandb_logger = WandbLogger(project="LeanVAE",
                               name=os.path.basename(args.default_root_dir),
                               save_dir=args.default_root_dir,
                               config=args)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        default_root_dir=args.default_root_dir,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=49,
        logger=wandb_logger,
        callbacks=callbacks,
        use_distributed_sampler=False,
        limit_val_batches=0,
        num_sanity_val_steps=0
    )

    ckpt_path = args.ckpt_path if args.ckpt_path else None
    trainer.fit(model=model,
                datamodule=data,
                ckpt_path=ckpt_path)


if __name__ == "__main__":
    args = add_model_specific_args()
    main(args)
    

    




