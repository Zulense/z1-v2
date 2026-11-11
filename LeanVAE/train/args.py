import argparse



def add_model_specific_args():
    parser = argparse.ArgumentParser(description="Vae Training param", add_help=True)

    # Model arch param
    parser.add_argument("--embedding_dim", type=int, default=512, help="Dim of the embedding space.")
    parser.add_argument("--latent_dim", type=int, default=4, help="Dim of the latent channel.")
    parser.add_argument("--ista_iter_num", type=int, default=2, help="Num of iteration in ISTA latent bottleneck.")
    parser.add_argument("--ista_layer_num", type=int, default=2, help="Num of layers in ISTA latent bottleneck.")

    parser.add_argument("--l_dim", type=int, default=128)
    parser.add_argument("--h_dim", type=int, default=384)
    parser.add_argument("--sep_num_layer", type=int, default=2, help="Num of separate processing layers in encoder/decoder.")
    parser.add_argument("--fusion_num_layer", type=int, default=4, help="Num of fusion layers in encoder/decoder.")

    # Tiling inference (for memory-efficient processing.)
    parser.add_argument("--use_tile_inference", action="store_true", help="Enable tiling inference to process video in chunks.")
    parser.add_argument("--chunksize_enc", type=int, default=9, help="Num of frames per chunk during tiled encoding.")
    parser.add_argument("--chunksize_dec", type=int, default=5, help="Num of frames per chunk during tiled decoding.")

    # training config
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--grad_clip_val", type=float, default=1.0)
    parser.add_argument("--grad_clip_val_disc", type=float, default=1.0)

    parser.add_argument("--kl_weight", type=float, default=1e-7)
    parser.add_argument("--perceptual_weight", type=float, default=4.0)
    parser.add_argument("--l1_weight", type=float, default=4.0)
    parser.add_argument("--disc_weight", type=float, default=0.2)

    # config for discriminator.
    parser.add_argument("--disc_warmup_steps", type=int, default=0)
    parser.add_argument("--disc_iter_start", type=int, default=0)
    parser.add_argument("--disc_lr_multiplier", type=float, default=1.0)

    # extra config.
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Trainer arguments
    parser.add_argument("--gpus", type=int, default=1, help="Num of GPUs to use.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Num of nodes for distributed training.")
    parser.add_argument("--max_steps", type=int, default=600000, help="Maximum training steps")
    parser.add_argument("--default_root_dir", type=str, default="./LeanVAE/logs", help="Directory for logs and checkpoints.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=None)

    # dataset param.
    parser.add_argument("--data_path", type=str, nargs="+", default=[''])
    parser.add_argument("--train_datalist", type=str, nargs="+", default=[''])
    parser.add_argument("--val_datalist", type=str, nargs="+", default=[''])
    parser.add_argument("--sequence_length", type=int, default=17)
    parser.add_argument("--sample_rate", type=int, default=1, help="Frame sampling rate.")
    parser.add_argument("--dynamic_sample", action='store_true', help="Enable dynamic sampling rate.")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[5])
    parser.add_argument("--num_workers", type=int, default=8)

    return parser