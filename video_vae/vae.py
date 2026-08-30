import torch 
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import Tuple, Optional, Union
from torch import nn 
from timm.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint
from diffusers.utils import logging
logger = logging.get_logger()


from .enc_dec import CausalVaeEncoder, CausalVaeDecoder, DecoderOutput, DiagonalGaussianDistribution
from .causal_conv import CausalConv3d

import sys 
from pathlib import Path
# Add the parent directory (z1-v2)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from context_parallel import is_context_parallel_initialized, conv_gather_from_context_parallel_region, get_context_parallel_rank


class CausalVideoVae(ModelMixin, ConfigMixin):

    _supports_group_offloading = True

    @register_to_config
    def __init__(self,
                 # Encoder related paramteres
                 encoder_in_channels: int = 3,
                 encoder_out_channels: int = 4,
                 encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
                 encoder_down_block_types: Tuple[str, ...] = (
                     "DownEncoderBlockCausal3D",
                     "DownEncoderBlockCausal3D",
                     "DownEncoderBlockCausal3D",
                     "DownEncoderBlockCausal3D",
                 ),
                 encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 encoder_spatial_down_sample: Tuple[bool, ...] = (True, True, True, False),
                 encoder_temporal_down_sample: Tuple[bool, ...] = (True, True, True, False),
                 encoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
                 encoder_act_fn: str = "silu",
                 encoder_norm_num_groups: int = 32,
                 encoder_double_z: bool = True,
                 encoder_type: str = 'Causal_vae_conv',
                 # DECODER RELATED PARAMETERS
                 decoder_in_channels: int = 4,
                 decoder_out_channels: int = 3,
                 decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
                 decoder_up_block_types: Tuple[str, ...] = (
                     "UpDecoderBlockCausal3D",
                     "UpDecoderBlockCausal3D",
                     "UpDecoderBlockCausal3D",
                     "UpDecoderBlockCausal3D"
                 ),
                 decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
                 decoder_spatial_up_sample: Tuple[bool, ...] = (True, True, Tuple, False),
                 decoder_temporal_up_sample: Tuple[bool, ...] = (True, True, True, False),
                 decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
                 decoder_act_fn: str = "silu",
                 decoder_norm_num_groups: int = 32,
                 decoder_type: str = "Causal_vae_conv",
                 # OTHER PARAMETERS 
                 sample_size = 256,
                 scaling_factor: float = 0.18215,
                 add_post_quant_conv: bool = True,
                 downsample_scale: int = 8
                 ):

        super().__init__()
        
        print(f"The latent dim channels is: {encoder_out_channels}")

        self.encoder = CausalVaeEncoder(in_channels=encoder_in_channels,
                                        out_channels=encoder_out_channels,
                                        down_black_types=encoder_down_block_types,
                                        spatial_down_sample=encoder_spatial_down_sample,
                                        temporal_down_sample=encoder_temporal_down_sample,
                                        block_out_channels=encoder_block_out_channels,
                                        layers_per_block=encoder_layers_per_block,
                                        norm_num_groups=encoder_norm_num_groups,
                                        act_fn=encoder_act_fn,
                                        double_z=encoder_double_z,
                                        block_dropout=encoder_block_dropout)

        self.decoder = CausalVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            spatial_up_sample=decoder_spatial_up_sample,
            temporal_up_sample=decoder_temporal_up_sample,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            mid_block_add_attention=True,
            block_dropout=decoder_block_dropout
        )

        self.quant_conv = CausalConv3d(input_channels=2*encoder_in_channels,
                                       output_chaannels=2*encoder_out_channels,
                                       kernel_size=1,
                                       stride=1
                                       )
        self.post_quant_conv = CausalConv3d(input_channels=encoder_out_channels,
                                            output_chaannels=encoder_out_channels,
                                            kernel_size=1,
                                            stride=1)

        self.downsample_scale = downsample_scale
        self.use_tiling = False 

        # only relevent if vae tiling is enabled 
        self.tile_sample_min_size = self.config.sample_size

        sample_size = (
            self.config.sample_size[0] if isinstance(self.config.sample_size, (list, tuple)) else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / downsample_scale)
        self.decode_tile_overlap_factor = 1 / 4 
        self.downsample_scale = downsample_scale

        self.apply(self._init_weight)


    def _init_weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (self.encoder, self.decoder)):
            module.gradient_checkpointing = value         




    def forward(self,
                sample: torch.FloatTensor,
                sample_posterior: bool = True,
                generator: Optional[torch.Generator] = None,
                freeze_encoder: bool = False,
                is_init_image = True,
                temporal_chunk = False) -> Union[DecoderOutput, torch.FloatTensor]:

        x = sample 

        if is_context_parallel_initialized():
            assert self.training, "Only supports during training"

            if freeze_encoder:
                logger.warning("THIS `freeze_encoder` work when you `finetune`. "
                "if you can train from `scratch` it's not work. " \
                "because `learnable parameters` are off and decoder are biased data take.")

            else:
                h = self.encoder(x, is_init_image=True, temporal_chunk=False)
                moments = self.quant_conv(h, is_init_image=True, temporal_chunk=False)
                posterior = DiagonalGaussianDistribution(moments)
                global_moments = conv_gather_from_context_parallel_region(moments, dim=2, kernel_size=1)
                global_posterior = DiagonalGaussianDistribution(global_moments)

            if sample_posterior:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()

            if get_context_parallel_rank() == 0:
                dec = self.decode(z, is_init_image=True).sample

            return global_posterior, dec 
        


    def decode(self, 
               z: torch.FloatTensor,
               is_init_image=True,
               temporal_chunk=False,
               return_dict: bool = True,
               window_size: int = 2,
               tile_sample_min_size: int = 256) -> Union[DecoderOutput, torch.FloatTensor]:

        self.tile_sample_min_size = tile_sample_min_size
        self.tile_latent_min_size = int(tile_sample_min_size / self.downsample_scale)

        # checks if spatial tiling is enabled AND if the width ([-1]) or height ([-2]) of the latent exceeds the safe limit.
        # if it is too big, it rotues to `tiled_decode` to prevent GPU memory crashed.
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            logger.warning("latent shape are more than: [:, :, :, 32, 32], But `tile_decode` Function does not Execute...")
            return self.tiled_decode()

        if temporal_chunk:
            # dec = self.chunk_decode(z, window_size=window_size)
            logger.warning("Temporal Chunk is Enable But Funcation does not Execute...")
        else:
            z = self.post_quant_conv(z, is_init_image=is_init_image, temporal_chunk=False)
            dec = self.decoder(z, is_init_image=is_init_image, temporal_chunk=False)


        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    


    def tiled_decode(self):

        r"""
        Decode High resolution videos.
        """

        pass 

        


    



            

                
