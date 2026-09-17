import torch 
from typing import Tuple, Optional, Union
from torch import nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin as FromOriginalVAEMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from .enc_dec import (EncoderCausal3D, 
                      DecoderCausal3D, 
                      DecoderOutput2, 
                      DiagonalGaussianDistribution,
                      DecoderOutput)


class AutoencoderKLCausal3D(ModelMixin, ConfigMixin, FromOriginalVAEMixin):

    _supports_gradient_checkpointing = True 

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str] = ("DownEncoderBlockCausal3D",),
                 up_block_types: Tuple[str] = ("UpDecoderBlockCausal3D",),
                 block_out_channels: Tuple[int] = (64,),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 4,
                 norm_num_groups: int = 32,
                 sample_size: int = 32,
                 sample_tsize: int = 64,
                 scaling_factor: float = 0.18215,
                 force_upcast: float = True,
                 spatial_compression_ratio: int = 8,
                 time_compression_ratio: int = 4,
                 mid_block_add_attention: bool = True
                 ):

        super().__init__()
        self.time_compression_ratio = time_compression_ratio

        self.encoder = EncoderCausal3D(in_channels=in_channels,
                                       out_channels=out_channels,
                                       down_block_type=down_block_types,
                                       block_out_channels=block_out_channels,
                                       layers_per_block=layers_per_block,
                                       norm_num_groups=norm_num_groups,
                                       act_fn=act_fn,
                                       double_z=True,
                                       mid_block_add_attention=mid_block_add_attention,
                                       time_compression_ratio=time_compression_ratio,
                                       spatial_compression_ratio=spatial_compression_ratio)

        self.decoder = DecoderCausal3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio
        )

        self.quant_conv = nn.Conv3d(in_channels=2*latent_channels,
                                    out_channels=2*latent_channels,
                                    kernel_size=1)
        self.post_quant_conv = nn.Conv3d(in_channels=latent_channels,
                                         out_channels=latent_channels,
                                         kernel_size=1)

        self.use_slicing = False 
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False 

        # only relevent if vae tiling is enabled 
        self.tile_sample_min_tsize = sample_tsize
        self.tile_sample_min_size = self.config.sample_size

        self.tile_latent_min_tsize = sample_size // time_compression_ratio
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) -1)))
        self.tile_overlap_factor = 0.25


    def forward(self,
                sample: torch.FloatTensor,
                sample_posterior: bool = False,
                return_dict: bool = True,
                return_posterior: bool = False,
                generator: Optional[torch.Generator] = None) -> Union[DecoderOutput2, torch.FloatTensor]:

        x = sample 
        posterior = self.encode(x).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()


        dec = self.decode(z).sample

        if not return_dict:
            if return_posterior:
                return (dec, posterior)
            else:
                return (dec,)

        if return_posterior:
            return DecoderOutput2(sample=dec, posterior=posterior)
        else:
            return DecoderOutput2(sample=dec)
        



    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to 
        compute decoding in several steps. This is useful to save memory and allow larger batch sizes.
        """
        self.use_slicing = True


    @apply_forward_hook
    def encode(
        self,
        x: torch.FloatTensor,
        return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:

        assert len(x.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and x.shape[2] > self.tile_sample_min_tsize:
            assert ValueError("Funcation Deos not Found.")

        if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            assert ValueError("Funcation Deos not Found.")

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)

        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)


    @apply_forward_hook
    def decode(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
        generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)

        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    


    def _decoded(self, 
                 z: torch.FloatTensor,
                 return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:

        assert len(z.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and z.shape[2] > self.tile_sample_min_tsize:
            assert ValueError("Funcation Deos not Found.")

        if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            assert ValueError("Funcation Deos not Found.")

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    



    

    

        


