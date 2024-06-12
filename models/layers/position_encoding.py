"""
Various positional encodings for the transformer.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import math
import torch
from torch import nn
import einops

from utils.util import NestedTensor

# position encoding for 3 dims
class PositionalEncodingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, embed_size=64, z_size=36, fit_mode='interpolate', temperature=10000, normalize=False, scale=None, device='cuda'):
        super().__init__()
        self.embed_size = embed_size
        self.fit_mode = fit_mode
        self.temperature = temperature
        self.normalize = normalize
        self.z_size = z_size
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, scan):
        d, embed, h, w = scan.size()
        mask = torch.ones((self.z_size, h, w))
        mask = mask.reshape(1, self.z_size,h,w).to(self.device)
        # assert mask is not None
        # not_mask = ~mask
        z_embed = mask.cumsum(1, dtype=torch.float32)
        y_embed = mask.cumsum(2, dtype=torch.float32)
        x_embed = mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale


        dim_t = torch.arange(embed// 3 + ((embed // 3) % 2), dtype=torch.float32, device=scan.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (embed // 3))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).flatten(2, 3).squeeze(0)
        # pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        if pos.shape[2] != embed or pos.shape[0] != d:
            if self.fit_mode == 'interpolate':
                pos = pos.reshape(1,1,self.z_size,h*w,-1)
                pos = nn.functional.interpolate(pos, size=(d, h * w, embed), mode='trilinear',align_corners=False).squeeze()
                pos = pos.reshape(d, h, w, -1)
            elif self.fit_mode == 'prune':
                pos = pos.reshape(self.z_size, h * w, -1)
                pos = nn.functional.interpolate(pos, size=(embed), mode='linear', align_corners=False)
                pos = pos.reshape(self.z_size, h, w, -1)
                pos = pos[:d,:,:,:]
            else:
                raise NotImplementedError('Unknown positional embedding mode')

        return pos

class PositionalEncodingSine2D(nn.Module):
    def __init__(self, embed_size):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncodingSine2D, self).__init__()
        self.org_channels = embed_size
        channels = int(np.ceil(embed_size / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1).permute(0, 2, 3, 1)
        return self.cached_penc

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class LearnedPositionalEmbedding3D(nn.Module):
    """
    Absolute position embedding, learned.
    """
    def __init__(self, embedding_dim, max_depth=40, max_height=256, max_width=256):
        super().__init__()
        self.row_embed = nn.Embedding(max_width, embedding_dim // 3 + int(embedding_dim % 3 > 0))
        self.col_embed = nn.Embedding(max_height, embedding_dim // 3 + int(embedding_dim % 3 == 2))
        self.depth_embed = nn.Embedding(max_depth, embedding_dim // 3)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.depth_embed.weight)

    def forward(self, scan: NestedTensor):
        # x = tensor_list.tensors
        d, em, h, w = scan.size()
        # d, h, w = x.shape[-3:]
        i = torch.arange(w, device=scan.device)
        j = torch.arange(h, device=scan.device)
        k = torch.arange(d, device=scan.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.depth_embed(k)
        pos = torch.cat([
            x_emb.unsqueeze(0).unsqueeze(0).repeat(d, h, 1, 1),
            y_emb.unsqueeze(1).unsqueeze(0).repeat(d, 1, w, 1),
            z_emb.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1),
        ], dim=-1)
        # pos = torch.cat([
        #     x_emb.unsqueeze(0).unsqueeze(0).repeat(d, h, 1, 1),
        #     y_emb.unsqueeze(1).unsqueeze(0).repeat(d, 1, w, 1),
        #     z_emb.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1),
        # ], dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(1, 1, 1, 1, 1)
        return pos


class LearnedPositionalEncoding2D(nn.Module):
  def __init__(self, embed_size, img_size=256, patch_size=16):
    super().__init__()
    num_patches = (img_size // patch_size) ** 2
    self.width = self.height = img_size // patch_size
    self.pos_embedding = nn.Parameter(torch.randn(num_patches, embed_size))
  def forward(self, x):
    batch_size = x.shape[0]
    pos = self.pos_embedding.data.reshape(self.width, self.height, -1)
    pos = einops.repeat(pos, 'w h e -> b w h e', b=batch_size)
    return pos


def build_position_encoding(args):
    if args.MODEL.POSITION_EMBEDDING.TYPE == 'sine':
        if args.MODEL.VIT_ENCODER.ATTENTION_3D:
            position_embedding = PositionalEncodingSine3D(embed_size=args.MODEL.VIT_ENCODER.EMBED_SIZE,
                                                       z_size=args.MODEL.POSITION_EMBEDDING.Z_SIZE,
                                                       fit_mode=args.MODEL.POSITION_EMBEDDING.FIT_MODE,
                                                       normalize=True,
                                                       device=args.DEVICE)
        else:
            position_embedding = PositionalEncodingSine2D(embed_size=args.MODEL.VIT_ENCODER.EMBED_SIZE)

    elif args.MODEL.POSITION_EMBEDDING.TYPE == 'learned':
        if args.MODEL.VIT_ENCODER.ATTENTION_3D:
            position_embedding = LearnedPositionalEmbedding3D(embedding_dim=args.MODEL.VIT_ENCODER.EMBED_SIZE)
        else:
            position_embedding = LearnedPositionalEncoding2D(embed_size=args.MODEL.VIT_ENCODER.EMBED_SIZE,
                                                             img_size=args.TRAINING.INPUT_SIZE,
                                                             patch_size=args.MODEL.PATCH_SIZE
                                                             )
    elif args.MODEL.POSITION_EMBEDDING.TYPE is None:
        position_embedding = None
    else:
        raise ValueError(f"not supported {args.MODEL.POSITION_EMBEDDING.TYPE}")

    return position_embedding
