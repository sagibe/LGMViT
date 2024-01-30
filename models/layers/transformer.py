import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.networks.layers import trunc_normal_
from timm.models.layers import DropPath
from typing import Sequence
from torch import Tensor
import copy
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from models.layers.max_vit import MaxViT


# from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange, Reduce
# from torchsummary import summary


class PatchEmbedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False,  store_layers_attn=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if store_layers_attn:
            self.attn_maps = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

        # if return_attention:
        #     return x, attn
        # else:
        #     return x, None

# class WindowAttention(nn.Module):
#     """
#     Window based multi-head self attention module with relative position bias based on: "Liu et al.,
#     Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
#     <https://arxiv.org/abs/2103.14030>"
#     https://github.com/microsoft/Swin-Transformer
#     """
#
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int,
#             window_size: Sequence[int],
#             qkv_bias: bool = False,
#             attn_drop: float = 0.0,
#             proj_drop: float = 0.0,
#     ) -> None:
#         """
#         Args:
#             dim: number of feature channels.
#             num_heads: number of attention heads.
#             window_size: local window size.
#             qkv_bias: add a learnable bias to query, key, value.
#             attn_drop: attention dropout rate.
#             proj_drop: dropout rate of output.
#         """
#
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         mesh_args = torch.meshgrid.__kwdefaults__
#
#         if len(self.window_size) == 3:
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros(
#                     (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
#                     num_heads,
#                 )
#             )
#             coords_d = torch.arange(self.window_size[0])
#             coords_h = torch.arange(self.window_size[1])
#             coords_w = torch.arange(self.window_size[2])
#             if mesh_args is not None:
#                 coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
#             else:
#                 coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
#             coords_flatten = torch.flatten(coords, 1)
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#             relative_coords[:, :, 0] += self.window_size[0] - 1
#             relative_coords[:, :, 1] += self.window_size[1] - 1
#             relative_coords[:, :, 2] += self.window_size[2] - 1
#             relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
#             relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
#         elif len(self.window_size) == 2:
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
#             )
#             coords_h = torch.arange(self.window_size[0])
#             coords_w = torch.arange(self.window_size[1])
#             if mesh_args is not None:
#                 coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
#             else:
#                 coords = torch.stack(torch.meshgrid(coords_h, coords_w))
#             coords_flatten = torch.flatten(coords, 1)
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#             relative_coords[:, :, 0] += self.window_size[0] - 1
#             relative_coords[:, :, 1] += self.window_size[1] - 1
#             relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#
#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         trunc_normal_(self.relative_position_bias_table, std=0.02)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask):
#         b, n, c = x.shape
#         qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
#         ].reshape(n, n, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         attn = attn + relative_position_bias.unsqueeze(0)
#         if mask is not None:
#             nw = mask.shape[0]
#             attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, n, n)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn).to(v.dtype)
#         x = (attn @ v).transpose(1, 2).reshape(b, n, c)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, attn


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embed_size=768, num_heads=8, drop_path=0., forward_expansion=4, forward_drop_p=0., norm_layer=nn.LayerNorm ):
        super().__init__()
        self.norm1 = norm_layer(embed_size)
        self.attn = Attention(embed_size, num_heads=num_heads)
        self.norm2 = norm_layer(embed_size)
        self.mlp = Mlp(in_features=embed_size, hidden_features=forward_expansion, drop=forward_drop_p)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.patch_embed = PatchEmbedding(patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=embed_size)

    def forward(self, x, return_attention=False, store_layers_attn=False):
        out_attn, attn_map = self.attn(self.norm1(x), store_layers_attn=store_layers_attn)
        x = x + self.drop_path(out_attn)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn_map
        else:
            return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, embed_size=768, num_heads=8, drop_path=0., forward_expansion=4, forward_drop_p=0.,
                 norm_layer=nn.LayerNorm, num_layers=6, norm_output=None, return_attention=False, store_layers_attn=False):
        super().__init__()
        encoder_layer = TransformerEncoderBlock(embed_size=embed_size,
                                                num_heads=num_heads,
                                                drop_path=drop_path,
                                                forward_expansion=forward_expansion,
                                                forward_drop_p=forward_drop_p,
                                                norm_layer=norm_layer)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.store_layers_attn = store_layers_attn
        self.norm_output = norm_output

    def forward(self, src):
        output = src
        for idx, layer in enumerate(self.layers):
            if idx < self.num_layers - 1:
                output = layer(output, store_layers_attn=self.store_layers_attn)
            else:
                output, attn = layer(output, return_attention=True, store_layers_attn=self.store_layers_attn)
        # for idx, layer in enumerate(self.layers):
        #     output, layer_attn = layer(output, return_attention=True)
        #     if idx == 0:
        #         attn = layer_attn.unsqueeze(1)
        #     else:
        #         attn = torch.cat([attn, layer_attn.unsqueeze(1)], dim=1)
        if self.norm_output is not None:
            output = self.norm_output(output)
        return output, attn

def build_transformer(args):
    store_layers_attn = args.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC == 'relevance_map'
    if args.MODEL.TRANSFORMER.TYPE == 'max_vit':
        return MaxViT(
                num_classes=2,
                dim=args.MODEL.TRANSFORMER.EMBED_SIZE//8,
                depth=(2,2,5,2),
                dim_head=args.MODEL.TRANSFORMER.EMBED_SIZE // args.MODEL.TRANSFORMER.HEADS,
                dim_conv_stem=None,
                window_size=7,
                mbconv_expansion_rate=4,
                mbconv_shrinkage_rate=0.25,
                dropout=0.1,
                channels=3
            )
    else:
        return TransformerEncoder(
            embed_size=args.MODEL.TRANSFORMER.EMBED_SIZE,
            num_heads=args.MODEL.TRANSFORMER.HEADS,
            drop_path=args.MODEL.TRANSFORMER.DROP_PATH,
            forward_expansion=args.MODEL.TRANSFORMER.FORWARD_EXPANSION_RATIO,
            forward_drop_p=args.MODEL.TRANSFORMER.FORWARD_DROP_P,
            norm_layer=nn.LayerNorm,
            num_layers=args.MODEL.TRANSFORMER.NUM_LAYERS,
            norm_output=None,
            store_layers_attn=store_layers_attn
        )
    # return TransformerEncoder(
    #     d_model=args.hidden_dim,
    #     dropout=args.dropout,
    #     nhead=args.nheads,
    #     dim_feedforward=args.dim_feedforward,
    #     num_encoder_layers=args.enc_layers,
    #     num_decoder_layers=args.dec_layers,
    #     normalize_before=args.pre_norm,
    #     return_intermediate_dec=True,
    # )

