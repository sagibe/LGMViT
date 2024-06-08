import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm.models.layers import DropPath
from typing import Sequence
from torch import Tensor
import copy
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

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

    #     self.attn_gradients = None
    # def save_attn_gradients(self, attn_gradients):
    #     self.attn_gradients = attn_gradients
    #
    # def get_attn_gradients(self):
    #     return self.attn_gradients
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
        #
        # if attn.requires_grad:
        #     attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

        # if return_attention:
        #     return x, attn
        # else:
        #     return x, None


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
        # self.mlp = Mlp(in_features=embed_size, hidden_features=forward_expansion, drop=forward_drop_p)
        mlp_hidden_dim = int(embed_size * forward_expansion)
        self.mlp = Mlp(in_features=embed_size, hidden_features=mlp_hidden_dim, drop=forward_drop_p)
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

    def __init__(self, embed_size=768, num_heads=12, drop_path=0., forward_expansion=4, forward_drop_p=0.,
                 norm_layer=nn.LayerNorm, num_layers=12, norm_output=None, return_attention=False, store_layers_attn=False):
        super().__init__()
        encoder_layer = TransformerEncoderBlock(embed_size=embed_size,
                                                num_heads=num_heads,
                                                drop_path=drop_path,
                                                forward_expansion=forward_expansion,
                                                forward_drop_p=forward_drop_p,
                                                norm_layer=norm_layer)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.store_layers_attn = store_layers_attn
        self.norm_output = norm_output

    def forward(self, src):
        output = src
        for idx, layer in enumerate(self.layers):
            if idx < self.num_layers - 1:
                output = layer(output, store_layers_attn=self.store_layers_attn)
            else:
                output, attn = layer(output, return_attention=True, store_layers_attn=self.store_layers_attn)
        if self.norm_output is not None:
            output = self.norm_output(output)
        return output, attn

def build_vit_encoder(args):
    # store_layers_attn = args.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC == 'relevance_map'
    store_layers_attn = args.TRAINING.LOSS.LOCALIZATION_LOSS.ATTENTION_METHOD in ['relevance_map', 'rollout']
    return TransformerEncoder(
        embed_size=args.MODEL.VIT_ENCODER.EMBED_SIZE,
        num_heads=args.MODEL.VIT_ENCODER.HEADS,
        drop_path=args.MODEL.VIT_ENCODER.DROP_PATH,
        forward_expansion=args.MODEL.VIT_ENCODER.FORWARD_EXPANSION_RATIO,
        forward_drop_p=args.MODEL.VIT_ENCODER.FORWARD_DROP_P,
        norm_layer=nn.LayerNorm,
        num_layers=args.MODEL.VIT_ENCODER.NUM_LAYERS,
        norm_output=None,
        store_layers_attn=store_layers_attn
    )

