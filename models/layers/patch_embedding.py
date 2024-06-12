"""
Backbone modules.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import torch
import torchvision
from torch import nn
from typing import Dict

from models.layers.position_encoding import build_position_encoding

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


def build_patch_embedding(args):
    patch_embed = PatchEmbedding(patch_size=args.MODEL.PATCH_SIZE,
                              stride=args.MODEL.PATCH_SIZE,
                              padding=0,
                              in_chans=3,
                              embed_dim=args.MODEL.VIT_ENCODER.EMBED_SIZE)
    return patch_embed
