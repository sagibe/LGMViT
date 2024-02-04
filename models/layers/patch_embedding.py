"""
Backbone modules.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict

from models.layers.backbones.convnext import *
from models.layers.backbones.resnet import ResNetBackbone
from utils.util import NestedTensor, is_main_process

from models.layers.position_encoding import build_position_encoding
# from backbones.convnext import *

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

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, scan, use_pos_embed=True):
        xs = self[0](scan)
        out = []
        if use_pos_embed:
            pos = []
        else:
            pos = None
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                # position encoding
                if use_pos_embed:
                    pos.append(self[1](x).to(x.dtype))
        else:
            out.append(xs)
            if use_pos_embed:
                pos.append(self[1](xs).to(xs.dtype))

        return out, pos


def build_patch_embedding(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.TRAINING.LR > 0
    if 'resnet' in args.MODEL.PATCH_EMBED.NAME:
        backbone = ResNetBackbone(name=args.MODEL.PATCH_EMBED.NAME,
                                  train_backbone=train_backbone,
                                  return_interm_layers=args.MODEL.PATCH_EMBED.RETURN_INTERM_LAYERS,
                                  dilation=args.MODEL.PATCH_EMBED.DILATION,
                                  backbone_stages=args.MODEL.PATCH_EMBED.BACKBONE_STAGES)
    elif 'convnext' in args.MODEL.PATCH_EMBED.NAME:
        backbone = convnext_tiny(backbone_only=True,
                                 backbone_stages=args.MODEL.PATCH_EMBED.BACKBONE_STAGES)
    elif 'patch_embed' in args.MODEL.PATCH_EMBED.NAME:
        backbone = PatchEmbedding(patch_size=args.MODEL.PATCH_SIZE,
                                  stride=args.MODEL.PATCH_SIZE,
                                  padding=0,
                                  in_chans=3,
                                  embed_dim=args.MODEL.TRANSFORMER.EMBED_SIZE)

    model = Joiner(backbone, position_embedding)
    if 'resnet' in args.MODEL.PATCH_EMBED.NAME:
        model.num_channels = backbone.num_channels
    return model
