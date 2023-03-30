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
from models.layers.backbones.patch_embedding import PatchEmbedding
from utils.util import NestedTensor, is_main_process

from models.layers.position_encoding import build_position_encoding
# from backbones.convnext import *


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool, backbone_stages: int):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {}
            for idx in range(backbone_stages):
                return_layers[f'layer{idx+1}'] = str(idx)
            print('hi')

        else:
            return_layers = {f'layer{backbone_stages}': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, scan):
        xs = self.body(scan)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out


class ResNetBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 backbone_stages: int):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, backbone_stages)


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


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.TRAINING.LR > 0
    if 'resnet' in args.MODEL.BACKBONE.NAME:
        backbone = ResNetBackbone(name=args.MODEL.BACKBONE.NAME,
                                  train_backbone=train_backbone,
                                  return_interm_layers=args.MODEL.BACKBONE.RETURN_INTERM_LAYERS,
                                  dilation=args.MODEL.BACKBONE.DILATION,
                                  backbone_stages=args.MODEL.BACKBONE.BACKBONE_STAGES)
    elif 'convnext' in args.MODEL.BACKBONE.NAME:
        backbone = convnext_tiny(backbone_only=True,
                                 backbone_stages=args.MODEL.BACKBONE.BACKBONE_STAGES)
    elif 'patch_embed' in args.MODEL.BACKBONE.NAME:
        backbone = PatchEmbedding(patch_size=args.MODEL.PATCH_SIZE,
                                  stride=args.MODEL.PATCH_SIZE,
                                  padding=0,
                                  in_chans=3,
                                  embed_dim=args.MODEL.TRANSFORMER.EMBED_SIZE)

    model = Joiner(backbone, position_embedding)
    if 'resnet' in args.MODEL.BACKBONE.NAME:
        model.num_channels = backbone.num_channels
    return model
