import torch
from scipy.ndimage import zoom
from torch import nn
import torch.nn.functional as F

from models.layers.backbone import build_backbone
from models.layers.transformer import build_transformer
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict

from models.layers.backbones.convnext import *
from utils.util import NestedTensor, is_main_process


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

class Resnet(nn.Module):
    def __init__(self, backbone, num_classes=2, backbone_stages=4):
        """ Initializes the model.
        Parameters:
            backbone:
            num_classes: number of object classes
        """
        super().__init__()
        self.backbone = backbone
        self.backbone_stages = backbone_stages
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1 if num_classes == 2 else num_classes)

        # self.classification_head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Linear(512, 1 if num_classes == 2 else num_classes),
        #     # nn.Softmax(dim=1)
        # )

    def forward(self, samples):
        features = self.backbone(samples)
        x = features['0']
        x = torch.flatten(self.avgpool(x), 1)
        out = self.fc(x)
        return out, None
        # outputs_class = self.mlp_head(self.avgpool(out_transformer).squeeze())
        # return outputs_class, attn_map #out

def build_resnet(args):
    device = torch.device(args.DEVICE)
    # feat_size = args.TRAINING.INPUT_SIZE // args.MODEL.PATCH_SIZE
    # pos_embed = args.MODEL.POSITION_EMBEDDING.TYPE is not None
    # backbone = build_backbone(args)
    backbone = ResNetBackbone(name=args.MODEL.PATCH_EMBED.NAME,
                              train_backbone=True,
                              return_interm_layers=False,
                              dilation=args.MODEL.PATCH_EMBED.DILATION,
                              backbone_stages=4)

    model = Resnet(
        backbone,
        backbone_stages=args.MODEL.PATCH_EMBED.BACKBONE_STAGES,
    )
    return model

