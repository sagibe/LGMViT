import torch
from torch import nn
from einops import repeat

from models.layers.patch_embedding import build_patch_embedding
from models.layers.position_encoding import build_position_encoding
from models.layers.vit_encoder import build_vit_encoder


class VisionTransformerLGM(nn.Module):
    """
    Localization-Guided Medical Vision Transformer (LGM-ViT) module.
    """
    def __init__(self, patch_embed, pos_encode, vit_encoder, feat_size, num_classes=2,
                 embed_dim=768, use_cls_token=True, attention_3d=False, store_layers_attn=False): # channel_reduction_srcs=[]
        """
        Initializes the model.
        Parameters:
        - patch_embed (nn.Module): Module for converting input images into patch embeddings.
        - pos_encode (nn.Module or None): Module for adding positional encoding to patch embeddings. If None, no positional encoding is applied.
        - vit_encoder (nn.Module): Transformer encoder that processes the sequence of patches.
        - feat_size (int): Feature size of each patch grid dimension.
        - num_classes (int): Number of classes for the classification task. Defaults to 2.
        - embed_dim (int): Dimensionality of the patch embeddings. Default is 768.
        - use_cls_token (bool): Flag to include a class token for classification. Default is True.
        - attention_3d (bool): Flag for enabling 3D attention in the encoder. Default is False.
        - store_layers_attn (bool): Whether to store attention maps from each encoder layer. Default is False.
        """
        super().__init__()
        self.feat_size = feat_size
        self.attention_3d = attention_3d
        self.patch_embed = patch_embed
        self.pos_encode = pos_encode
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.avgpool = nn.AvgPool1d(feat_size * feat_size)
        self.store_layers_attn = store_layers_attn

        self.vit_encoder = vit_encoder
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1 if num_classes == 2 else num_classes),
        )

    def forward(self, samples):
        """
        Forward pass of the LGM-ViT.

        Parameters:
        - samples (Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
        - out_class (Tensor): Classification output tensor of shape (batch_size, num_classes).
        - attn (Tensor): Attention map from the last encoder layer.
        - out_encoder (Tensor): Encoder's final representation tensor of shape (batch_size, embed_dim, seq_length).
        """
        tokens = self.patch_embed(samples)
        if self.pos_encode is not None:
            pos = self.pos_encode(tokens).flatten(1, 2)
        bs, em, h, w = tokens.size()
        x = tokens.flatten(-2).permute(0, 2, 1)

        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=bs)
            x = torch.cat([cls_tokens, x], dim=1)

        if self.pos_encode is not None:
            if self.use_cls_token:
                x[:, 1:, :] += pos
            else:
                x += pos

        if self.attention_3d:
            x = x.flatten(0,1).unsqueeze(0)

        out_encoder, attn = self.vit_encoder(x)

        if self.attention_3d:
            if self.use_cls_token:
                out_encoder = out_encoder.reshape(bs, h * w + 1, em)
            else:
                out_encoder = out_encoder.reshape(bs, h * w, em)
        out_encoder = out_encoder.permute(0, 2, 1)

        if self.use_cls_token:
            out_class = self.mlp_head(out_encoder[:, :, 0])
        else:
            out_class = self.mlp_head(self.avgpool(out_encoder).squeeze())
        return out_class, attn, out_encoder

def build_model(config):
    """
    Build the LGM-ViT model.
    """
    feat_size = config.TRAINING.INPUT_SIZE // config.MODEL.PATCH_SIZE
    patch_embed = build_patch_embedding(config)
    if config.MODEL.POSITION_EMBEDDING.TYPE is not None:
        pos_encode = build_position_encoding(config)
    else:
        pos_encode = None
    encoder = build_vit_encoder(config)

    model = VisionTransformerLGM(
        patch_embed,
        pos_encode,
        encoder,
        feat_size=feat_size,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.VIT_ENCODER.EMBED_SIZE,
        use_cls_token=config.MODEL.VIT_ENCODER.USE_CLS_TOKEN,
        attention_3d=config.MODEL.VIT_ENCODER.ATTENTION_3D,
    )

    if config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD == 'learned_S1':
        # shallow imputation without X
        model.imp = nn.Conv2d(1, 1, 32, stride=16, padding=8)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD == 'learned_S2':
        # shallow imputation with X as additional input
        model.imp = nn.Conv2d(4, 1, 32, stride=16, padding=8)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD == 'learned_D1':
        # deep imputation without X
        model.imp_conv1 = nn.Conv2d(1, 1, 7, stride=2, padding=3)
        model.imp_conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        model.imp_conv3 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        model.imp_conv4 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        # model.imp_conv5 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD == 'learned_D2':
        # deep imputation with X as residual input
        model.imp_conv1 = nn.Conv2d(4, 4, 7, stride=2, padding=3)
        model.imp_conv2 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        model.imp_conv3 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        model.imp_conv4 = nn.Conv2d(4, 1, 3, stride=2, padding=1)
        # model.imp_conv5 = nn.Conv2d(4, 1, 3, stride=2, padding=1)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD in ['learned_d1', 'learned_ds1']:
        input_dims = 1 if config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD == 'learned_d1' else 4
        kernel_size = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE
        padding = kernel_size // 2
        model.imp = nn.Conv2d(input_dims, 1, kernel_size, stride=1, padding=padding)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD in ['learned_d2', 'learned_ds2']:
        input_dims = 1 if config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD == 'learned_d2' else 4
        if isinstance(config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE, list):
            kernel_size_l1 = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE[0]
            kernel_size_l2 = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE[1]
        else:
            kernel_size_l1 = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE
            kernel_size_l2 = (kernel_size_l1 // 2) // 2 * 2 + 1
        padding_l1 = kernel_size_l1 // 2
        padding_l2 = kernel_size_l2 // 2
        model.imp_conv1 = nn.Conv2d(input_dims, input_dims, kernel_size_l1, stride=1, padding=padding_l1)
        model.imp_conv2 = nn.Conv2d(input_dims, 1, kernel_size_l2, stride=1, padding=padding_l2)
    elif config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD in ['learned_d3', 'learned_ds3']:
        input_dims = 1 if config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD == 'learned_d3' else 4
        if isinstance(config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE, list):
            kernel_size_l1 = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE[0]
            kernel_size_l2 = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE[1]
            kernel_size_l3 = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE[2]
        else:
            kernel_size_l1 = config.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE
            kernel_size_l2 = (kernel_size_l1 // 2) // 2 * 2 + 1
            kernel_size_l3 = (kernel_size_l2 // 2) // 2 * 2 + 1
        padding_l1 = kernel_size_l1 // 2
        padding_l2 = kernel_size_l2 // 2
        padding_l3 = kernel_size_l3 // 2
        model.imp_conv1 = nn.Conv2d(input_dims, input_dims, kernel_size_l1, stride=1, padding=padding_l1)
        model.imp_conv2 = nn.Conv2d(input_dims, input_dims, kernel_size_l2, stride=1, padding=padding_l2)
        model.imp_conv3 = nn.Conv2d(input_dims, 1, kernel_size_l3, stride=1, padding=padding_l3)

    return model
