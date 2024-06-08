import torch
from scipy.ndimage import zoom
from torch import nn
import torch.nn.functional as F
from einops import repeat

from models.layers.patch_embedding import build_patch_embedding
from models.layers.position_encoding import build_position_encoding
from models.layers.vit_encoder import build_vit_encoder


class VisionTransformerLGM(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, patch_embed, pos_encode, transformer, feat_size, num_classes=2, backbone_stages=4,
                 embed_dim=2048, use_cls_token=False, pos_embed_fit_mode='interpolate',
                 attention_3d=True, store_layers_attn=False, learned_beta=False, channel_reduction_srcs=[]):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbones to be used. See backbones.py
            transformer: torch module of the transformer architecture. See vit_encoder.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame,
                         thus 360 queries for 36 frames.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.feat_size = feat_size
        self.attention_3d = attention_3d
        self.patch_embed = patch_embed
        self.pos_encode = pos_encode
        self.backbone_stages = backbone_stages
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.avgpool = nn.AvgPool1d(feat_size * feat_size)
        self.store_layers_attn = store_layers_attn
        self.pos_embed_fit_mode = pos_embed_fit_mode

        if learned_beta:
            self.beta = nn.Parameter(torch.tensor(0.8))
        # if 'attn' in channel_reduction_srcs:
        #     self.channel_reduction_attn = nn.Conv2d(transformer.num_heads, 1, kernel_size=1, stride=1, padding=0)
        # if 'bb_feat' in channel_reduction_srcs:
        #     self.channel_reduction_embedding = nn.Conv2d(transformer.embed_size, 1, kernel_size=1, stride=1, padding=0)
        # if 'fusion_experimental' in channel_reduction_srcs:
        #     self.channel_reduction_exp_fusion = nn.Conv2d(transformer.num_heads+transformer.embed_size, 1, kernel_size=1, stride=1, padding=0)

        self.transformer = transformer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1 if num_classes == 2 else num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src = self.patch_embed(samples)
        if self.pos_encode is not None:
            pos = self.pos_encode(src).flatten(1, 2)
        bs, em, h, w = src.size()
        x = src.flatten(-2).permute(0, 2, 1)
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=bs)
            x = torch.cat([cls_tokens, x], dim=1)

        x = x
        if self.pos_encode is not None:
            if self.use_cls_token:
                x[:, 1:, :] += pos
            else:
                x += pos

            ###############
        # # x = torch.cat([self.cls_token.expand(f, -1, -1), x], dim=1)
        # out_transformer, attn_map = self.transformer(x)
        # out_transformer = out_transformer.permute(0,2,1)
        #
        # outputs_class = self.mlp_head(self.avgpool(out_transformer).squeeze())
        ##############
        # x = torch.cat([self.cls_token.expand(f, -1, -1), x], dim=1)


        if self.attention_3d:
            x = x.flatten(0,1).unsqueeze(0)
        out_transformer, attn = self.transformer(x)

        if self.attention_3d:
            out_transformer = out_transformer.reshape(bs, h * w, em)

        out_transformer = out_transformer.permute(0,2,1)

        if self.use_cls_token:
            outputs_class = self.mlp_head(out_transformer[:, :, 0])
        else:
            outputs_class = self.mlp_head(self.avgpool(out_transformer).squeeze())
        return outputs_class, attn, out_transformer

def build_model(config):
    device = torch.device(config.DEVICE)
    feat_size = config.TRAINING.INPUT_SIZE // config.MODEL.PATCH_SIZE
    pos_embed = config.MODEL.POSITION_EMBEDDING.TYPE is not None
    learned_beta = True if config.TRAINING.LOSS.LOCALIZATION_LOSS.FUSION_BETA == 'learned' else False
    if config.TRAINING.LOSS.LOCALIZATION_LOSS.FEAT_CHANNEL_REDUCTION == 'learned':
        if config.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC == 'attn':
            channel_reduction_srcs = ['attn']
        elif config.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC == 'bb_feat':
            channel_reduction_srcs = ['bb_feat']
        elif config.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC == 'fusion':
            channel_reduction_srcs = ['attn', 'bb_feat']
        elif config.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC == 'fusion_experimental':
            channel_reduction_srcs = ['fusion_experimental']
        else:
            channel_reduction_srcs = []
    else:
        channel_reduction_srcs = []

    patch_embed = build_patch_embedding(config)
    pos_encode = build_position_encoding(config)
    encoder = build_vit_encoder(config)

    model = VisionTransformerLGM(
        patch_embed,
        pos_encode,
        encoder,
        feat_size=feat_size,
        num_classes=config.MODEL.NUM_CLASSES,
        backbone_stages=config.MODEL.PATCH_EMBED.BACKBONE_STAGES,
        embed_dim=config.MODEL.TRANSFORMER.EMBED_SIZE,
        use_cls_token=config.MODEL.TRANSFORMER.USE_CLS_TOKEN,
        pos_embed_fit_mode=config.MODEL.POSITION_EMBEDDING.FIT_MODE,
        attention_3d=config.MODEL.TRANSFORMER.ATTENTION_3D,
        learned_beta=learned_beta,
        channel_reduction_srcs=channel_reduction_srcs
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
