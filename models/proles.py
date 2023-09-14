import torch
from scipy.ndimage import zoom
from torch import nn
import torch.nn.functional as F

from models.layers.backbone import build_backbone
from models.layers.transformer import build_transformer


class ProLesClassifier(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, backbone, transformer, feat_size, num_classes=2, backbone_stages=4,
                 embed_dim=2048, use_pos_embed=True, pos_embed_fit_mode='interpolate', attention_3d=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbones to be used. See backbones.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame,
                         thus 360 queries for 36 frames.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.feat_size = feat_size
        self.attention_3d = attention_3d
        self.backbone = backbone
        self.backbone_stages = backbone_stages
        self.use_pos_embed = use_pos_embed
        self.pos_embed_fit_mode = pos_embed_fit_mode
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.avgpool = nn.AvgPool1d(feat_size*feat_size)
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
        features, pos = self.backbone(samples, use_pos_embed=self.use_pos_embed)
        src = features[-1]
        f, em, h, w = src.size()
        src_proj = src
        src_proj = src_proj.flatten(-2).permute(0, 2, 1)

        if self.use_pos_embed:
            pos_last = pos[-1]
            pos_last = pos_last.flatten(1, 2)
            x = src_proj + pos_last
        else:
            x = src_proj
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
            out_transformer = out_transformer.reshape(f, h * w, em)

        out_transformer = out_transformer.permute(0,2,1)
        backbone_feat_maps = out_transformer.reshape(f, em, h, w)

        outputs_class = self.mlp_head(self.avgpool(out_transformer).squeeze())
        return outputs_class, attn, backbone_feat_maps

def build_model(args):
    device = torch.device(args.DEVICE)
    feat_size = args.DATA.INPUT_SIZE // args.MODEL.PATCH_SIZE
    pos_embed = args.MODEL.POSITION_EMBEDDING.TYPE is not None
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = ProLesClassifier(
        backbone,
        transformer,
        feat_size=feat_size,
        num_classes=args.MODEL.NUM_CLASSES,
        backbone_stages=args.MODEL.BACKBONE.BACKBONE_STAGES,
        embed_dim=args.MODEL.TRANSFORMER.EMBED_SIZE,
        use_pos_embed=pos_embed,
        pos_embed_fit_mode=args.MODEL.POSITION_EMBEDDING.FIT_MODE,
        attention_3d=args.MODEL.TRANSFORMER.ATTENTION_3D
    )
    return model

    # # matcher = build_matcher(args)
    # weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    # weight_dict['loss_giou'] = args.giou_loss_coef
    # if args.masks:
    #     weight_dict["loss_mask"] = args.mask_loss_coef
    #     weight_dict["loss_dice"] = args.dice_loss_coef
    # # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)
    #
    # losses = ['labels', 'boxes', 'cardinality']
    # if args.masks:
    #     losses += ["masks"]
    # criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
    #                          eos_coef=args.eos_coef, losses=losses)
    # criterion.to(device)
    # postprocessors = {'bbox': PostProcess()}
    # if args.masks:
    #     postprocessors['segm'] = PostProcessSegm()
    # return model