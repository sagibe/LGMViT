import torch
import torch.nn.functional as F
from torch import nn

from models.backbone import build_backbone
from models.transformer import build_transformer


class VisTRcls(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, backbone, transformer, num_frames, num_classes=2, embed_dim=2048):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame,
                         thus 360 queries for 36 frames.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer = transformer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1 if num_classes == 2 else num_classes),
            nn.Softmax(dim=1)
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
        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)
        # moved the frame to batch dimension for computation efficiency
        features, pos = self.backbone(samples)
        src = features[-1]
        f, em, h, w = src.size()
        pos = pos[-1]
        pos = pos[0,:f,:,:,:].flatten(-2).permute(0,2,1)
        # pos = pos.permute(0, 2, 1, 3, 4).flatten(-2)  # 1,0,2,3

        # src_proj = self.input_proj(src)
        src_proj = src
        # n,c,h,w = src_proj.shape
        # src_proj = src_proj.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(0,2,1,3,4).flatten(-2)
        src_proj = src_proj.flatten(-2).permute(0,2,1)
        # mask = mask.reshape(n//self.num_frames, self.num_frames, h*w)
        # pos = pos.permute(0,2,1,3,4).flatten(-2)

        ##### #TODO fix size of pos embeddings
        pos = nn.functional.interpolate(pos, size=(em), mode='linear', align_corners=False)
        #####
        x = src_proj + pos
        x = torch.cat([self.cls_token.expand(f, -1, -1), x], dim=1)
        out_transformer = self.transformer(x)

        outputs_class = self.mlp_head(out_transformer[:,0,:])
        # outputs_class = self.class_embed(hs)
        # outputs_coord = self.bbox_embed(hs).sigmoid()
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return outputs_class #out

def build_model(args):
    # if args.dataset_file == "ytvos":
    #     num_classes = 41
    device = torch.device(args.device)
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = VisTRcls(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
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