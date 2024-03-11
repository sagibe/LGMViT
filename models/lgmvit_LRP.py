import torch
from scipy.ndimage import zoom
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

from models.layers.position_encoding import PositionalEncodingSine2D
from utils.layers_lrp import *

# from models.layers.patch_embedding import build_patch_embedding
# from models.layers.transformer import build_transformer

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention
class PatchEmbedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, img_size=256, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size // self.patch_size), (self.img_size // self.patch_size))
        return self.proj.relprop(cam, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, return_attention=False,  store_layers_attn=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        self.save_v(v)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        dots = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        # if store_layers_attn:
        self.save_attn(attn)
        if attn.requires_grad is True:
            attn.register_hook(self.save_attn_gradients)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.matmul2([attn, v])
        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embed_size=768, num_heads=8, mlp_ratio=4., forward_drop_p=0., attn_drop=0., norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_size)
        self.attn = Attention(embed_size, num_heads=num_heads, attn_drop=attn_drop)
        self.norm2 = norm_layer(embed_size)
        mlp_hidden_dim = int(embed_size * mlp_ratio)
        self.mlp = Mlp(in_features=embed_size, hidden_features=mlp_hidden_dim, drop=forward_drop_p)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()
        # self.patch_embed = PatchEmbedding(patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=embed_size)

    def forward(self, x, return_attention=False, store_layers_attn=False):
        # out_attn, attn_map = self.attn(self.norm1(x), store_layers_attn=store_layers_attn)
        # x = x + self.drop_path(out_attn)
        # # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        x1, x2 = self.clone1(x, 2)
        out1, attn_mat = self.attn(self.norm1(x2), store_layers_attn=store_layers_attn)
        x = self.add1([x1, out1])
        x1, x2 = self.clone2(x, 2)
        out2 = self.mlp(self.norm2(x))
        x = self.add2([x1, out2])

        if return_attention:
            return x, attn_mat
        else:
            return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam

class VisionTransformerLGMLRP(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, img_size,patch_size, in_chans=3, num_classes=2, embed_dim=2048, depth=12,
                 num_heads=12, mlp_ratio=4., use_cls_token=False, use_pos_embed=True, pos_embed_fit_mode='interpolate',
                 attention_3d=True, store_layers_attn=False, drop_rate=0., attn_drop=0.):
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
        self.attention_3d = attention_3d
        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_pos_embed = use_pos_embed
        self.pos_embed_fit_mode = pos_embed_fit_mode
        self.store_layers_attn = store_layers_attn
        self.num_layers = depth

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, stride=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = PositionalEncodingSine2D(embed_dim)

        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderBlock(
                embed_size=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                forward_drop_p=drop_rate, attn_drop=attn_drop)
            for i in range(depth)])
        self.norm = LayerNorm(embed_dim)
        self.mlp_head = Linear(embed_dim, 1 if num_classes == 2 else num_classes)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, 1 if num_classes == 2 else num_classes),
        #     # nn.Softmax(dim=1)
        # )
        self.add = Add()
    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

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
        embeds = self.patch_embed(samples)
        # if self.use_pos_embed:
        #     pos = self.pos_embed(embeds)[0].unsqueeze(0)
        #     # embeds += pos.permute(0, 3, 1, 2)
        #     embeds = self.add([embeds, pos.permute(0, 3, 1, 2)])
        if self.use_pos_embed:
            pos = self.pos_embed(embeds)[0].flatten(start_dim=0, end_dim=1).unsqueeze(0)
            pos = torch.cat([pos, torch.zeros((1,1,pos.shape[-1]), device=pos.device, dtype=pos.dtype)],dim=1)
        src = embeds
        bs, em, h, w = src.size()
        src_proj = src
        src_proj = src_proj.flatten(-2).permute(0, 2, 1)

        # if self.use_pos_embed:
        #     # pos = self.pos_embed(src_proj)
        #     # embeds += pos.permute(0, 3, 1, 2)
        #     src_proj = self.add([src_proj, pos])

        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=bs)
            src_proj = torch.cat([cls_tokens, src_proj], dim=1)
            if self.use_pos_embed:
                src_proj = self.add([src_proj, pos])

        x = src_proj
        # if self.use_pos_embed:
        #     pos_last = pos[-1]
        #     pos_last = pos_last.flatten(1, 2)
        #     if self.use_cls_token:
        #         x[:, 1:, :] += pos_last
        #     else:
        #         x = src_proj + pos_last

            ###############
        # # x = torch.cat([self.cls_token.expand(f, -1, -1), x], dim=1)
        # out_transformer, attn_map = self.transformer(x)
        # out_transformer = out_transformer.permute(0,2,1)
        #
        # outputs_class = self.mlp_head(self.avgpool(out_transformer).squeeze())
        ##############
        # x = torch.cat([self.cls_token.expand(f, -1, -1), x], dim=1)
        if x.requires_grad:
            x.register_hook(self.save_inp_grad)
        if self.attention_3d:
            x = x.flatten(0,1).unsqueeze(0)

        ########
        for idx, blk in enumerate(self.transformer_encoder):
            if idx < self.num_layers - 1:
                x = blk(x, store_layers_attn=self.store_layers_attn)
            else:
                out_transformer, attn = blk(x, return_attention=True, store_layers_attn=self.store_layers_attn)
        #########
        # out_transformer, attn = self.transformer_encoder(x)

        if self.attention_3d:
            out_transformer = out_transformer.reshape(bs, h * w, em)

        out_transformer = out_transformer.permute(0,2,1)

        if self.use_cls_token:
            outputs_class = self.mlp_head(self.norm(out_transformer[:, :, 0]))
        else:
            outputs_class = self.mlp_head(self.avgpool(out_transformer).squeeze())
        return outputs_class, attn, out_transformer

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        # cam = self.mlp_head.relprop(cam, **kwargs)
        # cam = cam.unsqueeze(1)
        # # cam = self.pool.relprop(cam, **kwargs)
        # cam = self.norm.relprop(cam, **kwargs)
        # for blk in reversed(self.transformer_encoder):
        #     cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            cam = self.mlp_head.relprop(cam, **kwargs)
            cam = cam.unsqueeze(1)
            # cam = self.pool.relprop(cam, **kwargs)
            cam = self.norm.relprop(cam, **kwargs)
            for blk in reversed(self.transformer_encoder):
                cam = blk.relprop(cam, **kwargs)

            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.transformer_encoder:
                attn_heads = blk.attn.get_attn().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam

        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.transformer_encoder:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn()
                # cam = blk.attn.get_attn_cam()
                # cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                # grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=1)
                cams.append(cam)
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam

        elif method == "last_layer":
            cam = self.transformer_encoder[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.transformer_encoder[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.transformer_encoder[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.transformer_encoder[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.transformer_encoder[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

def build_model_with_LRP(config):
    device = torch.device(config.DEVICE)
    feat_size = config.TRAINING.INPUT_SIZE // config.MODEL.PATCH_SIZE
    pos_embed = config.MODEL.POSITION_EMBEDDING.TYPE is not None
    # patch_embed = PatchEmbedding(img_size=config.TRAINING.INPUT_SIZE,
    #                              patch_size=config.MODEL.PATCH_SIZE,
    #                              stride=config.MODEL.PATCH_SIZE,
    #                              in_chans=3,
    #                              embed_dim=config.MODEL.TRANSFORMER.EMBED_SIZE)

    # transformer = build_transformer(config)
    model = VisionTransformerLGMLRP(
        img_size=config.TRAINING.INPUT_SIZE,
        patch_size=config.MODEL.PATCH_SIZE,
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.TRANSFORMER.EMBED_SIZE,
        depth=config.MODEL.TRANSFORMER.NUM_LAYERS,
        num_heads=config.MODEL.TRANSFORMER.HEADS,
        mlp_ratio=config.MODEL.TRANSFORMER.FORWARD_EXPANSION_RATIO,
        use_cls_token=config.TRAINING.USE_CLS_TOKEN,
        use_pos_embed=pos_embed,
        pos_embed_fit_mode=config.MODEL.POSITION_EMBEDDING.FIT_MODE,
        attention_3d=config.MODEL.TRANSFORMER.ATTENTION_3D,
        store_layers_attn=True
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

    # # matcher = build_matcher(args)
    # weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    # weight_dict['loss_giou'] = args.giou_loss_coef
    # if args.masks:
    #     weight_dict["loss_mask"] = args.mask_loss_coef
    #     weight_dict["loss_dice"] = args.dice_loss_coef
    # # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_d ict = {}
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