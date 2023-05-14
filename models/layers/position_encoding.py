"""
Various positional encodings for the transformer.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
import torch
from torch import nn

from utils.util import NestedTensor

# position encoding for 3 dims
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, embed_size=64, z_size=36, fit_mode='interpolate', temperature=10000, normalize=False, scale=None, device='cuda'):
        super().__init__()
        self.embed_size = embed_size
        self.fit_mode = fit_mode
        self.temperature = temperature
        self.normalize = normalize
        self.z_size = z_size
        self.device = device
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, scan):
        d, embed, h, w = scan.size()
        mask = torch.ones((self.z_size, h, w))
        mask = mask.reshape(1, self.z_size,h,w).to(self.device)
        # assert mask is not None
        # not_mask = ~mask
        z_embed = mask.cumsum(1, dtype=torch.float32)
        y_embed = mask.cumsum(2, dtype=torch.float32)
        x_embed = mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        # dim_t = torch.arange(self.embed_size // 3 + ((self.embed_size // 3) % 2), dtype=torch.float32, device=scan.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / (self.embed_size // 3))

        dim_t = torch.arange(embed// 3 + ((embed // 3) % 2), dtype=torch.float32, device=scan.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (embed // 3))


        # #####
        # pos_x = x_embed[:, :, :, :, None] / dim_t
        # pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        # if self.embed_size % 3 == 2:
        #     pos_y = y_embed[:, :, :, :, None] / dim_t[:-1]
        #     pos_y = torch.stack((pos_y[:, :, :, :, 2::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        # else:
        #     pos_y = y_embed[:, :, :, :, None] / dim_t
        #     pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        # if self.embed_size % 3 > 0:
        #     pos_z = z_embed[:, :, :, :, None] / dim_t[:-1]
        #     pos_z = torch.stack((pos_z[:, :, :, :, 2::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        # else:
        #     pos_z = z_embed[:, :, :, :, None] / dim_t
        #     pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        # #####

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).flatten(2, 3).squeeze(0)
        # pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        if pos.shape[2] != embed or pos.shape[0] != d:
            if self.fit_mode == 'interpolate':
                pos = pos.reshape(1,1,self.z_size,h*w,-1)
                pos = nn.functional.interpolate(pos, size=(d, h * w, embed), mode='trilinear',align_corners=False).squeeze()
                pos = pos.reshape(d, h, w, -1)
            elif self.fit_mode == 'prune':
                pos = pos.reshape(self.z_size, h * w, -1)
                pos = nn.functional.interpolate(pos, size=(embed), mode='linear', align_corners=False)
                pos = pos.reshape(self.z_size, h, w, -1)
                pos = pos[:d,:,:,:]
            else:
                raise NotImplementedError('Unknown positional embedding mode')

        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.dep_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(self, scan: NestedTensor):
        d, em, h, w = scan.size()
        i = torch.arange(w, device=scan.device)
        j = torch.arange(h, device=scan.device)
        # k = torch.arange(d, device=scan.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        # z_emb = self.dep_embed(k)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
            # z_emb.unsqueeze(2).repeat(1, 1, d),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(scan.shape[0], 1, 1, 1)
        return pos


import torch.nn as nn


# class LearnedPositionalEmbedding3D(nn.Module):
#     def __init__(self, embedding_dim, max_depth=40, max_height=256, max_width=256):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.depth_embedding = nn.Embedding(max_depth, embedding_dim)
#         self.height_embedding = nn.Embedding(max_height, embedding_dim)
#         self.width_embedding = nn.Embedding(max_width, embedding_dim)
#         self.position_embeddings = nn.Parameter(torch.zeros(1, embedding_dim))
#
#     def forward(self, inputs):
#         inputs = inputs.permute(0,2,3,1).unsqueeze(0)
#         batch_size, depth, height, width, channels = inputs.shape
#         depth_positions = torch.arange(depth, device=inputs.device, dtype=torch.long).repeat(batch_size, height, width, 1).transpose(2, 3)
#         height_positions = torch.arange(height, device=inputs.device, dtype=torch.long).repeat(batch_size, depth, width, 1).transpose(2, 3)
#         width_positions = torch.arange(width, device=inputs.device, dtype=torch.long).repeat(batch_size, depth, height, 1)
#         depth_embeddings = self.depth_embedding(depth_positions)
#         height_embeddings = self.height_embedding(height_positions)
#         width_embeddings = self.width_embedding(width_positions)
#         position_embeddings = self.position_embeddings.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
#         embeddings = depth_embeddings + height_embeddings + width_embeddings + position_embeddings
#         embeddings = embeddings.permute(0, 4, 1, 2, 3)
#         return embeddings

class LearnedPositionalEmbedding3D(nn.Module):
    """
    Absolute position embedding, learned.
    """
    def __init__(self, embedding_dim, max_depth=40, max_height=256, max_width=256):
        super().__init__()
        self.row_embed = nn.Embedding(max_width, embedding_dim // 3 + int(embedding_dim % 3 > 0))
        self.col_embed = nn.Embedding(max_height, embedding_dim // 3 + int(embedding_dim % 3 == 2))
        self.depth_embed = nn.Embedding(max_depth, embedding_dim // 3)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.depth_embed.weight)

    def forward(self, scan: NestedTensor):
        # x = tensor_list.tensors
        d, em, h, w = scan.size()
        # d, h, w = x.shape[-3:]
        i = torch.arange(w, device=scan.device)
        j = torch.arange(h, device=scan.device)
        k = torch.arange(d, device=scan.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.depth_embed(k)
        pos = torch.cat([
            x_emb.unsqueeze(0).unsqueeze(0).repeat(d, h, 1, 1),
            y_emb.unsqueeze(1).unsqueeze(0).repeat(d, 1, w, 1),
            z_emb.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1),
        ], dim=-1)
        # pos = torch.cat([
        #     x_emb.unsqueeze(0).unsqueeze(0).repeat(d, h, 1, 1),
        #     y_emb.unsqueeze(1).unsqueeze(0).repeat(d, 1, w, 1),
        #     z_emb.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1),
        # ], dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(1, 1, 1, 1, 1)
        return pos



def build_position_encoding(args):
    # N_steps = args.MODEL.TRANSFORMER.EMBED_SIZE // 3
    # modulo = args.MODEL.TRANSFORMER.EMBED_SIZE % 3
    if args.MODEL.POSITION_EMBEDDING.TYPE in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(embed_size=args.MODEL.TRANSFORMER.EMBED_SIZE,
                                                   z_size=args.MODEL.POSITION_EMBEDDING.Z_SIZE,
                                                   fit_mode=args.MODEL.POSITION_EMBEDDING.FIT_MODE,
                                                   normalize=True,
                                                   device=args.DEVICE)
    elif args.MODEL.POSITION_EMBEDDING.TYPE in ('v3', 'learned'):
        position_embedding = LearnedPositionalEmbedding3D(embedding_dim=args.MODEL.TRANSFORMER.EMBED_SIZE)
    elif args.MODEL.POSITION_EMBEDDING.TYPE is None:
        position_embedding = None
    else:
        raise ValueError(f"not supported {args.MODEL.POSITION_EMBEDDING.TYPE}")

    return position_embedding
