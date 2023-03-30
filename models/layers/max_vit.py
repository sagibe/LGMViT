from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# MBConv

class SqueezeExcitation(nn.Module):
    # TODO add also 3D!!
    def __init__(self, dim, shrinkage_rate=0.25, spatial_dims=2):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)
        self.spatial_dims = spatial_dims
        self.create_gate(dim=dim, hidden_dim=hidden_dim)

    def create_gate(self, dim, hidden_dim):

        if self.spatial_dims == 2:
            self.gate = nn.Sequential(
                Reduce('b c h w -> b c', 'mean'),
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False),
                nn.Sigmoid(),
                Rearrange('b c -> b c 1 1')
            )
        else:
            self.gate = nn.Sequential(
                Reduce('b c d h w -> b c', 'mean'),
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False),
                nn.Sigmoid(),
                Rearrange('b c -> b c 1 1 1')
            )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(
        dim_in,
        dim_out,
        *,
        downsample,
        spatial_dims=2,
        expansion_rate=4,
        shrinkage_rate=0.25,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    convFunc = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
    batchFunc = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
    net = nn.Sequential(
        convFunc(dim_in, hidden_dim, 1),
        batchFunc(hidden_dim),
        nn.GELU(),
        convFunc(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        batchFunc(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate, spatial_dims=spatial_dims),
        convFunc(hidden_dim, dim_out, 1),
        batchFunc(dim_out)
    )
    # net = nn.Sequential(
    #     nn.Conv2d(dim_in, hidden_dim, 1),
    #     nn.BatchNorm2d(hidden_dim),
    #     nn.GELU(),
    #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
    #     nn.BatchNorm2d(hidden_dim),
    #     nn.GELU(),
    #     SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
    #     nn.Conv2d(hidden_dim, dim_out, 1),
    #     nn.BatchNorm2d(dim_out)
    # )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            spatial_dims=2,
            dropout=0.,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** spatial_dims,
                                         self.heads)

        self.spatial_dims = spatial_dims
        pos = torch.arange(window_size)
        if spatial_dims == 2:
            grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
        else:
            grid = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'))
            grid = rearrange(grid, 'c i j k -> (i j k) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        # We assume that the window size is equal in all axis
        rel_pos += window_size - 1
        if spatial_dims == 3:
            rel_pos[..., 0] *= (2 * window_size - 1) ** 2
            rel_pos[..., 1] *= 2 * window_size - 1
            rel_pos_indices = rel_pos.sum(dim=-1)
        else:
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        if x.ndim == 8:
            (batch,
             depth,
             height,
             width,
             window_depth,
             window_height,
             window_width,
             _,
             device,
             h) = (*x.shape,
                   x.device,
                   self.heads)
            x = rearrange(x, 'b z x y w1 w2 w3 c -> (b z x y) (w1 w2 w3) c')
            winlen = window_depth * window_height * window_width
        else:
            (batch,
             height,
             width,
             window_height,
             window_width,
             _,
             device,
             h) = (*x.shape,
                   x.device,
                   self.heads)
            x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
            winlen = window_height * window_width

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # bias = self.rel_pos_bias(self.rel_pos_indices)
        bias = self.rel_pos_bias(self.rel_pos_indices.clone()
                                 [:winlen, :winlen].reshape(-1))\
            .reshape(winlen, winlen, -1)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j c -> b h i c', attn, v)

        if self.spatial_dims == 3:
            out = rearrange(out, 'b h (w1 w2 w3) d -> b w1 w2 w3 (h d)',
                            w1=window_depth,
                            w2=window_height,
                            w3=window_width)
        else:
            out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out

        out = self.to_out(out)
        if self.spatial_dims == 3:
            return rearrange(out,
                             '(b z x y) ... -> b z x y ...',
                             z=depth,
                             x=height,
                             y=width)
        else:
            return rearrange(out,
                             '(b x y) ... -> b x y ...',
                             x=height,
                             y=width)


class MaxViT(nn.Module):
    def __init__(
            self,
            *,
            num_classes,
            dim,
            depth,
            dim_head=32,
            dim_conv_stem=None,
            window_size=7,
            mbconv_expansion_rate=4,
            mbconv_shrinkage_rate=0.25,
            dropout=0.1,
            channels=3
    ):
        super().__init__()
        assert isinstance(depth,
                          tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride=2, padding=1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=is_first,
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # block-like attention
                    PreNormResidual(layer_dim,
                                    Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
                    PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),  # grid-like attention
                    PreNormResidual(layer_dim,
                                    Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
                    PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x, extractFeatures=False):
        x = self.conv_stem(x)

        out = []
        for stage in self.layers:
            x = stage(x)
        if extractFeatures:
            return x, None
        else:
            return self.mlp_head(x)


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class MaxVitBlock(nn.Module):

    def __init__(self, stage_dim_in,
                 w,
                 dim_head,
                 spatial_dims,
                 layer_dim,
                 is_first,
                 mbconv_expansion_rate,
                 mbconv_shrinkage_rate,
                 dropout,
                 attn_block_z=False):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.attn_block_z = attn_block_z
        self.window_size = (w, w, w)
        self.mbconv = MBConv(
            stage_dim_in,
            layer_dim,
            spatial_dims=spatial_dims,
            downsample=is_first,
            expansion_rate=mbconv_expansion_rate,
            shrinkage_rate=mbconv_shrinkage_rate
        )
        self.layer1 = PreNormResidual(layer_dim,
                                      Attention(dim=layer_dim,
                                                dim_head=dim_head,
                                                spatial_dims=spatial_dims,
                                                dropout=dropout,
                                                window_size=w))
        self.layer2 = PreNormResidual(layer_dim,
                                      FeedForward(dim=layer_dim,
                                                  dropout=dropout))
        self.layer3 = PreNormResidual(layer_dim,
                                      Attention(dim=layer_dim,
                                                dim_head=dim_head,
                                                spatial_dims=spatial_dims,
                                                dropout=dropout,
                                                window_size=w))
        self.layer4 = PreNormResidual(layer_dim,
                                      FeedForward(dim=layer_dim,
                                                  dropout=dropout))

    def forward(self, x):
        x = self.mbconv(x)
        ws = get_window_size(x.shape[-self.spatial_dims:],
                             self.window_size)
        if self.spatial_dims == 2:
            x = rearrange(x, 'b c (x w1) (y w2) -> b x y w1 w2 c',
                          w1=ws[0],
                          w2=ws[1])
        else:
            x = rearrange(x, 'b c (z w1) (x w2) (y w3) -> b z x y w1 w2 w3 c',
                          w1=ws[0],
                          w2=ws[1],
                          w3=ws[2])

        # window attention
        x = self.layer1(x)
        x = self.layer2(x)

        if self.spatial_dims == 2:
            x = rearrange(x, 'b x y w1 w2 c -> b c (x w1) (y w2)',
                          w1=ws[0],
                          w2=ws[1])
        else:
            x = rearrange(x, 'b z x y w1 w2 w3 c -> b c (z w1) (x w2) (y w3)',
                          w1=ws[0],
                          w2=ws[1],
                          w3=ws[2])

        if self.spatial_dims == 2:
            x = rearrange(x, 'b c (w1 x) (w2 y) -> b x y w1 w2 c',
                          w1=ws[0],
                          w2=ws[1])
        else:
            # (w1 z) - grid
            # (z w1) - window (block)
            if self.attn_block_z:
                x = rearrange(x, 'b c (z w1) (w2 x) (w3 y) -> b z x y w1 w2 w3 c',
                              w1=ws[0],
                              w2=ws[1],
                              w3=ws[2])
            else:
                x = rearrange(x, 'b c (w1 z) (w2 x) (w3 y) -> b z x y w1 w2 w3 c',
                              w1=ws[0],
                              w2=ws[1],
                              w3=ws[2])

        # grid attention
        x = self.layer3(x)
        x = self.layer4(x)

        if self.spatial_dims == 2:
            x = rearrange(x, 'b x y w1 w2 c -> b c (w1 x) (w2 y)',
                          w1=ws[0],
                          w2=ws[1])
        else:
            x = rearrange(x, 'b z x y w1 w2 w3 c -> b c (w1 z) (w2 x) (w3 y)',
                          w1=ws[0],
                          w2=ws[1],
                          w3=ws[2])

        return x


class MaxViTSeg(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            spatial_dims=2,
            dim_head=32,
            dim_conv_stem=None,
            window_size=7,
            attn_block_z=False,
            mbconv_expansion_rate=4,
            mbconv_shrinkage_rate=0.25,
            dropout=0.1,
            channels=3
    ):
        super().__init__()
        assert isinstance(depth,
                          tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)
        self.spatial_dims = spatial_dims
        convFunc = nn.Conv2d if \
            spatial_dims == 2 else \
            nn.Conv3d

        self.conv_stem = nn.Sequential(
            convFunc(channels, dim_conv_stem, 3, stride=2, padding=1),
            convFunc(dim_conv_stem, dim_conv_stem, 3, padding=1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size
        self.outIndsToTake = []

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim
                block = MaxVitBlock(w=w,
                                    spatial_dims=spatial_dims,
                                    dim_head=dim_head,
                                    stage_dim_in=stage_dim_in,
                                    layer_dim=layer_dim,
                                    is_first=is_first,
                                    mbconv_expansion_rate=mbconv_expansion_rate,
                                    mbconv_shrinkage_rate=mbconv_shrinkage_rate,
                                    attn_block_z=attn_block_z,
                                    dropout=dropout)

                self.layers.append(block)
                if stage_ind == layer_depth - 1:
                    self.outIndsToTake.append(len(self.layers) - 1)

        # block = nn.Sequential(
        #     MBConv(
        #         stage_dim_in,
        #         layer_dim,
        #         downsample=is_first,
        #         expansion_rate=mbconv_expansion_rate,
        #         shrinkage_rate=mbconv_shrinkage_rate
        #     ),
        #     Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # block-like attention
        #     PreNormResidual(layer_dim,
        #                     Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
        #     PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
        #     Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
        #
        #     Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),  # grid-like attention
        #     PreNormResidual(layer_dim,
        #                     Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
        #     PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
        #     Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
        # )

    def forward(self, x, extractFeatures=False):
        x = self.conv_stem(x)
        out = [x]
        outFeatures = []
        for stageIdx, stage in enumerate(self.layers):
            x = stage(x)
            if stageIdx in self.outIndsToTake:
                out.append(x)
            elif extractFeatures:
                outFeatures.append(x)

        if extractFeatures:
            out += outFeatures

        return out
