from torch import nn
from timm.models.layers import DropPath
import copy

class Attention(nn.Module):
    """
    Implements multi-head self-attention module.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Initializes the Attention layer.

        Parameters:
        - dim (int): Dimensionality of the input features.
        - num_heads (int): Number of attention heads. Default is 8.
        - qkv_bias (bool): If True, includes a bias term in the linear layers for query, key, and value.
        - attn_drop (float): Dropout rate for the attention scores. Default is 0.
        - proj_drop (float): Dropout rate for the output projection. Default is 0.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,  store_layers_attn=False):
        """
        Forward pass for computing multi-head self-attention.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, num_tokens, dim).
        - store_layers_attn (bool): If True, stores the computed attention maps as `self.attn_maps`.

        Returns:
        - x (Tensor): Output tensor after applying attention and projection, of shape (batch_size, num_tokens, dim).
        - attn (Tensor): The attention weights of shape (batch_size, num_heads, num_tokens, num_tokens).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if store_layers_attn:
            self.attn_maps = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerEncoderBlock(nn.Sequential):
    """
    Vision Transformer encoder block.
    """
    def __init__(self, embed_size=768, num_heads=8, drop_path=0., forward_expansion=4, forward_drop_p=0., norm_layer=nn.LayerNorm ):
        '''
        Initializes the Vision Transformer encoder block.

        Parameters:
        - embed_size (int): The dimensionality of the input features. Default is 768.
        - num_heads (int): Number of attention heads in the attention layer. Default is 8.
        - drop_path (float): Dropout rate applied to residual connections, adding stochastic depth. Default is 0.
        - forward_expansion (int): Expansion factor for the hidden dimension in the MLP layer. Default is 4.
        - forward_drop_p (float): Dropout rate in the MLP layer. Default is 0.
        - norm_layer (nn.Module): Normalization layer to use before attention and MLP layers. Default is nn.LayerNorm.
        '''
        super().__init__()
        self.norm1 = norm_layer(embed_size)
        self.attn = Attention(embed_size, num_heads=num_heads)
        self.norm2 = norm_layer(embed_size)
        mlp_hidden_dim = int(embed_size * forward_expansion)
        self.mlp = Mlp(in_features=embed_size, hidden_features=mlp_hidden_dim, drop=forward_drop_p)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_attention=False, store_layers_attn=False):
        """
        Forward pass of the Vision Transformer encoder block.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, num_tokens, embed_size).
        - return_attention (bool): If True, returns the attention map.
        - store_layers_attn (bool): If True, stores the attention map in the attention layer.

        Returns:
        - x (Tensor): Output tensor after applying self-attention and MLP.
        - attn_map (Tensor, optional): The attention map if `return_attention` is True.
        """
        out_attn, attn_map = self.attn(self.norm1(x), store_layers_attn=store_layers_attn)
        x = x + self.drop_path(out_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn_map
        else:
            return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    """
    Vision Transformer encoder.
    """
    def __init__(self, embed_size=768, num_heads=12, drop_path=0., forward_expansion=4, forward_drop_p=0.,
                 norm_layer=nn.LayerNorm, num_layers=12, norm_output=None, store_layers_attn=False):
        '''
        Parameters:
        - embed_size (int): Dimensionality of the input embedding. Default is 768.
        - num_heads (int): Number of attention heads in each attention layer. Default is 12.
        - drop_path (float): Dropout rate for stochastic depth (residual connection dropout). Default is 0.
        - forward_expansion (int): Expansion factor for hidden dimensions in MLP layers. Default is 4.
        - forward_drop_p (float): Dropout probability for the MLP. Default is 0.
        - norm_layer (nn.Module): Normalization layer applied within each Transformer encoder block. Default is nn.LayerNorm.
        - num_layers (int): Number of Transformer encoder blocks. Default is 12.
        - norm_output (nn.Module or None): Optional normalization layer for the final output. Default is None.
        - store_layers_attn (bool): Whether to store attention maps from each encoder layer. Default is False.
        '''
        super().__init__()
        encoder_layer = TransformerEncoderBlock(embed_size=embed_size,
                                                num_heads=num_heads,
                                                drop_path=drop_path,
                                                forward_expansion=forward_expansion,
                                                forward_drop_p=forward_drop_p,
                                                norm_layer=norm_layer)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.store_layers_attn = store_layers_attn
        self.norm_output = norm_output

    def forward(self, src):
        """
        Forward pass of the Vision Transformer encoder.

        Parameters:
        - src (Tensor): Input tensor of shape (batch_size, seq_length, embed_size).

        Returns:
        - output (Tensor): Final output tensor after encoding, with shape (batch_size, seq_length, embed_size).
        - attn (Tensor): Attention map from the last encoder layer.
        """
        output = src
        for idx, layer in enumerate(self.layers):
            if idx < self.num_layers - 1:
                output = layer(output, store_layers_attn=self.store_layers_attn)
            else:
                output, attn = layer(output, return_attention=True, store_layers_attn=self.store_layers_attn)
        if self.norm_output is not None:
            output = self.norm_output(output)
        return output, attn

def build_vit_encoder(args):
    '''
    Builds the Vision Transformer encoder.
    '''
    store_layers_attn = args.TRAINING.LOSS.LOCALIZATION_LOSS.ATTENTION_METHOD in ['relevance_map', 'rollout']
    return TransformerEncoder(
        embed_size=args.MODEL.VIT_ENCODER.EMBED_SIZE,
        num_heads=args.MODEL.VIT_ENCODER.HEADS,
        drop_path=args.MODEL.VIT_ENCODER.DROP_PATH,
        forward_expansion=args.MODEL.VIT_ENCODER.FORWARD_EXPANSION_RATIO,
        forward_drop_p=args.MODEL.VIT_ENCODER.FORWARD_DROP_P,
        norm_layer=nn.LayerNorm,
        num_layers=args.MODEL.VIT_ENCODER.NUM_LAYERS,
        norm_output=None,
        store_layers_attn=store_layers_attn
    )

