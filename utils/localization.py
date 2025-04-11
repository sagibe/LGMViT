import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from typing import List, Optional, Tuple, Union
import warnings

def generate_spatial_attention(attn, mode='max_pool'):
    """
    Generates spatial attention maps from an attention tensor.

    Parameters:
    - attn (Tensor): Attention tensor with shape (batch_size, num_heads, num_tokens, num_tokens).
                      This tensor typically represents the attention weights computed by a multi-head
                      self-attention mechanism.
    - mode (str, optional): Specifies how to generate the spatial attention. Can be one of:
        - 'max_pool': Uses the maximum value of the attention across the spatial dimension (i.e., max pooling).
        - 'cls_token': Extracts the attention values corresponding to the class token and reshapes it.
                      This mode assumes that the first token corresponds to the class token.
        Default is 'max_pool'.

    Returns:
    - spat_attn (Tensor): Spatial attention map with shape (batch_size, num_heads, feat_size, feat_size),
                          where `feat_size` is the square root of the number of spatial tokens, representing the
                          height and width of the feature map.

    """
    bs, nh = attn.shape[0], attn.shape[1]
    if mode == 'max_pool':
        feat_size = int(np.sqrt(attn.shape[2]))
        spat_attn = attn.max(dim=-2)[0].view(bs, nh, feat_size, feat_size)
    elif mode == 'cls_token':
        feat_size = int(np.sqrt(attn.shape[2]-1))
        spat_attn = attn[:, :, 0, 1:].view(bs, nh, feat_size, feat_size)
    else:
        raise ValueError(f"{mode} spatial attention type not supported")
    return spat_attn

def generate_spatial_bb_map(bb_feats, mode='max_pool'): # TODO fix the max_pool naming error
    """
    Generates a spatial feature map from the backbone features.
    The spatial map can be computed by either using max pooling over the spatial dimension
    or by extracting the feature map corresponding to the class token.

    Parameters:
    - bb_feats (Tensor): Backbone feature tensor with shape (batch_size, embedding_dim, num_tokens),
                          where `num_tokens` is typically the flattened spatial dimension.
                          `embedding_dim` represents the size of the feature vector for each token.
    - mode (str, optional): Specifies how to generate the spatial feature map. Can be one of:
        - 'max_pool': Uses max pooling over the spatial dimension to compute the feature map.
        - 'cls_token': Extracts the feature corresponding to the class token and reshapes it.
        Default is 'max_pool'.

    Returns:
    - bb_feats (Tensor): A spatial feature map with shape (batch_size, embedding_dim, feat_size, feat_size),
                          where `feat_size` is the square root of the number of spatial tokens.
                          This map represents the spatial features from the backbone in a grid format.

    Raises:
    - ValueError: If an unsupported mode is provided.
    """
    bs, em = bb_feats.shape[0], bb_feats.shape[1]
    if mode == 'max_pool':
        feat_size = int(np.sqrt(bb_feats.shape[2]))
    elif mode == 'cls_token':
        feat_size = int(np.sqrt(bb_feats.shape[2] - 1))
        bb_feats = bb_feats[:, :, 1:]
    else:
        raise ValueError(f"{mode} spatial attention type not supported")
    return bb_feats.reshape(bs, em, feat_size, feat_size)

def extract_heatmap(featmap: torch.Tensor,
                 feat_interpolation = 'bilinear',
                 channel_reduction: Optional[str] = 'squeeze_mean',
                 topk: int = 20,
                 resize_shape: Optional[tuple] = None):  #TODO
    """Draw featmap.

    - If `overlaid_image` is not None, the final output image will be the
      weighted sum of img and featmap.

    - If `resize_shape` is specified, `featmap` and `overlaid_image`
      are interpolated.

    - If `resize_shape` is None and `overlaid_image` is not None,
      the feature map will be interpolated to the spatial size of the image
      in the case where the spatial dimensions of `overlaid_image` and
      `featmap` are different.

    - If `channel_reduction` is "squeeze_mean" and "select_max",
      it will compress featmap to single channel image and weighted
      sum to `overlaid_image`.

    - If `channel_reduction` is None

      - If topk <= 0, featmap is assert to be one or three
        channel and treated as image and will be weighted sum
        to ``overlaid_image``.
      - If topk > 0, it will select topk channel to show by the sum of
        each channel. At the same time, you can specify the `arrangement`
        to set the window layout.

    Args:
        featmap (torch.Tensor): The featmap to draw which format is
            (C, H, W).
        overlaid_image (np.ndarray, optional): The overlaid image.
            Defaults to None.
        channel_reduction (str, optional): Reduce multiple channels to a
            single channel. The optional value is 'squeeze_mean'
            or 'select_max'. Defaults to 'squeeze_mean'.
        topk (int): If channel_reduction is not None and topk > 0,
            it will select topk channel to show by the sum of each channel.
            if topk <= 0, tensor_chw is assert to be one or three.
            Defaults to 20.
        arrangement (Tuple[int, int]): The arrangement of featmap when
            channel_reduction is not None and topk > 0. Defaults to (4, 5).
        resize_shape (tuple, optional): The shape to scale the feature map.
            Defaults to None.
        alpha (Union[int, List[int]]): The transparency of featmap.
            Defaults to 0.5.

    Returns:
        np.ndarray: RGB image.
    """
    assert isinstance(featmap,
                      torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                      f' but got {type(featmap)}')
    assert featmap.ndim == 3 or featmap.ndim == 4, f'Input dimension must be 3 or 4, ' \
                              f'but got {featmap.ndim}'

    if featmap.ndim == 3:
        featmap = featmap.unsqueeze(0)

    if channel_reduction is not None:
        assert channel_reduction in [
            'squeeze_mean', 'select_max', 'squeeze_max'], \
            f'Mode only support "squeeze_mean", "select_max", "squeeze_max"' \
            f'but got {channel_reduction}'
        if channel_reduction == 'select_max':
            sum_channel_featmap = torch.sum(featmap, dim=(2, 3))
            _, indices = torch.topk(sum_channel_featmap, 1, dim=1)
            # feat_map = featmap[indices]
            expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, featmap.size(2), featmap.size(3))
            feat_map = torch.gather(featmap, 1, expanded_indices).squeeze()
        elif channel_reduction == 'squeeze_max':
            feat_map = torch.max(featmap, dim=1)[0]
        else:
            feat_map = torch.mean(featmap, dim=1)

        if resize_shape is not None:
            assert feat_interpolation in [
                'bilinear', 'nearest'], \
                f'feat_interpolation only support "bilinear", "nearest"' \
                f'but got {feat_interpolation}'
            feat_map = F.interpolate(
                feat_map.unsqueeze(0),
                resize_shape,
                mode=feat_interpolation,
                align_corners=False)
        return feat_map.squeeze(0)

    elif topk <= 0:
        featmap_channel = featmap.shape[0]
        assert featmap_channel in [
            1, 3
        ], ('The input tensor channel dimension must be 1 or 3 '
            'when topk is less than 1, but the channel '
            f'dimension you input is {featmap_channel}, you can use the'
            ' channel_reduction parameter or set topk greater than '
            '0 to solve the error')
        return featmap
    else:
        channel = featmap.shape[1]
        # Extract the feature map of topk
        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(featmap, dim=(2, 3))
        _, indices = torch.topk(sum_channel_featmap, topk, dim=1)
        expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, featmap.size(2), featmap.size(3))
        topk_featmap = torch.gather(featmap, 1, expanded_indices).squeeze()
        return topk_featmap

def generate_heatmap_over_img(heatmap, overlaid_img, alpha=0.5):
    """
    Overlays a heatmap onto an image with adjustable transparency.

    Parameters:
    - heatmap (numpy array or tensor): The heatmap to overlay. Expected shape is (H, W) or (1, H, W).
    - overlaid_img (numpy array): The base image onto which the heatmap will be applied.
      If grayscale (2D), it will be converted to RGB.
    - alpha (float): The transparency level for the heatmap overlay (0 = fully transparent, 1 = fully opaque).

    Returns:
    - numpy array: The RGB image with the heatmap overlay applied.
    """
    if len(overlaid_img.shape) == 2:
        overlaid_img = cv2.cvtColor(overlaid_img, cv2.COLOR_GRAY2RGB)

    if overlaid_img.shape[:2] != heatmap.shape[-2:]:
        warnings.warn(
            f'Since the spatial dimensions of '
            f'overlaid_image: {overlaid_img.shape[:2]} and '
            f'featmap: {heatmap.shape[1:]} are not same, '
            f'the feature map will be interpolated. '
            f'This may cause mismatch problems ！')
        heatmap = F.interpolate(
            heatmap[None],
            overlaid_img.shape[:2],
            mode='bilinear',
            align_corners=False)[0]
    return convert_overlay_heatmap(heatmap, overlaid_img, alpha)

def convert_overlay_heatmap(feat_map: Union[np.ndarray, torch.Tensor],
                            img: Optional[np.ndarray] = None,
                            alpha: float = 0.5) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    heat_norm_img = np.zeros(feat_map.shape)
    heat_norm_img = cv2.normalize(feat_map, heat_norm_img, 0, 255, cv2.NORM_MINMAX)
    heat_norm_img = np.asarray(heat_norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(heat_norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        norm_img = np.zeros(img.shape)
        norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)
        heat_img = cv2.addWeighted(norm_img, 1 - alpha, heat_img, alpha, 0)
    return heat_img

def generate_gauss_blur_annotations(binary_masks, kernel_size=5, sigma=None):
    """
    Apply Gaussian blur to a batch of binary masks and return normalized blurred masks.

    Args:
        binary_masks (torch.Tensor): Batch of binary masks with shape (..., batch_size, height, width).
        kernel_size (int, optional): Size of the Gaussian kernel for blurring. Default is 5.
        sigma (float, optional): Standard deviation of the Gaussian distribution for blurring. Default is 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8.

    Returns:
        torch.Tensor: Batch of normalized blurred masks with the same shape as input binary masks.
                      The values are between 0 and 1.
    """

    blurred_masks = gaussian_blur(binary_masks, kernel_size=kernel_size, sigma=sigma)
    return blurred_masks

def generate_learned_processed_annotations(model,binary_masks,input=None, mode='learned_S1'):
    target_map_pos_org = (binary_masks > 0).float().permute(1, 0, 2, 3)
    if mode in ['learned_S1', 'learned_d1']:
        input_imp = target_map_pos_org
        target_map_trans = model.imp(input_imp)
    elif mode in ['learned_S2', 'learned_ds1']:
        # input_imp = target_map_pos_org
        input_imp = torch.cat((target_map_pos_org, input), 1)
        target_map_trans = model.imp(input_imp)
    elif mode in ['learned_D1', 'learned_D2']:
        if mode == 'learned_D1':
            input_imp = target_map_pos_org
        else:
            input_imp = torch.cat((target_map_pos_org, input), 1)
        H1 = torch.relu(model.imp_conv1(input_imp))
        H2 = torch.relu(model.imp_conv2(H1))
        H3 = torch.relu(model.imp_conv3(H2))
        target_map_trans = torch.relu(model.imp_conv4(H3))
        # target_map_trans = model.imp_conv5(H4)
    elif mode in ['learned_d2', 'learned_ds2']:
        if mode == 'learned_d2':
            input_imp = target_map_pos_org
        else:
            input_imp = torch.cat((target_map_pos_org, input), 1)

        H1 = model.imp_conv1(input_imp)
        target_map_trans = model.imp_conv2(H1)
    elif mode in ['learned_d3', 'learned_ds3']:
        if mode == 'learned_d2':
            input_imp = target_map_pos_org
        else:
            input_imp = torch.cat((target_map_pos_org, input), 1)

        H1 = model.imp_conv1(input_imp)
        H2 = model.imp_conv2(H1)
        target_map_trans = model.imp_conv3(H2)
    else:
        raise ValueError(f"{mode} mode type not supported")

    learned_masks = target_map_trans - torch.min(target_map_trans)
    learned_masks = learned_masks / (torch.max(learned_masks) + 1e-6)

    return learned_masks.permute(1,0,2,3)

def generate_relevance(model, outputs, input_size=256, upscale=True):
    """
    Computes the relevance map (acoording to the GAE method) for a given model and output predictions. The relevance map highlights
    which parts of the input contribute most to the model's decision-making process.

    Parameters:
    - model (nn.Module): The trained ViT-based model.
    - outputs (Tensor): The model's output predictions, typically from the final layer.
    - input_size (int): size of input image.
    - upscale (bool, optional): If True, upscales the relevance map to the original input size. If False, keeps a fixed map size. Default is True.

    Returns:
    - relevance (Tensor): The relevance map, which highlights the spatial areas contributing most to the prediction.
                           The output tensor will have the shape (num_tokens, batch_size, feat_size, feat_size), where
                           `feat_size` is typically the spatial dimension of the feature map.
    """
    # a batch of samples
    batch_size = outputs.shape[0]
    feat_map_size = model.feat_size

    one_hot = torch.sum(outputs)
    model.zero_grad()

    num_tokens = model.vit_encoder.layers[0].attn.attn_maps.shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(model.vit_encoder.layers):
        grad = torch.autograd.grad(one_hot, [blk.attn.attn_maps], retain_graph=True)[0]
        cam = blk.attn.attn_maps
        cam = avg_heads(cam, grad)
        R = R + apply_self_attention_rules(R, cam)
    relevance = R[:, 0, 1:]
    if upscale:
        relevance = upscale_relevance(relevance, feat_map_size, scale_factor=input_size//feat_map_size)
    else:
        relevance = relevance.reshape(-1, 1, feat_map_size, feat_map_size)
        # normalize between 0 and 1
        relevance = relevance.reshape(relevance.shape[0], -1)
        min = relevance.min(1, keepdim=True)[0]
        max = relevance.max(1, keepdim=True)[0]
        relevance = (relevance - min) / (max - min)
        relevance = relevance.reshape(-1, 1, feat_map_size, feat_map_size)
    return relevance.permute(1,0,2,3)

def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-3], cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, cam.shape[-3], grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=1)
    return cam

def attention_rollout(batch_As):
    """
    Computes attention rollout from the given list of attention matrices for a batch of instances.

    Args:
        batch_As (torch.Tensor): A tensor of shape (B, L, H, W, W) where
                                 B is the batch size,
                                 L is the number of layers,
                                 H is the number of attention heads,
                                 W is the dimension of the attention matrices.

    Returns:
        torch.Tensor: A tensor of shape (B, W, W) representing the attention rollout for each instance in the batch.
    """
    # Average the attention heads for each instance and each layer
    batch_As = batch_As.mean(dim=2)  # Shape: (B, L, W, W)

    # Initialize the rollout for each instance in the batch
    rollout = batch_As[:, 0]  # Shape: (B, W, W)

    # Compute the rollout for each subsequent layer
    for i in range(1, batch_As.shape[1]):
        A = batch_As[:, i]  # Shape: (B, W, W)
        eye = torch.eye(A.shape[-1], device=A.device).unsqueeze(0)  # Shape: (1, W, W)
        A = 0.5 * A + 0.5 * eye  # Shape: (B, W, W)
        rollout = torch.matmul(A, rollout) # Shape: (B, W, W)

    spatial_rollout_attn = rollout[:, 0, 1:]
    min_vals = spatial_rollout_attn.min(dim=1)[0].unsqueeze(1)
    max_vals = spatial_rollout_attn.max(dim=1)[0].unsqueeze(1)
    spatial_rollout_attn = (spatial_rollout_attn - min_vals) / (max_vals - min_vals)
    width = int(spatial_rollout_attn.size(-1) ** 0.5)
    spatial_rollout_attn = spatial_rollout_attn.reshape(-1, width, width)

    return spatial_rollout_attn

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def upscale_relevance(relevance, feat_map_size=16, scale_factor=16):
    relevance = relevance.reshape(-1, 1, feat_map_size, feat_map_size)
    relevance = torch.nn.functional.interpolate(relevance, scale_factor=scale_factor, mode='bilinear')

    # normalize between 0 and 1
    relevance = relevance.reshape(relevance.shape[0], -1)
    min = relevance.min(1, keepdim=True)[0]
    max = relevance.max(1, keepdim=True)[0]
    relevance = (relevance - min) / (max - min)

    relevance = relevance.reshape(-1, 1, feat_map_size * scale_factor, feat_map_size * scale_factor)
    return relevance

def BF_solver(X, Y):
    epsilon = 1e-4

    with torch.no_grad():
        x = torch.flatten(X)
        y = torch.flatten(Y)
        g_idx = (y<0).nonzero(as_tuple=True)[0]
        le_idx = (y>0).nonzero(as_tuple=True)[0]
        len_g = len(g_idx)
        len_le = len(le_idx)
        a = 0
        a_ct = 0.0
        for idx in g_idx:
            v = x[idx] + epsilon # to avoid miss the constraint itself
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)
        for idx in le_idx:
            v = x[idx]
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)

    # print('optimal solution for batch, a=', a)
    # print('final threshold a is assigned as:', am)

    return torch.tensor([a]).cuda()