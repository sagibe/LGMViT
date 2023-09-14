import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import warnings

def generate_spatial_attetntion(attn):
    bs, nh, feat_size = attn.shape[0], attn.shape[1], int(np.sqrt(attn.shape[2]))
    return attn.max(dim=-2)[0].view(bs, nh, feat_size, feat_size)

def extract_heatmap(featmap: torch.Tensor,
                 feat_interpolation = 'bilinear',
                 channel_reduction: Optional[str] = 'squeeze_mean',
                 topk: int = 20,
                 resize_shape: Optional[tuple] = None):
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
    # if resize_shape is not None:
    #     assert feat_interpolation in [
    #         'bilinear', 'nearest'], \
    #         f'feat_interpolation only support "bilinear", "nearest"' \
    #         f'but got {feat_interpolation}'
    #     featmap = F.interpolate(
    #         featmap,
    #         resize_shape,
    #         mode=feat_interpolation,
    #         align_corners=False)

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
        # return feat_map
        # return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
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
        # return convert_overlay_heatmap(featmap, overlaid_image, alpha)
    else:
        channel = featmap.shape[1]
        # Extract the feature map of topk
        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(featmap, dim=(2, 3))
        _, indices = torch.topk(sum_channel_featmap, topk, dim=1)
        # feat_map = featmap[indices]
        expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, featmap.size(2), featmap.size(3))
        topk_featmap = torch.gather(featmap, 1, expanded_indices).squeeze()
        return topk_featmap

        # fig = plt.figure(frameon=False)
        # # Set the window layout
        # fig.subplots_adjust(
        #     left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        # dpi = fig.get_dpi()
        # fig.set_size_inches((width * col + 1e-2) / dpi,
        #                     (height * row + 1e-2) / dpi)
        # for i in range(topk):
        #     axes = fig.add_subplot(row, col, i + 1)
        #     axes.axis('off')
        #     axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
        #     axes.imshow(
        #         convert_overlay_heatmap(topk_featmap[i], overlaid_image,
        #                                 alpha))
        # image = img_from_canvas(fig.canvas)
        # plt.close(fig)
        # return image

def generate_heatmap_over_img(heatmap, overlaid_img, alpha=0.5):
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

def img_from_canvas(canvas: 'FigureCanvasAgg') -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """  # noqa: E501
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype('uint8')

def generate_blur_masks_normalized(binary_masks, kernel_size=5, sigma=None):
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
    # assert binary_masks.ndim == 3 or (binary_masks.ndim == 4 and binary_masks.shape[0] == 1), "Input binary masks should have 3 dimensions: (batch_size, height, width) or 3 dimensions: (1, batch_size, height, width)"
    # blurred_masks = []
    # num_dims = binary_masks.ndim
    # if (num_dims == 4 and binary_masks.shape[0] == 1):
    #     binary_masks = binary_masks.squeeze(0)

    blurred_masks = gaussian_blur(binary_masks, kernel_size=kernel_size, sigma=sigma)
    # for mask in binary_masks:
    #     # Convert mask to float tensor
    #     mask_float = mask.float()
    #
    #     # Apply Gaussian blur using PyTorch's functional interface
    #     blurred_mask = gaussian_blur(mask_float, kernel_size=kernel_size, sigma=sigma)
    #
    #     # Normalize blurred mask to have values between 0 and 1
    #     blurred_mask_normalized = (blurred_mask - blurred_mask.min()) / (blurred_mask.max() - blurred_mask.min())
    #
    #     blurred_masks.append(blurred_mask_normalized)
    # blurred_masks = torch.stack(blurred_masks)
    # if num_dims == 4:
    #     return blurred_masks.unsqueeze(0)
    return blurred_masks
