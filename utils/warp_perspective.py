#-*-coding:utf-8-*-
"""
kornia implemented warp perspective method
Only for backup
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))

def normal_transform_pixel(
    height: int, width: int, eps: float = 1e-14,
    device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].
    Args:
        height (int): image height.
        width (int): image width.
        eps (float): epsilon to prevent divide-by-zero errors
    Returns:
        torch.Tensor: normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0],
                           [0.0, 1.0, -1.0],
                           [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(dst_pix_trans_src_pix: torch.Tensor,
                         dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].
    Args:
        dst_pix_trans_src_pix (torch.Tensor): homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_dst (tuple): size of the destination image (height, width).
    Returns:
        torch.Tensor: the normalized homography of shape :math:`(B, 3, 3)`.
    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError("Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}"
                         .format(dst_pix_trans_src_pix.shape))

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(
        src_h, src_w).to(dst_pix_trans_src_pix)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(
        dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = (
        dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    )
    return dst_norm_trans_src_norm


def convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.
    Examples::
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_from_homogeneous(input)  # BxNx2
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))

    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    # we check for points at infinity
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1. / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.
    Examples::
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_to_homogeneous(input)  # BxNx4
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def transform_points(trans_01: torch.Tensor,
                     points_1: torch.Tensor) -> torch.Tensor:
    r"""Function that applies transformations to a set of points.
    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.
    Shape:
        - Output: :math:`(B, N, D)`
    Examples:
        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """
    check_is_tensor(trans_01)
    check_is_tensor(points_1)
    if not (trans_01.device == points_1.device and trans_01.dtype == points_1.dtype):
        raise TypeError(
            "Tensor must be in the same device and dtype. "
            f"Got trans_01 with ({trans_01.dtype}, {points_1.dtype}) and "
            f"points_1 with ({points_1.dtype}, {points_1.dtype})")
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError("Input batch size must be the same for both tensors or 1")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differ by one unit")

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h,
                           trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0



def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True,
        device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.
    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample
    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.
    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def warp_perspective(src: torch.Tensor, M: torch.Tensor, dsize: Tuple[int, int],
                     mode: str = 'bilinear', padding_mode: str = 'zeros',
                     align_corners: Optional[bool] = None) -> torch.Tensor:
    r"""Applies a perspective transformation to an image.

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
        \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}}
        \right )

    Args:
        src (torch.Tensor): input image with shape :math:`(B, C, H, W)`.
        M (torch.Tensor): transformation matrix with shape :math:`(B, 3, 3)`.
        dsize (tuple): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners(bool, optional): interpolation flag. Default: None.

    Returns:
        torch.Tensor: the warped input image :math:`(B, C, H, W)`.

    """
    if not isinstance(src, torch.Tensor):
        raise TypeError("Input src type is not a torch.Tensor. Got {}"
                        .format(type(src)))

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}"
                        .format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}"
                         .format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        print(
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpPerspective. In case you want to keep your previous "
            "behaviour set it to False. This warning will disappear in kornia > v0.6.")
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()
    h_out, w_out = dsize

    # we normalize the 3x3 transformation matrix and convert to 3x4
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(
        M, (H, W), (h_out, w_out))  # Bx3x3

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)  # Bx3x3

    # this piece of code substitutes F.affine_grid since it does not support 3x3
    grid = create_meshgrid(h_out, w_out, normalized_coordinates=True,
                           device=src.device).to(src.dtype).repeat(B, 1, 1, 1)
    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    return F.grid_sample(src, grid,
                         align_corners=align_corners,
                         mode=mode,
                         padding_mode=padding_mode)