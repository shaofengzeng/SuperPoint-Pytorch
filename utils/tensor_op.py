#-*-coding:utf-8-*-
import torch
import torch.nn.functional as f

def erosion2d(image, strel, origin=(0, 0), border_value=1e6):
    """
    :param image:BCHW
    :param strel: BHW
    :param origin: default (0,0)
    :param border_value: default 1e6
    :return:
    """
    image_pad = f.pad(image, [origin[0], strel.shape[1]-origin[0]-1, origin[1], strel.shape[2]-origin[1]-1], mode='constant', value=border_value)
    image_unfold = f.unfold(image_pad, kernel_size=strel.shape[1])#[B,C*sH*sW,L],L is the number of patches
    strel_flatten = torch.flatten(strel,start_dim=1).unsqueeze(-1)
    diff = image_unfold - strel_flatten
    # Take maximum over the neighborhood
    result, _ = diff.min(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(r*r), r*H, r*W],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert ch % (scale_factor * scale_factor) == 0

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor


def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (r*r)*C, H/r, W/r],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert height % scale_factor == 0
    assert width % scale_factor == 0

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute(0, 1, 3, 5, 2, 4)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor