#this code is mainly copied from Superpoint[https://github.com/rpautrat/SuperPoint]
#-*-coding:utf8-*-
from math import pi
import cv2
import numpy as np
from numpy.random import uniform
from scipy.stats import truncnorm
import kornia
import torch
from utils.params import dict_update
from utils.tensor_op import erosion2d
from utils.keypoint_op import *
from imgaug import augmenters as iaa



def homographic_aug_pipline(img, pts, config, device='cpu'):
    """
    :param img: [1,1,H,W]
    :param pts:[N,2]
    :param config:parameters
    :param device: cpu or cuda
    :return:
    """
    if len(img.shape)==2:
        img = img.unsqueeze(dim=0).unsqueeze(dim=0)
    image_shape = img.shape[2:]#HW
    homography = sample_homography(image_shape, config['params'], device=device)
    ##
    #warped_image = cv2.warpPerspective(img, homography, tuple(image_shape[::-1]))
    warped_image = kornia.warp_perspective(img, homography, image_shape, align_corners=True)

    warped_valid_mask = compute_valid_mask(image_shape, homography, config['valid_border_margin'], device=device)

    warped_points = warp_points(pts, homography, device=device)
    warped_points = filter_points(warped_points, image_shape, device=device)
    warped_points_map = compute_keypoint_map(warped_points, img.shape[2:], device=device)

    return {'warp':{'img': warped_image.squeeze(),
                    'kpts': warped_points,
                    'kpts_map': warped_points_map.squeeze(),#some point maybe filtered
                    'mask':warped_valid_mask.squeeze()},
            'homography':homography.squeeze(),
            }
    #return warpped_image, warped_points, valid_mask, homography



def compute_valid_mask(image_shape, homographies, erosion_radius=0, device='cpu'):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: `[H, W]`, tuple, list or ndarray
        homography: B*3*3 homography
        erosion_radius: radius of the margin to be discarded.

    Returns: mask with values 0 or 1
    """
    if len(homographies.shape)==2:
        homographies = homographies.unsqueeze(0)
    # TODO:uncomment this line if your want to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    B = homographies.shape[0]
    img_one = torch.ones(tuple([B,1,*image_shape]),device=device, dtype=torch.float32)#B,C,H,W
    mask = kornia.warp_perspective(img_one, homographies, tuple(image_shape), align_corners=True)
    mask = mask.round()#B1HW
    #mask = cv2.warpPerspective(np.ones(image_shape), homography, dsize=tuple(image_shape[::-1]))#dsize=tuple([w,h])
    if erosion_radius > 0:
        # TODO: validation & debug
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        kernel = torch.as_tensor(kernel[np.newaxis,:,:],device=device)
        _, kH, kW = kernel.shape
        origin = ((kH-1)//2, (kW-1)//2)
        mask = erosion2d(mask, torch.flip(kernel, dims=[1,2]), origin=origin) + 1.# flip kernel so perform as tf.nn.erosion2d

    return mask.squeeze(dim=1)#BHW


def sample_homography(shape, config=None, device='cpu'):
    """
    ------------------m1----------------------
    import tensorflow as tf
    import matplotlib.pyplot as plt

    c = tf.truncated_normal(shape=[10000, ], mean=0, stddev=0.05)

    with tf.Session() as sess:
        sess.run(c)
        data = c.eval()
    plt.hist(x=data, bins=100, color='steelblue', edgecolor='black')
    plt.show()
    ------------------m2----------------------
    import scipy.stats as stats

    mu, sigma = 0, 0.05
    lower, upper = mu - 2 * sigma, mu + 2 * sigma

    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    plt.hist(X.rvs(10000), bins=100, color='red', edgecolor='black')
    plt.show()
    ------------------------------------------
    c1==c2
    """
    default_config = {'perspective':True, 'scaling':True, 'rotation':True, 'translation':True,
    'n_scales':5, 'n_angles':25, 'scaling_amplitude':0.2, 'perspective_amplitude_x':0.1,
    'perspective_amplitude_y':0.1, 'patch_ratio':0.5, 'max_angle':pi / 2,
    'allow_artifacts': False, 'translation_overflow': 0.}

    #TODO: not tested
    if config is not None:
        config = dict_update(default_config, config)
    else:
        config = default_config

    std_trunc = 2

    # Corners of the input patch
    margin = (1 - config['patch_ratio']) / 2
    pts1 = margin + np.array([[0, 0],
                              [0, config['patch_ratio']],
                              [config['patch_ratio'], config['patch_ratio']],
                              [config['patch_ratio'], 0]])
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if config['perspective']:
        if not config['allow_artifacts']:
            perspective_amplitude_x = min(config['perspective_amplitude_x'], margin)
            perspective_amplitude_y = min(config['perspective_amplitude_y'], margin)
        else:
            perspective_amplitude_x = config['perspective_amplitude_x']
            perspective_amplitude_y = config['perspective_amplitude_y']
        perspective_displacement = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if config['scaling']:
        scales = truncnorm(-std_trunc, std_trunc, loc=1, scale=config['scaling_amplitude']/2).rvs(config['n_scales'])
        #scales = np.random.uniform(0.8, 2, config['n_scales'])
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if config['allow_artifacts']:
            valid = np.arange(config['n_scales'])  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if config['translation']:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if config['allow_artifacts']:
            t_min += config['translation_overflow']
            t_max += config['translation_overflow']
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if config['rotation']:
        angles = np.linspace(-config['max_angle'], config['max_angle'], num=config['n_angles'])
        angles = np.concatenate((np.array([0.]),angles), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center

        if config['allow_artifacts']:
            valid = np.arange(config['n_angles'])  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]

    # Rescale to actual size
    shape = np.array(shape[::-1])  # different convention [y, x]
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    # this homography is the same with tf version and this line
    #homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    ## equals the following 3 lines
    pts1 = torch.tensor(pts1[np.newaxis,:], device=device, dtype=torch.float32)
    pts2 = torch.tensor(pts2[np.newaxis,:], device=device, dtype=torch.float32)
    homography = kornia.get_perspective_transform(pts1, pts2)

    #TODO: comment the follwing line if you want same result as tf version
    # since if we use homography directly ofr opencv function, for example warpPerspective
    # the result we get is different from tf version. In order to get a same result, we have to
    # apply inverse operation,like this
    #homography = np.linalg.inv(homography)
    homography = torch.inverse(homography)#inverse here to be consistent with tf version
    #debug
    #homography = torch.eye(3,device=device).unsqueeze(dim=0)
    return homography#[1,3,3]

def ratio_preserving_resize(img, target_size):
    '''
    :param img: raw img
    :param dest_size: (w,h)
    :return:
    '''
    scales = np.array((target_size[1]/img.shape[0], target_size[0]/img.shape[1]))##h_s,w_s

    new_size = np.round(np.array(img.shape)*np.max(scales)).astype(np.int)#
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape
    target_w, target_h = target_size
    ##
    hp = (target_h-curr_h)//2
    wp = (target_w-curr_w)//2
    aug = iaa.Sequential([iaa.CropAndPad(px=(hp, wp, target_h-curr_h-hp, target_w-curr_w-wp),keep_size=False),])
    new_img = aug(images=temp_img)
    return new_img




if __name__=='__main__':
    import cv2
    img = cv2.imread('./data/icl_snippet/250.png',0)






