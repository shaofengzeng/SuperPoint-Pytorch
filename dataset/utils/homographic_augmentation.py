#this code is mainly copied from Superpoint[https://github.com/rpautrat/SuperPoint]
#-*-coding:utf8-*-
import cv2
from math import pi
from numpy.random import uniform
from scipy import stats
from utils.params import dict_update
from utils.keypoint_op import *
from imgaug import augmenters as iaa



def homographic_aug_pipline(img, pts, config):
    """
    :param img: np.array, H*W
    :param pts: np.array,N*2,yx format
    """
    H,W = img.shape

    homography = sample_homography((H,W), config['params'])
    warped_image = cv2.warpPerspective(img, homography, (W,H))

    warped_valid_mask = compute_valid_mask((H,W), homography, config['valid_border_margin'])

    warped_points = warp_points(pts, homography)
    warped_points = filter_points(warped_points, (H,W))
    warped_points_map = compute_keypoint_map(warped_points, (H,W))

    return warped_image, warped_points, warped_points_map, warped_valid_mask, homography


def compute_valid_mask(shape, homography, erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        shape: `[H, W]`, tuple, list or ndarray
        homography: 3*3 homography
        erosion_radius: radius of the margin to be discarded.

    Returns: mask with values 0 or 1
    """
    H,W = shape
    img_one = np.ones(shape,dtype=homography.dtype)
    mask = cv2.warpPerspective(img_one, homography, (W,H))
    #TODO: 是否及何时需要round
    mask = mask.round()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def sample_homography(shape, config=None):

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
        
        tnorm_y = stats.truncnorm(-2, 2, loc=0, scale=perspective_amplitude_y/2)
        tnorm_x = stats.truncnorm(-2, 2, loc=0, scale=perspective_amplitude_x/2)
        perspective_displacement = tnorm_y.rvs(1)
        h_displacement_left = tnorm_x.rvs(1)
        h_displacement_right = tnorm_x.rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if config['scaling']:
        mu, sigma = 1, config['scaling_amplitude']/2
        lower, upper = mu - 2 * sigma, mu + 2 * sigma
        tnorm_s = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        scales = tnorm_s.rvs(config['n_scales'])
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
    homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    #homography = [[1., 0., 0.],[0.,1.0,0.],[0.,0.,1.0]]
    return homography.astype(np.float32)#[3,3]


def ratio_preserving_resize(img, target_size):
    '''
    :param img: raw img
    :param target_size: (h,w)
    :return:
    '''
    scales = np.array((target_size[0]/img.shape[0], target_size[1]/img.shape[1]))##h_s,w_s

    new_size = np.round(np.array(img.shape)*np.max(scales)).astype(np.int)#
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape
    target_h, target_w = target_size
    ##
    hp = (target_h-curr_h)//2
    wp = (target_w-curr_w)//2
    aug = iaa.Sequential([iaa.CropAndPad(px=(hp, wp, target_h-curr_h-hp, target_w-curr_w-wp),keep_size=False),])
    new_img = aug(images=temp_img)
    return new_img




if __name__=='__main__':
    h = sample_homography([7,7], config=None)
    ones = np.ones((7,7),np.uint8)*255
    wones = cv2.warpPerspective(ones, h, (7,7))
    print(wones)
    mask = compute_valid_mask((7,7), h,0)


    print(mask)






