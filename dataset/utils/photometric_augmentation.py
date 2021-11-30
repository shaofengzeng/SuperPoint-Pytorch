#this code is mainly copied from pytorch-superpoint[https://github.com/eric-yyjau/pytorch-superpoint]
#-*-coding:utf8-*-

import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa
from numpy.random import uniform
from numpy.random import randint
import torchvision.transforms as transforms
from PIL import Image

class PhotoAugmentor:
    '''
    '''

    def __init__(self, config):
        self.config = config
        self.primitives = config['primitives']

        self.brightness_max_change = config['params']['random_brightness']['max_abs_change']
        self.contrast_factors = tuple(config['params']['random_contrast']['strength_range'])
        # self.colorjitter_brightness = transforms.ColorJitter(
        #     brightness=config['params']['random_brightness']['max_abs_change']
        # )
        # self.colorjitter_contrast = transforms.ColorJitter(
        #     contrast=tuple(config['params']['random_contrast']['strength_range'])
        # )

    def additive_gaussian_noise(self, image):
        stddev_range = self.config['params']['additive_gaussian_noise']['stddev_range']

        # stddev = tf.random_uniform((), *stddev_range)
        # noise = tf.random_normal(tf.shape(image), stddev=stddev)
        # noisy_image = tf.clip_by_value(image + noise, 0, 255)
        stddev = np.random.uniform(stddev_range[0], stddev_range[1])
        noise = np.random.normal(scale=stddev,size=image.shape)
        noisy_image = np.clip(image+noise, 0, 255)
        return noisy_image


    def additive_speckle_noise(self, image):
        prob_range = self.config['params']['additive_speckle_noise']['prob_range']

        # prob = tf.random_uniform((), *prob_range)
        # sample = tf.random_uniform(tf.shape(image))
        # noisy_image = tf.where(sample <= prob, tf.zeros_like(image), image)
        # noisy_image = tf.where(sample >= (1. - prob), 255.*tf.ones_like(image), noisy_image)
        prob = np.random.uniform(prob_range[0], prob_range[1])
        sample = np.random.uniform(size=image.shape)
        noisy_image = np.where(sample<=prob, np.zeros_like(image), image)
        noisy_image = np.where(sample>=(1. - prob), 255.*np.ones_like(image), noisy_image)
        noisy_image = np.clip(noisy_image.round(),0,255)
        return noisy_image


    def random_brightness(self, image):
        # if not isinstance(image.dtype, np.uint8):
        #     image = image.round().astype(np.uint8)
        # if isinstance(image, np.ndarray):
        #     image = Image.fromarray(image)
        # image = self.colorjitter_brightness(image)
        # image = np.asarray(image)  # to numpy

        delta = np.random.uniform(low=-self.brightness_max_change,high=self.brightness_max_change, size=1)[0]
        image = image + delta
        image = np.clip(image, 0, 255.0)
        return image.astype(np.float32)


    def random_contrast(self, image):
        # if not isinstance(image.dtype, np.uint8):
        #     image = image.round().astype(np.uint8)
        # if isinstance(image, np.ndarray):
        #     image = Image.fromarray(image)
        # image = self.colorjitter_contrast(image)
        # image = np.asarray(image)#to numpy

        contrast_factor = np.random.uniform(low=self.contrast_factors[0],
                                            high=self.contrast_factors[1],
                                            size=1)[0]
        mean = image.mean()
        image = (image-mean)*contrast_factor+mean
        image = np.clip(image, 0, 255.)
        return image.astype(np.float32)

    def additive_shade(self, image):
        nb_ellipses = self.config['params']['additive_shade']['nb_ellipses']
        transparency_range = self.config['params']['additive_shade']['transparency_range']
        kernel_size_range = self.config['params']['additive_shade']['kernel_size_range']

        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
            shaded = img * (1 - transparency * mask/255.)
            return np.clip(shaded, 0, 255)

        # shaded = tf.py_func(_py_additive_shade, [image], tf.float32)
        # res = tf.reshape(shaded, tf.shape(image))
        shaded = _py_additive_shade(image)
        res = np.reshape(shaded, image.shape)

        return np.clip(res.round(),0,255)


    def motion_blur(self, image):
        max_kernel_size = self.config['params']['motion_blur']['max_kernel_size']
        def _py_motion_blur(img):
            # Either vertial, hozirontal or diagonal blur
            mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
            ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
            center = int((ksize-1)/2)
            kernel = np.zeros((ksize, ksize))
            if mode == 'h':
                kernel[center, :] = 1.
            elif mode == 'v':
                kernel[:, center] = 1.
            elif mode == 'diag_down':
                kernel = np.eye(ksize)
            elif mode == 'diag_up':
                kernel = np.flip(np.eye(ksize), 0)
            var = ksize * ksize / 16.
            grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
            gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
            kernel *= gaussian
            kernel /= np.sum(kernel)
            img = cv2.filter2D(img, -1, kernel)
            return img

        # blurred = tf.py_func(_py_motion_blur, [image], tf.float32)
        # tf.reshape(blurred, tf.shape(image))
        blurred = _py_motion_blur(image)
        res = np.reshape(blurred, image.shape)
        return np.clip(res,0,255)


    def __call__(self, image):

        indices = np.arange(len(self.primitives))
        np.random.shuffle(indices)

        def step(i, image):
            for j in range(len(self.primitives)):
                if j == indices[i]:
                    image = getattr(self, self.primitives[j])(image)
            return image

        # if image.dtype!=np.uint8:
        #     image = image.astype(np.int).astype(np.uint8)
        for i in range(len(self.primitives)):
            image = step(i, image)
        return image.astype(np.float32)


if __name__=='__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    import yaml
    img = Image.open('../../data/synthetic_shapes/draw_cube/images/training/0.png')
    img = np.array(img)
    print(type(img))
    #
    config_path = '../../configs/magic-point_shapes.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f)



