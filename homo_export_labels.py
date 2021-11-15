import yaml
import os
import torch
from tqdm import tqdm
from math import pi
import kornia
import cv2
import numpy as np

from utils.params import dict_update
from solver.nms import box_nms
from utils.tensor_op import erosion2d
from dataset.utils.homographic_augmentation import sample_homography,ratio_preserving_resize
from model.magic_point import MagicPoint


homography_adaptation_default_config = {
        'num': 50,
        'aggregation': 'sum',
        'valid_border_margin': 3,
        'homographies': {
            'translation': True,
            'rotation': True,
            'scaling': True,
            'perspective': True,
            'scaling_amplitude': 0.1,
            'perspective_amplitude_x': 0.1,
            'perspective_amplitude_y': 0.1,
            'patch_ratio': 0.5,
            'max_angle': pi,
        },
        'filter_counts': 0
}


def read_image(img_path):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
    Returns
      grayim: grayscale image
    """
    grayim = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if grayim is None:
        raise Exception('Error reading image %s' % img_path)
    return grayim

def to_tensor(image, device):
    H,W = image.shape
    image = image.astype('float32') / 255.
    image = image.reshape(1, H, W)
    image = torch.from_numpy(image).view(1,1,H,W).to(device)
    return image

def one_adaptation(net, raw_image, probs, counts, images, config, device='cpu'):
    """
    :param probs:[B,1,H,W]
    :param counts: [B,1,H,W]
    :param images: [B,1,H,W,N]
    :return:
    """
    B, C, H, W, _ = images.shape
    #sample image patch
    M = sample_homography(shape=[H, W], config=config['homographies'],device=device)
    M_inv = torch.inverse(M)
    ##
    warped = kornia.warp_perspective(raw_image, M, dsize=(H,W), align_corners=True)
    mask = kornia.warp_perspective(torch.ones([B,1,H,W], device=device), M, dsize=(H, W), mode='nearest',align_corners=True)
    count = kornia.warp_perspective(torch.ones([B,1,H,W],device=device), M_inv, dsize=(H,W), mode='nearest',align_corners=True)

    # Ignore the detections too close to the border to avoid artifacts
    if config['valid_border_margin']:
        ##TODO: validation & debug
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['valid_border_margin'] * 2,) * 2)
        kernel = torch.as_tensor(kernel[np.newaxis,:,:], device=device)#BHW
        kernel = torch.flip(kernel, dims=[1,2])
        _, kH, kW = kernel.shape
        origin = ((kH-1)//2, (kW-1)//2)
        count = erosion2d(count, kernel, origin=origin) + 1.
        mask = erosion2d(mask, kernel, origin=origin) + 1.
    mask = mask.squeeze(dim=1)#B,H,W
    count = count.squeeze(dim=1)#B,H,W

    # Predict detection probabilities
    prob = net(warped)
    prob = prob['prob']
    prob = prob * mask
    prob_proj = kornia.warp_perspective(prob.unsqueeze(dim=1), M_inv, dsize=(H,W), align_corners=True)
    prob_proj = prob_proj.squeeze(dim=1)#B,H,W
    prob_proj = prob_proj * count#project back
    ##

    probs = torch.cat([probs, prob_proj.unsqueeze(dim=1)], dim=1)#the probabilities of each pixels on raw image
    counts = torch.cat([counts, count.unsqueeze(dim=1)], dim=1)
    images = torch.cat([images, warped.unsqueeze(dim=-1)], dim=-1)

    return probs, counts, images

@torch.no_grad()
def homography_adaptation(net, raw_image, config, device='cpu'):
    """
    :param raw_image: [B,1,H,W]
    :param net: MagicPointNet
    :param config:
    :return:
    """
    probs = net(raw_image)#B,H,W
    probs = probs['prob']
    ## probs = torch.tensor(np.load('./prob.npy'), dtype=torch.float32)#debug
    ## warped_prob = torch.tensor(np.load('./warped_prob.npy'), dtype=torch.float32)#debug

    counts = torch.ones_like(probs)
    #TODO: attention dim expand
    probs = probs.unsqueeze(dim=1)
    counts = counts.unsqueeze(dim=1)
    images = raw_image.unsqueeze(dim=-1)#maybe no need
    #
    H,W = raw_image.shape[2:4]#H,W
    config = dict_update(homography_adaptation_default_config, config)

    for _ in range(config['num']-1):
        probs, counts, images = one_adaptation(net, raw_image, probs, counts, images, config, device=device)

    counts = torch.sum(counts, dim=1)
    max_prob, _ = torch.max(probs, dim=1)
    mean_prob = torch.sum(probs, dim=1)/counts

    if config['aggregation'] == 'max':
        prob = max_prob
    elif config['aggregation'] == 'sum':
        prob = mean_prob
    else:
        raise ValueError('Unkown aggregation method: {}'.format(config['aggregation']))

    if config['filter_counts']:
        prob = torch.where(counts>=config['filter_counts'], prob, torch.zeros_like(prob))

    return {'prob': prob, 'counts': counts,
            'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}



if __name__=='__main__':

    with open('./config/homo_export_labels.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)

    if not os.path.exists(config['data']['dst_label_path']):
        os.makedirs(config['data']['dst_label_path'])
    if not os.path.exists(config['data']['dst_image_path']):
        os.makedirs(config['data']['dst_image_path'])

    image_list = os.listdir(config['data']['src_image_path'])
    image_list = [os.path.join(config['data']['src_image_path'], fname) for fname in image_list]

    # image_list = []
    # with open('./coco_train_list.txt', 'r') as fin:
    #     for line in fin:
    #         image_list.append(line.strip())
    # image_list = image_list[0:int(len(image_list)*0.5)]

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    net = MagicPoint(config['model'], input_channel=1, grid_size=8,device=device)
    net.load_state_dict(torch.load(config['model']['pretrained_model']))
    net.to(device).eval()

    batch_fnames,batch_imgs,batch_raw_imgs = [],[],[]
    for idx, fpath in tqdm(enumerate(image_list)):
        root_dir, fname = os.path.split(fpath)
        ##
        img = read_image(fpath)
        img = ratio_preserving_resize(img, config['data']['resize'])
        t_img = to_tensor(img, device)
        ##
        batch_imgs.append(t_img)
        batch_fnames.append(fname)
        batch_raw_imgs.append(img)
        ##
        if len(batch_imgs)<1 and ((idx+1)!=len(image_list)):
            continue

        batch_imgs = torch.cat(batch_imgs)
        outputs = homography_adaptation(net, batch_imgs, config['data']['homography_adaptation'], device=device)
        prob = outputs['prob']
        ##nms or threshold filter
        if config['model']['nms']:
            prob = [box_nms(p.unsqueeze(dim=0),#to 1HW
                            config['model']['nms'],
                            min_prob=config['model']['det_thresh'],
                            keep_top_k=config['model']['topk']).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
        pred = (prob>=config['model']['det_thresh']).int()
        ##
        points = [torch.stack(torch.where(e)).T for e in pred]
        points = [pt.cpu().numpy() for pt in points]
        ##save points
        for fname, pt in zip(batch_fnames, points):
            if len(pt)==0:
                continue
            cv2.imwrite(os.path.join(config['data']['dst_image_path'], fname), img)
            np.save(os.path.join(config['data']['dst_label_path'], fname+'.npy'), pt)
            print('{}, {}'.format(os.path.join(config['data']['dst_label_path'], fname+'.npy'), len(pt)))

        # ## debug
        # import matplotlib.pyplot as plt
        # for img, pts in zip(batch_raw_imgs,points):
        #     debug_img = cv2.merge([img, img, img])
        #     for pt in pts:
        #         cv2.circle(debug_img, (int(pt[1]),int(pt[0])), 1, (0,255,0), thickness=-1)
        #     plt.imshow(debug_img)
        #     plt.show()
        # if idx>2:
        #     break

        batch_fnames,batch_imgs,batch_raw_imgs = [],[],[]
    print('Done')
