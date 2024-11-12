import yaml
import os
import torch
from tqdm import tqdm
from math import pi
import cv2
import numpy as np
from utils.params import dict_update
from solver.nms import box_nms
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


@torch.no_grad()
def homography_adaptation(net, cv_image, config, device='cpu'):
    """
    :param cv image: cv2 image
    :param net: MagicPointNet
    :param config:
    :return:
    """
    H, W = cv_image.shape
    tensor_images = cv_image.astype('float32') / 255.
    tensor_images = torch.from_numpy(tensor_images[np.newaxis,np.newaxis,:,:]).to(device)

    pred_res = net(tensor_images)
    probs = pred_res['prob']#BHW
    probs = probs.cpu().numpy()

    counts = np.ones_like(probs)
    probs = probs[:,:,:,np.newaxis]#BHW1
    counts = counts[:,:,:,np.newaxis]#BHW1
    images = cv_image[np.newaxis,np.newaxis,:,:,np.newaxis]#BCHW1
    config = dict_update(homography_adaptation_default_config, config)

    def step(images, probs, counts):
        '''
        :param net: magicpoint net
        :param images: np array, 11HW1
        :param probs: np array, BHW1
        :param counts: np array, BHW1
        :return
        '''
        B, C, H, W, _ = images.shape
        M = sample_homography(shape=[H, W], config=config['homographies'])
        M_inv = np.linalg.inv(M)

        warped = cv2.warpPerspective(cv_image, M, dsize=(W,H))#注意opencv使用的十xy格式
        count = cv2.warpPerspective(np.ones((H, W), dtype=np.float32), M_inv, dsize=(W,H))
        mask = cv2.warpPerspective(np.ones((H, W), dtype=np.float32), M, dsize=(W,H))

        # Ignore the detections too close to the border to avoid artifacts
        if config['valid_border_margin']:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['valid_border_margin'] * 2,) * 2)
            count = cv2.erode(count, kernel, iterations=1)  # +1
            mask = cv2.erode(mask, kernel, iterations=1)  # +1

        # Predict detection probabilities
        tensor_warped = warped.astype('float32') / 255.
        tensor_warped = torch.from_numpy(tensor_warped[np.newaxis,np.newaxis,:,:]).to(device)
        net_output = net(tensor_warped)
        prob = net_output['prob']

        prob = prob.cpu().numpy()#BHW
        prob = prob.squeeze(axis=0)#HW
        prob = prob * mask
        prob_proj = cv2.warpPerspective(prob, M_inv, dsize=(W,H))
        prob_proj = prob_proj * count  # project back

        # the probabilities of each pixels on raw image
        probs = np.concatenate([probs, prob_proj[np.newaxis,:,:,np.newaxis]],axis=-1)
        counts = np.concatenate([counts, count[np.newaxis,:,:,np.newaxis]], axis=-1)
        images = np.concatenate([images, warped[np.newaxis,np.newaxis,:,:,np.newaxis]], axis=-1)

        # #deubg
        # for i,(dprob, dcount) in enumerate(zip(probs.transpose(3,0,1,2), counts.transpose(3,0,1,2))):
        #     dprob = dprob.squeeze()
        #     dcount = dcount.squeeze()
        #     dimage = images[0,0,:,:,0]
        #
        #     dprob = (dprob*255).astype(int).astype(np.uint8)
        #     dcount = (dcount * 255).astype(int).astype(np.uint8)
        #
        #     kpts = np.stack(np.where(dprob > 0.1)).T
        #
        #     dprob = cv2.merge([np.zeros_like(dprob),dprob,np.zeros_like(dprob)])
        #     dcount = cv2.merge([dcount, dcount, dcount])
        #     dimage = cv2.merge([dimage, dimage, dimage])
        #     dimage[kpts[:,0],kpts[:,1],:] = np.array([0,255,0],dtype=np.uint8)
        #
        #     cv2.imshow("dprob{}".format(i), dprob)
        #     cv2.imshow("dcount{}".format(i), dcount)
        #     cv2.imshow("dimage{}".format(i), dimage)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return images, probs, counts

    for _ in range(config['num']-1):
        images, probs, counts = step(images, probs, counts)

    counts = np.sum(counts, axis=-1)
    max_prob = np.max(probs, axis=-1)
    mean_prob = np.sum(probs, axis=-1)/counts

    if config['aggregation'] == 'max':
        prob = max_prob
    elif config['aggregation'] == 'sum':
        prob = mean_prob
    else:
        raise ValueError('Unkown aggregation method: {}'.format(config['aggregation']))

    if config['filter_counts']:
        prob = np.where(counts>=config['filter_counts'], prob, np.zeros_like(prob))

    return {'prob': prob, 'counts': counts,
            'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}


if __name__=='__main__':
    import matplotlib.pyplot as plt

    with open('./config/homo_export_labels.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)

    if not os.path.exists(config['data']['dst_label_path']):
        os.makedirs(config['data']['dst_label_path'])
    if not os.path.exists(config['data']['dst_image_path']):
        os.makedirs(config['data']['dst_image_path'])


    image_list = os.listdir(config['data']['src_image_path'])
    image_list = [os.path.join(config['data']['src_image_path'], fname) for fname in image_list]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = MagicPoint(config['model'], input_channel=1, grid_size=8,device=device)
    net.load_state_dict(torch.load(config['model']['pretrained_model']))
    net.to(device).eval()

    for idx, fpath in tqdm(enumerate(image_list)):
        root_dir, fname = os.path.split(fpath)

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        img = ratio_preserving_resize(img, config['data']['resize'])

        homo_adap_outputs = homography_adaptation(net, img, config['data']['homography_adaptation'], device=device)
        prob = homo_adap_outputs['prob']

        #apply nms
        if config['model']['nms']:
            prob = torch.tensor(prob, device='cpu')
            prob = box_nms(prob,#to 1HW
                            config['model']['nms'],
                            min_prob=config['model']['det_thresh'],
                            keep_top_k=config['model']['topk'])
            prob = prob.numpy()

        pred = (prob>=config['model']['det_thresh']).astype(int)
        points = np.stack(np.where(pred.squeeze(axis=0))).T

        ##save points
        cv2.imwrite(os.path.join(config['data']['dst_image_path'], fname), img)
        np.save(os.path.join(config['data']['dst_label_path'], fname.split('.')[0]+'.npy'), points)
        print('{}, {}'.format(os.path.join(config['data']['dst_label_path'], fname.split('.')[0]+'.npy'), len(points)))

        # #debug
        # debug_img = cv2.merge([img, img, img])
        # for pt in points:
        #     cv2.circle(debug_img, (int(pt[1]),int(pt[0])), 1, (0,255,0), thickness=-1)
        # cv2.imshow("debug homo adap", debug_img)
        # cv2.waitKey()

    print('Done')
