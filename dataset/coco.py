#-*-coding:utf8-*-
import os
import glob
from copy import deepcopy

import cv2
import numpy as np
from torch.utils.data import DataLoader
from utils.params import dict_update
from dataset.utils.homographic_augmentation import homographic_aug_pipline
from dataset.utils.photometric_augmentation import PhotoAugmentor
from utils.keypoint_op import compute_keypoint_map
from dataset.utils.photometric_augmentation import *


class COCODataset(torch.utils.data.Dataset):

    def __init__(self, config, is_train, device='cpu'):

        super(COCODataset, self).__init__()
        self.device = device
        self.is_train = is_train
        self.resize = tuple(config['resize'])
        self.photo_augmentor = PhotoAugmentor(config['augmentation']['photometric'])
        # load config
        self.config = config
        # get images
        if self.is_train:
            self.samples = self._init_data(config['image_train_path'], config['label_train_path'])
        else:
            self.samples = self._init_data(config['image_test_path'], config['label_test_path'])


    def _init_data(self, image_path, label_path=None):
        ##
        if not isinstance(image_path,list):
            image_paths, label_paths = [image_path,], [label_path,]
        else:
            image_paths, label_paths = image_path, label_path

        samples = []
        for im_path, lb_path in zip(image_paths, label_paths):
            im_names = [os.path.join(im_path, im_name) for im_name in os.listdir(im_path)]
            if lb_path is not None:
                label_names = [os.path.join(lb_path, os.path.basename(imp).split('.')[0]+'.npy') for imp in im_names]
            else:
                label_names = [None,]*len(im_names)
            temp = [{'image':imp, 'label':lb} for imp, lb in zip(im_names, label_names)]
            samples += temp

        return samples

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        '''load raw data'''
        data_path = self.samples[idx]#raw image path of processed image and point path
        img = cv2.imread(data_path['image'], 0)#Gray image
        img = cv2.resize(img, self.resize[::-1])
        H,W = img.shape
        kpts = None if data_path['label'] is None else np.load(data_path['label'])#N*2,yx

        # init data dict
        kpts_map = None if kpts is None else compute_keypoint_map(kpts, img.shape)
        valid_mask = np.ones((H,W), dtype=int)

        data = {'raw':{'img': img,
                       'kpts': kpts.astype(np.float32),
                       'kpts_map':kpts_map,
                       'mask':valid_mask.astype(np.float32)},
                'warp':None,
                'homography':np.eye(3,dtype=np.float32)}
        data['warp'] = deepcopy(data['raw'])

        if self.is_train:
            photo_enable = self.config['augmentation']['photometric']['train_enable']
            homo_enable = self.config['augmentation']['homographic']['train_enable']
        else:
            photo_enable = self.config['augmentation']['photometric']['test_enable']
            homo_enable = self.config['augmentation']['homographic']['test_enable']

        if homo_enable and data['raw']['kpts'] is not None:#apply homographic augmentation
            (warped_img, warped_kpts, warped_kpts_map, warped_mask,homo_mat) = \
                homographic_aug_pipline(data['warp']['img'], data['warp']['kpts'],self.config['augmentation']['homographic'])
            homo_data = {'warp':{'img':warped_img, 'kpts':warped_kpts, 'kpts_map':warped_kpts_map, 'mask':warped_mask}, 'homography':homo_mat}
            data.update(homo_data)

        if photo_enable:
            aug_img = self.photo_augmentor(data['warp']['img'])
            data['warp']['img'] = aug_img

        ##normalize
        data['raw']['img'] = (data['raw']['img']).astype(np.float32)/255.
        data['warp']['img'] = (data['warp']['img']).astype(np.float32)/255.

        return data#img:HW, kpts:N2(yx format), kpts_map:HW, valid_mask:HW, homography:3*3, all are np.array

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys like
        sub_data = {'img': X, 'kpts': X, 'kpts_map': X,'mask': X}
        batch = {'raw':sub_data0, 'warp':sub_data1, 'homography': X}
        img:H*W, kpts:N*2(yx format), kpts_map:HW, valid_mask:HW, homography:HW
        :return:
        """
        sub_data = {'img': [], 'kpts_map': [],'mask': []}#remove kpts
        batch = {'raw':sub_data, 'warp':deepcopy(sub_data), 'homography': []}
        for s in samples:
            batch['homography'].append(s['homography'])
            for k in sub_data:
                if k=='img':
                    batch['raw'][k].append(s['raw'][k][np.newaxis,:,:])#1HW
                    if 'warp' in s:
                        batch['warp'][k].append(s['warp'][k][np.newaxis,:,:])#1HW
                else:
                    batch['raw'][k].append(s['raw'][k])
                    if 'warp' in s:
                        batch['warp'][k].append(s['warp'][k])

        batch['homography'] = np.stack(batch['homography'])
        batch['homography'] = torch.tensor(batch['homography'],device=self.device)
        for k0 in ('raw','warp'):
            for k1 in sub_data:#`img`, `img_name`, `kpts`, `kpts_map`...
                if k1=='kpts' or k1=='img_name':
                    continue
                batch[k0][k1] = np.stack(batch[k0][k1])
                batch[k0][k1] = torch.tensor(batch[k0][k1],device=self.device)#to tensor

        return batch#img:B1HW, kpts_map:BHW, mask:BHW, homography:B33


if __name__=='__main__':
    import yaml
    from dataset.utils.photometric_augmentation import *
    with open('../config/superpoint_train.yaml','r') as fin:
        config = yaml.safe_load(fin)

    coco = COCODataset(config['data'],True)
    cdataloader = DataLoader(coco,collate_fn=coco.batch_collator,batch_size=1,shuffle=False)

    for i,d in enumerate(cdataloader):
        if i>=10:
            break
        img = (d['raw']['img']*255).cpu().numpy().squeeze().astype(int).astype(np.uint8)
        img_warp = (d['warp']['img']*255).cpu().numpy().squeeze().astype(int).astype(np.uint8)
        img = cv2.merge([img, img, img])
        img_warp = cv2.merge([img_warp, img_warp, img_warp])

        kpts = np.where(d['raw']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(int)
        for kp in kpts:
            cv2.circle(img, (kp[1], kp[0]), radius=1, color=(0,255,0))
        kpts = np.where(d['warp']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(int)
        for kp in kpts:
            cv2.circle(img_warp, (kp[1], kp[0]), radius=1, color=(0,255,0))

        mask = d['raw']['mask'].cpu().numpy().squeeze().astype(int).astype(np.uint8)*255
        warp_mask = d['warp']['mask'].cpu().numpy().squeeze().astype(int).astype(np.uint8)*255


        cv2.imshow("img",img)
        cv2.imshow("warp img", img_warp)
        cv2.imshow("mask", mask)
        cv2.imshow("warp mask", warp_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()

    print('Done')
