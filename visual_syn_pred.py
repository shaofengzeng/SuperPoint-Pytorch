#-*-coding:utf8-*-
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import cv2
import yaml
from dataset.synthetic_shapes import SyntheticShapes
from model.magic_point import MagicPoint
from torch.utils.data import DataLoader


with open('./config/magic_point_syn_train.yaml', 'r', encoding='utf8') as fin:
    config = yaml.safe_load(fin)

device = 'cpu' #'cuda:2' if torch.cuda.is_available() else 'cpu'
dataset_ = SyntheticShapes(config['data'], task='training', device=device)
dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=False, collate_fn=dataset_.batch_collator)

net = MagicPoint(config['model'], device=device)

net.load_state_dict(torch.load(config['model']['pretrained_model']))
net.to(device).eval()

with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader_)):
        ret = net(data['raw']['img'])
        ##debug
        if i > 10:
            break
        warp_img = (data['raw']['img'] * 255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        warp_img = cv2.merge((warp_img, warp_img, warp_img))
        prob = ret['prob_nms'].cpu().numpy().squeeze()
        keypoints = np.where(prob > 0)
        keypoints = np.stack(keypoints).T
        for kp in keypoints:
            cv2.circle(warp_img, (int(kp[1]), int(kp[0])), radius=3, color=(0, 255, 0))
        plt.imshow(warp_img)
        plt.show()

print('Done')