#-*-coding:utf-8-*-
import os
import yaml
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from dataset.patch import PatchesDataset
from model.magic_point import MagicPoint

if __name__=="__main__":
    ##
    with open('./config/magic_point_repeatability.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)

    output_dir = config['data']['export_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ##
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    p_dataset = PatchesDataset(config['data'],device=device)
    p_dataloader = DataLoader(p_dataset,batch_size=1,shuffle=False, collate_fn=p_dataset.batch_collator)

    net = MagicPoint(config['model'], input_channel=1, grid_size=8, device=device)
    net.load_state_dict(torch.load(config['model']['pretrained_model']))
    net.to(device).eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(p_dataloader)):
            prob1 = net(data['img'])
            prob2 = net(data['warp_img'])
            ##
            pred = {'prob':prob1['prob_nms'], 'warp_prob':prob2['prob_nms'],
                    'homography': data['homography']}

            if not ('name' in data):
                pred.update(data)
            #to numpy
            pred = {k:v.cpu().numpy().squeeze() for k,v in pred.items()}
            filename = data['name'] if 'name' in data else str(i)
            filepath = os.path.join(output_dir, '{}.npz'.format(filename))
            np.savez_compressed(filepath, **pred)

    print('Done')