import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.patch import PatchesDataset
from model.superpoint_bn import SuperPointBNNet

if __name__ == '__main__':
    ##
    with open('./config/export_descriptors.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)

    output_dir = config['data']['export_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ##
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    p_dataset = PatchesDataset(config['data'], device=device)
    p_dataloader = DataLoader(p_dataset, batch_size=1, shuffle=False, collate_fn=p_dataset.batch_collator)

    net = SuperPointBNNet(config['model'], device=device)
    net.load_state_dict(torch.load(config['model']['pretrained_model']))
    net.to(device).eval()

    ##
    for i, data in tqdm(enumerate(p_dataloader)):
        pred1 = net(data['img'])
        pred2 = net(data['warp_img'])

        pred = {'prob': pred1['det']['prob_nms'],
                'warped_prob': pred2['prob_nms'],
                'desc': pred1['descriptors'],
                'warped_desc': pred2['descriptors'],
                'homography': data['homography']}

        if not ('name' in data):
            pred.update(data)
        # to numpy
        pred = {k: v.cpu().numpy().squeeze() for k, v in pred.items()}
        filename = data['name'] if 'name' in data else str(i)
        filepath = os.path.join(output_dir, '{}.npz'.format(filename))
        np.savez_compressed(filepath, **pred)
    print('Done')



