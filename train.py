#-*-coding:utf8-*-
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import cv2
import os
import yaml
import argparse
from tqdm import tqdm
from dataset.coco import COCODataset
from dataset.synthetic_shapes import SyntheticShapes
from torch.utils.data import DataLoader
from model.magic_point import MagicPoint
from model.superpoint_bn import SuperPointBNNet
from solver.loss import loss_func
from solver.nms import box_nms


def train_eval(model, dataloader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['solver']['base_lr'])
    #lr_sch = StepLR(optimizer, step_size=60000, gamma=0.5)

    # start training
    for epoch in range(config['solver']['epoch']):
        model.train()
        mean_loss, best_loss = [], 9999
        for i, data in tqdm(enumerate(dataloader['train'])):

            prob, desc, prob_warp, desc_warp = None, None, None, None
            if config['model']['name']=='magicpoint' and config['data']['name']=='coco':
                raw_outputs = model(data['warp'])
            else:
                raw_outputs = model(data['raw'])
            if config['model']['name']=='superpoint':
                warp_outputs = model(data['warp'])
                prob, desc, prob_warp, desc_warp = raw_outputs['det_info'], \
                                                   raw_outputs['desc_info'], \
                                                   warp_outputs['det_info'],\
                                                   warp_outputs['desc_info']
            else:
                prob = raw_outputs

            ##loss
            loss = loss_func(config['solver'], data, prob, desc,
                             prob_warp, desc_warp, device)

            mean_loss.append(loss.item())
            # reset
            model.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_sch.step()

            # for every 1000 images, print progress and visualize the matches
            if i % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR [{}], Loss: {:.3f}'
                      .format(epoch, config['solver']['epoch'], i, len(dataloader['train']),
                              optimizer.state_dict()['param_groups'][0]['lr'], np.mean(mean_loss)))
                mean_loss = []
            # do evaluation
            if (i%60000==0 and i!=0) or (i+1)==len(dataloader['train']):
                eval_loss = do_eval(model, dataloader['test'], config, device)
                model.train()
                if eval_loss < best_loss:
                    save_path = os.path.join(config['solver']['save_dir'],
                                             config['solver']['model_name'] + '_{}_{}.pth').format(epoch, round(eval_loss, 4))
                    torch.save(model.state_dict(), save_path)
                    print('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}'
                          .format(epoch, config['solver']['epoch'], i, len(dataloader['train']), save_path))
                    best_loss = eval_loss
                mean_loss = []

@torch.no_grad()
def do_eval(model, dataloader, config, device):
    model.eval()
    mean_loss = []
    for ind, data in tqdm(enumerate(dataloader)):
        prob, desc, prob_warp, desc_warp = None, None, None, None
        if config['model']['name'] == 'magicpoint' and config['data']['name'] == 'coco':
            raw_outputs = model(data['warp'])
        else:
            raw_outputs = model(data['raw'])
        if config['model']['name'] == 'superpoint':
            warp_outputs = model(data['warp'])
            prob, desc, prob_warp, desc_warp = raw_outputs['det_info'], \
                                               raw_outputs['desc_info'], \
                                               warp_outputs['det_info'], \
                                               warp_outputs['desc_info']
        else:
            prob = raw_outputs

        # compute loss
        loss = loss_func(config['solver'], data, prob, desc,
                         prob_warp, desc_warp, device)
        mean_loss.append(loss.item())
    mean_loss = np.mean(mean_loss)

    return mean_loss


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()

    config_file = args.config
    assert (os.path.exists(config_file))
    ##
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    if not os.path.exists(config['solver']['save_dir']):
        os.makedirs(config['solver']['save_dir'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_loaders = None
    if config['data']['name'] == 'coco':
        datasets = {k: COCODataset(config['data'], is_train=True if k == 'train' else False, device=device) for k in
                    ['test', 'train']}
        data_loaders = {k: DataLoader(datasets[k],
                                      config['solver']['{}_batch_size'.format(k)],
                                      collate_fn=datasets[k].batch_collator,
                                      shuffle=True) for k in ['train', 'test']}
    elif config['data']['name'] == 'synthetic':
        datasets = {'train': SyntheticShapes(config['data'], task=['training', 'validation'], device=device),
                    'test': SyntheticShapes(config['data'], task=['test', ], device=device)}
        data_loaders = {'train': DataLoader(datasets['train'], batch_size=16, shuffle=True,
                                            collate_fn=datasets['train'].batch_collator),
                        'test': DataLoader(datasets['test'], batch_size=16, shuffle=False,
                                           collate_fn=datasets['test'].batch_collator)}

    if config['model']['name'] == 'superpoint':
        model = SuperPointBNNet(config['model'], device=device)
    else:
        model = MagicPoint(config['model'], device=device)

    if os.path.exists(config['model']['pretrained_model']):
        model.load_state_dict(torch.load(config['model']['pretrained_model']))
    model.to(device)

    train_eval(model, data_loaders, config)
    print('Done')

