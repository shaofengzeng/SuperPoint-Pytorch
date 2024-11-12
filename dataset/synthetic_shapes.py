import random

import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from utils.params import dict_update, parse_primitives
from utils.keypoint_op import compute_keypoint_map
from dataset.utils import synthetic_dataset
from dataset.utils.homographic_augmentation import homographic_aug_pipline
from dataset.utils.photometric_augmentation import PhotoAugmentor


class SyntheticShapes(Dataset):
    default_config = {
        'primitives': 'all',
        'truncate': {},
        'validation_size': -1,
        'test_size': -1,
        'on-the-fly': True,
        'cache_in_memory': False,
        'suffix': None,
        'add_augmentation_to_test_set': False,
        'num_parallel_calls': 10,
        'generation': {
            'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
            'image_size': [960, 1280],
            'random_seed': 0,
            'params': {
                'generate_background': {
                    'min_kernel_size': 150, 'max_kernel_size': 500,
                    'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                'draw_stripes': {'transform_params': (0.1, 0.1)},
                'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
            },
        },
        'preprocessing': {
            'resize': [240, 320],
            'blur_size': 11,
        },
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        }
    }
    drawing_primitives = [
        'draw_lines',
        'draw_polygon',
        'draw_multiple_polygons',
        'draw_ellipses',
        'draw_star',
        'draw_checkerboard',
        'draw_stripes',
        'draw_cube',
        'gaussian_noise'
    ]

    def __init__(self, config, task, device='cpu'):
        """
        Args:
            config: hyper-parameters
            task: 'train','validation','test' or their combinations
            transforms: transformation for samples
        """
        self.device = device
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self.task = task if isinstance(task, (list, tuple)) else [task, ]
        self.photo_aug = PhotoAugmentor(config['augmentation']['photometric'])
        #load data, if no data generate some
        self.samples = self._init_dataset()

    def dump_primitive_data(self, primitive):
        output_dir = Path(self.config['data_dir'], primitive)

        synthetic_dataset.set_random_state(np.random.RandomState(self.config['generation']['random_seed']))
        for split, size in self.config['generation']['split_sizes'].items():
            im_dir, pts_dir = [Path(output_dir, i, split) for i in ['images', 'points']]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size),desc='Generate {}({})'.format(primitive, split)):
                image = synthetic_dataset.generate_background(
                        self.config['generation']['image_size'],
                        **self.config['generation']['params']['generate_background'])
                points = np.array(getattr(synthetic_dataset, primitive)(
                        image, **self.config['generation']['params'].get(primitive, {})))
                points = np.flip(points, 1)  # reverse convention with opencv

                b = self.config['preprocessing']['blur_size']
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (points * np.array(self.config['preprocessing']['resize'], np.float)
                          / np.array(self.config['generation']['image_size'], np.float))
                image = cv2.resize(image, tuple(self.config['preprocessing']['resize'][::-1]),
                                   interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(str(Path(im_dir, '{}.png'.format(i))), image)
                np.save(Path(pts_dir, '{}.npy'.format(i)), points)

    def _init_dataset(self,):
        # Parse drawing primitives
        primitives = parse_primitives(self.config['primitives'], self.drawing_primitives)

        basepath = Path(self.config['data_dir'])
        basepath.mkdir(parents=True, exist_ok=True)

        data_path = []
        for primitive in primitives:
            primitive_dir = Path(self.config['data_dir'], primitive)
            if not primitive_dir.exists():
                #try generate data
                self.dump_primitive_data(primitive)
            # Gather filenames in all splits, optionally truncate
            truncate = self.config['truncate'].get(primitive, 1)
            path = Path(basepath, primitive)
            for t in self.task:
                e = [str(p) for p in Path(path, 'images', t).iterdir()]
                f = [p.replace('images', 'points') for p in e]
                f = [p.replace('.png', '.npy') for p in f]
                data_path.extend([{'image': _im, 'point': _pt}
                                  for _im, _pt in
                                  zip(e[:int(truncate*len(e))], f[:int(truncate*len(f))])])
        return data_path

    def __len__(self,):
        return len(self.samples)


    def on_the_fly(self,):
        primitives = parse_primitives(self.config['primitives'], self.drawing_primitives)
        primitive = random.choice(primitives)
        image = synthetic_dataset.generate_background(
            self.config['generation']['image_size'],
            **self.config['generation']['params']['generate_background'])
        points = np.array(getattr(synthetic_dataset, primitive)(
            image, **self.config['generation']['params'].get(primitive, {})))
        points = np.flip(points, 1)#reverse convention with opencv

        b = self.config['preprocessing']['blur_size']
        image = cv2.GaussianBlur(image, (b, b), 0)
        points = (points * np.array(self.config['preprocessing']['resize'], float)
                  / np.array(self.config['generation']['image_size'], float))
        image = cv2.resize(image, tuple(self.config['preprocessing']['resize'][::-1]),
                           interpolation=cv2.INTER_LINEAR)

        return image, points


    def __getitem__(self, idx):

        if not self.config['on-the-fly']:
            img_path = self.samples[idx]['image']
            pts_path = self.samples[idx]['point']
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)#shape=(h,w)
            pts = np.load(pts_path)  #  (y,x) formated vector
        else:
            img, pts = self.on_the_fly()

        # #debug
        # for pt in pts:
        #     img = cv2.circle(img, (int(pt[1]),int(pt[0])),2, (0,255,0),1,1)
        # cv2.imshow("debug", img)
        # cv2.waitKey()
        # print(pts)


        H,W = img.shape

        if self.config['augmentation']['photometric']['enable']:#image augmentation
            img = self.photo_aug(img)
            pts_map = compute_keypoint_map(pts, (H,W))
            valid_mask = np.ones((H,W),dtype=int)
            homography = np.eye(3,dtype=np.float32)

        if self.config['augmentation']['homographic']['enable']:##homographic augmentation
            img, pts, pts_map, valid_mask, homography = homographic_aug_pipline(img, pts, self.config['augmentation']['homographic'])


        data = {'raw':{'img':img.astype(np.float32),#H,W
                       'kpts':pts.astype(np.float32),#N,2, yx format
                       'kpts_map':pts_map,#H,W
                       'mask':valid_mask,#H,W
                        },
                'homography': homography}#3,3

        ##normalize
        data['raw']['img'] = data['raw']['img']/255.

        return data

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `image`, `mask`, `homography`...
        Note that image_name and point cannot be batch as a tensor,
        so will be kept as list
        :return:
        """
        assert(len(samples)>0 and isinstance(samples[0], dict))
        batch = {'raw':{'img':[], 'kpts':[], 'kpts_map':[], 'mask':[]}, 'homography':[]}

        for item in samples:
            batch['raw']['img'].append(item['raw']['img'])
            batch['raw']['kpts'].append(item['raw']['kpts'])
            batch['raw']['kpts_map'].append(item['raw']['kpts_map'])
            batch['raw']['mask'].append(item['raw']['mask'])
            batch['homography'].append(item['homography'])

        batch['raw']['img'] = np.stack(batch['raw']['img'])#BHW
        batch['raw']['kpts_map'] = np.stack(batch['raw']['kpts_map'])#BHW
        batch['raw']['mask'] = np.stack(batch['raw']['mask'])#BHW
        batch['homography'] = np.stack(batch['homography'])#BHW

        #to tensor
        batch['raw']['img'] = torch.tensor(batch['raw']['img'][:,np.newaxis,:,:], dtype=torch.float32, device=self.device)#B1HW
        batch['raw']['kpts_map'] = torch.tensor(batch['raw']['kpts_map'], dtype=torch.float32, device=self.device)#BHW
        batch['raw']['mask'] = torch.tensor(batch['raw']['mask'], dtype=torch.float32, device=self.device)#BHW
        batch['homography'] = torch.tensor(batch['homography'], dtype=torch.float32, device=self.device)#B33

        return batch



if __name__=="__main__":
    import yaml
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    config_file = '../config/magic_point_syn_train.yaml'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    syn_datasets = {'train': SyntheticShapes(config['data'], task=['training', 'validation'], device=device),
                    'test': SyntheticShapes(config['data'], task=['test', ], device=device)}
    data_loaders = {'train': DataLoader(syn_datasets['train'], batch_size=1, shuffle=True,
                                        collate_fn=syn_datasets['train'].batch_collator),
                    'test': DataLoader(syn_datasets['test'], batch_size=1, shuffle=True,
                                       collate_fn=syn_datasets['test'].batch_collator)}
    for i, d in enumerate(data_loaders['train']):
        if i >= 100:
            break
        img = (d['raw']['img'][0] * 255).cpu().numpy().squeeze().astype(int).astype(np.uint8)
        img = cv2.merge([img, img, img])
        ##
        kpts = np.where(d['raw']['kpts_map'][0].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int)
        for kp in kpts:
            cv2.circle(img, (kp[1], kp[0]), radius=3, color=(0, 255, 0))
        print("points on the image")
        print(kpts)

        mask = d['raw']['mask'][0].cpu().numpy().squeeze().astype(int).astype(np.uint8)*255

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()
        print("show one")

