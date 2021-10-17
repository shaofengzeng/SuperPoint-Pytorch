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
            'suffix': None,
            'add_augmentation_to_test_set': False,
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
        # Update config
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


    def __getitem__(self, idx):
        img_path = self.samples[idx]['image']
        pts_path = self.samples[idx]['point']
        #
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)#shape=(h,w)
        pts = np.load(pts_path)  # attention: each point is a (y,x) formated vector

        kp_map, valid_mask, homography = None, None, None
        #augmentations
        if 'training' in self.task:
            if self.config['augmentation']['photometric']['enable']:#image augmentation
                img = self.photo_aug(img)
            if self.config['augmentation']['homographic']['enable']:##homographic augmentation
                # first to tensor
                img = torch.as_tensor(img[np.newaxis, np.newaxis,:,:],device=self.device, dtype=torch.float32)
                pts = torch.as_tensor(pts, dtype=torch.float32, device=self.device)

                # img:[1,1,H,W], point:[N,2], valid_mask:1,H,W, homography: 1,3,3; tensors
                homo_data = homographic_aug_pipline(img, pts, self.config['augmentation']['homographic'], device=self.device)

                img,pts,kp_map,valid_mask,homography = homo_data['warp']['img'],\
                                                homo_data['warp']['kpts'],\
                                                homo_data['warp']['kpts_map'],\
                                                homo_data['warp']['mask'],\
                                                homo_data['homography']


        if 'test' in self.task or self.config['augmentation']['homographic']['enable']==False:#no homography transformation
            img = torch.as_tensor(img[np.newaxis, np.newaxis, :, :], device=self.device,dtype=torch.float32)
            pts = torch.as_tensor(pts, dtype=torch.float32, device=self.device)
            kp_map = compute_keypoint_map(pts, img.shape[2:], device=self.device)
            valid_mask = torch.ones_like(img).squeeze(dim=1)
            homography = torch.eye(3, device=self.device)

        img = img/255.

        data = {'raw':{'img':img.squeeze(dim=0),#1,H,W
                       'kpts':pts,#N,2
                       'kpts_map':kp_map,#H,W
                       'mask':valid_mask.squeeze(dim=0),#H,W
                        },
                'homography': homography.squeeze(dim=0)  # 3,3 or None

                }
        return data

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `image`, `image_name`, `point`, `keypoint_map`,
        `valid_mask`, `homography`...
        Note that image_name and point cannot be batch as a tensor,
        so will be kept as list
        :return:
        """
        assert(len(samples)>0 and isinstance(samples[0], dict))
        batch = {'raw':{'img':[], 'kpts':[], 'kpts_map':[], 'mask':[]}, 'homography':[]}
        ##
        for item in samples:
            batch['raw']['img'].append(item['raw']['img'])
            batch['raw']['kpts'].append(item['raw']['kpts'])
            batch['raw']['kpts_map'].append(item['raw']['kpts_map'])
            batch['raw']['mask'].append(item['raw']['mask'])
            batch['homography'].append(item['homography'])
        ##
        batch['raw']['img'] = torch.stack(batch['raw']['img'])
        batch['raw']['kpts_map'] = torch.stack(batch['raw']['kpts_map'])
        batch['raw']['mask'] = torch.stack(batch['raw']['mask'])
        batch['homography'] = torch.stack(batch['homography'])
        return batch



if __name__=="__main__":
    import yaml
    from torch.utils.data import DataLoader

    config_file = '../config/magic_point_train.yaml'
    device = 'cpu'#'cuda:3' if torch.cuda.is_available() else 'cpu'
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    syn_datasets = {'train': SyntheticShapes(config['data'], task=['training', 'validation'], device=device),
                    'test': SyntheticShapes(config['data'], task=['test', ], device=device)}
    data_loaders = {'train': DataLoader(syn_datasets['train'], batch_size=1, shuffle=False,
                                        collate_fn=syn_datasets['train'].batch_collator),
                    'test': DataLoader(syn_datasets['test'], batch_size=1, shuffle=False,
                                       collate_fn=syn_datasets['test'].batch_collator)}
    for i, data in enumerate(data_loaders['train']):
        img = data['raw']['img'][0]
        print(img)
        img = (img*255).squeeze().cpu().numpy().astype(np.int).astype(np.uint8)
        pt_map = data['raw']['kpts_map'][0].squeeze().cpu().numpy()
        mask = data['raw']['mask'][0].squeeze().cpu().numpy()
        pts = np.vstack(np.where(pt_map==1)).T
        print(img.shape)
        print(i)

