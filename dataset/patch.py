import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from dataset.utils.homographic_augmentation import sample_homography,ratio_preserving_resize


class PatchesDataset(Dataset):
    default_config = {
        'dataset': 'hpatches',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': False
        }
    }

    def __init__(self, config, device='cpu'):

        super(PatchesDataset, self).__init__()
        self.device = device
        self.config = config
        self.files = self._init_dataset()

    def _init_dataset(self,):
        dataset_folder = self.config['data_dir']
        sub_folders = [x for x in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder,x))]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for sfolder in sub_folders:
            if self.config['alteration'] == 'i' and sfolder[0] != 'i':
                continue
            if self.config['alteration'] == 'v' and sfolder[0] != 'v':
                continue
            num_images = 1 if 'coco' in self.config['data_dir'] else 5
            file_ext = '.jpg' if 'coco' in self.config['data_dir'] else '.ppm'
            for i in range(2, 2 + num_images):
                image_paths.append(os.path.join(dataset_folder, sfolder, "1" + file_ext))
                warped_image_paths.append(os.path.join(dataset_folder, sfolder, str(i) + file_ext))
                homographies.append(np.loadtxt(os.path.join(dataset_folder, sfolder, "H_1_" + str(i))))

        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies}
        return files

    def _preprocess(self, image):
        if len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = ratio_preserving_resize(image, self.config['preprocessing']['resize'])
        return image

    def _adapt_homography_to_preprocessing(self, zip_data):
        '''缩放后对应的图像的homography矩阵
        :param zip_data:{'shape':原图像HW,
                         'warped_shape':warped图像HW,
                         'homography':原始变换矩阵}
        :return:对应当前图像尺寸的homography矩阵
        '''
        H = zip_data['homography'].astype(np.float32)
        source_size = zip_data['shape'].astype(np.float32)#h,w
        source_warped_size = zip_data['warped_shape'].astype(np.float32)#h,w
        target_size = np.array(self.config['preprocessing']['resize'],dtype=np.float32)[::-1]#h,w

        # Compute the scaling ratio due to the resizing for both images
        s = np.max(target_size/source_size)
        up_scale = np.diag([1./s, 1./s, 1])
        warped_s = np.max(target_size/source_warped_size)
        down_scale = np.diag([warped_s, warped_s, 1])

        # Compute the translation due to the crop for both images
        pad_y, pad_x = (source_size*s - target_size)//2.0

        translation = np.array([[1, 0, pad_x],
                                [0, 1, pad_y],
                                [0, 0, 1]],dtype=np.float32)
        pad_y, pad_x = (source_warped_size*warped_s - target_size) //2.0

        warped_translation = np.array([[1,0, -pad_x],
                                       [0,1, -pad_y],
                                       [0,0,1]], dtype=np.float32)
        H = warped_translation @ down_scale @ H @ up_scale @ translation
        return H

    def __len__(self):
        return len(self.files['image_paths'])

    def __getitem__(self, idx):
        im_path = self.files['image_paths'][idx]
        img = cv2.imread(im_path,0)#H,W
        warped_im_path = self.files['warped_image_paths'][idx]
        warped_img = cv2.imread(warped_im_path,0)#H,W
        homography = self.files['homography'][idx]

        if self.config['preprocessing']['resize']:
            img_shape = img.shape
            warped_shape = warped_img.shape
            homography = {'homography': homography, 'shape': np.array(img_shape), 'warped_shape': np.array(warped_shape)}
            homography = self._adapt_homography_to_preprocessing(homography)

        img = self._preprocess(img)
        warped_img = self._preprocess(warped_img)

        img = img/255.
        warped_img = warped_img/255.
        ##to tenosr

        img = torch.as_tensor(img,dtype=torch.float32, device=self.device)#HW
        warped_img = torch.as_tensor(warped_img, dtype=torch.float32, device=self.device)#HW
        homography = torch.as_tensor(homography, device=self.device)#HW

        data = {'img': img, 'warp_img': warped_img, 'homography': homography}

        return data

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `img`, `img_name`, `kpts`, `kpts_map`,
        `valid_mask`, `homography`...
        img:H*W, kpts:N*2, kpts_map:HW, valid_mask:HW, homography:HW
        :return:batch data
        """
        assert (len(samples) > 0 and isinstance(samples[0], dict))
        batch = {'img':[], 'warp_img':[], 'homography': []}
        for s in samples:
            for k,v in s.items():
                if 'img' in k:
                    batch[k].append(v.unsqueeze(dim=0))
                else:
                    batch[k].append(v)
        ##
        for k in batch:
            batch[k] = torch.stack(batch[k],dim=0)
        return batch



if __name__=='__main__':
    import yaml
    with open('../config/magic_point_repeatability.yaml','r') as fin:
        config = yaml.safe_load(fin)

    datas = PatchesDataset(config['data'])
    print('Done')