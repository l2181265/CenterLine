from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, centerlines_dir, points_dir, scale=1, centerlines_suffix='', points_suffix=''):
        self.imgs_dir = imgs_dir
        self.centerlines_dir = centerlines_dir
        self.points_dir = points_dir
        self.scale = scale
        self.centerlines_suffix = centerlines_suffix
        self.points_suffix = points_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        ###################  显示一共多少数据集  ##################
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        centerlines_file = glob(self.centerlines_dir + idx + self.centerlines_suffix + '.*')
        points_file = glob(self.points_dir + idx + self.points_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(centerlines_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {centerlines_file}'
        assert len(points_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {points_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        centerlines = Image.open(centerlines_file[0])
        points = Image.open(points_file[0])
        img = Image.open(img_file[0])

        assert img.size == centerlines.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {centerlines.size}'
        assert img.size == points.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {points.size}'

        img = self.preprocess(img, self.scale)
        centerlines = self.preprocess(centerlines, self.scale)
        points = self.preprocess(points, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'centerlines': torch.from_numpy(centerlines).type(torch.FloatTensor),
            'points': torch.from_numpy(points).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, centerlines_dir, points_dir, scale=1):
        super().__init__(imgs_dir, centerlines_dir, points_dir, scale, centerlines_suffix='_centerlines', points_suffix='_points')
