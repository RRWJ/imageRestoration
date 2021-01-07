import random
import numpy as np
import torch
import data.util as util
import torch.utils.data as data
import cv2

class HRDataset(data.Dataset):
    def __init__(self , opt):
        super(HRDataset, self).__init__()
        print('Get L/H for image-to-image mapping. Only "HR_paths" are needed.sythesize bicubicly downsampled L on-the-fly.')
        self.opt = opt
        self.patch_size = opt.patch_size if opt.patch_size else 48
        self.n_channels = opt.n_channels if opt.n_channels else 3
        self.scale = opt.scale if opt.scale else 4
        self.use_flip = opt.use_flip if opt.scale else False
        self.use_rot = opt.use_rot if opt.scale else False
        self.HR_paths = util.get_image_paths_(opt.dataroot_HR)
        self.LR_paths , self.LR_size = None,None
        assert self.HR_paths, 'Error: HR paths are empty.'

    def __getitem__(self, index ):
        # 从文件中读出HR图片，返回HR、LR图片对
        HR_path , LR_path = None, None
        HR_path = self.HR_paths[index]
        img_HR = util.read_img_(HR_path, self.n_channels)  # return: Numpy float32, HWC, BGR, [0,1]
        scale = self.scale
        patch_size = self.patch_size
        if self.opt.phase == 'train':
            H_s, W_s, _ = img_HR.shape
            img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
            # force to 3 channels
            if img_HR.ndim == 2:
                img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)
        # using matlab imresize
        img_LR = util.imresize_np(img_HR, 1 / scale, True)
        H, W, C = img_LR.shape
        if img_LR.ndim == 2:
            img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt.phase == 'train':
            patch_H, patch_L = util.paired_random_crop(img_HR, img_LR, patch_size, scale, gt_path)
            # augmentation - flip, rotate
            img_HR , img_LR = util.augment([patch_H, patch_L], self.opt.use_flip,self.opt.use_rot)


        img_HR  = util.img2tensor(img_HR)
        img_LR = util.img2tensor(img_LR)

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.HR_paths)
