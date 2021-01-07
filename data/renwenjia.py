import random
import numpy as np
import torch
import data.util as util
import torch.utils.data as data
import torch
import numpy as np
import random
from utils.utils_deblur import fspecial

class HRkerdataset(data.Dataset):
    def __init__(self , opt):
        super(HRkerdataset, self).__init__()
        print('Get L/H for image-to-image mapping. Only "HR_paths" are needed.sythesize bicubicly downsampled L on-the-fly.')
        self.opt = opt
        self.n_colors = opt.n_colors
        self.scale = opt.scale
        self.patch_size = opt.patch_size
        self.sigma_min, self.sigma_max = self.opt.sigma[0], self.opt.sigma[1]
        self.sigma_test = opt.sigma_test
        self.idx_scale = 0
        if self.opt.phase == 'train':
            self.HR_paths  = util.get_image_paths_(opt.dataroot_HR)

    def __getitem__(self, index ):
        # 从文件中读出HR图片，返回HR、LR图片对
        HR_path , LR_path = None, None
        HR_path = self.HR_paths[index]
        img_HR = util.read_img_(HR_path, self.n_colors)
        # 归一化
        img_HR = util.unit2single(img_HR)
        img_HR = util.modcrop(img_HR, self.scale)
        kernel = fspecial('gaussian', 15, 2.6)
        H, W, _ = img_HR.shape
        img_LR = util.srmd_degradation(img_HR, kernel, self.scale)
        if self.opt.phase == 'train':
            patch_H, patch_L = util.paired_random_crop(img_HR, img_LR, self.patch_size, self.scale)
            # augmentation - flip, rotate
            img_HR , img_LR = util.augment([patch_H, patch_L], self.opt.use_flip,self.opt.use_rot)
            img_HR = util.img_single2tensor(img_HR)
            img_LR = util.img_single2tensor(img_LR)
            if random.random() < 0.1:
                noise_level = torch.zeros(1).float()
            else:
                noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0
        else:
            img_HR = util.img_single2tensor(img_HR)
            img_LR = util.img_single2tensor(img_LR)
            noise_level = torch.FloatTensor([self.sigma_test])
        noise = torch.randn(img_LR.size()).mul_(noise_level).float()
        img_LR.add_(noise)
        kernel = util.single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        noise_level = torch.FloatTensor([noise_level]).view([1,1,1])
        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'k': kernel, 'sigma': noise_level, 'sf': self.scale,'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.HR_paths)

