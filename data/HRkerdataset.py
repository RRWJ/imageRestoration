import random
import numpy as np
import torch
import data.util as util
import torch.utils.data as data
import os
import hdf5storage
from torchvision import transforms

class HRkerdataset(data.Dataset):
    def __init__(self , opt):
        super(HRkerdataset, self).__init__()
        print('Get L/H for image-to-image mapping. Only "HR_paths" are needed.sythesize bicubicly downsampled L on-the-fly.')
        self.opt = opt
        self.n_colors = opt.n_colors
        self.scale = opt.scale
        self.patch_size = opt.patch_size
        self.sigma = opt.sigma
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt.sigma_test
        self.idx_scale = 0
        # load PCA matrix of enough kernel
        # print('load PCA matrix')
        self.p = hdf5storage.loadmat('D:/NEU/ImageRestoration/codes/srmd_pca_matlab.mat')['P']
        self.ksize = int(np.sqrt(self.p.shape[-1]))  # kernel size
        # print('PCA matrix shape: {}'.format(pca_matrix.shape))
        # self.transform = transforms.Compose([transforms.ToTensor()])
        self.HR_paths  = util.get_image_paths_(opt.dataroot_HR)

    def __getitem__(self, index ):
        # 从文件中读出HR图片，返回HR、LR图片对
        HR_path , LR_path = None, None
        HR_path = self.HR_paths[index]
        img_HR = util.read_img_(HR_path, self.n_colors)
        # 归一化
        img_HR = util.unit2single(img_HR)
        img_HR = util.modcrop(img_HR, self.scale)

        if self.opt.phase == 'train':
            l_max = 50
            theta = np.pi * np.random.rand(1)
            l1 = 0.1 + l_max * np.random.rand(1)
            l2 = 0.1 + (l1 - 0.1) * np.random.rand(1)
            kernel = util.anisotropic_Gaussian(ksize=self.ksize, theta=theta[0], l1=l1[0], l2=l2[0])
        else:
            kernel = util.anisotropic_Gaussian(ksize=self.ksize, theta=np.pi, l1=0.1, l2=0.1)
        k = np.reshape(kernel, (-1), order="F")
        k_reduced = np.dot(self.p, k)
        k_reduced = torch.from_numpy(k_reduced).float()
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
        M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)
        M = M_vector.repeat(1, img_LR.size()[-2], img_LR.size()[-1])

        img_LR = torch.cat((img_LR, M), 0)
        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}
        # return img_LR, img_HR, LR_path, HR_path
        # 1102之前训练batch类型报错，记录一下
        # TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'data.div2k.DIV2K'>
        # return self.transform(img_LR), self.transform(img_HR),LR_path,HR_path
    def __len__(self):
        return len(self.HR_paths)

