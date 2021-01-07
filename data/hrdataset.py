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
        self.images_hr, self.images_lr = [], [[] for _ in self.scale]
        list_hr = self._scan()
        if opt.ext.find('img')>= 0:
            self.images_hr = list_hr
        for h in list_hr:
            self.images_hr.append(h)
        if self.opt.phase == 'train':
            n_patches = opt.batch_size * opt.test_every
            n_images = len(opt.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
    def _scan(self):
        path = self.dir_hr
        names_hr = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                img_path = dirpath + '/' + fname
                names_hr.append(img_path)
        # 读取HR图片，生成 LR/BlurMap
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            img_HR = util.read_img_(f, self.n_colors)
            # 归一化
            img_HR = util.unit2single(img_HR)
            img_HR = util.modcrop(img_HR, self.scale)

            if self.opt['phase'] == 'train':
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

            if not self.opt.test_only:
                patch_H, patch_L = util.paired_random_crop(img_HR, img_LR, patch_size, scale)
                # augmentation - flip, rotate
                img_HR, img_LR = util.augment([patch_H, patch_L], self.opt.use_flip, self.opt.use_rot)
                if random.random < 0.1:
                    noise_level = np.zeros(1).float()
                else:
                    noise_level = np.random.uniform(0, 15)
            else:
                noise_level = np.zeros(self.sigma_test).float

            img_LR = img_LR + np.random.normal(0, noise_level / 255., img_LR.shape)

            img_HR = util.img_single2tensor(img_HR)
            img_LR = util.img_single2tensor(img_LR)

            M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)
            M = M_vector.repeat(1, img_LR.size()[-2], img_LR.size()[-1])

            img_LR = torch.cat((img_LR, M), 0)
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):

                # 1018测试的图片格式与训练图片格式不符，记得修改！
                if self.name == 'DIV2K':
                    names_lr[si].append(self.dir_lr + '/' + 'X{}'.format(s) + '/' + 'sig2.6_' + filename + '.jpg')
                elif self.name == 'Set5':
                    names_lr[si].append(self.dir_lr + '/' + 'X{}'.format(s) + '/' + 'sig2.6_' + filename + '.png')
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        # 1018在Windows上路径保存如下：div2k\DIV2K\DIV2K_train_LR_bicubic\x4
        # option中设置得文件名大类：dir_data
        # 训练集名称：self.name 为 DIV2K、Set5', 'Set14', 'B100', 'Urban100等，同时调用不同得类函数
        # dir_hr dir_lr表示HR/LR图片对得路径
        self.apath = dir_data+'/'+self.name
        self.dir_hr = self.apath +'/'+ 'DIV2K_train_HR'
        self.dir_lr = self.apath +'/'+'DIV2K_train_LR_bicubic'
        if self.name == 'DIV2K':
            self.ext = ('.jpg', '.jpg')
        elif self.name == 'Set5':
            self.ext = ('.png', '.png')

    def __getitem__(self, index ):
        # 从文件中读出HR图片，返回HR、LR图片对
        HR_path , LR_path = None, None
        HR_path = self.HR_paths[index]
        scale = self.scale[self.idx_scale]
        patch_size = self.patch_size

        img_HR = util.read_img_(HR_path, self.n_colors)
        # 归一化
        img_HR = util.unit2single(img_HR)
        img_HR = util.modcrop(img_HR, self.scale)

        if self.opt['phase'] == 'train':
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

        if not self.opt.test_only:
            patch_H, patch_L = util.paired_random_crop(img_HR, img_LR, patch_size, scale)
            # augmentation - flip, rotate
            img_HR , img_LR = util.augment([patch_H, patch_L], self.opt.use_flip,self.opt.use_rot)
            if random.random<0.1:
                noise_level = np.zeros(1).float()
            else:
                noise_level = np.random.uniform(0, 15)
        else:
            noise_level = np.zeros(self.sigma_test).float

        img_LR = img_LR + np.random.normal(0, noise_level / 255., img_LR.shape)

        img_HR = util.img_single2tensor(img_HR)
        img_LR = util.img_single2tensor(img_LR)

        M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)
        M = M_vector.repeat(1, img_LR.size()[-2], img_LR.size()[-1])

        img_LR = torch.cat((img_LR, M), 0)
        if LR_path is None:
            LR_path = HR_path
        return img_LR, img_HR, LR_path, HR_path
        #
        # return self.transform(img_LR), self.transform(img_HR),LR_path,HR_path
    def __len__(self):
        return len(self.HR_paths)

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
