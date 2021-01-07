import os
import glob
import random
import pickle
import hdf5storage
from data import common
import numpy as np
import imageio
import torch
import data.util as util
import torch.utils.data as data
from utils.utils_deblur import fspecial
import utils.utils_image as util
from utils import utils_deblur
from utils import utils_sisr

class SRData(data.Dataset):
    def __init__(self, args ,name='DIV2K', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.sigma_max = args.sigma_max
        self.n_colors = args.n_color
        self.patch_size = args.patch_size
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))

        self._set_filesystem(args.dir_data)

        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'train')
            os.makedirs(path_bin, exist_ok=True)

        list_hr , list_lr  = self._scan()

        if args.ext.find('dataset') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )

            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                self.images_hr.append(h)
            for i, ll in enumerate(list_lr):
                for l in ll:
                    self.images_lr[i].append(l)
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        """
        # 1013在window环境下路径地址不正确 "./datasets\\Xsacle/0001xscale.png"
        # names_hr = util.get_image_paths_(os.path.join(self.dir_data, self.name))
        # 暂时修改
        names_hr = util.get_image_paths_(self.dir_data+'/'+self.name)
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))
        """
        """
        path = self.dir_hr+'/'+self.name
        names_hr = []
        names_lr = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                # 1012在windows下修改的代码，在Linux记得改回来！
                # img_path = os.path.join(dirpath, fname)
                img_path = dirpath + '/' + fname
                img_hr = util.read_img_(img_path,self.n_colors)
                img_hr = util.unit2single(img_hr)
                img_hr = util.modcrop(img_hr, self.scale)
                kernel = fspecial('gaussian', 15, 2.6)

                H, W, _ = img_hr.shape
                img_lr = util.srmd_degradation(img_hr, kernel, self.scale)

                if not self.opt.test_only:
                    patch_H, patch_L = util.paired_random_crop(img_HR, img_LR, self.patch_size, self.scale)
                    # augmentation - flip, rotate
                    img_HR, img_LR = util.augment([patch_H, patch_L], self.opt.use_flip, self.opt.use_rot)
                    img_HR = util.img_single2tensor(img_HR)
                    img_LR = util.img_single2tensor(img_LR)

                    if random.random() < 0.1:
                        noise_level = torch.zeros(1).float()
                    else:
                        noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)]) / 255.0
                else:
                    img_HR = util.img_single2tensor(img_HR)
                    img_LR = util.img_single2tensor(img_LR)
                    noise_level = torch.FloatTensor([self.sigma_test])

                noise = torch.randn(img_LR.size()).mul_(noise_level).float()
                img_LR.add_(noise)
                kernel = util.single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
                noise_level = torch.FloatTensor([noise_level]).view([1, 1, 1])

                names_hr.append(img_path)
        return names_hr, names_lr
        """
        path = self.dir_hr
        names_hr = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                img_path = dirpath + '/' + fname
                names_hr.append(img_path)
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                # 1018测试的图片格式与训练图片格式不符，记得修改！
                if self.name == 'DIV2K':
                    names_lr[si].append(self.dir_lr + '/' + 'X{}'.format(s) + '/' + 'sig2.6_' + filename + '.jpg')
                elif self.name == 'Set5':
                    names_lr[si].append(self.dir_lr + '/' + 'X{}'.format(s) + '/' + 'sig2.6_' + filename + '.bmp')
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/' + self.name
        self.dir_hr = self.apath + '/' + 'DIV2K_train_HR'
        self.dir_lr = self.apath + '/' + 'DIV2K_train_LR_bicubic'
        if self.name == 'DIV2K':
            self.ext = ('.jpg', '.jpg')
        elif self.name == 'Set5':
            self.ext = ('.png', '.png')

    def __getitem__(self, idx):
        # 将二进制文件转为imageio
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        # 生成kernel
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        k = utils_sisr.gen_kernel(scale_factor=np.array([self.scale, self.scale]))  # Gaussian blur
        r_value = np.random.randint(0, 8)
        if r_value > 3:
            k = utils_deblur.blurkernel_synthesis(h=25)  # motion blur
        else:
            sf_k = random.choice(self.scale)
            k = utils_sisr.gen_kernel(scale_factor=np.array([sf_k, sf_k]))  # Gaussian blur
            mode_k = np.random.randint(0, 8)
            k = util.augment_img(k, mode=mode_k)
            if np.random.randint(0, 8) == 1:
                noise_level = 0/255.0
            else:
                noise_level = np.random.randint(0, self.sigma_max)/255.0

            # ---------------------------
            # Low-quality image
            # ---------------------------
            img_L = ndimage.filters.convolve(patch_H, np.expand_dims(k, axis=2), mode='wrap')
            img_L = img_L[0::self.sf, 0::self.sf, ...]
            # add Gaussian noise
            img_L = util.uint2single(img_L) + np.random.normal(0, noise_level, img_L.shape)
            img_H = patch_H

        return pair_t[0], pair_t[1], filename


    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = random.randint(0, len(self.scale) - 1)