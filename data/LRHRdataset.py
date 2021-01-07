import random
import numpy as np
import torch
import data.util as util
import torch.utils.data as data
import cv2

class LRHRdataset(data.Dataset):
    def __init__(self , opt):
        super(LRHRdataset, self).__init__()
        print('Get L/H for image-to-image mapping.')
        self.opt = opt
        self.patch_size = opt.patch_size
        self.n_colors = opt.n_colors
        self.scale = opt.scale
        self._set_filesystem(opt.dir_data)
        self.images_hr, self.images_lr = [], [[] for _ in self.scale]
        list_hr, list_lr = self._scan()
        if opt.ext.find('img') >= 0:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif opt.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(self.dir_lr.replace(self.apath, path_bin) + '/' + 'X{}'.format(s), exist_ok=True)
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
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
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                # 1018测试的图片格式与训练图片格式不符，记得修改！
                if self.name == 'DIV2K':
                    names_lr[si].append(self.dir_lr + '/' + 'X{}'.format(s) + '/' + 'sig2.6_' + filename + '.jpg')
                elif self.name == 'Set5':
                    names_lr[si].append(self.dir_lr + '/' + 'X{}'.format(s) + '/' + 'sig2.6_' + filename + '.png')
        return names_hr, names_lr

    def __getitem__(self, index):
        # 从文件中读出HR图片，返回HR、LR图片对
        HR_path = self.HR_paths[index]
        LR_path = self.LR_paths[index]
        img_HR = util.read_img_(HR_path, self.n_colors)
        img_LR = util.read_img_(LR_path,self.n_colors)
        if not self.opt.test_only == 'train':
            patch_H, patch_L = util.paired_random_crop(img_HR, img_LR, self.patch_size, self.scale)
            img_HR , img_LR = util.augment([patch_H, patch_L], self.opt.use_flip,self.opt.use_rot)
        img_HR  = util.img2tensor(img_HR)
        img_LR = util.img2tensor(img_LR)
        return {'LR': img_LR, 'HR': img_HR}

    def __len__(self):
        return len(self.HR_paths)
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
    def set_scale(self, idx_scale):
        self.idx_scale = random.randint(0, len(self.scale) - 1)
