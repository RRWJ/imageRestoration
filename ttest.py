import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import data.util as util
import cv2
from utils.util import to_pil_image,to_tensor,resize
from PIL import Image
import scipy
import scipy.stats as ss
import scipy.io as io
from scipy import ndimage
import os
import functools
import torch
from torch.nn import init
import os.path as osp
import sys
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
except ImportError:
    pass
import random

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def read_img_(path, n_channels=3):
    '''
    :param path:
    :param n_channels:
    :return numpy.ndarray:
    '''
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.savefig('img_LR.png', bbox_inches='tight')
    plt.show()

def GPU_Bicubic(variable, scale):
    tensor = variable.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_view[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_view = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_view

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def srmd_degradation(x, k, sf=3):
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)
    return x

def bicubic_degradation(x, sf=3):
    x = util.imresize_np(x, scale=1/sf)
    return x

def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)
    return k
def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
    k = k / np.sum(k)
    return k

"""
path = '/home/dl/Desktop/users/renwenjia/ImageRestoration/datasets/Set5/original'
img_HR_np = read_img_(os.path.join(path,'baby.png'), n_channels=3)
img_HR= uint2single(img_HR_np)
img_L = modcrop(img_HR,scale=2)
kernel = anisotropic_Gaussian(ksize=15,theta=np.pi, l1=2, l2=2)
img_L_blur = srmd_degradation(img_L, kernel , sf =2)
noise_level = np.random.uniform(0,15)
img_LR = img_L_blur+ np.random.normal(0, noise_level/255., img_L_blur.shape)
imshow(single2uint(img_LR))
# img_path = os.path.join(path, 'img_LR.png')
# cv2.imwrite(img_path,img_LR,'RGB')
# plt.savefig('img_LR.png',path= img_path, color = 'RGB')
print('Processing img_LR')
"""
"""
path = '/home/dl/Desktop/users/renwenjia/ImageRestoration/datasets/Set5/original/baby.png'
img_HR_np = read_img_(path,n_channels=3)
img_HR= uint2single(img_HR_np)
img_L = modcrop(img_HR,scale=2)
kernel = anisotropic_Gaussian(ksize=15,theta=np.pi, l1=2, l2=2)
img_L_blur = srmd_degradation(img_L, kernel , sf =2)
noise_level = np.random.uniform(0,15)
img_LR = img_L_blur+ np.random.normal(0, noise_level/255., img_L_blur.shape)
# imshow(img_LR)
"""
"""
# 0925采用添加batch后的四维tensor处理图片，无法正常显示
kernel_size = 15
batchsize = 1
pad = nn.ReflectionPad2d(kernel_size // 2)
path = '/home/dl/Desktop/users/renwenjia/ImageRestoration/datasets/Set5/original/baby.png'
img_HR_np = read_img_(path,n_channels=3)
H , W , C = img_HR_np.shape
img_batch_HR = np.zeros((batchsize,H , W , C))
for  i in range(batchsize):
    img_batch_HR[i] = read_img_(path,3)
img_HR = torch.from_numpy(img_batch_HR.transpose((0, 3, 1, 2))).float().div(255)
# batch_kernel = random_batch_kernel(1, kernel_size, 0.2, 2.0, tensor=True)
# kernel_var = batch_kernel.contiguous().view((batchsize, 1, kernel_size, kernel_size)).repeat(1, C, 1, 1).view((batchsize * C, 1, kernel_size, kernel_size))
# img_HR = torch.from_numpy(img_HR_np.transpose((2, 0, 1))).float().div(255)
kernel = random_isotropic_gaussian_kernel(sig_min = 0.2, sig_max=2.0, k = kernel_size, tensor=True)
# print(kernel.size())
blur_act = BatchBlur(k=kernel_size)
img_LR_blur = blur_act(img_HR.cuda(), kernel.cuda())
# 降维后
# img_LR_blur = img_LR_blur.squeeze()
img_LR_blur_ds = GPU_Bicubic(img_LR_blur,scale = 2)
# print(img_LR_blur_ds.size())
# img_HR  = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR , (2, 0, 1)))).float()
# img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR , (2, 0, 1)))).float()
noise_level = torch.FloatTensor([np.random.uniform(0,75)])/255.0
noise_LR = torch.randn(img_LR_blur_ds.size()).mul_(noise_level).float()
from codes.utils.util import tensor2img
img_LR = tensor2img(noise_LR)
imshow(single2uint(tensor2img(img_HR_np)) , title='HR image')
imshow(single2uint(tensor2img(img_LR_blur)) , title='LR image with blur')
imshow(single2uint(tensor2img(img_LR_blur_ds)) , title='LR image with downsample')
imshow(single2uint(tensor2img(noise_LR)), title='LR image with noise level {}'.format(75))
"""


# 0924 模型训练有问题导致加载出错
# 0928 用生成的HR、LR图片对测试loss
from model.tinynet import tinyNet
import torch.nn as nn

def adjust_learning_rate(optimizer, epoch,lr,step):
    lr = lr * (0.1 ** (epoch // step))
    return lr

def img2tensor(img):
    '''
    # BGR to RGB, HWC to CHW, numpy to tensor
    Input: img(H, W, C), [0,255], np.uint8 (default)
    Output: 3D(C,H,W), RGB order, float tensor
    '''
    img = img.astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img

with torch.autograd.set_detect_anomaly(True):
    path = 'D:/NEU/ImageRestoration/datasets/Set5/original'
    # 读取图片，RGB三通道 （512, 512, 3）
    img_HR_np = read_img_(os.path.join(path, 'baby.png'), n_channels=3)
    img_LR_np = read_img_(os.path.join(path, 'img_LR.png'), n_channels=3)

    patch_size = 48
    scale = 2
    HR_size = patch_size * scale
    H, W, C = img_LR_np.shape
    # randomly crop
    rnd_h = random.randint(0, max(0, H - patch_size))
    rnd_w = random.randint(0, max(0, W - patch_size))
    patch_L = img_LR_np[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
    rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
    patch_H = img_HR_np[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

    cv2.imshow('img1', patch_L)
    cv2.imshow('img2', patch_H)
    cv2.waitKey(0)

    # 1004升维，可自己手写函数
    img_HR_4tensor = img_HR_tensor.unsqueeze(0)
    img_LR_4tensor = img_LR_tensor.unsqueeze(0)
    img_HR_t = img_HR_4tensor.numpy()
    img_HR_te = img_HR_tensor.numpy()
    cv2.imshow('img1', img_HR_t[0,1, :, :])
    cv2.imshow('img2', img_HR_te[1, :, :])
    cv2.waitKey(0)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 将数据集放到GPU上
    img_HR = img_HR_4tensor.to(device)
    img_LR = img_LR_4tensor.to(device)

    criterion = nn.MSELoss(size_average=False)
    criterion = criterion.to(device)
    # 0930损失值过大导致数值溢出，盲猜模型是不是有错误
    # 选择network
    model = tinyNet(in_nc=3, out_nc=3, nc=[8, 16, 32, 32], nb=2, act_mode='R', downsample_mode='strideconv',upsample_mode='convtranspose')
    # 初始化权重
    # init_weights(model,init_type=opt_net['init_type'],init_bn_type=opt_net['init_bn_type'],
    #              gain=opt_net['init_gain'])

    model = model.to(device)
    model.eval()
    model.train()

    optimizer_G = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, betas=(0.9, 0.99),eps=1e-8)
    for i in range(100):
        out = model(img_LR)
        loss = criterion(out, img_HR)
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()