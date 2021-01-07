import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math
import cv2
import os
import sys
try:
    import accimage
except ImportError:
    accimage = None

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def isotropic_gaussian_kernel(l, sigma, tensor):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)

def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=2.0, l=15, tensor = False):
    x = np.random.rand() * (sig_max - sig_min) + sig_min
    y = np.random.rand() * (sig_max - sig_min) + sig_min
    if x != y:
        k1 = isotropic_gaussian_kernel(l, x, tensor=tensor)
        k2 = isotropic_gaussian_kernel(l, y, tensor=tensor)
    return k1,k2

def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=3.0, l=15, tensor=False):
    x = np.random.rand() * (sig_max - sig_min) + sig_min
    y = np.random.rand() * (sig_max - sig_min) + sig_min
    if x != y :
        k3 = isotropic_gaussian_kernel(l, x, tensor=tensor)
        k4 = isotropic_gaussian_kernel(l, y, tensor=tensor)
    return k3,k4

def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4, l=15, tensor=False):
    x = np.random.rand() * (sig_max - sig_min) + sig_min
    y = np.random.rand() * (sig_max - sig_min) + sig_min
    if x != y :
        k5 = isotropic_gaussian_kernel(l, x, tensor=tensor)
        k6 = isotropic_gaussian_kernel(l, y, tensor=tensor)
    return k5,k6

def stable_isotropic_gaussian_kernel(sig=2.6, l=15, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def imshow(x, title='inaga', cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def save_img(img, img_path):
    cv2.imwrite(img_path, img)

# savedir = '/home/dl/Desktop/users/renwenjia/0821/datasets/kernel/'
k1 , k2 = random_isotropic_gaussian_kernel(0.2, 2.0, 15, False)
k3,k4 = random_isotropic_gaussian_kernel(0.2, 3.0, 15, False)
k5,k6 = random_isotropic_gaussian_kernel(0.2, 4.0, 15, False)
# img1 = tensor2img(k1, out_type=np.uint8, min_max=(0, 1))
# img2 = tensor2img(k2, out_type=np.uint8, min_max=(0, 1))
# save_img(k1, savedir+'1')
# save_img(k2, savedir+'2')
imshow(k1)
imshow(k2)
imshow(k3)
imshow(k4)
imshow(k5)
imshow(k6)


"""
def PCA(data, dim=2):
    # dim = 0 取均值。样本中心化
    X_mean = torch.mean(data, 0)
    X = data - X_mean.expand_as(data)
    # 奇异值分解函数：torch.svd , X = U * S * V
    U, S, V = torch.svd(torch.t(X))
    return U[:, :dim] # PCA matrix

print(PCA(k1,dim=2))
"""