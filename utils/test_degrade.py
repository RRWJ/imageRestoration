# 测试resUNet网络处理模糊图像的能力
import torch
import cv2
from model.tinydemo import renwenjiaMode
import numpy as np
import random
from PIL import Image
import math
from torchvision.utils import make_grid
import utils.util as util
import matplotlib.pyplot as plt
from scipy import ndimage
import os

# 分别生成三种不同大小的高斯模糊核
k1 = util.stable_isotropic_gaussian_kernel(sig=0.6, k=15, tensor=False)
k2 = util.stable_isotropic_gaussian_kernel(sig=1.6, k=15, tensor=False)
k3 = util.stable_isotropic_gaussian_kernel(sig=2.6, k=15, tensor=False)
# 采用PCA降维，并添加同样大小的噪声，放入模型中看输出结果
# k1_resize = np.reshape(k1, (-1), order="F")
k1_pca = util.PCA(k1,dim=48)
k2_pca = util.PCA(k2,dim=48)
k3_pca = util.PCA(k2,dim=48)

from model.renwenjia import tinyNet
model = tinyNet()
output = model(k1_pca)
plt.subplot(1, 3, 3)

plt.title(label='resunet HR ')
plt.imshow(output)
plt.show()

plt.imshow(output, cmap='gray')