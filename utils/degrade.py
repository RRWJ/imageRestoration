import matplotlib.pyplot as plt
import numpy as np
import torch
import codes.data.util as util
import cv2
from codes.utils.util import to_pil_image,to_tensor,resize
from PIL import Image
import scipy
import scipy.stats as ss
import scipy.io as io
from scipy import ndimage
import os

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

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

def read_img(path):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def read_img_(path, n_channels=3):
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
    # plt.savefig('img_LR.png', bbox_inches='tight')
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

path = '/home/dl/Desktop/users/renwenjia/ImageRestoration/datasets/Set5/original'
# path ='/home/dl/Desktop/users/renwenjia/ImageRestoration/datasets/Set5/original/baby.png'
# img_HR_np = read_img_(path, n_channels=3)
img_HR_np = read_img_( os.path.join(path,'baby.png') , n_channels=3 )

img_HR= uint2single(img_HR_np)
img_L = modcrop(img_HR,scale=2)
kernel = anisotropic_Gaussian(ksize=15,theta=np.pi, l1=2, l2=2)
img_L_blur = srmd_degradation(img_L, kernel , sf =2)

noise_level = np.random.uniform(0,15)
img_LR = img_L_blur+ np.random.normal(0, noise_level/255., img_L_blur.shape)

img = single2uint(img_LR)
imshow(single2uint(img_LR))
img_path = os.path.join(path, 'img_LR.png')
imsave(img , img_path)
# plt.savefig('img_LR.png',path= img_path, color = 'RGB')
print('Processing img_LR')

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

"""
# 0924 模型训练有问题导致加载出错
with torch.autograd.set_detect_anomaly(True):
    model = tinyNet(in_nc=3, out_nc=3, nc=[8, 16, 32, 32], nb=2, act_mode='R', downsample_mode='strideconv',upsample_mode='convtranspose')
    x = torch.randn((2, 3, 48, 48),requires_grad=True)
    x = x/255.
    y = torch.randn((2, 3, 96, 96))
    out = model(x)
    G_optim_params = []
    loss = nn.MSELoss(size_average=False)
    for i in range(4):
        for k, v in model.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer_G = torch.optim.Adam(G_optim_params, lr=2e-4, weight_decay=0, betas=(0.9, 0.99))
        error = loss(out, y)
        nn.utils.clip_grad_norm(model.parameters(), 0.4)
        optimizer_G.zero_grad()
        error.backward(retain_graph= True)
        optimizer_G.step()
        print(error)
"""