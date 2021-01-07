import torch
import cv2
import numpy as np
import random
from PIL import Image
import math
from torchvision.utils import make_grid
from model.tiny_demo2 import renwenjiaMode2
from utils.utils_deblur import fspecial
from data.util import single2tensor3 , read_img_,img_single2tensor
from utils.util import tensor2uint , to_pil_image
import os
import matplotlib.pyplot as plt
import torchvision
import data.util as util

"""
sourcedir = 'D:/NEU/ImageRestoration/experiment/sig1.6epoch300/renwenjia'
log_file = torch.load(os.path.join(sourcedir+ '/'+'loss_log.pt'))
log_psnr = torch.load(os.path.join(sourcedir+ '/' +'train_psnr_log.pt'))

def plot_psnr(log_psnr,log_file,epoch):
    fig = plt.figure()
    # plt.title("Train")
    axis = np.linspace(1, epoch,epoch)
    num1 = np.zeros((300, 1))
    # num2 = np.zeros((300, 1))
    for i in range(log_file.shape[0]):
        num1[i, :] = log_file[i].numpy()
    # for i in range(log_psnr.shape[0]):
    #     num2[i, :] = log_psnr[i].numpy()
    # num2[300,:] =
    min_loss = np.argmin(num1)
    show_loss = f'Min: {num1[min_loss]}'
    plt.plot(axis, num1, color='blue')
    plt.plot(axis, num1, '.', color='blue')
    plt.plot(min_loss, num1[min_loss], 'o', color='red')
    plt.annotate(show_loss, xy=(min_loss, num1[min_loss]))
    # plt.plot(axis, num2, color='red')
    # plt.plot(axis, num2,'+', color='red')
    # plt.legend(["Loss","PSNR"])
    plt.legend(["Loss"])
    plt.xlim(1, epoch)  # x轴范围
    # plt.xticks(range(1, epoch + 1,))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
plot_psnr(log_psnr,log_file,epoch=300)
# print('Continue from epoch {}...'.format(len(log)))

"""
"""
from imageio import imread,imsave
sourcedir = 'D:/NEU/ImageRestoration/div2k/Set14/HR/'
savedir = 'D:/NEU/ImageRestoration/div2k/Set14/HR/'
img_H1= cv2.imread(os.path.join(sourcedir, 'zebra.bmp'))
img_H1 = cv2.resize(img_H1,(512,360))
cv2.imwrite(os.path.join(savedir+'zebra.bmp'),img_H1)
"""
"""
kernel = fspecial('gaussian', 15, 2.6)
H, W, _ = image_HR.shape
scale = 4
img_LR = util.srmd_degradation(image_HR, kernel, scale)
cv2.imwrite(os.path.join(savedir+'sig2.6_img092.jpg'), img_LR)

img_H1 = np.float32(img_H1/255.)
HR_img_tensor = img_single2tensor(img_H1)
HR_img_4tensor = HR_img_tensor.unsqueeze(0)
print(HR_img_4tensor.shape)

saveLRblurpath = 'D:/NEU/ImageRestoration/div2k/Set14'
img_L1= read_img_(os.path.join(saveLRblurpath, 'sig2.6orginal_butterfly.jpg'),n_colors=3)
img_L1 = np.float32(img_L1/255.)
LR_img_tensor = img_single2tensor(img_L1)
LR_img_4tensor = LR_img_tensor.unsqueeze(0)

def feature_imshow(inp, title=None):
    inp = inp.detach().numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

k = fspecial('gaussian', 15, 2.6)
k = single2tensor3(np.expand_dims(np.float32(k), axis=2))
k = k.unsqueeze(0)
sigma_max = 25
sf = 4
noise_level = np.random.randint(0, sigma_max)/255.0
sigma = torch.tensor(noise_level).float().view([1, 1, 1, 1])
from model.tinydemo import DataNet , p2o , cconj,cabs2,r2c,cmul,upsample
import torch.nn as nn
model = renwenjiaMode2()
# z= model(LR_img_4tensor,k = k,sf= 4,sigma=sigma,n_iter = 6)
# z = tensor2uint(z.squeeze().detach().numpy())
# plt.imshow(z)
# print(z.shape)
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


feature_output1 = z.transpose(1,0).cpu()
out1 = torchvision.utils.make_grid(feature_output1,nrow=16)
feature_imshow(out1)


feature_output2 = out_fea.transpose(1,0).cpu()
out2 = torchvision.utils.make_grid(feature_output2,nrow=16)
feature_imshow(out2)
plt.show()
"""