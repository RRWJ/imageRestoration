import os
import sys
import cv2
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import util

def degradation_model():
    # set parameters
    scale = 4
    mod_scale = 4

    # set data dir
    sourcedir = 'D:/NEU/ImageRestoration/div2k/Set5/HR/'
    savedir = 'D:/NEU/ImageRestoration/div2k/Set5/'
    # set random seed和
    util.set_random_seed(0)

    # load PCA matrix of enough kernelh
    # print('load PCA matrix')
    # pca_matrix = torch.load('/media/sdc/yjchai/IKC/codes/pca_matrix.pth', map_location=lambda storage, loc: storage)
    # print('PCA matrix shape: {}'.format(pca_matrix.shape))
    # 初始化kernelmap
    batch_ker = util.random_batch_kernel(batch=1000, k=15, sig_min=0.2, sig_max=4.0, rate_iso=1.0,
                                         scaling=3, tensor=False)
    print('batch kernel shape: {}'.format(batch_ker.shape))
    b = np.size(batch_ker, 0)
    batch_ker = batch_ker.reshape((b, -1))

    dim_pca = 15
    # torch.Size([225, 15])
    pca_matrix = util.PCA(batch_ker, dim_pca).float()
    print('PCA matrix shape: {}'.format(pca_matrix.shape))
    # saveHRpath = os.path.join(savedir, 'HR')
    # saveLRpath = os.path.join(savedir, 'LR', 'x' + str(scale))
    # saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(scale))
    saveLRblurpath = os.path.join(savedir, 'LRblur_sig0.6', 'X' + str(scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # if not os.path.isdir(os.path.join(savedir, 'HR')):
    #     os.mkdir(os.path.join(savedir, 'HR'))

    # if not os.path.isdir(os.path.join(savedir, 'LR')):
    #     os.mkdir(os.path.join(savedir, 'LR'))
    # if not os.path.isdir(os.path.join(savedir, 'Bic')):
    #     os.mkdir(os.path.join(savedir, 'Bic'))
    if not os.path.isdir(os.path.join(savedir, 'LRblur_sig0.6')):
        os.mkdir(os.path.join(savedir, 'LRblur_sig0.6'))

    # if not os.path.isdir(saveHRpath):
    #     os.mkdir(saveHRpath)
    # else:
    #     print('It will cover ' + str(saveHRpath))

    # if not os.path.isdir(saveLRpath):
    #     os.mkdir(saveLRpath)
    # else:
    #     print('It will cover ' + str(saveLRpath))

    # if not os.path.isdir(saveBicpath):
    #     os.mkdir(saveBicpath)
    # else:
    #     print('It will cover ' + str(saveBicpath))

    if not os.path.isdir(saveLRblurpath):
        os.mkdir(saveLRblurpath)
    else:
        print('It will cover '+ str(saveLRblurpath))

    # 记得更改图片格式，不然无法生成
    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith('.bmp')])
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format((i+1), filename))
        # read image
        image_HR = cv2.imread(os.path.join(sourcedir, filename))

        image = cv2.imread(os.path.join(sourcedir, filename))
        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]

        # LR_blur, by random gaussian kernel
        img_HR = util.img2tensor(image_HR)
        C, H, W = img_HR.size()
        # sig_list = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
        sig = 0.6

        prepro = util.SRMDPreprocessing( scale, pca_matrix, random=False, kernel=15, noise=False,
                                        cuda=True, sig=sig, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=4,
                                        rate_cln=0.2, noise_high=0.0) #random(sig_min, sig_max) | stable kernel(sig)

        LR_img, ker_map = prepro(img_HR.view(1, C, H, W))
        image_LR_blur = util.tensor2img(LR_img)
        cv2.imwrite(os.path.join(saveLRblurpath, 'sig{}_'.format(str(sig)) + filename), image_LR_blur)
        # LR
        # image_LR = imresize_np(image_HR, 1 / scale, True)
        # bic
        # image_Bic = imresize_np(image_LR, scale, True)

        # cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)

        # cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        # cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)

        # kernel_map_tensor[i] = ker_map

    # save dataset corresponding kernel maps
    # 1016windows下路径不可含特殊符号，
    # torch.save(kernel_map_tensor, 'D:/NEU/ImageRestoration/datasets/try/try_sig2.6_kermap.pth')
    print("Image Blurring & Down smaple Done: X"+str(scale))

if __name__ == "__main__":
    degradation_model()
