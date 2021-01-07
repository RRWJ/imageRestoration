import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
def otf2psf(otf, outsize=None):
    insize = np.array(otf.shape)
    psf = np.fft.ifftn(otf, axes=(0, 1))
    for axis, axis_size in enumerate(insize):
        psf = np.roll(psf, np.floor(axis_size / 2).astype(int), axis=axis)
    if type(outsize) != type(None):
        insize = np.array(otf.shape)
        outsize = np.array(outsize)
        n = max(np.size(outsize), np.size(insize))
        # outsize = postpad(outsize(:), n, 1);
        # insize = postpad(insize(:) , n, 1);
        colvec_out = outsize.flatten().reshape((np.size(outsize), 1))
        colvec_in = insize.flatten().reshape((np.size(insize), 1))
        outsize = np.pad(colvec_out, ((0, max(0, n - np.size(colvec_out))), (0, 0)), mode="constant")
        insize = np.pad(colvec_in, ((0, max(0, n - np.size(colvec_in))), (0, 0)), mode="constant")

        pad = (insize - outsize) / 2
        if np.any(pad < 0):
            print("otf2psf error: OUTSIZE must be smaller than or equal than OTF size")
        prepad = np.floor(pad)
        postpad = np.ceil(pad)
        dims_start = prepad.astype(int)
        dims_end = (insize - postpad).astype(int)
        for i in range(len(dims_start.shape)):
            psf = np.take(psf, range(dims_start[i][0], dims_end[i][0]), axis=i)
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    return psf
def p2o(psf, shape):
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf
def cdiv(x, y):
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)
def splits(a, sf):
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b
def r2c(x):
    return torch.stack([x, torch.zeros_like(x)], -1)
def csum(x, y):
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)
def cabs(x):
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)
def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2
def cmul(t1, t2):
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)
def get_rho_sigma(sigma=2.55/255, iter_num=15):
    modelSigma1 = 49.0
    modelSigma2 = 2.55
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num)
    sigmas = modelSigmaS/255.
    mus = list(map(lambda x: (sigma**2)/(x**2)/3, sigmas))
    rhos = mus
    return rhos, sigmas

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
def p2o(psf, shape):
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf
def cconj(t, inplace=False):
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c
def rfft(t):
    return torch.rfft(t, 2, onesided=False)
def irfft(t):
    return torch.irfft(t, 2, onesided=False)
def fft(t):
    return torch.fft(t, 2)
def ifft(t):
    return torch.ifft(t, 2)
def upsample(x, sf=3):
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z

class SLModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(SLModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.deblur = DataNet()

        # Defining learnable parameters
        self.alpha_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha_5 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha_6 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.alpha_1.data = torch.tensor(0.1)
        self.alpha_2.data = torch.tensor(0.1)
        self.alpha_3.data = torch.tensor(0.1)
        self.alpha_4.data = torch.tensor(0.1)
        self.alpha_5.data = torch.tensor(0.1)
        self.alpha_6.data = torch.tensor(0.1)

    def forward(self,x, k, sf, sigma):

        for i in range(6):
            out_c1 = self.act(self.c1(input))
            distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
            out_c2 = self.act(self.c2(remaining_c1))
            distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
            out_c3 = self.act(self.c3(remaining_c2))
            distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
            out_c4 = self.c4(remaining_c3)
            out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
            out = self.deblur
            out_fused = self.c5(self.cca(out)) + input
        return out_fused
    def reconnect(self,v, x, y, i):
        i = i + 1
        if i == 1:
            alpha= self.delta_1
        if i == 2:
            delta = self.delta_2
            eta = self.eta_2
        if i == 3:
            delta = self.delta_3
            eta = self.eta_3
        if i == 4:
            delta = self.delta_4
            eta = self.eta_4
        if i == 5:
            delta = self.delta_5
            eta = self.eta_5
        if i == 6:
            delta = self.delta_6
            eta = self.eta_6
        # recon = torch.mul((1 - delta - eta * delta), x) + torch.mul(eta * delta, v) + torch.mul(delta, y)
        recon = torch.mul((1 - delta - eta), v) + torch.mul(eta, x) + torch.mul(delta, y)
        return recon
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, input, FB, FBC, F2B, FBFy, alpha, sf):
        alpha = alpha[:, 1:2, ...]
        FR = FBFy + torch.rfft(alpha*input, 2, onesided=False)
        x1 = cmul(FB, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, alpha))
        FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
        Xest = torch.irfft(FX, 2, onesided=False)
        return Xest

class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=2, channel=96):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


class SLModule_(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(SLModule_, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        # self.c5 = conv_layer(in_channels, in_channels, 1)
        self.c5 = conv_layer(self.distilled_channels, in_channels, 1)
    def forward(self,input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        remaining_c4 = self.c4(remaining_c3)
        distilled_out = torch.cat([distilled_c1, distilled_c2, distilled_c3], dim=1)
        remaining_out = self.act(self.c5(remaining_c4))+input
        return distilled_out, remaining_out

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=96):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)
    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

def make_model():
    return renwenjiaMode()

class renwenjiaMode(nn.Module):
    def __init__(self, upscale=4, in_nc=3 , nf=128 , out_nc=3):
        super(renwenjiaMode, self).__init__()
        self.upscale = upscale
        self.fea_conv = nn.Sequential(conv_layer(in_nc, nf, 3),
                                      nn.ReLU(inplace=True),
                                      conv_layer(nf, nf, 3, stride=2, bias=False))
        self.IMDB = SLModule_(in_channels=nf)
        self.deblur1 = DataNet()
        self.deblur2 = ResidualBlock_noBN(nf = 96)
        self.hypanet = HyPaNet()
        self.act = activation('lrelu', neg_slope=0.05)
        self.lr_conv = conv_layer(nf,out_nc,kernel_size=3)
        self.fit_channel = nn.Sequential(conv_layer(96,nf,3),
                                         nn.ReLU(inplace=True),
                                         conv_layer(nf,nf,3,2,bias=False),
                                         nn.ReLU(inplace=True),
                                         conv_layer(nf,nf,3,2,bias=False))
        self.upsampler1 = pixelshuffle_block(nf, out_nc,upscale)
        self.upsampler2 = pixelshuffle_block(out_nc, out_nc, upscale//2)
    def forward(self,input, k, sf, sigma):
        out_fea = self.fea_conv(input)
        out1,out2 = self.IMDB(out_fea)
        w, h = out1.shape[-2:]
        FB = p2o(k, (w * sf, h * sf))
        FBC = cconj(FB, inplace=False)
        F2B = r2c(cabs2(FB))
        STy = upsample(out1, sf=sf)
        FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
        x = nn.functional.interpolate(out1, scale_factor=sf, mode='nearest')
        alpha = self.hypanet(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        predata = self.deblur1(x,FB, FBC, F2B, FBFy,alpha,sf)
        deblur = self.deblur2(predata)
        lr = self.fit_channel(deblur) + out2
        output1 = self.upsampler1(lr)
        output2 = self.upsampler2(output1)
        return lr
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
