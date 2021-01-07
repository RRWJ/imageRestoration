import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        self.m_head = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=2, stride=2, padding=0, bias=False)
        )
        self.m_down1 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0, bias=False)
            )
        self.m_down2 = nn.Sequential(
            ResBlock(in_channels=128, out_channels=128,kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False)
            )
        self.m_down3 = nn.Sequential(
            ResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0, bias=False)
        )
        self.m_body = nn.Sequential(
            ResBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True),
            ResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            ResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=False),
            ResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False),
            ResBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.m_tail = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),

    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        # kernel_correct = x
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[..., :h, :w]
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias)
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
