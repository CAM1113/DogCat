import torch
from torch import nn
import numpy as np


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, padding_mode='zero'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 64
        self.block2 = nn.Sequential(
            ConBlock(input_channel=64, kernel_size=3, channels=[64, 64, 256], strides=1),  # 256
            IdentityBlock(input_channel=256, channels=[64, 64, 256], kernel_size=3),
            IdentityBlock(input_channel=256, channels=[64, 64, 256], kernel_size=3),
            IdentityBlock(input_channel=256, channels=[64, 64, 256], kernel_size=3),
        )

        self.block3 = nn.Sequential(
            ConBlock(input_channel=256, kernel_size=3, channels=[128, 128, 512]),
            IdentityBlock(input_channel=512, channels=[128, 128, 512], kernel_size=3),
            IdentityBlock(input_channel=512, channels=[128, 128, 512], kernel_size=3),
            IdentityBlock(input_channel=512, channels=[128, 128, 512], kernel_size=3),
        )

        self.block4 = nn.Sequential(
            ConBlock(input_channel=512, kernel_size=3, channels=[256, 256, 1024]),
            IdentityBlock(input_channel=1024, channels=[256, 256, 1024], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, 1024], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, 1024], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, 1024], kernel_size=3),
            IdentityBlock(input_channel=1024, channels=[256, 256, 1024], kernel_size=3),
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        return y


class ConBlock(nn.Module):
    def __init__(self, input_channel, kernel_size, channels, strides=2):
        super(ConBlock, self).__init__()

        self.mainNet = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channels[0], kernel_size=1, stride=strides),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=1),
            nn.BatchNorm2d(channels[2])
        )

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channels[2], kernel_size=1, stride=strides),
            nn.BatchNorm2d(channels[2])
        )

        self.finalReLU = nn.ReLU()

    def forward(self, x):
        y1 = self.mainNet(x)
        y2 = self.short_cut(x)
        return self.finalReLU(torch.add(y1, y2))


class IdentityBlock(nn.Module):
    def __init__(self, input_channel, channels, kernel_size):
        super(IdentityBlock, self).__init__()
        assert input_channel == channels[2], ["input_channel != channels[2],残差无法相加"]

        self.mainNet = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=channels[0], kernel_size=1),
            nn.BatchNorm2d(num_features=channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=1),
            nn.BatchNorm2d(num_features=channels[2]),
        )

        self.finalReLU = nn.ReLU()

    def forward(self, x):
        y = self.mainNet(x)
        z = torch.add(x, y)
        return self.finalReLU(z)


class ClassifyNet(nn.Module):
    def __init__(self):
        super(ClassifyNet, self).__init__()
        self.feature_net = Res50()
        self.fc = nn.Sequential(
            nn.Linear(in_features=65536, out_features=1478, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1478, out_features=100, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=100, out_features=2, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.feature_net(x)
        y = y.view(y.shape[0],-1)
        return self.fc(y)


