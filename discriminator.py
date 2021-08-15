import torch
import torch.nn as nn


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels*2, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.layers = nn.Sequential(
            CNNBlock(64, 128, stride=2),
            CNNBlock(128, 256, stride=2),
            CNNBlock(256, 512, stride=1)
        )

        self.C_last = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.C1(x)
        x = self.layers(x)
        x = self.C_last(x)
        return x


# x = torch.rand((1, 3, 256, 256))
# y = torch.rand((1, 3, 256, 256))
#
# model = Discriminator()
#
# print(model(x, y).shape)
