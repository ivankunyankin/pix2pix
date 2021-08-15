import torch
import torch.nn as nn


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=True, dropout=False, activation="relu"):
        super(CNNBlock, self).__init__()

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

        if activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.LeakyReLU(0.2)

        self.layers = nn.Sequential(
            conv,
            nn.InstanceNorm2d(out_channels),
            act
        )

        self.dropout = dropout
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.layers(x)
        return self.drop(x) if self.dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()

        self.init_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.down1 = CNNBlock(features, features*2, downsample=True, dropout=False, activation="leaky")
        self.down2 = CNNBlock(features*2, features*4, downsample=True, dropout=False, activation="leaky")
        self.down3 = CNNBlock(features*4, features*8, downsample=True, dropout=False, activation="leaky")
        self.down4 = CNNBlock(features*8, features*8, downsample=True, dropout=False, activation="leaky")
        self.down5 = CNNBlock(features*8, features*8, downsample=True, dropout=False, activation="leaky")
        self.down6 = CNNBlock(features*8, features*8, downsample=True, dropout=False, activation="leaky")
        self.bottleneck = nn.Sequential(nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"), nn.ReLU(True))
        self.up1 = CNNBlock(features*8, features*8, downsample=False, dropout=True, activation="relu")
        self.up2 = CNNBlock(features*8*2, features*8, downsample=False, dropout=True, activation="relu")
        self.up3 = CNNBlock(features*8*2, features*8, downsample=False, dropout=True, activation="relu")
        self.up4 = CNNBlock(features*8*2, features*8, downsample=False, dropout=False, activation="relu")
        self.up5 = CNNBlock(features*8*2, features*4, downsample=False, dropout=False, activation="relu")
        self.up6 = CNNBlock(features*4*2, features*2, downsample=False, dropout=False, activation="relu")
        self.up7 = CNNBlock(features*2*2, features, downsample=False, dropout=False, activation="relu")
        self.last = nn.Sequential(nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.init_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.last(torch.cat([up7, d1], 1))


# model = Generator()
#
# x = torch.rand((1, 6, 256, 256))
# y = model(x)
# print(y.shape)
