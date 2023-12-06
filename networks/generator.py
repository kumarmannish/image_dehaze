import torch
import torch.nn as nn
from networks.block import Block

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        # (384 - 4 + 2*1)/2 + 1 = 192

        self.down1 = Block(features, features*2, down=True, activation="leaky", use_dropout=False)

        # (192 - 4 + 2*1)/2 + 1 = 96

        self.down2 = Block(features*2, features*4, down=True, activation="leaky", use_dropout=False)

        # (96 - 4 + 2*1)/2 + 1 = 48

        self.down3 = Block(features*4, features*8, down=True, activation="leaky", use_dropout=False)

        # (48 - 4 + 2*1)/2 + 1 = 24

        self.down4 = Block(features*8, features*8, down=True, activation="leaky", use_dropout=False)

        # (24 - 4 + 2*1)/2 + 1 = 12

        self.down5 = Block(features*8, features*8, down=True, activation="leaky", use_dropout=False)

        # (12 - 4 + 2*1)/2 + 1 = 6

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # (6 - 4 + 2*1)/2 + 1 = 3

        self.up1 = Block(features*8, features*8, down=False, activation="relu", use_dropout=True)

        # (3 - 1)*2 - 2*1 + (4-1) + 1 = 6

        self.up2 = Block(features*8*2, features*8, down=False, activation="relu", use_dropout=True)

        # (6 - 1)*2 - 2*1 + (4-1) + 1 = 12

        self.up3 = Block(features*8*2, features*8, down=False, activation="relu", use_dropout=True)

        # (12 - 1)*2 - 2*1 + (4-1) + 1 = 24

        self.up4 = Block(features*8*2, features*4, down=False, activation="relu", use_dropout=False)

        # (24 - 1)*2 - 2*1 + (4-1) + 1 = 48

        self.up5 = Block(features*8, features*2, down=False, activation="relu", use_dropout=False)

        # (48 - 1)*2 - 2*1 + (4-1) + 1 = 96

        self.up6 = Block(features*4, features, down=False, activation="relu", use_dropout=False)

        # (96 - 1)*2 - 2*1 + (4-1) + 1 = 192

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # (192 - 1)*2 - 2*1 + (4-1) + 1 = 384

    def forward(self, data):
        down_1 = self.initial_down(data)
        down_2 = self.down1(down_1)
        down_3 = self.down2(down_2)
        down_4 = self.down3(down_3)
        down_5 = self.down4(down_4)
        down_6 = self.down5(down_5)
        bottleneck = self.bottleneck(down_6)
        up_1 = self.up1(bottleneck)
        up_2 = self.up2(torch.cat([up_1, down_6], 1))
        up_3 = self.up3(torch.cat([up_2, down_5], 1))
        up_4 = self.up4(torch.cat([up_3, down_4], 1))
        up_5 = self.up5(torch.cat([up_4, down_3], 1))
        up_6 = self.up6(torch.cat([up_5, down_2], 1))
        return self.final_up(torch.cat([up_6, down_1], 1))
