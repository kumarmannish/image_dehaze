import torch
import torch.nn as nn
from networks.cnn_block import CNNBlock

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super(Discriminator, self).__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, data_x, data_y):
        data_x = torch.cat([data_x, data_y], dim=1)
        data_x = self.initial(data_x)
        return self.model(data_x)
