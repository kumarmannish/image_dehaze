import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, activation="relu", use_dropout=False):
        super(Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2)
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, data):
        data = self.conv(data)
        return self.dropout(data) if self.use_dropout else data
