import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    # Double the length, concatenate the skip and then refinate
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


class Ludwig(nn.Module):
    # U-Net 1D that predicts the high band residual loss after the 4 kHz low-pass
    def __init__(self, channels=32, depth=5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.GELU(),
        )

        downs = []
        ups = []

        for level in range(depth):
            next_channels = channels * (2 ** (level + 1))
            downs.append(DownBlock(channels if level == 0 else channels * (2 ** level), next_channels))

        for level in reversed(range(depth)):
            out_channels = channels if level == 0 else channels * (2 ** level)
            in_channels = channels * (2 ** (level + 1))
            ups.append(UpBlock(in_channels, out_channels))

        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

        # Emit the residual so it can be sum onto the input
        self.head = nn.Conv1d(channels, 1, kernel_size=7, padding=3)

    def forward(self, x):
        skips = []

        h = self.stem(x)

        for down in self.downs:
            skips.append(h)
            h = down(h)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)

        return self.head(h)
