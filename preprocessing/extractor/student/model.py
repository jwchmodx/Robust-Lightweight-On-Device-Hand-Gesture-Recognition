"""Compact hand-landmark regressor (the lightweight student).

Image crop (3x128x128) -> 42 landmark coords (21 x,y) in [0,1] (crop frame).
Depthwise-separable conv backbone; ~0.2-0.4M params depending on `width`
(vs MediaPipe's ~2.0M landmark model + 1.76M palm detector).
"""
import torch
import torch.nn as nn


class DSConv(nn.Module):
    """Depthwise-separable conv + BN + ReLU, optional stride."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class HandLandmarkNet(nn.Module):
    def __init__(self, width=1.0, num_landmarks=21):
        super().__init__()
        c = lambda n: max(8, int(n * width))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c(16), 3, 2, 1, bias=False),   # 128 -> 64
            nn.BatchNorm2d(c(16)), nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            DSConv(c(16), c(32), stride=2),   # 64 -> 32
            DSConv(c(32), c(32)),
            DSConv(c(32), c(64), stride=2),   # 32 -> 16
            DSConv(c(64), c(64)),
            DSConv(c(64), c(128), stride=2),  # 16 -> 8
            DSConv(c(128), c(128)),
            DSConv(c(128), c(128), stride=2), # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c(128), num_landmarks * 2),
            nn.Sigmoid(),                      # coords in [0,1]
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x)))   # (B, 42)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for w in (0.5, 1.0):
        m = HandLandmarkNet(width=w)
        x = torch.randn(2, 3, 128, 128)
        print(f"width={w}: params={count_params(m):,} out={tuple(m(x).shape)}")
