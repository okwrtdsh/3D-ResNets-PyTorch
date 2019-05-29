import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class PatternConv(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            bias=True, pattern_size=(8, 8)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.pattern_size = pattern_size
        hs, ws = self.pattern_size
        self.convs = [[
            nn.Conv2d(
                self.in_channels, self.out_channels,
                kernel_size=self.kernel_size,
                stride=(hs, ws), bias=self.bias
            ).to("cuda")
            for _ in range(ws)
        ] for _ in range(hs)]
        self.reset_parameters()

    def forward(self, x):
        hs, ws = self.pattern_size
        paded = F.pad(x, (0, hs-1, 0, ws-1))
        B, C, H, W = x.shape
        out = torch.zeros(self.in_channels*B, self.out_channels*C, H, W).float().to('cuda')
        for h in range(hs):
            for w in range(ws):
                out[:, :, h::hs, w::ws] = self.convs[h][w](paded[:, :, h:H+h, w:W+w])
        return out

    def reset_parameters(self):
        hs, ws = self.pattern_size
        for h in range(hs):
            for w in range(ws):
                init.kaiming_uniform_(self.convs[h][w].weight, a=math.sqrt(2))
                if self.convs[h][w].bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.convs[h][w].weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(self.convs[h][w].bias, -bound, bound)

    def extra_repr(self):
        return 'pattern_size=%s, %s' % (repr(tuple(self.pattern_size)), repr(self.convs[0][0]))
