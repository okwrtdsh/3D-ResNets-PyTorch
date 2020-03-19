import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'SPC'
]


class SinglePixelCamera(nn.Module):
    def __init__(self, s=112, t=16, requires_grad=True):
        super().__init__()
        self.s = s
        self.t = t
        self.weight = nn.Parameter(torch.Tensor(self.s*self.s//self.t, self.t, self.s, self.s), requires_grad=requires_grad)
        self.weight.data.normal_(0.5, math.sqrt(2. / self.s**4))

    def forward(self, x):
        # x: (B, C, T, H, W) -> (B, H*W)
        # x = torch.repeat_interleave(x, self.s*self.s//self.t, dim=1)
        x = x.repeat(1, self.s*self.s//self.t, 1, 1, 1)
        return torch.clamp(torch.sum(x * torch.clamp(self.weight, min=0, max=1), dim=(3,4)).view(-1, self.s*self.s),  min=0, max=1)


class SPC(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=400):
        super().__init__()
        self.camera = SinglePixelCamera(s=112, t=16)
        self.fc6 = nn.Linear(112*112, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.camera(x)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return F.log_softmax(self.fc8(x), dim=1)
