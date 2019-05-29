import numpy as np
import torch
from random import randint


class Coded(object):

    def __init__(self, mask_path):
        self.mask = np.load(mask_path)

    def __call__(self, clip):
        length = len(clip)
        image = torch.zeros(1, 112, 112).float()
        for t, m in zip(clip, self.mask):
            m = torch.from_numpy(np.transpose(m, (2, 0, 1))).float()
            image += t * m
        image /= length
        return image


class ToTemporal(object):

    def __init__(self, mask_path, size=4, duration=16):
        self.mask = np.load(mask_path)
        self.size = size
        self.duration = duration

    def __call__(self, tensor):
        img = tensor.numpy()
        size = self.size
        duration = self.duration

        out = []
        for i in range(duration):
            out.append(
                img.reshape(112, 112)[(self.mask[i] == 1).reshape(112, 112)]
                .reshape(112//size, 112//size)
                .repeat(size, axis=0).repeat(size, axis=1)
                .reshape(1, 112, 112)
            )

        out = torch.from_numpy(
            np.array(out).astype(np.float32)
        ).float().permute(1, 0, 2, 3)
        return out

    def randomize_parameters(self):
        pass


class Averaged(object):

    def __call__(self, clip):
        length = len(clip)
        image = torch.zeros(1, 112, 112).float()
        for t in clip:
            image += t
        image /= length
        return image


class OneFrame(object):

    def __call__(self, clip, fixed=False):
        image = torch.zeros(1, 112, 112).float()
        if fixed:
            image = clip[0]
        else:
            image = clip[randint(0, len(clip)-1)]
        return image
