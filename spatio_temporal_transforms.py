import numpy as np
import torch


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


class Averaged(object):

    def __call__(self, clip):
        length = len(clip)
        image = torch.zeros(1, 112, 112).float()
        for t in clip:
            image += t
        image /= length
        return image


class OneFrame(object):

    def __call__(self, clip):
        image = torch.zeros(1, 112, 112).float()
        image = clip[0]
        return image
