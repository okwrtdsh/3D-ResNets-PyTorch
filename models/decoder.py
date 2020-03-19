# import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
# from functools import partial
from .binarized_modules import Exposuref
from .pattern_conv_modules import PatternConv

__all__ = [
    'STSRResNetExp'
]

##############################################################################
import torch
from torch import nn


def pixel_shuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.
    See :class:`~torch.nn.PixelShuffle` for details.
    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by
    Examples::
        # 1D example
        >>> input = torch.Tensor(1, 4, 8)
        >>> output = F.pixel_shuffle(input, 2)
        >>> print(output.size())
        torch.Size([1, 2, 16])
        # 2D example
        >>> input = torch.Tensor(1, 9, 8, 8)
        >>> output = F.pixel_shuffle(input, 3)
        >>> print(output.size())
        torch.Size([1, 1, 24, 24])
        # 3D example
        >>> input = torch.Tensor(1, 8, 16, 16, 16)
        >>> output = F.pixel_shuffle(input, 2)
        >>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    """
    input_size = list(input.size())
    dimensionality = len(input_size) - 2
    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]
    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )
    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[0::2]
    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    return shuffle_out.view(input_size[0], input_size[1], *output_size)


class PixelShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.
    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.
    Input Tensor must have at least 3 dimensions, e.g. :math:`(N, C, d_{1})` for 1D data,
    but Tensors with any number of dimensions after :math:`(N, C, ...)` (where N is mini-batch size,
    and C is channels) are supported.
    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details
    Args:
        upscale_factor (int): factor to increase spatial resolution by
    Shape:
        - Input: :math:`(N, C, d_{1}, d_{2}, ..., d_{n})`
        - Output: :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`
        Where :math:`n` is the dimensionality of the data, e.g. :math:`n-1` for 1D audio,
        :math:`n=2` for 2D images, etc.
    Examples::
        # 1D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = torch.Tensor(1, 4, 8)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 2, 16])
        # 2D example
        >>> ps = nn.PixelShuffle(3)
        >>> input = torch.Tensor(1, 9, 8, 8)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 24, 24])
        # 3D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = torch.Tensor(1, 8, 16, 16, 16)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input):
        return pixel_shuffle(input, self.upscale_factor)
    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
##############################################################################



class ResidualBlock(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(n_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(n_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output


class ResidualBlock3D(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(n_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(n_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output
##############################################################################


class STSRResNetExp(nn.Module):
    """
    Spatio-temporal super-resolution ResNet for a pixel-wise coded exposure image
    """
    def __init__(self,
                 sample_size,
                 sample_duration,
                 n_classes=101,
                 upscale=2,
                 n_features_base=256,
                 n_features_up=16,
                 n_features_clf=1024
                 ):
        super().__init__()
        assert sample_size == 112
        assert upscale == 2
        self.activation = F.sigmoid
        self.n_classes = n_classes
        self.upscale = upscale
        self.duration = sample_duration
        self.n_features_base = n_features_base
        self.n_features_up = n_features_up
        self.n_features_clf = n_features_clf

        n_batchs = 1193
        self.exp = Exposuref(t=sample_duration, c=1, s=8, block=sample_size//upscale//8, noise_count=0, pass_count=n_batchs*3)

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=self.n_features_base, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(ResidualBlock, 16, self.n_features_base)

        self.conv_mid = nn.Conv2d(in_channels=self.n_features_base, out_channels=self.n_features_base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(self.n_features_base, affine=True)
        self.upscale2x = nn.Sequential(
            nn.Conv3d(in_channels=self.n_features_base//(self.duration//self.upscale), out_channels=self.n_features_up*self.upscale**3, kernel_size=3, stride=1, padding=1, bias=False),
            PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.residual2 = self.make_layer(ResidualBlock3D, 3, self.n_features_up)
        self.conv_output = nn.Conv3d(in_channels=self.n_features_up, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

        self.conv_clf = nn.Conv2d(in_channels=self.n_features_base, out_channels=self.n_features_clf, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_clf = nn.InstanceNorm2d(self.n_features_clf, affine=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_features_clf, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer, *args):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.exp(x)
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)

        clf_out = torch.flatten(self.avgpool(self.conv_clf(out)), 1)
        clf_out = F.dropout2d(clf_out, training=self.training)
        clf_out = self.fc(clf_out)

        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        shape = out.shape
        out = out.view(shape[0], shape[1]//(self.duration//self.upscale), self.duration//self.upscale, *shape[2:])
        out = self.upscale2x(out)
        out = self.residual2(out)
        out = self.conv_output(out)
        return self.activation(out), F.log_softmax(clf_out, dim=1)


class SVSTSRResNetExp(STSRResNetExp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_input = PatternConv(in_channels=1, out_channels=self.n_features_base, kernel_size=9, stride=1, padding=4, bias=True)


class TSRResNetExp(nn.Module):
    """
    Temporal super-resolution ResNet for a pixel-wise coded exposure image
    """
    def __init__(self,
                 sample_size,
                 sample_duration,
                 n_classes=101,
                 upscale=1,
                 n_features_base=128,
                 n_features_up=16,
                 n_features_clf=1024
                 ):
        super().__init__()
        assert sample_size == 112
        assert upscale == 1
        self.activation = F.sigmoid
        self.n_classes = n_classes
        self.upscale = upscale
        self.duration = sample_duration
        self.n_features_base = n_features_base
        self.n_features_up = n_features_up
        self.n_features_clf = n_features_clf

        self.n_batchs = 1193
        self.exp = Exposuref(t=sample_duration, c=1, s=8, block=sample_size//upscale//8, noise_count=0, pass_count=self.n_batchs*3)

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=self.n_features_base, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(ResidualBlock, 16, self.n_features_base)

        self.conv_mid = nn.Conv2d(in_channels=self.n_features_base, out_channels=self.n_features_base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(self.n_features_base, affine=True)
        self.upscale1x = nn.Sequential(
            nn.Conv3d(in_channels=self.n_features_base//(self.duration//self.upscale), out_channels=self.n_features_up*self.upscale**3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.residual2 = self.make_layer(ResidualBlock3D, 3, self.n_features_up)
        self.conv_output = nn.Conv3d(in_channels=self.n_features_up, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

        self.conv_clf = nn.Conv2d(in_channels=self.n_features_base, out_channels=self.n_features_clf, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_clf = nn.InstanceNorm2d(self.n_features_clf, affine=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_features_clf, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer, *args):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.exp(x)
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)

        clf_out = torch.flatten(self.avgpool(self.conv_clf(out)), 1)
        clf_out = F.dropout2d(clf_out, training=self.training)
        clf_out = self.fc(clf_out)

        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        shape = out.shape
        out = out.view(shape[0], shape[1]//(self.duration//self.upscale), self.duration//self.upscale, *shape[2:])
        out = self.upscale1x(out)
        out = self.residual2(out)
        out = self.conv_output(out)
        return self.activation(out), F.log_softmax(clf_out, dim=1)


class SVTSRResNetExp(TSRResNetExp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_input = PatternConv(in_channels=1, out_channels=self.n_features_base, kernel_size=9, stride=1, padding=4, bias=True)
