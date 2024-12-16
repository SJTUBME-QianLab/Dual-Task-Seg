import collections
import math
import random
from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Conv2d, Conv3d

from config.config import msd_resample_data_path, LOW_RANGE, HIGH_RANGE


class RandConvModule(nn.Module):
    def __init__(
            self,
            kernel_size=3,
            in_channels=1,
            out_channels=1,
            rand_bias=False,
            mixing=True,
            identity_prob=0.0,
            distribution="kaiming_normal",
            data_mean=None,
            data_std=None,
            clamp_output=False,
    ):
        """

        :param net:
        :param kernel_size:
        :param in_channels:
        :param out_channels:
        :param rand_bias:
        :param mixing: "random": output = (1-alpha)*input + alpha* randconv(input) where alpha is a random number sampled
                            from a distribution defined by res_dist
        :param identity_prob:
        :param distribution:
        :param data_mean:
        :param data_std:
        :param clamp_output:
        """

        super(RandConvModule, self).__init__()

        # if the input is not normalized, we need to normalized with given mean and std (tensor of size 1)
        self.register_buffer(
            "data_mean",
            None if data_mean is None else torch.tensor(data_mean).reshape(1, 1, 1),
        )
        self.register_buffer(
            "data_std",
            None if data_std is None else torch.tensor(data_std).reshape(1, 1, 1),
        )

        # adjust output range based on given data mean and std, (clamp or norm)
        # clamp with clamp the value given that the was image pixel values [0,1]
        # normalize will linearly rescale the values to the allowed range
        # The allowed range is ([0, 1]-data_mean)/data_std in each color channel
        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (
                    self.data_std is not None
            ), "Need data mean/std to do output range adjust"
        self.register_buffer(
            "range_up",
            None
            if not self.clamp_output
            else (torch.ones(1).reshape(1, 1, 1) - self.data_mean) / self.data_std,
        )
        self.register_buffer(
            "range_low",
            None
            if not self.clamp_output
            else (torch.zeros(1).reshape(1, 1, 1) - self.data_mean) / self.data_std,
        )

        self.randconv = MultiScaleRandConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_size,
            stride=1,
            rand_bias=rand_bias,
            distribution=distribution,
            clamp_output=self.clamp_output,
            range_low=self.range_low,
            range_up=self.range_up,
        )

        if isinstance(kernel_size, collections.Sequence) and len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        if mixing:
            out_channels = in_channels

        # generate random conv layer
        print(
            "Add RandConv layer with kernel size {}, output channel {}".format(
                kernel_size, out_channels
            )
        )

        # mixing mode
        self.mixing = (
            mixing
        )  # In the mixing mode, a mixing connection exists between input and output of random conv layer
        # self.res_dist = res_dist
        self.res_test_weight = None
        if self.mixing:
            assert (
                    in_channels == out_channels or out_channels == 1
            ), "In mixing mode, in/out channels have to be equal or out channels is 1"
            self.alpha = (
                random.random()
            )  # sample mixing weights from uniform distributin (0, 1)

        self.identity_prob = identity_prob  # the probability that use original input

    def forward(self, input):
        """assume that the input is whightened"""

        ######## random conv ##########
        if not (self.identity_prob > 0 and torch.rand(1) < self.identity_prob):
            # whiten input and go through randconv
            output = self.randconv(input)

            if self.mixing:
                output = self.alpha * output + (1 - self.alpha) * input

            if self.clamp_output:
                output = torch.max(torch.min(output, self.range_up), self.range_low)
        else:
            output = input

        return output

    def parameters(self, recurse=True):
        return self.randconv.parameters()

    def trainable_parameters(self, recurse=True):
        return self.randconv.trainable_parameters()

    def randomize(self):
        self.randconv.randomize()

        if self.mixing:
            self.alpha = random.random()

    def set_test_res_weight(self, w):
        self.res_test_weight = w


class RandConv3d(Conv3d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            rand_bias=True,
            distribution="kaiming_normal",
            clamp_output=None,
            range_up=None,
            range_low=None,
            **kwargs
    ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param rand_bias:
        :param distribution:
        :param clamp_output:
        :param range_up:
        :param range_low:
        :param kwargs:
        """
        super(RandConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=rand_bias,
            **kwargs
        )

        self.rand_bias = rand_bias
        self.distribution = distribution

        self.clamp_output = clamp_output
        self.register_buffer("range_up", None if not self.clamp_output else range_up)
        self.register_buffer("range_low", None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (
                    self.range_low is not None
            ), "No up/low range given for adjust"
        self.randomize()

    def randomize(self):
        new_weight = torch.zeros_like(self.weight)
        with torch.no_grad():
            if self.distribution == "kaiming_uniform":
                nn.init.kaiming_uniform_(new_weight, nonlinearity="conv3d")
            elif self.distribution == "kaiming_normal":
                nn.init.kaiming_normal_(new_weight, nonlinearity="conv3d")
            elif self.distribution == "kaiming_normal_clamp":
                fan = nn.init._calculate_correct_fan(new_weight, "fan_in")
                gain = nn.init.calculate_gain("conv3d", 0)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    new_weight.normal_(0, std)
                    new_weight = new_weight.clamp(-2 * std, 2 * std)
            elif self.distribution == "xavier_normal":
                nn.init.xavier_normal_(new_weight)
            else:
                raise NotImplementedError()

        self.weight = nn.Parameter(new_weight.detach())
        if self.bias is not None and self.rand_bias:
            # new_bias = self.bias.clone().detach()
            new_bias = torch.zeros_like(self.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)
            self.bias = nn.Parameter(new_bias)

    def forward(self, input):
        output = super(RandConv3d, self).forward(input)

        if self.clamp_output == "clamp":
            print("clamp")
            output = torch.max(torch.min(output, self.range_up), self.range_low)
        elif self.clamp_output == "norm":
            print("norm")
            output_low = torch.min(
                torch.min(output, dim=3, keepdim=True)[0], dim=2, keepdim=True
            )[0]
            output_up = torch.max(
                torch.max(output, dim=3, keepdim=True)[0], dim=2, keepdim=True
            )[0]
            output = (output - output_low) / (output_up - output_low) * (
                    self.range_up - self.range_low
            ) + self.range_low

        return output


class MultiScaleRandConv3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            rand_bias=True,
            distribution="kaiming_normal",
            clamp_output=False,
            range_up=None,
            range_low=None,
            **kwargs
    ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size: sequence of kernel size, e.g. (1,3,5)
        :param bias:
        """
        super(MultiScaleRandConv3d, self).__init__()

        self.clamp_output = clamp_output
        self.register_buffer("range_up", None if not self.clamp_output else range_up)
        self.register_buffer("range_low", None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (
                    self.range_low is not None
            ), "No up/low range given for adjust"

        self.multiscale_rand_convs = nn.ModuleDict(
            {
                str(kernel_size): RandConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    rand_bias=rand_bias,
                    distribution=distribution,
                    clamp_output=self.clamp_output,
                    range_low=self.range_low,
                    range_up=self.range_up,
                    **kwargs
                )
                for kernel_size in kernel_sizes
            }
        )

        self.scales = kernel_sizes
        self.n_scales = len(kernel_sizes)
        self.randomize()

    def randomize(self):
        self.current_scale = str(self.scales[random.randint(0, self.n_scales - 1)])
        self.multiscale_rand_convs[self.current_scale].randomize()

    def forward(self, input):
        output = self.multiscale_rand_convs[self.current_scale](input)
        return output


class data_whiten_layer(nn.Module):
    def __init__(self, data_mean, data_std):
        super(data_whiten_layer, self).__init__()
        self.register_buffer(
            "data_mean",
            None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1),
        )
        self.register_buffer(
            "data_std",
            None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1),
        )

    def forward(self, input):
        return (input - self.data_mean) / self.data_std


def get_random_module(data_mean, data_std):
    return RandConvModule(
        in_channels=1,
        out_channels=1,
        kernel_size=[3],
        mixing=True,
        identity_prob=0.0,
        rand_bias=True,
        distribution="kaiming_normal",
        data_mean=data_mean,
        data_std=data_std,
        clamp_output=True,
    )


if __name__ == "__main__":

    x = (
        np.load(join(msd_resample_data_path, "{}.npy".format(1)))
        .transpose((2, 0, 1))
        .astype(np.float)
    )
    if np.max(x) > 1:
        np.minimum(np.maximum(x, LOW_RANGE, x), HIGH_RANGE, x)
        x -= LOW_RANGE
        x /= HIGH_RANGE - LOW_RANGE
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=0)
    x = np.ascontiguousarray(x)
    x = torch.from_numpy(x).float().cuda()
    # x = torch.randn(1, 1, 8, 64, 64).cuda()
    data_mean = x.mean()
    data_std = x.std()
    print(data_mean, data_std)
    net = RandConvModule(
        in_channels=1,
        out_channels=1,
        kernel_size=[3],
        mixing=True,
        identity_prob=0.0,
        rand_bias=True,
        distribution="kaiming_normal",
        data_mean=data_mean,
        data_std=data_std,
        clamp_output=True,
    )
    net = net.cuda()
    net.randomize()
    # y1 = net(x)
    # print(y1.size(), y1.mean(), y1.std())
    # plt.imshow(y1.cpu().detach().numpy()[0, 0, 80], "gray")
    # plt.show()
    # net.randomize()
    # y1 = net(x)
    # print(y1.size(), y1.mean(), y1.std())
    # plt.imshow(y1.cpu().detach().numpy()[0, 0, 80], "gray")
    # plt.show()
    # net.randomize()
    # y1 = net(x)
    # print(y1.size(), y1.mean(), y1.std())
    # plt.imshow(y1.cpu().detach().numpy()[0, 0, 80], "gray")
    # plt.show()
    # net.randomize()
    # y1 = net(x)
    # print(y1.size(), y1.mean(), y1.std())
    # plt.imshow(y1.cpu().detach().numpy()[0, 0, 80], "gray")
    # plt.show()
    # net.randomize()
    # y1 = net(x)
    # print(y1.size(), y1.mean(), y1.std())
    # plt.imshow(y1.cpu().detach().numpy()[0, 0, 80], "gray")
    # plt.show()

    a = torch.ones((1, 1, 128, 128, 128))
