#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import sys
import torch
from torch import nn, tanh, sigmoid
from torch.nn.functional import layer_norm, log_softmax, pad
from torch.nn import Conv2d, BatchNorm2d, Dropout, Sequential, MaxPool2d, InstanceNorm2d, GroupNorm, ZeroPad2d
from torch.nn import ReLU
from torch.nn import ModuleList


class GFCN(nn.Module):

    def __init__(self, params):
        super(GFCN, self).__init__()

        if params["normalization"]["type"] == "batch":
            self.norm_layer = BatchNorm2d
        elif params["normalization"]["type"] == "instance":
            self.norm_layer = InstanceNorm2d
        elif params["normalization"]["type"] == "group":
            self.norm_layer = GroupNorm
        elif params["normalization"]["type"] == "layer":
            self.norm_layer = CustomLayerNorm
        else:
            print("invalid normalization option")
            sys.exit(0)

        self.activation = ReLU

        self.eps = 0.00001 if params["normalization"]["eps"] is None else params["normalization"]["eps"]
        self.momentum = 0.1 if params["normalization"]["momentum"] is None else params["normalization"]["momentum"]
        self.track = params["normalization"]["track_running_stats"]
        self.num_group = params["normalization"]["num_group"]

        self.device = params["device"]
        self.vocab_size = params["vocab_size"]

        self.gn = GaussianNoise()
        self.init_blocks = ModuleList([self.init_block(params["input_channels"], 32), self.init_block(32, 64)])
        self.blocks = ModuleList([self.block(64, 64, hor_pool=2), self.block(64, 128, hor_pool=2), self.block(128, 128),
                                  self.block(128, 128), self.block(128, 128)])
        self.trans_conv = Sequential(Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 1), groups=128),
                                     Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
                                     self.activation(),
                                     )
        self.end_blocks = ModuleList([self.end_block() for _ in range(params["nb_gate"])])
        self.end_conv = Conv2d(in_channels=256, out_channels=self.vocab_size+1, kernel_size=(1, 1))

    def forward(self, x):
        x = self.gn(x)
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            x = b(x)
        x = self.trans_conv(x)
        for b in self.end_blocks:
            x = b(x)
        x = self.end_conv(x)
        x = torch.squeeze(x, dim=2)
        return log_softmax(x, dim=1)

    def init_block(self, in_, out_):
        if self.norm_layer is GroupNorm:
            layer_norm = self.norm_layer(self.num_group, num_channels=out_, eps=self.eps)
        elif self.norm_layer is CustomLayerNorm:
            layer_norm = self.norm_layer(eps=self.eps)
        else:
            layer_norm = self.norm_layer(out_, eps=self.eps, momentum=self.momentum, track_running_stats=self.track)
        return Sequential(Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3, 3), padding=(1, 1)),
                          self.activation(),
                          Conv2d(in_channels=out_, out_channels=out_, kernel_size=(3, 3), padding=(1, 1)),
                          self.activation(),
                          layer_norm,
                          Dropout(p=0.4)
                          )

    def block(self, in_, out_, hor_pool=1):
        if self.norm_layer is GroupNorm:
            layer_norm = self.norm_layer(self.num_group, num_channels=2*out_, eps=self.eps)
        elif self.norm_layer is CustomLayerNorm:
            layer_norm = self.norm_layer(eps=self.eps)
        else:
            layer_norm = self.norm_layer(out_, eps=self.eps, momentum=self.momentum, track_running_stats=self.track)
        return Sequential(DepthSepConv2D(in_, 2 * out_, kernel_size=(3, 3)),
                          self.activation(),
                          DepthSepConv2D(2 * out_, 2 * out_, kernel_size=(3, 3)),
                          self.activation(),
                          layer_norm,
                          MaxPool2d((2, hor_pool)),
                          Gate(),
                          Dropout(p=0.4)
                          )

    def end_block(self):
        return Sequential(
            Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 8), groups=256),
            ZeroPad2d((4, 3, 0, 0)),
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1)),
            self.activation(),
            Gate(),
            Dropout(p=0.4)
        )


class CustomLayerNorm(nn.Module):
    def __init__(self, eps=0.00001):
        super(CustomLayerNorm, self).__init__()

        self.eps = eps

    def forward(self, x):
        return layer_norm(x, x.size()[1:], eps=self.eps)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            x = x + torch.autograd.Variable(torch.randn(x.size()).to(x.get_device()) * self.std)
        return x


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, x):
        tensor_1, tensor_2 = torch.split(x, int(x.size()[1]/2), dim=1)
        act_1 = tanh(tensor_1)
        act_2 = sigmoid(tensor_2)
        norm_1 = layer_norm(act_1, act_1.size()[1:])
        norm_2 = layer_norm(act_2, act_2.size()[1:])
        return torch.mul(norm_1, norm_2)


class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        if self.activation:
            x = self.activation(x)
        return x
