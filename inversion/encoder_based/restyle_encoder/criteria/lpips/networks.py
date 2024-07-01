from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models as mod0
from criteria.lpips.utils import normalize_activation
from models.stylegan2 import model




Means, Stds = torch.Tensor([0.00323179, -0.00636883, -0.00447128]),\
                torch.Tensor([0.09430821,0.10637118, 0.1566861]) 

def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    elif net_type == 'discrim' :
        return StyleDiscrim(mean=Means, std=Stds)
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg, discrim].')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Identity()
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self, mean = torch.Tensor([-.030, -.088, -.188]),\
                       std =torch.Tensor([.458, .448, .450]) ):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', mean[None, :, None, None])
        self.register_buffer(
            'std', std[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = mod0.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = mod0.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = mod0.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
        
class StyleDiscrim(BaseNet):
    def __init__(self, mean, std):
        super(StyleDiscrim, self).__init__(mean, std)
        self.layers = model.Discriminator(128).convs
        self.target_layers = [1,2,3,4]
        self.n_channels_list = [256, 512, 512, 512]
        
        self.set_requires_grad(False)
        
        