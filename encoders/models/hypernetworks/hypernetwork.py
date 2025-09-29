from torch import nn
from torch.nn import BatchNorm2d, PReLU, Sequential, Module
from torchvision.models import resnet34

from encoders.models.hypernetworks.refinement_blocks import HyperRefinementBlock, RefinementBlock, RefinementBlockSeparable
from encoders.models.hypernetworks.shared_weights_hypernet import SharedWeightsHypernet


class SharedWeightsHyperNetResNet(Module):

    def __init__(self, config):
        super(SharedWeightsHyperNetResNet, self).__init__()

        self.conv1 = nn.Conv2d(config.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=False)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        if len(config.layers_to_tune) == 0:
            self.layers_to_tune = list(range(config.n_hypernet_outputs))
        else:
            self.layers_to_tune = [int(l) for l in config.layers_to_tune.split(',')]

        self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12]
        self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)

        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = config.n_hypernet_outputs
        for layer_idx in range(self.n_outputs):
            if layer_idx in self.layers_to_tune:
                if layer_idx in self.shared_layers:
                    refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
                else:
                    refinement_block = RefinementBlock(layer_idx, config, n_channels=512, inner_c=256)
            else:
                refinement_block = None
            self.refinement_blocks.append(refinement_block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        weight_deltas = []
        for j in range(self.n_outputs):
            if self.refinement_blocks[j] is not None:
                delta = self.refinement_blocks[j](x)
            else:
                delta = None
            weight_deltas.append(delta)
        return weight_deltas


class SharedWeightsHyperNetResNetSeparable(Module):

    def __init__(self, config):
        super(SharedWeightsHyperNetResNetSeparable, self).__init__()

        self.conv1 = nn.Conv2d(config.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=False)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        if len(config.layers_to_tune) == 0:
            self.layers_to_tune = list(range(config.n_hypernet_outputs))
        else:
            self.layers_to_tune = [int(l) for l in config.layers_to_tune.split(',')]

        self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12]
        self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)

        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = config.n_hypernet_outputs
        for layer_idx in range(self.n_outputs):
            if layer_idx in self.layers_to_tune:
                if layer_idx in self.shared_layers:
                    refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
                else:
                    refinement_block = RefinementBlockSeparable(layer_idx, config, n_channels=512, inner_c=256)
            else:
                refinement_block = None
            self.refinement_blocks.append(refinement_block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        weight_deltas = []
        for j in range(self.n_outputs):
            if self.refinement_blocks[j] is not None:
                delta = self.refinement_blocks[j](x)
            else:
                delta = None
            weight_deltas.append(delta)
        return weight_deltas