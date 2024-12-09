import math
import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from gan.model.op.conv3d_gradfix import conv3d_gradfix
from gan.model.op.upfirdn3d import upfirdn3d

# TODO : The code needs to be reviewed by an other person than Victor (ex : @clement))

library = {'stylegan2_3d' : {'G' :  'Generator3D', 'D' : 'Discriminator3D'}}


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # # print("####### PIXEL NORM #######")
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k.unsqueeze(0)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[1] - factor
        # print('p for Upsample class', p)
        pad0 = (p + 1) // 2 + factor - 1 # width padding
        pad1 = p // 2 # heigt padding
        pad2 = 0 # depth padding 
        

        self.pad = (pad0, pad1, pad2) # 



    def forward(self, input):
        # print('Upsample class')
        out = upfirdn3d(input, self.kernel, up=(self.factor,self.factor, 1), down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[1] - factor
        # print('p for Downsample class', p)
        pad0 = (p + 1) // 2 # heigt padding
        pad1 = p // 2 # width padding
        pad2 = 0 # depth padding 

        self.pad = (pad0, pad1, pad2)

    def forward(self, input):
        out = upfirdn3d(input, self.kernel, up=1, down=(1,self.factor,self.factor), pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
            # # print('kernel upsampled ', kernel)
        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        # print("#### BLUR ####")
        
        # print("Before Bluring", input.shape)
        out = upfirdn3d(input, self.kernel, pad=self.pad)
        # print("After Bluring", out.shape)

        return out

class EqualConv3d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, *kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size[-1] ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv3d_gradfix.conv3d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        # # print("####### EQUAL LINEAR #######")

        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        if self.activation:
            out = F.leaky_relu(out)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

class ModulatedConv3d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if isinstance(kernel_size, tuple):
            kernel = kernel_size[-1]

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel - 1)
            # print('p for upsample', p)
            pad0 = (p + 1) // 2 + factor - 1 # heigt padding 
            pad1 = p // 2 + 1 # width padding
            pad2 = 0 # depth padding

            self.blur = Blur(blur_kernel, pad=(pad0, pad1, pad2), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel - 1)
            # print('p for downsample', p)
            pad0 = (p + 1) // 2 # heigt padding
            pad1 = p // 2 # width padding
            pad2 = 0 # depth padding 

            self.blur = Blur(blur_kernel, pad=(pad0, pad1, pad2))

        if isinstance(kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)

        fan_in = in_channel * max(list(kernel_size)) ** 2
        self.padding = [kernel // 2 for kernel in kernel_size]

        self.scale = 1 / math.sqrt(fan_in)
        
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def forward(self, input, style):
        # print("####### MODULATED CONV #######")
        batch, in_channel, depth, height, width = input.shape
        # print('a1 input shape', input.shape)

        if not self.fused:
            
            # print('self.fused necessary')
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4, 5)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv3d_gradfix.conv_transpose3d(
                    input, weight, padding=0, stride=(1,2,2)
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv3d_gradfix.conv3d(input, weight, padding=0, stride=(1,2,2))

            else:
                out = conv3d_gradfix.conv3d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1, 1)

            return out
        
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1, 1)
        # print("a2 style shape", np.shape(style))

        weight = self.scale * self.weight * style
        # print("a3 weight shape", weight.shape)

        if self.demodulate:
            # print('################ DEMODULATE ################')
            # print("demodulate a4 weight shape", weight.shape)
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4, 5]) + 1e-8)
            # print("demodulate a5 weight shape", demod.shape)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1, 1)
            # print("demodulate a6 weight shape", weight.shape)

        weight = weight.view(
            batch * self.out_channel, in_channel, *self.kernel_size
        )

        
        # print("a7 weight shape", weight.shape)

        if self.upsample:
            # print('################ UPSAMPLE ################')
            # print("upsample a8 input shape", np.shape(input))
            input = input.view(1, batch * in_channel, depth, height, width)
            # print("upsample a9 weight shape", weight.shape)

            weight = weight.view(
            batch * self.out_channel, in_channel, *self.kernel_size
            )
            # print("upsample a10 weight shape", weight.shape)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, *self.kernel_size
            )
            # print("upsample a11 weight shape", weight.shape)
            # print("upsample a11 input shape", input.shape)
            out = conv3d_gradfix.conv_transpose3d(
                input, weight, padding=0, stride=(1,2,2), groups=batch
            )
            # print("upsample a11 out shape", out.shape)
            # print("upsample a12 weight shape", weight.shape)
            _, _, depth, height, width = out.shape
            # print("upsample a13 weight shape", weight.shape)
            out = out.view(batch, self.out_channel, depth, height, width)
            # print("upsample a14 weight shape", weight.shape)
            # print("upsample a14 out shape", out.shape)
            out = self.blur(out)
            # print("upsample a15 weight shape", weight.shape)
            # print("upsample a15 out shape", out.shape)

        elif self.downsample:
            # print('################ DOWNSAMPLE ################')
            # print("a16 shape before interpolate ", input.shape)
            input= self.blur(input) #, scale=2)
            # print("a17 shape after interpolate ", input.shape)
            _, _, depth, height, width = input.shape
            # print("a18")
            input = input.view(1, batch * in_channel, depth, height, width)
            # print("a19 resize input", input.shape)
            out = conv3d_gradfix.conv3d(
                input, weight, padding=0, stride=2, groups=batch
            )
            # print("a20 out after conv3d", out.shape)
            _, _, depth, height, width = out.shape
            # print("a21")
            out = out.view(batch, self.out_channel, depth, height, width)
            # print("a22 final out shape", out.shape)

        else:
            # print('################ OTHER ################')
            # print("a23 shape before resize ", input.shape)
            input = input.view(1, batch * in_channel, depth, height, width)
            # print("a24 shape after resize ", input.shape,   weight.shape)
            out = conv3d_gradfix.conv3d(
                input, weight, padding=self.padding, groups=batch
            )
            # print("a25 shape after conv3d ", out.shape)
            _, _, depth, height, width = out.shape
            # print("a26")
            out = out.view(batch, self.out_channel, depth, height, width)
            # print("a27 shape after resize ", out.shape)
        # print('Output Modulated Conv', out.shape)
        return out


    
class NoiseInjection(nn.Module):
    def __init__(self, use_noise=True):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))
        self.use_noise = use_noise

    def forward(self, image, noise=None):
        # # print("####### NOISE INJECTION #######")
        if noise is None:
            batch, _, depth, height, width = image.shape
            noise = image.new_empty(batch, 1, depth, height, width).normal_()
        return image + float(int(self.use_noise)) * self.weight * noise

class ConstantInput(nn.Module):
    r''' Constant input at the top of StyleGAN Generator '''
    def __init__(self, channel, size=4, nb_frames=15):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, nb_frames, size, size))

    def forward(self, input):
        # # print("####### CONSTANT INPUT #######")
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel=[1, 3, 3, 1],
        upsample=False,
        demodulate=True,
        use_noise=True,
    ):
        super().__init__()

        self.conv = ModulatedConv3d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection(use_noise=use_noise)

        # classic version
        self.activate = nn.LeakyReLU()

    def forward(self, input, style, noise=None):
        # # print("####### STYLED CONV #######")
        # # print("input styled", input.shape)
        out = self.conv(input, style)
        # # print("modulated styled ", out.shape)
        out = self.noise(out, noise=noise)
        # # print("noise styled", out.shape)
        # out = out + self.bias
        out = self.activate(out)
        # # print("output styled", out.shape)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], nb_var=3, nb_frames=15):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv3d(
                                    in_channel=in_channel,
                                    out_channel=nb_var,
                                    kernel_size=(1,1,1),
                                    style_dim=style_dim,
                                    demodulate=False,
                                    upsample=False,
                                    downsample=False,
                                    fused=True
        )
        # Restrict the ouput to have nb_frames as depth
        self.bias = nn.Parameter(torch.zeros(1, nb_var, nb_frames, 1, 1))

    def forward(self, input, style, skip=None):
        # print("####### TORGB #######")
        # print('From ToRGB', np.shape(input))
        out = self.conv(input, style)
        # print('Size of ModConv RGB : ', out.shape)
        out = out + self.bias
        input_conved = torch.clone(out)
        # print('Shape input_conved', input_conved.shape)

        if skip is not None:
            prev_rgb = torch.clone(skip)
            # print('SKIP before Upsampling', prev_rgb.shape)
            skip = self.upsample(skip)
            # print('SKIP after Upsampling', skip.shape)
            out = out + skip

        return (out, input_conved, skip, prev_rgb) if skip  is not None else (out, input_conved)


class Generator3D(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        nb_var=3,
        nb_frames=15,
        progressive_nb_frames=False, # TODO : Version to progressively increase the depth dimension
        var_rr=False,
        tanh_output=False,
        use_noise=True,
    ):
        super().__init__()

        self.size = size
        self.var_rr = var_rr
        self.tanh_output = tanh_output

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.chm = channel_multiplier

        if progressive_nb_frames:
            self.input = ConstantInput(self.channels[4], nb_frames=4)
        else :
            self.input = ConstantInput(self.channels[4], nb_frames=nb_frames)

        self.conv1 = StyledConv(
            in_channel=self.channels[4],
            out_channel=self.channels[4],
            kernel_size=(1,3,3),
            style_dim=style_dim,
            blur_kernel=blur_kernel,
            upsample=False,
            demodulate=True,
            use_noise=use_noise,
        )

        if progressive_nb_frames:
            self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, nb_var=nb_var, nb_frames=4)
        else :
            self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, nb_var=nb_var, nb_frames=nb_frames)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        if progressive_nb_frames:
            nb_frames_list = [int(i) for i in np.linspace(4,nb_frames,(self.log_size+1)-3)]
            # print(nb_frames_list)

        for id, i in enumerate(range(3, self.log_size + 1)):
            
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=(1,3,3),
                    style_dim=style_dim,
                    blur_kernel=blur_kernel,
                    upsample=True,
                    use_noise=use_noise
                )
            )

            self.convs.append(
                StyledConv(
                    in_channel=out_channel,
                    out_channel=out_channel,
                    kernel_size=(1,3,3),
                    style_dim=style_dim,
                    blur_kernel=blur_kernel,
                    use_noise=use_noise
                )
            )

            if progressive_nb_frames:
                self.to_rgbs.append(ToRGB(out_channel, style_dim, nb_var=nb_var, nb_frames=nb_frames_list[id]))
            else :
                self.to_rgbs.append(ToRGB(out_channel, style_dim, nb_var=nb_var, nb_frames=nb_frames))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        return_rgb=False,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        # # print("From Generator : latent ", latent.shape)
        out = self.input(latent)
        # # print("From Generator : input gen ", out.shape)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        # # print("From Generator : conv1 gen ", out.shape)

        skip, input_conved = self.to_rgb1(out, latent[:, 1])

        # print("From Generator :rgb1", skip.shape)
        if return_rgb:
            rgbs_saved = {}
            rgbs_saved['prev_rgb'] = {}
            rgbs_saved['prev_rgb_upsampled'] = {}
            rgbs_saved['input_conved'] = {}
            rgbs_saved['current_rgb_out'] = {}
            
            rgbs_saved['prev_rgb'][1] = input_conved.detach().cpu().numpy()
            rgbs_saved['prev_rgb_upsampled'][1] = input_conved.detach().cpu().numpy()
            rgbs_saved['input_conved'][1] = input_conved.detach().cpu().numpy()
            rgbs_saved['current_rgb_out'][1] = skip.detach().cpu().numpy()

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):  
            # print('From Generator :Next layer conv')
            out = conv1(out, latent[:, i], noise=noise1)
            # print("From Generator :conv1 gen ", out.shape)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            # print("From Generator :conv2 gen ", out.shape)
            skip, input_conved, prev_rgb_upsampled, prev_rgb = to_rgb(out, latent[:, i + 2], skip)
            # print("From Generator :rgb ", skip.shape)
            if return_rgb:
                rgbs_saved['prev_rgb'][i//2 + 2] = prev_rgb.detach().cpu().numpy()
                rgbs_saved['prev_rgb_upsampled'][i//2 + 2] = prev_rgb_upsampled.detach().cpu().numpy()
                rgbs_saved['input_conved'][i//2 + 2] = input_conved.detach().cpu().numpy()
                rgbs_saved['current_rgb_out'][i//2 + 2] = skip.detach().cpu().numpy()

            i += 2

        image = skip
        if self.var_rr and self.tanh_output:
            image[:, 0] = torch.tanh(image[:, 0]) # assuming rr is the first variable ['rr', ...]
        if return_latents:
            if return_rgb:
                return image, latent, rgbs_saved
            else:
                return image, latent, None

        else:
            if return_rgb:
                return image, None, rgbs_saved
            else:
                return image, None, None


class ConvLayer(nn.Sequential):
    r''' Convolutional Layer '''
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        super().__init__()
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size[-1] - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            pad2 = 0

            layers.append(Blur(blur_kernel, pad=(pad0, pad1, pad2)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = (0, kernel_size[-2] // 2, kernel_size[-1] // 2) 

        layers.append(
            EqualConv3d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )
        

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        super().__init__(*layers)

class ResBlock(nn.Module):
    r''' Residual Block based on ConvLayer'''
    def __init__(self, in_channel, out_channel,  blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, (1,3,3),  blur_kernel=blur_kernel)
        self.conv2 = ConvLayer(in_channel, out_channel, (1,3,3), downsample=True,  blur_kernel=blur_kernel)

        self.skip = ConvLayer(
            in_channel, out_channel, (1,3,3), downsample=True, activate=False, bias=False, blur_kernel=blur_kernel
        )

    def forward(self, input):
        # print("resblock, input ", input.shape)
        # print("###### RESBLOCK IN ######")
        out = self.conv1(input)
        # print("resblock, out1 ", out.shape)
        out = self.conv2(out)
        # print("resblock, out2 ", out.shape)
        skip = self.skip(input)
        # print("resblock, skip ", skip.shape)
        out = (out + skip) / math.sqrt(2)
        # print("###### RESBLOCK OUT ######")
        return out


class Discriminator3D(nn.Module):
    def __init__(self, size, channel_multiplier=2, nb_var=3, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(nb_var, channels[size], (1,3,3), blur_kernel)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)
        
        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], (1,3,3))
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )
        # # print(self.convs, self.final_conv, self.final_linear)

    def forward(self, input):
        # print("input ", input.shape)
        out = self.convs(input)
        # print("before std ", out.shape)
        batch, channel, depth, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, depth, height, width
        )
        # print("stddev shape before", stddev.shape)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4, 5], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, depth, height, width)
        # print("stddev shape after ", stddev.shape)
        out = torch.cat([out, stddev], 1)
        # print('std ', out.shape)
        out = self.final_conv(out)
        # print('final_conv ',out.shape)
        out = out.view(batch, -1)
        # print('out ',out.shape, self.final_linear[0].weight.shape, self.final_linear[1].weight.shape)
        out = self.final_linear(out)

        return out

