from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function


def upfirdn3d(input, kernel, up=(1, 1, 1), down=(1, 1, 1), pad=(0, 0, 0, 0, 0, 0)):
    # Ensure up and down are iterable tuples
    if not isinstance(up, abc.Iterable):
        up = (up, up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down, down)
    # print(pad)
    if len(pad) == 3:
        pad = (pad[0], pad[1], pad[0], pad[1], pad[2], pad[2])
    
    # Check if input is on CPU or GPU and call appropriate function
    out = upfirdn3d_native(input, kernel, *up, *down, *pad)


    return out



def upfirdn3d_native(
    input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, 
    pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1
):
    # print(up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1)

    # Unpack input shape
    _, channel, in_d, in_h, in_w = input.shape
    # print('input shape', input.shape)
    # Reshape input to add an extra dimension for later operations
    input = input.reshape(-1, in_d, in_h, in_w, 1)

    # Get new shape after the reshaping (we unpack again for convenience)
    _, in_d, in_h, in_w, minor = input.shape  # `minor` is effectively the number of channels now
    kernel_d, kernel_h, kernel_w = kernel.shape  # Get dimensions of the kernel

    # Reshape input to prepare for upsampling along x, y, and z axes
    out = input.view(-1, in_d, 1, in_h, 1, in_w, 1, minor)
    
    # print('before 1st padding', out.shape)
    # Apply padding to simulate upsampling by inserting zeros between pixels (in depth, height, and width)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0, 0, up_z - 1])
    # print('after 1st padding', out.shape)
    # Reshape the output after upsampling (by padding) to merge the padded dimensions
    out = out.view(-1, in_d * up_z, in_h * up_y, in_w * up_x, minor)

    # print('before 2nd padding', out.shape)
    # Apply padding to the upsampled image (padding for the edges), 'padding:', [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_z0, 0), max(pad_z1, 0)]
    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_z0, 0), max(pad_z1, 0)]
    )
    # print('after padding', out.shape)
    # Remove any extra padding beyond the boundaries (negative padding handling)
    out = out[
        :,
        max(-pad_z0, 0) : out.shape[1] - max(-pad_z1, 0),
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
        :
    ]

    # Permute the dimensions to get the channels back in the second position
    out = out.permute(0, 4, 1, 2, 3)
    
    # # Reshape the output to prepare it for 3D convolution (grouping the spatial dimensions)
    out = out.reshape(
        [-1, 1, in_d * up_z + pad_z0 + pad_z1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    
    # Flip the kernel in all three dimensions (depth, height, width) and reshape it for applying 3D convolution
    w = torch.flip(kernel, [1, 2]).view(1, 1, kernel_d, kernel_h, kernel_w)
    # print('before conv', out.shape)
    # Apply 3D convolution with the flipped kernel on the padded and upsampled input
    out = F.conv3d(out, w)
    # print('after conv', out.shape)
    # # Reshape the output after convolution to match the expected depth/height/width
    out = out.reshape(
        -1,
        minor,
        in_d * up_z + pad_z0 + pad_z1 - kernel_d + 1,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    
    # Permute the output again to return it to (batch, depth, height, width, channels) format
    out = out.permute(0, 2, 3, 4, 1)
    # print('after permute and reshape', out.shape)
    # Downsample by taking every `down_x`, `down_y`, and `down_z`th pixel along the x, y, and z axes
    out = out[:, ::down_z, ::down_y, ::down_x, :]
    
    # # Calculate output depth, height, and width based on upsampling, padding, and downsampling
    out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d + down_z) // down_z
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    # print(out_d, out_h, out_w)
    

    _, out_d, out_h, out_w, _ = out.shape
    # Reshape the final output to have the original channel count
    out=out.view(-1, channel, out_d, out_h, out_w)
    
    return out

