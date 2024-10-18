from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)

upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)

class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(
                grad_output,
                kernel,
                grad_kernel,
                ctx.up,
                ctx.down,
                ctx.pad,
                ctx.g_pad,
                ctx.in_size,
                ctx.out_size,
            )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    if input.device.type == "cpu":
        out = upfirdn2d_native(input, kernel, *up, *down, *pad)

    else:
        out = UpFirDn2d.apply(input, kernel, up, down, pad)

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    # Unpack input shape (batch size is ignored here with _)
    _, channel, in_h, in_w = input.shape
    
    # Reshape input to add an extra dimension for later operations
    input = input.reshape(-1, in_h, in_w, 1)

    # Get new shape after the reshaping (we unpack again for convenience)
    _, in_h, in_w, minor = input.shape  # `minor` is effectively the number of channels now
    kernel_h, kernel_w = kernel.shape  # Get dimensions of the kernel

    # Reshape input to prepare for upsampling along x and y axes
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    
    # Apply padding to simulate upsampling by inserting zeros between pixels
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    
    # Reshape the output after upsampling (by padding) to merge the padded dimensions
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    # Apply padding to the upsampled image (padding for the edges)
    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )

    # Remove any extra padding beyond the boundaries (negative padding handling)
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    # Permute the dimensions to get the channels back in the second position
    out = out.permute(0, 3, 1, 2)
    
    # Reshape the output to prepare it for convolution (grouping the spatial dimensions)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    
    # Flip the kernel horizontally and vertically (as done in convolution) 
    # and reshape it for applying 2D convolution
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    
    # Apply 2D convolution with the flipped kernel on the padded and upsampled input
    out = F.conv2d(out, w)
    
    # Reshape the output after convolution to match the expected height/width
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )

    # Permute the output again to return it to (batch, height, width, channels) format
    out = out.permute(0, 2, 3, 1)
    
    # Downsample by taking every `down_x`th and `down_y`th pixel along the x and y axes
    out = out[:, ::down_y, ::down_x, :]

    # Calculate output height and width based on upsampling, padding, and downsampling
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    # Reshape the final output to have the original channel count
    return out.view(-1, channel, out_h, out_w)

