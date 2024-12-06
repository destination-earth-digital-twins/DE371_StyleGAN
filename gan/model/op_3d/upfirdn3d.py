from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function
# from torch.utils.cpp_extension import load

# # Load the 3D version of the upfirdn operation
# module_path = os.path.dirname(__file__)
# upfirdn3d_op = load(
#     "upfirdn3d",
#     sources=[
#         os.path.join(module_path, "upfirdn3d.cpp"),
#         os.path.join(module_path, "upfirdn3d_kernel.cu"),
#     ],
# )

# class UpFirDn3dBackward(Function):
#     @staticmethod
#     def forward(
#         ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
#     ):

#         # Unpack the up and down scaling factors for x, y, and z
#         up_x, up_y, up_z = up
#         down_x, down_y, down_z = down
#         g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1, g_pad_z0, g_pad_z1 = g_pad

#         # Reshape grad_output to account for the output size in x, y, and z dimensions
#         grad_output = grad_output.reshape(-1, out_size[0], out_size[1], out_size[2], 1)

#         # Apply the 3D upfirdn operation
#         grad_input = upfirdn3d_op.upfirdn3d(
#             grad_output,
#             grad_kernel,
#             down_x,
#             down_y,
#             down_z,
#             up_x,
#             up_y,
#             up_z,
#             g_pad_x0,
#             g_pad_x1,
#             g_pad_y0,
#             g_pad_y1,
#             g_pad_z0,
#             g_pad_z1,
#         )

#         # Reshape grad_input back to the original input size
#         grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3], in_size[4])

#         # Save necessary tensors and context variables for the backward pass
#         ctx.save_for_backward(kernel)

#         pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1 = pad

#         ctx.up_x = up_x
#         ctx.up_y = up_y
#         ctx.up_z = up_z
#         ctx.down_x = down_x
#         ctx.down_y = down_y
#         ctx.down_z = down_z
#         ctx.pad_x0 = pad_x0
#         ctx.pad_x1 = pad_x1
#         ctx.pad_y0 = pad_y0
#         ctx.pad_y1 = pad_y1
#         ctx.pad_z0 = pad_z0
#         ctx.pad_z1 = pad_z1
#         ctx.in_size = in_size
#         ctx.out_size = out_size

#         return grad_input

#     @staticmethod
#     def backward(ctx, gradgrad_input):
#         # Retrieve the saved kernel tensor
#         kernel, = ctx.saved_tensors

#         # Reshape gradgrad_input to match the input size from the forward pass
#         gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], ctx.in_size[4], 1)

#         # Apply the 3D upfirdn operation for the backward pass
#         gradgrad_out = upfirdn3d_op.upfirdn3d(
#             gradgrad_input,
#             kernel,
#             ctx.up_x,
#             ctx.up_y,
#             ctx.up_z,
#             ctx.down_x,
#             ctx.down_y,
#             ctx.down_z,
#             ctx.pad_x0,
#             ctx.pad_x1,
#             ctx.pad_y0,
#             ctx.pad_y1,
#             ctx.pad_z0,
#             ctx.pad_z1,
#         )

#         # Reshape the gradgrad_out tensor back to the original shape
#         gradgrad_out = gradgrad_out.view(
#             ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1], ctx.out_size[2]
#         )

#         return gradgrad_out, None, None, None, None, None, None, None, None




# class UpFirDn3d(Function):
#     @staticmethod
#     def forward(ctx, input, kernel, up, down, pad):
#         # Unpack up, down, and pad dimensions
#         up_x, up_y, up_z = up
#         down_x, down_y, down_z = down
#         pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1 = pad

#         kernel_d, kernel_h, kernel_w = kernel.shape  # 3D kernel dimensions
#         batch, channel, in_d, in_h, in_w = input.shape
#         ctx.in_size = input.shape

#         # Reshape input for the operation
#         input = input.reshape(-1, in_d, in_h, in_w, 1)

#         # Save kernel and its flipped version for backward
#         ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1, 2]))  # Flip along depth, height, and width

#         # Calculate output dimensions
#         out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d + down_z) // down_z
#         out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
#         out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
#         ctx.out_size = (out_d, out_h, out_w)

#         # Save up, down, and pad factors in the context
#         ctx.up = (up_x, up_y, up_z)
#         ctx.down = (down_x, down_y, down_z)
#         ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1)

#         # Calculate gradients for padding
#         g_pad_x0 = kernel_w - pad_x0 - 1
#         g_pad_y0 = kernel_h - pad_y0 - 1
#         g_pad_z0 = kernel_d - pad_z0 - 1
#         g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
#         g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
#         g_pad_z1 = in_d * up_z - out_d * down_z + pad_z0 - up_z + 1

#         ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1, g_pad_z0, g_pad_z1)

#         # Perform the 3D upfirdn operation
#         out = upfirdn3d_op.upfirdn3d(
#             input, kernel, up_x, up_y, up_z, down_x, down_y, down_z,
#             pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1
#         )
        
#         # Reshape the output tensor to the correct shape
#         out = out.view(-1, channel, out_d, out_h, out_w)

#         return out

    # @staticmethod
    # def backward(ctx, grad_output):
    #     kernel, grad_kernel = ctx.saved_tensors

    #     grad_input = None

    #     if ctx.needs_input_grad[0]:
    #         grad_input = UpFirDn3dBackward.apply(
    #             grad_output,
    #             kernel,
    #             grad_kernel,
    #             ctx.up,
    #             ctx.down,
    #             ctx.pad,
    #             ctx.g_pad,
    #             ctx.in_size,
    #             ctx.out_size,
    #         )

    #     return grad_input, None, None, None, None


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
    # if input.device.type == "cpu":
    #     out = upfirdn3d_native(input, kernel, *up, *down, *pad)
    # else:
    #     out = UpFirDn3d.apply(input, kernel, up, down, pad)

    return out


# def upfirdn3d_native(
#     input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1
# ):
#     _, channel, in_d, in_h, in_w = input.shape
#     input = input.reshape(-1, in_d, in_h, in_w, 1)


#     _, in_d, in_h, in_w, minor = input.shape
#     kernel_d, kernel_h, kernel_w = kernel.shape

#     out = input.view(-1, in_d, 1, in_h, 1, in_w, 1, minor)
    
#     out = F.pad(out, [0, 0, 0, up_x-1, 0, 0, up_y - 1, 0, 0, up_z - 1])
#     out = out.view(-1, in_d * up_z, in_h * up_y, in_w * up_x, minor)
    
#     # Apply padding
#     out = F.pad(out, [0, 0, max(pad_z0, 0), max(pad_z1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_x0, 0), max(pad_x1, 0)])
    
#     out = out[
#         :, 
#         max(-pad_z0, 0): out.shape[1] - max(-pad_z1, 0),
#         max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
#         max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0),
#         :,
#     ]

#     # Permute to have the channel as the second dimension
#     out = out.permute(0, 4, 1, 2, 3)  # Move the last dimension to the front
#     out = out.reshape(
#         -1, 1, in_d * up_x + pad_x0 + pad_x1, in_h * up_y + pad_y0 + pad_y1, in_w * up_z + pad_z0 + pad_z1
#     )

#     # Prepare kernel for convolution
#     w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_d, kernel_h, kernel_w)

#     # Perform 3D convolution
#     out = F.conv3d(out, w)

#     out = out.reshape(
#         -1,
#         minor,
#         (in_d * up_x + pad_x0 + pad_x1 - kernel_d + 1),
#         (in_h * up_y + pad_y0 + pad_y1 - kernel_h+ 1),
#         (in_w * up_z + pad_z0 + pad_z1 - kernel_w + 1),
#     )
#     out = out.permute(0, 2, 3, 4, 1)

#     # Downsample the output
#     out = out[:, ::down_x, ::down_y, ::down_z, :]

#     # Calculate output size after downsampling
#     out_d = (in_d * up_x + pad_x0 + pad_x1 - kernel_d + down_x) // down_x
#     out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
#     out_w = (in_w * up_z + pad_z0 + pad_z1 - kernel_w + down_z) // down_z
#     out = out.view(-1, channel, out_d, out_h, out_w)
# #     print(out.shape)
#     return out

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

