# ----------------------------------------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
#
# 为了让自定义算子能够正常正向传播、反向传播，我们需要继承torch.autograd.Function进行算子包装
# ----------------------------------------------------------------------------------------

from typing import Any
import DCNv3

import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd


class DCNv3Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        offset,
        mask,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        group_channels,
        offset_scale,
        im2col_step,
    ):
        # 常量保存到ctx中，以便反向传播使用
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.groups = groups
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step

        output = DCNv3.dcnv3_forward(
            input,
            offset,
            mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            group_channels,
            offset_scale,
            im2col_step,
        )

        # 张量的保存方式与常量不同
        ctx.save_for_backward(input, offset, mask)

        return output

    @staticmethod
    # If you wrap your 's method with this wrapper, the Tensors that you will get as input will never require gradients and you don’t have to write a backward function that computes the gradients in a differentiable manner. For example, you can use other libraries to do the computation. If you try to backward through the backward pass of such Function, an error will be raised stating that this Function is only differentiable once.Functionbackward
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors
        grad_input, grad_offset, grad_mask = DCNv3.dcnv3_backward(
            input,
            offset,
            mask,
            grad_output.contiguous(),
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_h,
            ctx.stride_w,
            ctx.pad_h,
            ctx.pad_w,
            ctx.dilation_h,
            ctx.dilation_w,
            ctx.groups,
            ctx.group_channels,
            ctx.offset_scale,
            ctx.im2col_step,
        )

        return (
            grad_input,
            grad_offset,
            grad_mask,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def symbolic(
        g,
        input,
        offset,
        mask,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_channels,
        offset_scale,
        im2col_step,
    ):
        """Symbolic function for mmdeploy::DCNv3.

        Returns:
            DCNv3 op for onnx.
        """
        return g.op(
            "mmdeploy::TRTDCNv3",
            input,
            offset,
            mask,
            kernel_h_i=int(kernel_h),
            kernel_w_i=int(kernel_w),
            stride_h_i=int(stride_h),
            stride_w_i=int(stride_w),
            pad_h_i=int(pad_h),
            pad_w_i=int(pad_w),
            dilation_h_i=int(dilation_h),
            dilation_w_i=int(dilation_w),
            group_i=int(group),
            group_channels_i=int(group_channels),
            offset_scale_f=float(offset_scale),
            im2col_step_i=int(im2col_step),
        )

class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )

    return (n & (n - 1) == 0) and n != 0