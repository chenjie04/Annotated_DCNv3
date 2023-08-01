/*!
**************************************************************************************************
* InternImage
* Copyright (c) 2022 OpenGVLab
* Licensed under The MIT License [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

// ***********************************************************************************************
//
//  cuda算子的C++封装，编写包装函数并调用PYBIND11_MODULE对算子进行封装。
//
// ***********************************************************************************************

// 请注意，setuptools不能处理具有相同名称但扩展名不同的文件，则必须为CUDA文件提供与c++文件不同的名称

#pragma once
#include <torch/extension.h>

#ifdef WITH_CUDA
#include "cuda/dcnv3_cuda.h"
#endif

at::Tensor dcnv3_forward(const at::Tensor &input,
                         const at::Tensor &offset,
                         const at::Tensor &mask,
                         const int kernel_h, const int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int groups, const int group_channels,
                         const float offset_scale,
                         const int im2col_step)
{
    if (input.device().is_cuda())
    {
#ifdef WITH_CUDA
        return dcnv3_cuda_forward(input,
                                  offset,
                                  mask,
                                  kernel_h, kernel_w,
                                  stride_h, stride_w,
                                  pad_h, pad_w,
                                  dilation_h, dilation_w,
                                  groups, group_channels,
                                  offset_scale,
                                  im2col_step);
#else
        TORCH_INTERNAL_ASSERT("Not compiled with GPU support");
#endif
    }
    TORCH_INTERNAL_ASSERT("Not implemented on the CPU");
    return at::Tensor();
}

std::vector<at::Tensor> dcnv3_backward(const at::Tensor &input,
                                       const at::Tensor &offset,
                                       const at::Tensor &mask,
                                       const at::Tensor &grad_output,
                                       const int kernel_h, const int kernel_w,
                                       const int stride_h, const int stride_w,
                                       const int pad_h, const int pad_w,
                                       const int dilation_h, const int dilation_w,
                                       const int groups, const int group_channels,
                                       const float offset_scale,
                                       const int im2col_step)
{
    if (input.device().is_cuda())
    {
#ifdef WITH_CUDA
        return dcnv3_cuda_backward(input,
                                   offset,
                                   mask,
                                   grad_output,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   groups, group_channels,
                                   offset_scale,
                                   im2col_step);
#else
        TORCH_INTERNAL_ASSERT("Not compiled with GPU support");
#endif
    }
    TORCH_INTERNAL_ASSERT("Not implemented on the CPU");

    return std::vector<at::Tensor>();
}