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
//  cuda算子的头文件，声明算子的函数接口
//
// ***********************************************************************************************

#pragma once                 // 在头文件的最开始加入这条指令，能够保证头文件只被编译一次
#include <torch/extension.h> //pytorch的头文件,这个头文件里面包含很多重要的模块。如用于python和C++11交互的pybind11，以及包含Tensor的一系列定义操作等，导入后就可以在C++ 中调用所有PyTorch 支持的功能。

// ***********************************************************************************************
// at即ATen，Pytorch的C++前端，是一个支持张量运算和自动求导的库，是Pytorch的核心库。
// ::是作用域运算符，如：A,B表示两个类，在A,B中都有成员member。那么A::member就表示类A中的成员member，B::member就表示类B中的成员member
// at::Tensor 表示张量，是Pytorch中的基本数据类型，是一个多维数组，可以用来存储和变换数据，
// 在这里表示算子的forward函数返回一个张量。
// const 表示常量，不可修改
// ***********************************************************************************************
at::Tensor dcnv3_cuda_forward(const at::Tensor &input,  // 输入张量 维度为[batch_size, height_in, width_in, channels]
                              const at::Tensor &offset, // 偏移量张量 维度为[batch_size, height_out, width_out, 2 * kernel_h * kernel_w]
                              const at::Tensor &mask,   // 掩码张量（ modulation scalar） 维度为[batch_size, height_out, width_out, kernel_h * kernel_w], 用于控制每个样本点的重要性影响
                              const int kernel_h,       // 卷积核
                              const int kernel_w,
                              const int stride_h,       // 步长
                              const int stride_w,
                              const int pad_h,          // 填充
                              const int pad_w,
                              const int dilation_h,     // 空洞卷积的膨胀率
                              const int dilation_w,
                              const int groups,         // DCN_v3采用分组卷积的方式，将输入的通道数分为groups组，每组group_channels个通道
                              const int group_channels,
                              const float offset_scale, // 偏移量的缩放因子
                              const int im2col_step);   // im2col的步长

// ***********************************************************************************************
// 反向传播函数 dcnv3_cuda_backward()的功能是:
// 根据输入函数dcnv3_cuda_forward()函数的输入input,offset,mask,以及它的输出grad_output，
// 计算出input,offset,mask对应的梯度grad_input, grad_offset, grad_mask，并返回
// 当然,也需要kernel_h,kernel_w,stride_h,...,offset_scale,im2col_step等这些参数辅助计算,一并传入
// 
// std::vector 是 C++ 标准库中的一个动态数组容器，可以方便地动态添加、删除元素，而无需手动管理内存。
// 在这里表示算子的backward函数返回一个存储 tensor 的 vector 容器
// ***********************************************************************************************
std::vector<at::Tensor> dcnv3_cuda_backward(const at::Tensor &input,
                                            const at::Tensor &offset,
                                            const at::Tensor &mask,
                                            const at::Tensor &grad_output, // 与forward函数相比多了一个&grad_output参数，这正是反向传播上一级计算得到的梯度值
                                            const int kernel_h,
                                            const int kernel_w,
                                            const int stride_h,
                                            const int stride_w,
                                            const int pad_h,
                                            const int pad_w,
                                            const int dilation_h,
                                            const int dilation_w,
                                            const int groups,
                                            const int group_channels,
                                            const float offset_scale,
                                            const int im2col_step);