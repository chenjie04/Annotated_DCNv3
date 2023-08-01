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

// ************************************************************************************************
// 
// DCN v3 cuda算子的具体实现
// 
// ************************************************************************************************

#include <vector>                       // 向量（Vector）是一个封装了动态大小数组的顺序容器（Sequence Container）。
#include <ATen/ATen.h>                  // ATen是一个用于张量操作的库，它提供了CPU和GPU的张量和其他基本张量操作的支持。
#include <ATen/cuda/CUDAContext.h>      // ATen CUDA上下文
#include <cuda.h>                       // CUDA头文件，定义了CUDA驱动API的public主机函数和类型。cuda开发中在cpu上运行的称为主机（host）代码，在gpu上运行的称为设备（device）代码。
#include <cuda_runtime.h>               // CUDA运行时API
#include <torch/torch.h>                // PyTorch C++前端API

#include "cuda/dcnv3_im2col_cuda.cuh" // DCN v3 im2col核函数,卷积操作的真正实现
#include "dcnv3_cuda.h"

at::Tensor dcnv3_cuda_forward(const at::Tensor &input,  // 输入张量 维度为[batch_size, height_in, width_in, channels]
                              const at::Tensor &offset, // 偏移量张量 维度为[batch_size, height_out, width_out, group * 2 * kernel_h * kernel_w]
                              const at::Tensor &mask,   // 掩码张量（ modulation scalar） 维度为[batch_size, height_out, width_out, group * kernel_h * kernel_w], 用于控制每个样本点的重要性影响
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
                              const int im2col_step)    // im2col的步长
{   
    // 检查input，offset，mask是否连续，是否在GPU上
    TORCH_INTERNAL_ASSERT(input.is_contiguous(), "input tensor has to be contiguous");
    TORCH_INTERNAL_ASSERT(offset.is_contiguous(), "offset tensor has to be contiguous");
    TORCH_INTERNAL_ASSERT(mask.is_contiguous(), "mask tensor has to be contiguous");
    TORCH_INTERNAL_ASSERT(input.device().is_cuda(), "input tensor has to be on GPU");
    TORCH_INTERNAL_ASSERT(offset.device().is_cuda(), "offset tensor has to be on GPU");
    TORCH_INTERNAL_ASSERT(mask.device().is_cuda(), "mask tensor has to be on GPU");

    // 获取输入张量的维度
    const int batch = input.size(0);        // batch_size
    const int height_in = input.size(1);       // height
    const int width_in = input.size(2);        // width
    const int channels = input.size(3);     // channels

    // 确定im2col的步长
    const int im2col_step_ = std::min(im2col_step, batch); 
    TORCH_INTERNAL_ASSERT(batch % im2col_step_ == 0, "batchsize (%d) must divide im2col step (%d)", batch, im2col_step_);
    TORCH_INTERNAL_ASSERT(channels == groups * group_channels, "input channels and groups * group_channels wont match: (%d vs %d)", channels, groups * group_channels);

    // 计算输出张量的维度
    const int height_out = (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;   // 输出的高度
    const int width_out = (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;    // 输出的宽度

    // 构建输出张量，维度为[batch_size, height_out, width_out, groups * group_channels]
    // auto用作类型推导，可以根据等号右边的表达式推导出等号左边的变量类型
    // inline at::Tensor torch::zeros(at::IntArrayRef size, at::TensorOptions options = {})
    auto output = at::zeros({batch, height_out, width_out, groups * group_channels}, input.options()); 

    // im2col_step_的目的是为了加速im2col这块的速度
    // im2col核函数一次只处理一个样本，因此在一个batch中就行需要调用batch_size次im2col核函数，
    // 在dcnv2中为了加速im2col，引入im2col_step_，将输入拆分成[batch_size/im2col_step_, im2col_step_, height, width, channels]，
    // 将[im2col_step_, height, width, channels]作为一个样本处理，这样就可以减少调用im2col核函数的次数，具体参考下面for循环调用im2col核函数的代码
    const int batch_n = im2col_step_;
    auto output_n = output.view({batch / batch_n, batch_n, height_out, width_out, groups * group_channels}); //将输出张量维度做相应维度调整

    // 因为后面要通过指针移动来获取每个样本的输入，偏移量，掩码，输出，因此需要计算每个样本的大小
    auto per_input_size = height_in * width_in * groups * group_channels; // 每个输入样本的大小
    auto per_offset_size = height_out * width_out * groups * kernel_h * kernel_w * 2; // 每个偏移量的大小
    auto per_mask_size = height_out * width_out * groups * kernel_h * kernel_w; // 每个掩码的大小

    // 循环调用im2col核函数处理每个样本，完成卷积操作
    for (int n = 0; n < batch / im2col_step_; ++n)
    {
        // at::Tensor Tensor::select(int64_t dim, int64_t index): 选择张量的一个子张量，dim为选择的维度，index为选择的索引
        auto columns = output_n.select(0, n); // 选择第n个样本对应的输出张量

        // AT_DISPATCH_FLOATING_TYPES_AND_HALF 包装了一个 switch 语句来完成针对张量类型的分派
        // 该宏接受三个参数：
        // 第一个为数据类型可以通过输入tensor调用.type()得到，
        // 第二个为一个字符串，没什么特别要求，
        // 第三个参数接受一个匿名函数，调用cuda kernel 函数逻辑都在该匿名函数里实现。

        // C++11提供了对匿名函数的支持,称为Lambda函数(也叫Lambda表达式). Lambda表达式具体形式如下:
        // [capture](parameters)->return-type{body}
        // 如果没有参数,空的圆括号()可以省略.返回值也可以省略,如果函数体只由一条return语句组成或返回类型为void的话.形如:
        // [capture](parameters){body}
        // C++的匿名函数请参考：https://www.cnblogs.com/pzhfei/archive/2013/01/14/lambda_expression.html

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), 
                                            "ms_deformable_convolution_forward_cuda",
                                            // [&]表示用到的任何外部变量都隐式按引用捕获，因此可以访问input，offset,mask,也可以改变columns
                                            ([&] 
                                            {
                                                dcnv3_im2col_cuda(
                                                    // CUDA流能封装一系列异步CUDA操作，比如我们常见的套路，在主机端分配设备主存（cudaMalloc），
                                                    // 主机向设备传输数据（cudaMemcpy），核函数启动，复制数据回主机（Memcpy）等，
                                                    // 这些操作中有些是异步的，执行顺序也是按照主机代码中的顺序执行的（但是异步操作的结束不一定是按照代码中的顺序的），
                                                    // 流能封装这些异步操作，并保持操作顺序，允许操作在流中排队。保证其在前面所有操作启动之后启动
                                                    at::cuda::getCurrentCUDAStream(),                               // 获取当前CUDA流
                                                    // input.data<scalar_t>()返回一个指向input张量数据的指针，scalar_t为输入张量的数据类型
                                                    input.data_ptr<scalar_t>() + n * im2col_step_ * per_input_size,     // 选择第n个样本对应的输入张量
                                                    offset.data_ptr<scalar_t>() + n * im2col_step_ * per_offset_size,   // 选择第n个样本对应的偏移量张量
                                                    mask.data_ptr<scalar_t>() + n * im2col_step_ * per_mask_size,       // 选择第n个样本对应的掩码张量
                                                    columns.data_ptr<scalar_t>(),                                       // 第n个样本对应的输出张量
                                                    kernel_h, kernel_w,
                                                    stride_h, stride_w,
                                                    pad_h, pad_w,
                                                    dilation_h, dilation_w,
                                                    groups, group_channels,
                                                    batch_n,
                                                    height_in, width_in,
                                                    height_out, width_out,
                                                    offset_scale);
                                            }));
    }

    return output;
}


std::vector<at::Tensor> dcnv3_cuda_backward(const at::Tensor &input,        // forward()函数的输入: 输入张量 维度为[batch_size, height_in, width_in, channels]
                                            const at::Tensor &offset,       // forward()函数的输入: 偏移量张量 维度为[batch_size, height_out, width_out, group * 2 * kernel_h * kernel_w]
                                            const at::Tensor &mask,         // forward()函数的输入: 掩码张量 维度为[batch_size, height_out, width_out, group * kernel_h * kernel_w]
                                            const at::Tensor &grad_output,  //反向传播上一级计算得到的梯度值
                                            const int kernel_h, const int kernel_w, 
                                            const int stride_h, const int stride_w, 
                                            const int pad_h, const int pad_w, 
                                            const int dilation_h, const int dilation_w, 
                                            const int groups, const int group_channels, 
                                            const float offset_scale, 
                                            const int im2col_step)
{
    TORCH_INTERNAL_ASSERT(input.is_contiguous(), "input tensor has to be contiguous");
    TORCH_INTERNAL_ASSERT(offset.is_contiguous(), "offset tensor has to be contiguous");
    TORCH_INTERNAL_ASSERT(mask.is_contiguous(), "mask tensor has to be contiguous");
    TORCH_INTERNAL_ASSERT(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    TORCH_INTERNAL_ASSERT(input.device().is_cuda(), "input tensor has to be on GPU");
    TORCH_INTERNAL_ASSERT(offset.device().is_cuda(), "offset tensor has to be on GPU");
    TORCH_INTERNAL_ASSERT(mask.device().is_cuda(), "mask tensor has to be on GPU");
    TORCH_INTERNAL_ASSERT(grad_output.device().is_cuda(), "grad_output tensor has to be on GPU");

    const int batch = input.size(0);        // batch_size
    const int height_in = input.size(1);       // height
    const int width_in = input.size(2);        // width
    const int channels = input.size(3);     // channels
    TORCH_INTERNAL_ASSERT(channels == groups * group_channels, "input channels and groups * group_channels wont match: (%d vs %d)", channels, groups * group_channels);


    const int height_out = (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;   // 输出的高度
    const int width_out = (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;    // 输出的宽度

    const int im2col_step_ = std::min(im2col_step, batch);
    TORCH_INTERNAL_ASSERT(batch % im2col_step_ == 0, "batchsize (%d) must divide im2col step (%d)", batch, im2col_step_);
    const int batch_n = im2col_step_;
    auto grad_output_n = grad_output.view({batch / batch_n, batch_n, height_out * width_out, groups, group_channels});  // 注意这里的维度调整,与forward()函数中的output_n的维度调整是不一样的

    auto per_input_size = height_in * width_in * groups * group_channels; // 每个输入样本的大小
    auto per_offset_size = height_out * width_out * groups * kernel_h * kernel_w * 2; // 每个偏移量的大小
    auto per_mask_size = height_out * width_out * groups * kernel_h * kernel_w; // 每个掩码的大小

    // 初始化要返回的梯度张量: grad_input, grad_offset, grad_mask
    // ****************************************************************************************************************************************************************
    auto dtype = input.dtype();

    if (dtype == at::kHalf)
    {
        dtype = at::kFloat;
    }

    auto grad_input = at::zeros_like(input, dtype); 
    auto grad_offset = at::zeros_like(offset, dtype); 
    auto grad_mask = at::zeros_like(mask, dtype); 
    // ****************************************************************************************************************************************************************

    for (int n = 0; n < batch / im2col_step_; ++n)
    {
        auto grad_output_n_ = grad_output_n.select(0, n); // 选择第n个样本
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
                                            "ms_deformable_convolution_backward_cuda",
                                            ([&] {
                                                dcnv3_col2im_cuda(
                                                        at::cuda::getCurrentCUDAStream(),
                                                        grad_output_n_.data_ptr<scalar_t>(), // 第n个样本对应的反向传播上一级计算得到的梯度值
                                                        input.data_ptr<scalar_t>() + n * im2col_step_ * per_input_size, // 第n个样本对应的forward()函数输入张量
                                                        offset.data_ptr<scalar_t>() + n * im2col_step_ * per_offset_size, // 第n个样本对应的forward()函数偏移量张量
                                                        mask.data_ptr<scalar_t>() + n * im2col_step_ * per_mask_size, // 第n个样本对应的forward()函数掩码张量
                                                        kernel_h, kernel_w,
                                                        stride_h, stride_w,
                                                        pad_h, pad_w,
                                                        dilation_h, dilation_w,
                                                        groups, group_channels,
                                                        batch_n,
                                                        height_in, width_in,
                                                        height_out, width_out,
                                                        offset_scale,
                                                        grad_input.data_ptr<opmath_t>() + n * im2col_step_ * per_input_size, // 第n个样本的input梯度张量
                                                        grad_offset.data_ptr<opmath_t>() + n * im2col_step_ * per_offset_size, // 第n个样本的offset梯度张量
                                                        grad_mask.data_ptr<opmath_t>() + n * im2col_step_ * per_mask_size // 第n个样本的mask梯度张量
                                                    );
                                                }
                                            )
                                            );
    }

    if (input.dtype() == torch::kHalf)
    {
        return {grad_input.to(torch::kHalf), grad_offset.to(torch::kHalf), grad_mask.to(torch::kHalf)};
    } else 
    {
        return {grad_input, grad_offset, grad_mask};
    }
}