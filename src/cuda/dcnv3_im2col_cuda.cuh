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
#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh> // Torch CUDA张量计算库

// 这里都是模板函数，由于输入张量可能是float或者double，甚至是half类型，所以需要模板化

// CUDA_KERNEL_LOOP(i, n)表示线程数大于当前grid开启上限时，一直在block中循环线程计算直到完成任务。后面会传入参数实例化；
// 当前开辟的所有线程数是blockDim.x * gridDim.x ；
// 当需要并行的任务总数超过了当前开辟的所有线程数时，可以让线程循环的完成任务。一种常见的用法；
// 比如，一共开辟了5*2共十个线程，一共有30个任务，0号线程在干完任务0后，可以继续干任务0+10，之后可以继续干任务0+10+10；
// 同理1号线程可以按顺序去做任务1,11,21。
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
inline int GET_BLOCKS(const int N, const int num_threads)
{
    // 这里采用一维网格、一维块的执行配置方案。
    // 每个线程块中有num_threads个线程，N是需要启动的线程总数。
    // 为了保证每个任务都能分配到一个线程，这里采用了向上取整的方式，即如果N不能被num_threads整除，那么就多启动一个线程块。
    return (N + num_threads - 1) / num_threads;
}

#define opmath_t at::opmath_type<scalar_t>

// ******************************************************************************************************************************************************************************
// forward()函数的相关计算

// *&的类型是对指针的引用如果传一个指针a的值给指针类型变量p，
// 例如int*p =a那么改变*p的值，*a的值会发生变化；但改变p的值，a的值不会改变
// 如果传一个指针a的值给指针引用类型变量p，例如int*& p=a那么不仅改变*p的值，*a的值会发生变化；改变p本身的值，a的值也会发生改变。
// 引用类型相当于给变量起了别名，例如int &a=b，a是一个整形引用类型变量，也相当于b的别名。那么改变b的值，a的值也会改变。

template <typename scalar_t>
__device__ opmath_t dcnv3_im2col_bilinear(const scalar_t *&bottom_data, // 输入张量的起始指针, *&表示引用传递
                                          const int &height, const int &width,
                                          const int &groups, const int &group_channels,
                                          const opmath_t &h, const opmath_t &w,
                                          const int &g, const int &c)
{
    /*
    ***********************************************************************************************************************
    下面的计算是为了方便理解, 但是存在很多计算冗余，会极大地拖慢我们的训练过程，所以在实际的代码中，我们会后面优化过的代码


    ***********************************************************************************************************************
    * 一定要参照 https://chenjie04.github.io/post/shuang-xian-xing-cha-zhi-fa-de-zhi-guan-li-jie/  理解双线性插值法
    * 特别理解为什么加权的权重是这样计算的
    *
    * 双线性插值法的计算公式：
    *    f(x, y) = f(Q11)w1 + f(Q21)w2 + f(Q12)w3 + f(Q22)w4
    *
    * *********************************************************************************************************************

    // batom_data是输入张量的起始指针的引用，这里张量的维度是[height_in, width_in, groups * group_channels]

    // h和w是当前像素点加上偏移量之后的坐标
    // 下面找出当前像素点周围的四个像素点的坐标
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    // Q11(h_low, w_low)是左下角的像素点坐标，Q22(h_high, w_high)是右上角的像素点坐标，Q12(h_low, w_high)是右下角的像素点坐标，Q21(h_high, w_low)是左上角的像素点坐标

    // 计算当前像素点周围四个像素点的权重元素(用来计算权重的量)
    const opmath_t lh = h - h_low;
    const opmath_t lw = w - w_low;
    const opmath_t hh = 1 - lh;
    const opmath_t hw = 1 - lw;
    // 根据权重元素计算四个点的权重
    const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;



    // 计算当前像素点周围四个像素点的索引（全局一维索引）并取得对应的像素值
    // 输入张量维度是[height_in, width_in, groups * group_channels]
    // 根据高维张量的索引计算公式：index = h * width * groups * group_channels + w * groups * group_channels + g * group_channels + c
    h_stride = width * groups * group_channels; // 输入张量的高度方向的步长
    w_stride = groups * group_channels;         // 输入张量的宽度方向的步长
    base_ptr = g * group_channels + c;          // 最后一个维度(groups * group_channels)的索引

    // 左下角
    opmath_t v1 = 0 // 初始化左下角像素点的像素值为0
    if (h_low >= 0 && w_low >= 0) // 检查左下角像素点的坐标是否在输入张量的范围内
    {
        v1 = bottom_data[h_low * h_stride + w_low * w_stride + base_ptr]; // 取得左下角像素点的像素值
    }

    // 左上角
    opmath_t v2 = 0 // 初始化右下角像素点的像素值为0
    if (h_low >= 0 && w_high <= width - 1) // 检查右下角像素点的坐标是否在输入张量的范围内
    {
        v2 = bottom_data[h_low * h_stride + w_high * w_stride + base_ptr]; // 取得右下角像素点的像素值
    }

    // 右下角
    opmath_t v3 = 0 // 初始化左上角像素点的像素值为0
    if (h_high <= height -1 && w_low >= 0) // 检查左上角像素点的坐标是否在输入张量的范围内
    {
        v3 = bottom_data[h_high * h_stride + w_low * w_stride + base_ptr]; // 取得左上角像素点的像素值
    }

    // 右上角
    opmath_t v4 = 0 // 初始化右上角像素点的像素值为0
    if (h_high <= height -1 && w_high <= width - 1) // 检查右上角像素点的坐标是否在输入张量的范围内
    {
        v4 = bottom_data[h_high * h_stride + w_high * w_stride + base_ptr]; // 取得右上角像素点的像素值
    }

    // 计算当前像素点的像素值
    const opmath_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;

    ***********************************************************************************************************************
    */

    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const opmath_t lh = h - h_low;
    const opmath_t lw = w - w_low;
    const opmath_t hh = 1 - lh, hw = 1 - lw;

    const int w_stride = groups * group_channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = g * group_channels + c;

    opmath_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    opmath_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    opmath_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    opmath_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }
    const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const opmath_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

// 这是真正实现卷积的和函数
// __global__表示这是一个CUDA核函数，在主机上调用，在设备上运行
template <typename scalar_t>
__global__ void dcnv3_im2col_gpu_kernel(const int num_kernels,
                                        const scalar_t *data_im,     // 输入张量的起始指针
                                        const scalar_t *data_offset, // 偏移量的起始指针
                                        const scalar_t *data_mask,   // 掩码的起始指针
                                        scalar_t *data_col,          // 输出张量的起始指针
                                        const int kernel_h, const int kernel_w,
                                        const int stride_h, const int stride_w,
                                        const int pad_h, const int pad_w,
                                        const int dilation_h, const int dilation_w,
                                        const int groups, const int group_channels,
                                        const int batch_n,
                                        const int height_in, const int width_in,
                                        const int height_out, const int width_out,
                                        const opmath_t offset_scale)
{
    CUDA_KERNEL_LOOP(index, num_kernels)
    {
        // num_kernels = batch_n * height_out * width_out * groups * group_channels;
        // 相当于把输出张量一维展开了，下面的代码就是把一维的index还原成多维的索引
        // 整不明白啊！！！！

        /*
        ***********************************************************************************************************************
        * 对于一个高维张量：[batch_n, height_out, width_out, groups, group_channels]，
        * 假设某个像素点的索引是[b, h, w, g, c]，那么它的一维索引为：
        * index = b * height_out * width_out * groups * group_channels
        *         + h * width_out * groups * group_channels
        *         + w * groups * group_channels
        *         + g * group_channels
        *         + c
        ***********************************************************************************************************************
        */

        int _tmp = index;                        // 临时变量，用于计算当前线程处理的像素点的位置，也就是在输出张量中的一维位置索引
        const int c_col = _tmp % group_channels; // 通道索引

        _tmp /= group_channels;          // 除以通道数，相当于去掉通道维度
        const int sampling_index = _tmp; // sampling_index = b * height_out * width_out * groups + h * width_out * groups + w * groups + g
                                         // 因为是分组卷积，所以这相当于空间位置的坐标索引，
                                         // 假如是在RGB图像的三维张量[Height, Width, Channel]中, 其中某个像素点的索引是[h, w, c]，
                                         // 在上一步除以Channel之后，相当于去掉了通道维度，那么索引就变成了[h, w]，

        const int g_col = _tmp % groups; // 组索引

        _tmp /= groups; // 除以组数，相当于去掉组维度

        // 根据当前像素点在输出张量的坐标反求其在输入张量上的卷积中心：
        //
        // (_tmp % width_out) 是当前像素点在输出张量的 w 坐标，我们需要据此求出卷积中心在输入张量中对应的 w 坐标
        // ((dilation_w * (kernel_w - 1)) >> 1) 是实际感受野大小的一半，
        // 假如当前像素点在输出张量的 w 坐标为 0 ，那么感受野大小的一半还要减去 pad_w 才是卷积中心的 w 坐标，
        // 那么，当输出张量上的 w 坐标增 1，输入张量上卷积中心的 w 坐标就要增加 stride_w 个像素点，
        // 结合 https://chenjie04.github.io/post/juan-ji-shu-chu-da-xiao-de-ji-suan-xiang-jie/ 中的图示，可以简单明了地理解这个过程
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_tmp % width_out) * stride_w;
        _tmp /= width_out; // 除以宽度，相当于去掉宽度维度
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_tmp % height_out) * stride_h;
        _tmp /= height_out; // 除以高度，相当于去掉高度维度

        const int b_col = _tmp; // 批次索引

        // 开始定位当前像素点在各个张量对应的位置
        // **********************************************************************************************************************************************************
        scalar_t *data_col_ptr = data_col + index; // 当前像素点在输出张量的指针

        const int input_size = height_in * width_in;                             // 输入张量的像素点数量
        const int qid_stride = groups * group_channels;                          // 就是原始通道数
        const scalar_t *data_im_ptr = data_im + b_col * input_size * qid_stride; // 当前像素对应输入张量样本的起始指针

        const int kernel_size = kernel_h * kernel_w;        // 卷积核的大小
        int data_weight_ptr = sampling_index * kernel_size; // mask权重索引
                                                            // sampling_index * kernel_size = b * height_out * width_out * groups * (kernel_h * kernel_w)
                                                            //                                + h * width_out * groups * (kernel_h * kernel_w)
                                                            //                                + w * groups * (kernel_h * kernel_w)
                                                            //                                + g * (kernel_h * kernel_w)

        int data_loc_w_ptr = data_weight_ptr << 1; // offset索引。'<< 1'左移一位，相当于乘以2
                                                   // sampling_index * kernel_size * 2 = b * height_out * width_out * groups * (kernel_h * kernel_w) * 2
                                                   //                                    + h * width_out * groups * (kernel_h * kernel_w) * 2
                                                   //                                    + w * groups * (kernel_h * kernel_w) * 2
                                                   //                                    + g * (kernel_h * kernel_w) * 2
        // **************************************************************************************************************************************************************

        // 卷积在输入张量上的左上角坐标
        // 已知卷积中心在输入张量上的坐标为 (p0_w, p0_h)，
        // 那么卷积在输入张量上的左上角坐标为 (p0_w - ((dilation_w * (kernel_w - 1)) >> 1), p0_h - ((dilation_h * (kernel_h - 1)) >> 1))
        const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        opmath_t col = 0; // 输出张量的像素点的值初始化为0
        // 从左上角开始遍历卷积核
        for (int i = 0; i < kernel_w; i++)
        {
            for (int j = 0; j < kernel_h; j++)
            {
                // 获取在输入张量上当前像素点的偏移量
                const opmath_t offset_w = data_offset[data_loc_w_ptr];
                const opmath_t offset_h = data_offset[data_loc_w_ptr + 1];

                // 在默认offset_scale为1.0的情况下，p0_w_ + i * dilation_w 是当前位置， 加上offset_w是偏移后的位置
                const opmath_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                const opmath_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;

                // 获取当前像素点的权重
                const opmath_t weight = data_mask[data_weight_ptr]; // 掩码权重，也就是公式中的 m_gk

                if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in) // 检查偏移后的位置是否在输入张量的范围内
                {
                    // 利用双线性插值法计算偏移后的位置的像素值，然后乘上权重，最后累加到col上
                    col += weight * dcnv3_im2col_bilinear(data_im_ptr, height_in, width_in, groups, group_channels, loc_h, loc_w, g_col, c_col);
                }
                data_weight_ptr += 1; // 权重索引加1
                data_loc_w_ptr += 2;  // 偏移量索引得加2
            }
        }
        *data_col_ptr = col; // 将计算得到的像素值赋值给输出张量对应位置的像素点
    }
}

// ******************************************************************************************************************************************************************************
// * dcnv3_cuda_forward()函数入口
// ******************************************************************************************************************************************************************************
template <typename scalar_t>
void dcnv3_im2col_cuda(cudaStream_t stream,
                       const scalar_t *data_im,     // 指向输入张量的初始位置，输出张量的维度应该是[batch_n, height_in, width_in, groups * group_channels]
                       const scalar_t *data_offset, // 偏移量指针，偏移张量的维度应该是[batch_n, height_out, width_out, 2 * kernel_h * kernel_w * groups]
                       const scalar_t *data_mask,   // 掩码指针，掩码张量的维度应该是[batch_n, height_out, width_out, kernel_h * kernel_w * groups]
                       scalar_t *data_col,          // 输出张量，输出张量的维度应该是batch_n, height_out, width_out, groups * group_channels]
                       const int kernel_h, const int kernel_w,
                       const int stride_h, const int stride_w,
                       const int pad_h, const int pad_w,
                       const int dilation_h, const int dilation_w,
                       const int groups, const int group_channels,
                       const int batch_n,
                       const int height_in, const int width_in,
                       const int height_out, const int width_out,
                       const opmath_t offset_scale) // 偏移量缩放系数，默认为1.0（即不缩放），索引我还不清楚它是怎么用的
{
    const int num_kernels = batch_n * height_out * width_out * groups * group_channels; // 实际上这是输出张量所有像素点的数量
    const int num_actual_kernels = batch_n * height_out * width_out * groups * group_channels;
    const int num_threads = CUDA_NUM_THREADS;

    // 这里的num_kernels是输出张量的像素点数量，也就是说，每个线程处理一个像素点
    //
    // CUDA核函数调用，<<<dim_grid, dim_block, Ns, stream>>>是核函数并行线程执行配置，
    // 这里的dim_grid是网格维度, 可以是一维、二维或三维，指明有多少线程块以及线程块怎么排列；dim_block是线程块的维度，知名每个线程块的线程数量以及排列方式；
    // Ns：size_t类型，指明共享内存的大小，可选项，默认0；stream是CUDA流，cudaStream_t类型（实质为CUstream_st *），可选项，默认0。
    //
    // 根据上面GET_BLOCKS宏定义，这里采用1维的网格，网格里有(num_actual_kernels + num_threads - 1) / num_threads 个线程块，线程块也是1维，每个线程块有num_threads个线程
    // 函数中可以使用 scalar_t 代指目标类型。而 ATEN 支持我们使用 Tensor.data<类型> 将 Tensor.data 转换为某个类型。
    dcnv3_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                               data_im,
                                                                                                               data_offset,
                                                                                                               data_mask,
                                                                                                               data_col,
                                                                                                               kernel_h, kernel_w,
                                                                                                               stride_h, stride_w,
                                                                                                               pad_h, pad_w,
                                                                                                               dilation_h, dilation_w,
                                                                                                               groups, group_channels,
                                                                                                               batch_n,
                                                                                                               height_in, width_in,
                                                                                                               height_out, width_out,
                                                                                                               offset_scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in dcnv3_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}

// ******************************************************************************************************************************************************************************

// ******************************************************************************************************************************************************************************
// backward()函数的相关计算

// 双线性插值法的反向传播
template <typename scalar_t>
__device__ void dcnv3_col2im_bilinear(const scalar_t *&bottom_data,
                                      const int &height, const int &width,
                                      const int &nheads, const int &group_channels,
                                      const opmath_t &h, const opmath_t &w,
                                      const int &m, const int &c,
                                      const opmath_t offset_scale,
                                      const opmath_t &top_grad,
                                      const opmath_t &mask,
                                      opmath_t *grad_im,
                                      opmath_t *grad_offset,
                                      opmath_t *grad_mask)
{
    /*
    ***********************************************************************************************************************
    * 双线性插值法的计算公式：
    *    f(x, y) = f(Q11)w1 + f(Q21)w2 + f(Q12)w3 + f(Q22)w4
    * 反向传播的计算公式：
    */

    // 周围四个像素点的坐标元素
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    // 周围四个像素点的权重元素
    const opmath_t lh = h - h_low;
    const opmath_t lw = w - w_low;
    const opmath_t hh = 1 - lh;
    const opmath_t hw = 1 - lw;

    // 定位周围四个像素点在输入张量上的一维索引元素
    const int w_stride = nheads * group_channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * group_channels + c;

    // 根据权重元素计算四个点的权重
    const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const opmath_t top_grad_im = top_grad * mask;
    opmath_t grad_h_weight = 0, grad_w_weight = 0;

    // 左下角
    opmath_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
        grad_h_weight -= v1 * hw;
        grad_w_weight -= v1 * hh;
        atomicAdd(grad_im + ptr1, top_grad_im * w1);
    }

    // 左上角
    opmath_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
        grad_h_weight -= v2 * lw;
        grad_w_weight += v2 * hh;
        atomicAdd(grad_im + ptr2, top_grad_im * w2);
    }

    // 右下角
    opmath_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
        grad_h_weight += v3 * hw;
        grad_w_weight -= v3 * lh;
        atomicAdd(grad_im + ptr3, top_grad_im * w3);
    }

    // 右上角
    opmath_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
        grad_h_weight += v4 * lw;
        grad_w_weight += v4 * lh;
        atomicAdd(grad_im + ptr4, top_grad_im * w4);
    }

    const opmath_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    *grad_mask = top_grad * val;
    *grad_offset = offset_scale * grad_w_weight * top_grad_im;
    *(grad_offset + 1) = offset_scale * grad_h_weight * top_grad_im;
}

// 双线性插值法的反向传播(仅有 group_channels > 1024 且为不是 1024 的整数倍时调用该算法)
template <typename scalar_t>
__device__ void dcnv3_col2im_bilinear_gm(const scalar_t *&bottom_data,
                                         const int &height, const int &width,
                                         const int &nheads, const int &group_channels,
                                         const opmath_t &h, const opmath_t &w,
                                         const int &m, const int &c,
                                         const opmath_t offset_scale,
                                         const opmath_t &top_grad,
                                         const opmath_t &mask,
                                         opmath_t *grad_im,
                                         opmath_t *grad_offset,
                                         opmath_t *grad_mask)
{
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const opmath_t lh = h - h_low;
    const opmath_t lw = w - w_low;
    const opmath_t hh = 1 - lh;
    const opmath_t hw = 1 - lw;

    const int w_stride = nheads * group_channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * group_channels + c;

    const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const opmath_t top_grad_im = top_grad * mask;
    opmath_t grad_h_weight = 0, grad_w_weight = 0;

    opmath_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
        grad_h_weight -= v1 * hw;
        grad_w_weight -= v1 * hh;
        atomicAdd(grad_im + ptr1, top_grad_im * w1);
    }

    opmath_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
        grad_h_weight -= v2 * lw;
        grad_w_weight += v2 * hh;
        atomicAdd(grad_im + ptr2, top_grad_im * w2);
    }

    opmath_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
        grad_h_weight += v3 * hw;
        grad_w_weight -= v3 * lh;
        atomicAdd(grad_im + ptr3, top_grad_im * w3);
    }

    opmath_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
        grad_h_weight += v4 * lw;
        grad_w_weight += v4 * lh;
        atomicAdd(grad_im + ptr4, top_grad_im * w4);
    }

    const opmath_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    atomicAdd(grad_mask, top_grad * val);
    atomicAdd(grad_offset, offset_scale * grad_w_weight * top_grad_im);
    atomicAdd(grad_offset + 1, offset_scale * grad_h_weight * top_grad_im);
}

// 如果 group_channels < 64 且不为2的整数倍，那么就采用该函数进行梯度计算
template <typename scalar_t>
__global__ void dcnv3_col2im_gpu_kernel_shm_reduce_v1(const int num_kernels,
                                                      const scalar_t *grad_col,
                                                      const scalar_t *data_im,
                                                      const scalar_t *data_offset,
                                                      const scalar_t *data_mask,
                                                      const int kernel_h, const int kernel_w,
                                                      const int stride_h, const int stride_w,
                                                      const int pad_h, const int pad_w,
                                                      const int dilation_h, const int dilation_w,
                                                      const int groups, const int group_channels,
                                                      const int height_in, const int width_in,
                                                      const int height_out, const int width_out,
                                                      const opmath_t offset_scale,
                                                      opmath_t *grad_im,
                                                      opmath_t *grad_offset,
                                                      opmath_t *grad_mask)
{
    CUDA_KERNEL_LOOP(index, num_kernels)
    {
        extern __shared__ int _s[];
        opmath_t *cache_grad_offset = (opmath_t *)_s;
        opmath_t *cache_grad_mask = cache_grad_offset + blockDim.x * 2;

        unsigned int tid = threadIdx.x;

        int _tmp = index;
        const int c_col = _tmp % group_channels;
        _tmp /= group_channels;
        const int sampling_index = _tmp;
        const int g_col = _tmp % groups;
        _tmp /= groups;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_tmp % width_out) * stride_w;
        _tmp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_tmp % height_out) * stride_h;
        _tmp /= height_out;
        const int b_col = _tmp;

        const opmath_t top_grad = grad_col[index];

        const int input_size = height_in * width_in;
        const int qid_stride = groups * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;

        const int kernel_size = kernel_h * kernel_w;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr << 1;

        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;

        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr << 1;
        grad_mask += grad_sampling_ptr;

        const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        for (int i = 0; i < kernel_w; i++)
        {
            for (int j = 0; j < kernel_h; j++)
            {
                const opmath_t offset_w = data_offset[data_loc_w_ptr];
                const opmath_t offset_h = data_offset[data_loc_w_ptr + 1];

                const opmath_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                const opmath_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;

                const opmath_t weight = data_mask[data_weight_ptr];

                *(cache_grad_mask + threadIdx.x) = 0;
                *(cache_grad_offset + (threadIdx.x << 2)) = 0;
                *(cache_grad_offset + (threadIdx.x << 2) + 1) = 0;

                if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in)
                {
                    dcnv3_col2im_bilinear(data_im_ptr,
                                          height_in, width_in,
                                          groups, group_channels,
                                          loc_h, loc_w,
                                          g_col, c_col,
                                          offset_scale,
                                          top_grad,
                                          weight,
                                          grad_im_ptr,
                                          cache_grad_offset + (threadIdx.x << 2),
                                          cache_grad_mask + threadIdx.x);
                }

                __syncthreads();

                if (tid == 0)
                {
                    opmath_t _grad_w = cache_grad_offset[0];
                    opmath_t _grad_h = cache_grad_offset[1];
                    opmath_t _grad_a = cache_grad_mask[0];

                    int sid = 2;
                    for (unsigned int tid = 0; tid < blockDim.x; tid++)
                    {
                        _grad_w += cache_grad_offset[sid];
                        _grad_h += cache_grad_offset[sid + 1];
                        _grad_a += cache_grad_mask[tid];
                        sid += 2;
                    }

                    *grad_offset = _grad_w;
                    *(grad_offset + 1) = _grad_h;
                    *grad_mask = _grad_a;
                }

                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_mask += 1;
                grad_offset += 2;
            }
        }
    }
}

// 如果 64 < group_channels < 1024 且不为2的整数倍，那么就采用该函数进行梯度计算
template <typename scalar_t>
__global__ void dcnv3_col2im_gpu_kernel_shm_reduce_v2(const int num_kernels,
                                                      const scalar_t *grad_col,
                                                      const scalar_t *data_im,
                                                      const scalar_t *data_offset,
                                                      const scalar_t *data_mask,
                                                      const int kernel_h, const int kernel_w,
                                                      const int stride_h, const int stride_w,
                                                      const int pad_h, const int pad_w,
                                                      const int dilation_h, const int dilation_w,
                                                      const int groups, const int group_channels,
                                                      const int height_in, const int width_in,
                                                      const int height_out, const int width_out,
                                                      const opmath_t offset_scale,
                                                      opmath_t *grad_im,
                                                      opmath_t *grad_offset,
                                                      opmath_t *grad_mask)
{
    CUDA_KERNEL_LOOP(index, num_kernels)
    {
        extern __shared__ int _s[];
        opmath_t *cache_grad_offset = (opmath_t *)_s;
        opmath_t *cache_grad_mask = cache_grad_offset + blockDim.x * 2;

        unsigned int tid = threadIdx.x;

        int _tmp = index;
        const int c_col = _tmp % group_channels;
        _tmp /= group_channels;
        const int sampling_index = _tmp;
        const int g_col = _tmp % groups;
        _tmp /= groups;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_tmp % width_out) * stride_w;
        _tmp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_tmp % height_out) * stride_h;
        _tmp /= height_out;
        const int b_col = _tmp;

        const opmath_t top_grad = grad_col[index];

        const int input_size = height_in * width_in;
        const int qid_stride = groups * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;

        const int kernel_size = kernel_h * kernel_w;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr << 1;

        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;

        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr << 1;
        grad_mask += grad_sampling_ptr;

        const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        for (int i = 0; i < kernel_w; i++)
        {
            for (int j = 0; j < kernel_h; j++)
            {
                const opmath_t offset_w = data_offset[data_loc_w_ptr];
                const opmath_t offset_h = data_offset[data_loc_w_ptr + 1];

                const opmath_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                const opmath_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;

                const opmath_t weight = data_mask[data_weight_ptr];

                *(cache_grad_mask + threadIdx.x) = 0;
                *(cache_grad_offset + (threadIdx.x << 2)) = 0;
                *(cache_grad_offset + (threadIdx.x << 2) + 1) = 0;

                if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in)
                {
                    dcnv3_col2im_bilinear(data_im_ptr,
                                          height_in, width_in,
                                          groups, group_channels,
                                          loc_h, loc_w,
                                          g_col, c_col,
                                          offset_scale,
                                          top_grad,
                                          weight,
                                          grad_im_ptr,
                                          cache_grad_offset + (threadIdx.x << 2),
                                          cache_grad_mask + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;
                        cache_grad_mask[tid] += cache_grad_mask[tid + s];
                        cache_grad_offset[xid1] += cache_grad_offset[xid2];
                        cache_grad_offset[xid1 + 1] += cache_grad_offset[xid2 + 1];
                        if (tid + (s << 1) < spre)
                        {
                            cache_grad_mask[tid] += cache_grad_mask[tid + (s << 1)];
                            cache_grad_offset[xid1] += cache_grad_offset[xid2 + (s << 1)];
                            cache_grad_offset[xid1 + 1] += cache_grad_offset[xid2 + (s << 1) + 1];
                        }
                    }

                    __syncthreads();
                }

                if (tid == 0)
                {
                    *grad_mask = cache_grad_mask[0];
                    *grad_offset = cache_grad_offset[0];
                    *(grad_offset + 1) = cache_grad_offset[1];
                }

                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_mask += 1;
                grad_offset += 2;
            }
        }
    }
}

// 如果group_channels 是 1024 的整数倍，那么就采用该函数进行梯度计算
template <typename scalar_t>
__global__ void dcnv3_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(const int num_kernels,
                                                                   const scalar_t *grad_col,
                                                                   const scalar_t *data_im,
                                                                   const scalar_t *data_offset,
                                                                   const scalar_t *data_mask,
                                                                   const int kernel_h, const int kernel_w,
                                                                   const int stride_h, const int stride_w,
                                                                   const int pad_h, const int pad_w,
                                                                   const int dilation_h, const int dilation_w,
                                                                   const int groups, const int group_channels,
                                                                   const int height_in, const int width_in,
                                                                   const int height_out, const int width_out,
                                                                   const opmath_t offset_scale,
                                                                   opmath_t *grad_im,
                                                                   opmath_t *grad_offset,
                                                                   opmath_t *grad_mask)
{
    // num_actual_kernels = batch_n * height_out * width_out * groups * group_channels;
    // 线程数以久按照forward()函数输出张量的像素点数量来设置，每个线程处理一个像素点

    CUDA_KERNEL_LOOP(index, num_kernels)
    {
        extern __shared__ int _s[];                                     // 程序调用的时候指明是 num_threads * 3 * sizeof(opmath_t) 大小的共享内存,
                                                                        // 共享内存是每个线程块独享的，线程块中的所有线程都可以访问共享内存，共享内存的访问速度比全局内存快得多
                                                                        // extern 说明该cuda 共享内存采用动态分配，这样可以在程序运行时动态分配共享内存的大小
        opmath_t *cache_grad_offset = (opmath_t *)_s;                   // 先进行类型转换cache_grad_offset，然后将指针赋给，用于偏移量梯度缓存
        opmath_t *cache_grad_mask = cache_grad_offset + blockDim.x * 2; // blockDim.x 即 num_threads，移动指针到掩码梯度缓存的起始位置

        unsigned int tid = threadIdx.x; // 当前线程的索引

        // 把一维的index还原成多维的索引
        int _tmp = index;                        // 当前线程的全局索引
        const int c_col = _tmp % group_channels; // 通道索引
        _tmp /= group_channels;                  // 除以通道数，相当于去掉通道维度
        const int sampling_index = _tmp;         // 相当于空间位置的坐标索引，sampling_index = b * height_out * width_out * groups + h * width_out * groups + w * groups + g
        const int g_col = _tmp % groups;         // 组索引
        _tmp /= groups;                          // 除以组数，相当于去掉组维度
        // 当前像素点在输入张量上的卷积中心
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_tmp % width_out) * stride_w;
        _tmp /= width_out; // 除以宽度，相当于去掉宽度维度
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_tmp % height_out) * stride_h;
        _tmp /= height_out;     // 除以高度，相当于去掉高度维度
        const int b_col = _tmp; // 批次索引

        // 开始定位当前像素点在各个张量对应的位置
        // grad_col:
        const opmath_t top_grad = grad_col[index]; // 获取当前像素点在反向传播上一级计算得到的梯度张量的值, 即当前像素点的顶端梯度

        // data_im:
        const int input_size = height_in * width_in;               // 输入张量的像素点数量
        const int qid_stride = groups * group_channels;            // 就是原始通道数
        const int im_ptr_offset = b_col * input_size * qid_stride; // 一个样本的大小，即指针偏移量
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;     // 当前像素对应输入张量样本的起始指针

        // data_mask:
        const int kernel_size = kernel_h * kernel_w;        // 卷积核的大小
        int data_weight_ptr = sampling_index * kernel_size; // mask权重索引
                                                            // sampling_index * kernel_size = b * height_out * width_out * groups * (kernel_h * kernel_w)
                                                            //                                + h * width_out * groups * (kernel_h * kernel_w)
                                                            //                                + w * groups * (kernel_h * kernel_w)
                                                            //                                + g * (kernel_h * kernel_w)
        // data_offset:
        int data_loc_w_ptr = data_weight_ptr << 1; // offset索引。'<< 1'左移一位，相当于乘以2

        // grad_im:
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset; // 当前像素点在输入张量的梯度张量的起始指针

        // grad_offset and grad_mask:
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr << 1; // 当前像素点在偏移量梯度张量的起始指针
        grad_mask += grad_sampling_ptr;        // 当前像素点在掩码梯度张量的起始指针

        // 卷积在输入张量上的左上角坐标
        const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        for (int i = 0; i < kernel_w; i++)
        {
            for (int j = 0; j < kernel_h; j++)
            {
                const opmath_t offset_w = data_offset[data_loc_w_ptr]; // 获取在输入张量上当前像素点的偏移量
                const opmath_t offset_h = data_offset[data_loc_w_ptr + 1];

                const opmath_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale; // 在默认offset_scale为1.0的情况下，p0_w_ + i * dilation_w 是当前位置， 加上offset_w是偏移后的位置
                const opmath_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;

                const opmath_t weight = data_mask[data_weight_ptr]; // 获取当前像素点的权重

                // 将缓存清零
                *(cache_grad_offset + (threadIdx.x << 2)) = 0;
                *(cache_grad_offset + (threadIdx.x << 2) + 1) = 0;
                *(cache_grad_mask + threadIdx.x) = 0;

                if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in) // 检查偏移后的位置是否在输入张量的范围内
                {
                    dcnv3_col2im_bilinear(data_im_ptr,
                                          height_in, width_in,
                                          groups, group_channels,
                                          loc_h, loc_w,
                                          g_col, c_col,
                                          offset_scale,
                                          top_grad,
                                          weight,
                                          grad_im_ptr,
                                          cache_grad_offset + (threadIdx.x << 2),
                                          cache_grad_mask + threadIdx.x);
                }

                __syncthreads(); // 等待所有线程都执行完上面的代码

                for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1) // 二分法进行梯度累加
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 2;
                        const unsigned int xid2 = (tid + s) << 2;

                        cache_grad_mask[tid] += cache_grad_mask[tid + s];
                        cache_grad_offset[xid1] += cache_grad_offset[xid2];
                        cache_grad_offset[xid1 + 1] += cache_grad_offset[xid2 + 1];

                        if (tid + (s << 1) < spre)
                        {
                            cache_grad_mask[tid] += cache_grad_mask[tid + (s << 1)];
                            cache_grad_offset[xid1] += cache_grad_offset[xid2 + (s << 1)];
                            cache_grad_offset[xid1 + 1] += cache_grad_offset[xid2 + (s << 1) + 1];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    atomicAdd(grad_mask, cache_grad_mask[0]);
                    atomicAdd(grad_offset, cache_grad_offset[0]);
                    atomicAdd(grad_offset + 1, cache_grad_offset[1]);
                }
                __syncthreads();

                data_weight_ptr += 1; // 权重索引加1
                data_loc_w_ptr += 2;  // 偏移量索引得加2
                grad_mask += 1;       // 掩码梯度索引加1
                grad_offset += 2;     // 偏移量梯度索引加2
            }
        }
    }
}

// 如果group_channels 大于 1024 但不是 1024 的整数倍，那么就采用该函数进行梯度计算
template <typename scalar_t>
__global__ void dcnv3_col2im_gpu_gm(const int num_kernels,
                                    const scalar_t *grad_col,
                                    const scalar_t *data_im,
                                    const scalar_t *data_offset,
                                    const scalar_t *data_mask,
                                    const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w,
                                    const int pad_h, const int pad_w,
                                    const int dilation_h, const int dilation_w,
                                    const int groups, const int group_channels,
                                    const int height_in, const int width_in,
                                    const int height_out, const int width_out,
                                    const opmath_t offset_scale,
                                    opmath_t *grad_im,
                                    opmath_t *grad_offset,
                                    opmath_t *grad_mask)
{
    CUDA_KERNEL_LOOP(index, num_kernels)
    {
        // 把一维的index还原成多维的索引
        int _tmp = index;
        const int c_col = _tmp % group_channels; // 通道索引
        _tmp /= group_channels;                  // 除以通道数，相当于去掉通道维度
        const int sampling_index = _tmp;         // 相当于空间位置的坐标索引，sampling_index = b * height_out * width_out * groups + h * width_out * groups + w * groups + g
        const int g_col = _tmp % groups;         // 组索引
        _tmp /= groups;                          // 除以组数，相当于去掉组维度
        // 当前像素点在输入张量上的卷积中心
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_tmp % width_out) * stride_w;
        _tmp /= width_out; // 除以宽度，相当于去掉宽度维度
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_tmp % height_out) * stride_h;
        _tmp /= height_out;     // 除以高度，相当于去掉高度维度
        const int b_col = _tmp; // 批次索引

        // 开始定位当前像素点在各个张量对应的位置
        // grad_col:
        const opmath_t top_grad = grad_col[index]; // 获取当前像素点在反向传播上一级计算得到的梯度张量的值, 即当前像素点的顶端梯度

        // data_im:
        const int input_size = height_in * width_in;               // 输入张量的像素点数量
        const int qid_stride = groups * group_channels;            // 就是原始通道数
        const int im_ptr_offset = b_col * input_size * qid_stride; // 一个样本的大小，即指针偏移量
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;     // 当前像素对应输入张量样本的起始指针

        // data_mask:
        const int kernel_size = kernel_h * kernel_w;        // 卷积核的大小
        int data_weight_ptr = sampling_index * kernel_size; // mask权重索引
                                                            // sampling_index * kernel_size = b * height_out * width_out * groups * (kernel_h * kernel_w)
                                                            //                                + h * width_out * groups * (kernel_h * kernel_w)
                                                            //                                + w * groups * (kernel_h * kernel_w)
                                                            //                                + g * (kernel_h * kernel_w)
        // data_offset:
        int data_loc_w_ptr = data_weight_ptr << 1; // offset索引。'<< 1'左移一位，相当于乘以2

        // grad_im:
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset; // 当前像素点在输入张量的梯度张量的起始指针

        // grad_offset and grad_mask:
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr << 1; // 当前像素点在偏移量梯度张量的起始指针
        grad_mask += grad_sampling_ptr;        // 当前像素点在掩码梯度张量的起始指针

        // 卷积在输入张量上的左上角坐标
        const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        for (int i = 0; i < kernel_w; i++)
        {
            for (int j = 0; j < kernel_h; j++)
            {
                const opmath_t offset_w = data_offset[data_loc_w_ptr]; // 获取在输入张量上当前像素点的偏移量
                const opmath_t offset_h = data_offset[data_loc_w_ptr + 1];

                const opmath_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale; // 在默认offset_scale为1.0的情况下，p0_w_ + i * dilation_w 是当前位置， 加上offset_w是偏移后的位置
                const opmath_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;

                const opmath_t weight = data_mask[data_weight_ptr]; // 获取当前像素点的权重

                if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in) // 检查偏移后的位置是否在输入张量的范围内
                {
                    dcnv3_col2im_bilinear_gm(data_im_ptr,
                                             height_in, width_in,
                                             groups, group_channels,
                                             loc_h, loc_w,
                                             g_col, c_col,
                                             offset_scale,
                                             top_grad,
                                             weight,
                                             grad_im_ptr,
                                             grad_offset,
                                             grad_mask);
                }
                data_weight_ptr += 1; // 权重索引加1
                data_loc_w_ptr += 2;  // 偏移量索引得加2
                grad_mask += 1;       // 掩码梯度索引加1
                grad_offset += 2;     // 偏移量梯度索引加2
            }
        }
    }
}

// 3、如果group_channels 为（1，2，4，8，16，32），那么就采用该函数进行梯度计算
template <typename scalar_t, unsigned int blockSize>
__global__ void dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(const int num_kernels,
                                                                      const scalar_t *grad_col,
                                                                      const scalar_t *data_im,
                                                                      const scalar_t *data_offset,
                                                                      const scalar_t *data_mask,
                                                                      const int kernel_h, const int kernel_w,
                                                                      const int stride_h, const int stride_w,
                                                                      const int pad_h, const int pad_w,
                                                                      const int dilation_h, const int dilation_w,
                                                                      const int groups, const int group_channels,
                                                                      const int height_in, const int width_in,
                                                                      const int height_out, const int width_out,
                                                                      const opmath_t offset_scale,
                                                                      opmath_t *grad_im,
                                                                      opmath_t *grad_offset,
                                                                      opmath_t *grad_mask)
{
    CUDA_KERNEL_LOOP(index, num_kernels)
    {
        __shared__ opmath_t cache_grad_offset[blockSize * 2];
        __shared__ opmath_t cache_grad_mask[blockSize];

        unsigned int tid = threadIdx.x;

        int _tmp = index;
        const int c_col = _tmp % group_channels;
        _tmp /= group_channels;
        const int sampling_index = _tmp;
        const int g_col = _tmp % groups;
        _tmp /= groups;

        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_tmp % width_out) * stride_w;
        _tmp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_tmp % height_out) * stride_h;
        _tmp /= height_out;

        const int b_col = _tmp;

        const opmath_t top_grad = grad_col[index];

        const int input_size = height_in * width_in;
        const int qid_stride = groups * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;

        const int kernel_size = kernel_h * kernel_w;
        int data_weight_ptr = sampling_index * kernel_size;

        int data_loc_w_ptr = data_weight_ptr << 1;

        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;

        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr << 1;
        grad_mask += grad_sampling_ptr;

        const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        for (int i = 0; i < kernel_w; i++)
        {
            for (int j = 0; j < kernel_h; j++)
            {
                const opmath_t offset_w = data_offset[data_loc_w_ptr];
                const opmath_t offset_h = data_offset[data_loc_w_ptr + 1];

                const opmath_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                const opmath_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;

                const opmath_t weight = data_mask[data_weight_ptr];

                *(cache_grad_offset + (threadIdx.x << 1)) = 0;
                *(cache_grad_offset + (threadIdx.x << 1) + 1) = 0;
                *(cache_grad_mask + threadIdx.x) = 0;

                if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in)
                {
                    dcnv3_col2im_bilinear(data_im_ptr,
                                          height_in, width_in,
                                          groups, group_channels,
                                          loc_h, loc_w,
                                          g_col, c_col,
                                          offset_scale,
                                          top_grad,
                                          weight,
                                          grad_im_ptr,
                                          cache_grad_offset + (threadIdx.x << 1),
                                          cache_grad_mask + threadIdx.x);
                }

                __syncthreads();

                if (tid == 0)
                {
                    opmath_t _grad_w = cache_grad_offset[0],
                             _grad_h = cache_grad_offset[1],
                             _grad_a = cache_grad_mask[0];

                    int sid = 2;
                    for (unsigned int tid = 1; tid < blockSize; tid++)
                    {
                        _grad_w += cache_grad_offset[sid];
                        _grad_h += cache_grad_offset[sid + 1];
                        _grad_a += cache_grad_mask[tid];
                        sid += 2;
                    }

                    *grad_offset = _grad_w;
                    *(grad_offset + 1) = _grad_h;
                    *grad_mask = _grad_a;
                }
                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_mask += 1;
                grad_offset += 2;
            }
        }
    }
}

// 4、如果group_channels 为（64, 128, 256, 512, 1024），那么就采用该函数进行梯度计算
template <typename scalar_t, unsigned int blockSize>
__global__ void dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(const int num_kernels,
                                                                      const scalar_t *grad_col,
                                                                      const scalar_t *data_im,
                                                                      const scalar_t *data_offset,
                                                                      const scalar_t *data_mask,
                                                                      const int kernel_h, const int kernel_w,
                                                                      const int stride_h, const int stride_w,
                                                                      const int pad_h, const int pad_w,
                                                                      const int dilation_h, const int dilation_w,
                                                                      const int groups, const int group_channels,
                                                                      const int height_in, const int width_in,
                                                                      const int height_out, const int width_out,
                                                                      const opmath_t offset_scale,
                                                                      opmath_t *grad_im,
                                                                      opmath_t *grad_offset,
                                                                      opmath_t *grad_mask)
{
    CUDA_KERNEL_LOOP(index, num_kernels)
    {
        __shared__ opmath_t cache_grad_offset[blockSize * 2];
        __shared__ opmath_t cache_grad_mask[blockSize];

        unsigned int tid = threadIdx.x;

        int _tmp = index;
        const int c_col = _tmp % group_channels;
        _tmp /= group_channels;
        const int sampling_index = _tmp;
        const int g_col = _tmp % groups;
        _tmp /= groups;

        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (_tmp % width_out) * stride_w;
        _tmp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (_tmp % height_out) * stride_h;
        _tmp /= height_out;

        const int b_col = _tmp;

        const opmath_t top_grad = grad_col[index];

        const int input_size = height_in * width_in;
        const int qid_stride = groups * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;

        const int kernel_size = kernel_h * kernel_w;
        int data_weight_ptr = sampling_index * kernel_size;

        int data_loc_w_ptr = data_weight_ptr << 1;

        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;

        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr << 1;
        grad_mask += grad_sampling_ptr;

        const opmath_t p0_w_ = p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ = p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        for (int i = 0; i < kernel_w; i++)
        {
            for (int j = 0; j < kernel_h; j++)
            {
                const opmath_t offset_w = data_offset[data_loc_w_ptr];
                const opmath_t offset_h = data_offset[data_loc_w_ptr + 1];

                const opmath_t loc_w = p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                const opmath_t loc_h = p0_h_ + (j * dilation_h + offset_h) * offset_scale;

                const opmath_t weight = data_mask[data_weight_ptr];

                *(cache_grad_offset + (threadIdx.x << 1)) = 0;
                *(cache_grad_offset + (threadIdx.x << 1) + 1) = 0;
                *(cache_grad_mask + threadIdx.x) = 0;

                if (loc_h > -1 && loc_w > -1 && loc_h < height_in && loc_w < width_in)
                {
                    dcnv3_col2im_bilinear(data_im_ptr,
                                          height_in, width_in,
                                          groups, group_channels,
                                          loc_h, loc_w,
                                          g_col, c_col,
                                          offset_scale,
                                          top_grad,
                                          weight,
                                          grad_im_ptr,
                                          cache_grad_offset + (threadIdx.x << 1),
                                          cache_grad_mask + threadIdx.x);
                }

                __syncthreads();

                for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        const unsigned int xid1 = tid << 1;
                        const unsigned int xid2 = (tid + s) << 1;

                        cache_grad_mask[tid] += cache_grad_mask[tid + s];
                        cache_grad_offset[xid1] += cache_grad_offset[xid2];
                        cache_grad_offset[xid1 + 1] += cache_grad_offset[xid2 + 1];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    *grad_offset = cache_grad_offset[0];
                    *(grad_offset + 1) = cache_grad_offset[1];
                    *grad_mask = cache_grad_mask[0];
                }

                __syncthreads();

                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_mask += 1;
                grad_offset += 2;
            }
        }
    }
}

// ******************************************************************************************************************************************************************************
// * dcnv3_cuda_backward()函数入口
// ******************************************************************************************************************************************************************************
template <typename scalar_t>
void dcnv3_col2im_cuda(
    cudaStream_t stream,
    const scalar_t *grad_col,    // 指向第n个样本对应的反向传播上一级计算得到的梯度值
    const scalar_t *data_im,     // 指向第n个样本对应的输入张量，维度应该是[batch_n, height_in, width_in, groups * group_channels]
    const scalar_t *data_offset, // 偏移量指针，偏移张量的维度应该是[batch_n, height_out, width_out, 2 * kernel_h * kernel_w * groups]
    const scalar_t *data_mask,   // 掩码指针，掩码张量的维度应该是[batch_n, height_out, width_out, kernel_h * kernel_w * groups]
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int groups, const int group_channels,
    const int batch_n,
    const int height_in, const int width_in,
    const int height_out, const int width_out,
    const opmath_t offset_scale,
    opmath_t *grad_im,
    opmath_t *grad_offset,
    opmath_t *grad_mask)
{
    // 在 CUDA_NUM_THREADS 和 group_channels 中取最小的一个作为线程数
    const int num_threads = (group_channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : group_channels;
    const int num_kernels = batch_n * height_out * width_out * groups * group_channels;
    const int num_actual_kernels = batch_n * height_out * width_out * groups * group_channels;

    if (group_channels > 1024)
    {
        if ((group_channels & 1023) == 0)
        // 1023的二进制形式是'0b1111111111'，1024的二进制形式为'0b10000000000'
        // 两者进行与运算，结果为0，1024的任意整数倍的二进制形式的右边10位都是0，与1023进行与运算，结果还是0
        // 但是如果group_channels不是1024的整数倍，最右边的10位中就会出现1，那么与1023进行与运算，结果就不是0了
        // 因此，这一步是判断group_channels是不是1024的整数倍
        {
            dcnv3_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, num_threads * 3 * sizeof(opmath_t), stream>>>(num_kernels,
                                                                                                                           grad_col,
                                                                                                                           data_im,
                                                                                                                           data_offset,
                                                                                                                           data_mask,
                                                                                                                           kernel_h, kernel_w,
                                                                                                                           stride_h, stride_w,
                                                                                                                           pad_h, pad_w,
                                                                                                                           dilation_h, dilation_w,
                                                                                                                           groups, group_channels,
                                                                                                                           height_in, width_in,
                                                                                                                           height_out, width_out,
                                                                                                                           offset_scale,
                                                                                                                           grad_im,
                                                                                                                           grad_offset,
                                                                                                                           grad_mask);
        }
        else
        {
            dcnv3_col2im_gpu_gm<scalar_t>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
        }
    }
    else
    {
        switch (group_channels)
        {
        case 1:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 2:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 4:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 8:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 16:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 32:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 64:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 128:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 256:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 512:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        case 1024:
            dcnv3_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                          grad_col,
                                                                                          data_im,
                                                                                          data_offset,
                                                                                          data_mask,
                                                                                          kernel_h, kernel_w,
                                                                                          stride_h, stride_w,
                                                                                          pad_h, pad_w,
                                                                                          dilation_h, dilation_w,
                                                                                          groups, group_channels,
                                                                                          height_in, width_in,
                                                                                          height_out, width_out,
                                                                                          offset_scale,
                                                                                          grad_im,
                                                                                          grad_offset,
                                                                                          grad_mask);
            break;

        default:
            if (group_channels < 64)
            {
                dcnv3_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, num_threads * 3 * sizeof(opmath_t), stream>>>(num_kernels,
                                                                                                                               grad_col,
                                                                                                                               data_im,
                                                                                                                               data_offset,
                                                                                                                               data_mask,
                                                                                                                               kernel_h, kernel_w,
                                                                                                                               stride_h, stride_w,
                                                                                                                               pad_h, pad_w,
                                                                                                                               dilation_h, dilation_w,
                                                                                                                               groups, group_channels,
                                                                                                                               height_in, width_in,
                                                                                                                               height_out, width_out,
                                                                                                                               offset_scale,
                                                                                                                               grad_im,
                                                                                                                               grad_offset,
                                                                                                                               grad_mask);
            }
            else
            {
                dcnv3_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, num_threads * 3 * sizeof(opmath_t), stream>>>(num_kernels,
                                                                                                                               grad_col,
                                                                                                                               data_im,
                                                                                                                               data_offset,
                                                                                                                               data_mask,
                                                                                                                               kernel_h, kernel_w,
                                                                                                                               stride_h, stride_w,
                                                                                                                               pad_h, pad_w,
                                                                                                                               dilation_h, dilation_w,
                                                                                                                               groups, group_channels,
                                                                                                                               height_in, width_in,
                                                                                                                               height_out, width_out,
                                                                                                                               offset_scale,
                                                                                                                               grad_im,
                                                                                                                               grad_offset,
                                                                                                                               grad_mask);
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in dcnv3_col2im_cuda: %s\n", cudaGetErrorString(err));
    }
}

// ******************************************************************************************************************************************************************************
