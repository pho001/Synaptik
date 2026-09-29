#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef __HAVE_TENSOR__
#error "MPP qualification requires __HAVE_TENSOR__"
#endif

kernel void mpp_matmul_f16_f32(
        tensor<device half, dextents<int32_t, 2>> left,
        tensor<device half, dextents<int32_t, 2>> right,
        tensor<device float, dextents<int32_t, 2>> destination,
        uint2 tgid [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = matmul2d_descriptor(
            64, 32, static_cast<int>(dynamic_extent), false, false, false);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto tileLeft = left.slice(0, tgid.y * 64);
    auto tileRight = right.slice(tgid.x * 32, 0);
    auto tileDestination = destination.slice(tgid.x * 32, tgid.y * 64);
    operation.run(tileLeft, tileRight, tileDestination);
}

kernel void mpp_matmul_bf16_f32(
        tensor<device bfloat, dextents<int32_t, 2>> left,
        tensor<device bfloat, dextents<int32_t, 2>> right,
        tensor<device float, dextents<int32_t, 2>> destination,
        uint2 tgid [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = matmul2d_descriptor(
            64, 32, static_cast<int>(dynamic_extent), false, false, false);
    static_assert(!descriptor.relaxed_precision);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto tileLeft = left.slice(0, tgid.y * 64);
    auto tileRight = right.slice(tgid.x * 32, 0);
    auto tileDestination = destination.slice(tgid.x * 32, tgid.y * 64);
    operation.run(tileLeft, tileRight, tileDestination);
}

kernel void mpp_convolution_f16_f32(
        tensor<device half, dextents<int32_t, 4>> activation,
        tensor<device half, dextents<int32_t, 4>> weights,
        tensor<device float, dextents<int32_t, 4>> destination) {
    constexpr auto descriptor = convolution2d_descriptor(
            int4(16, 8, 8, 1), int4(16, 8, 8, 1), int2(1, 1),
            convolution2d_activation_layout::nhwc,
            convolution2d_weights_layout::hwio,
            int2(1, 1), int2(1, 1), 1, false);
    static_assert(!descriptor.relaxed_precision);
    convolution2d<descriptor, execution_simdgroups<4>> operation;
    operation.run(activation, weights, destination);
}

kernel void mpp_convolution_bf16_f32(
        tensor<device bfloat, dextents<int32_t, 4>> activation,
        tensor<device bfloat, dextents<int32_t, 4>> weights,
        tensor<device float, dextents<int32_t, 4>> destination) {
    constexpr auto descriptor = convolution2d_descriptor(
            int4(16, 8, 8, 1), int4(16, 8, 8, 1), int2(1, 1),
            convolution2d_activation_layout::nhwc,
            convolution2d_weights_layout::hwio,
            int2(1, 1), int2(1, 1), 1, false);
    static_assert(!descriptor.relaxed_precision);
    convolution2d<descriptor, execution_simdgroups<4>> operation;
    operation.run(activation, weights, destination);
}
