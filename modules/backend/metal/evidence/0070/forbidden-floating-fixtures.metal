#include <metal_stdlib>
using namespace metal;

kernel void audit_forbidden_fadd(
        device const float *left [[buffer(0)]],
        device const float *right [[buffer(1)]],
        device float *output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    output[index] = left[index] + right[index];
}

kernel void audit_forbidden_fmul(
        device const float *left [[buffer(0)]],
        device const float *right [[buffer(1)]],
        device float *output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    output[index] = left[index] * right[index];
}

kernel void audit_forbidden_fdiv(
        device const float *left [[buffer(0)]],
        device const float *right [[buffer(1)]],
        device float *output [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    output[index] = left[index] / right[index];
}

kernel void audit_forbidden_fma(
        device const float *left [[buffer(0)]],
        device const float *right [[buffer(1)]],
        device const float *addend [[buffer(2)]],
        device float *output [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    output[index] = fma(left[index], right[index], addend[index]);
}
