#include <metal_stdlib>
using namespace metal;

#define TASK0053_MSL 1
#include "task0053_integer_core.h"
#undef TASK0053_MSL

struct Task0053PointMeta {
    ulong elementCount;
    ulong gridWidth;
    ulong gridHeight;
    uint reserved0;
    uint reserved1;
};

inline ulong task0053_linear_id(uint3 gid, ulong width, ulong height) {
    return ulong(gid.x) + width * (ulong(gid.y) + height * ulong(gid.z));
}

kernel void task0053_candidate_exp(
        device const uint *input [[buffer(0)]],
        device uint *output [[buffer(1)]],
        constant Task0053PointMeta &meta [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]) {
    ulong linear = task0053_linear_id(gid, meta.gridWidth, meta.gridHeight);
    if (linear < meta.elementCount) output[linear] = task0053_exp_word(input[linear]);
}

kernel void task0053_candidate_sigmoid(
        device const uint *input [[buffer(0)]],
        device uint *output [[buffer(1)]],
        constant Task0053PointMeta &meta [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]) {
    ulong linear = task0053_linear_id(gid, meta.gridWidth, meta.gridHeight);
    if (linear < meta.elementCount) output[linear] = task0053_sigmoid_word(input[linear]);
}
