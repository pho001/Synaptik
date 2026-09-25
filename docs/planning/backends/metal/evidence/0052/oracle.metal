#include <metal_stdlib>
using namespace metal;

struct BinaryMeta {
    ulong dims[16];
    ulong leftStrides[16];
    ulong rightStrides[16];
    ulong elementCount;
    ulong gridWidth;
    ulong gridHeight;
    uint rank;
    uint reserved;
};

struct PointMeta {
    ulong elementCount;
    ulong gridWidth;
    ulong gridHeight;
    uint scalar;
    uint reserved;
};

struct ClampMeta {
    ulong elementCount;
    ulong gridWidth;
    ulong gridHeight;
    uint lower;
    uint upper;
};

struct ReductionMeta {
    ulong dims[16];
    ulong strides[16];
    ulong outputCount;
    ulong reducedCount;
    ulong gridWidth;
    ulong gridHeight;
    uint rank;
    uint axesMask;
};

struct ScanMeta {
    ulong elementCount;
    ulong axisLength;
    ulong inner;
    ulong lineCount;
    ulong gridWidth;
    ulong gridHeight;
    uint exclusive;
    uint reverse;
};

inline ulong linear_id(uint3 gid, ulong width, ulong height) {
    return ulong(gid.x) + width * (ulong(gid.y) + height * ulong(gid.z));
}

inline bool nan_bits(uint value) {
    return (value & 0x7f800000u) == 0x7f800000u && (value & 0x007fffffu) != 0u;
}

inline bool zero_bits(uint value) {
    return (value & 0x7fffffffu) == 0u;
}

inline uint ordered_key(uint value) {
    return (value & 0x80000000u) != 0u ? ~value : (value ^ 0x80000000u);
}

inline bool exact_equal(uint left, uint right) {
    if (nan_bits(left) || nan_bits(right)) return false;
    if (zero_bits(left) && zero_bits(right)) return true;
    return left == right;
}

inline bool exact_less(uint left, uint right) {
    if (nan_bits(left) || nan_bits(right) || exact_equal(left, right)) return false;
    return ordered_key(left) < ordered_key(right);
}

inline bool exact_compare(uint left, uint right, uint operation) {
    switch (operation) {
        case 0u: return exact_less(right, left);                 // GT
        case 1u: return exact_less(right, left) || exact_equal(left, right); // GE
        case 2u: return exact_less(left, right);                 // LT
        case 3u: return exact_less(left, right) || exact_equal(left, right); // LE
        case 4u: return exact_equal(left, right);                // EQ
        default: return !exact_equal(left, right);               // NE
    }
}

inline uint exact_extreme(uint left, uint right, bool maximum) {
    if (nan_bits(left)) return left;
    if (nan_bits(right)) return right;
    if (zero_bits(left) && zero_bits(right)) {
        uint sign = maximum ? ((left & right) & 0x80000000u)
                            : ((left | right) & 0x80000000u);
        return sign;
    }
    if (exact_equal(left, right)) return left;
    bool leftLess = exact_less(left, right);
    return maximum ? (leftLess ? right : left) : (leftLess ? left : right);
}

inline void binary_offsets(ulong linear, constant BinaryMeta &meta,
                           thread ulong &leftOffset, thread ulong &rightOffset) {
    ulong remaining = linear;
    leftOffset = 0ul;
    rightOffset = 0ul;
    for (int axis = int(meta.rank) - 1; axis >= 0; --axis) {
        ulong coordinate = remaining % meta.dims[axis];
        remaining /= meta.dims[axis];
        leftOffset += coordinate * meta.leftStrides[axis];
        rightOffset += coordinate * meta.rightStrides[axis];
    }
}

#define COMPARISON_KERNEL(NAME, OPERATION) \
kernel void NAME(device const uint *left [[buffer(0)]], \
                 device const uint *right [[buffer(1)]], \
                 device uchar *output [[buffer(2)]], \
                 constant BinaryMeta &meta [[buffer(3)]], \
                 uint3 gid [[thread_position_in_grid]]) { \
    ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight); \
    if (linear >= meta.elementCount) return; \
    ulong leftOffset, rightOffset; \
    binary_offsets(linear, meta, leftOffset, rightOffset); \
    output[linear] = exact_compare(left[leftOffset], right[rightOffset], OPERATION) ? uchar(1) : uchar(0); \
}

COMPARISON_KERNEL(cmp_gt, 0u)
COMPARISON_KERNEL(cmp_ge, 1u)
COMPARISON_KERNEL(cmp_lt, 2u)
COMPARISON_KERNEL(cmp_le, 3u)
COMPARISON_KERNEL(cmp_eq, 4u)
COMPARISON_KERNEL(cmp_ne, 5u)

#define BINARY_EXTREME_KERNEL(NAME, MAXIMUM) \
kernel void NAME(device const uint *left [[buffer(0)]], \
                 device const uint *right [[buffer(1)]], \
                 device uint *output [[buffer(2)]], \
                 constant BinaryMeta &meta [[buffer(3)]], \
                 uint3 gid [[thread_position_in_grid]]) { \
    ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight); \
    if (linear >= meta.elementCount) return; \
    ulong leftOffset, rightOffset; \
    binary_offsets(linear, meta, leftOffset, rightOffset); \
    output[linear] = exact_extreme(left[leftOffset], right[rightOffset], MAXIMUM); \
}

BINARY_EXTREME_KERNEL(tensor_min, false)
BINARY_EXTREME_KERNEL(tensor_max, true)

#define SCALAR_EXTREME_KERNEL(NAME, MAXIMUM) \
kernel void NAME(device const uint *input [[buffer(0)]], \
                 device uint *output [[buffer(1)]], \
                 constant PointMeta &meta [[buffer(2)]], \
                 uint3 gid [[thread_position_in_grid]]) { \
    ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight); \
    if (linear >= meta.elementCount) return; \
    output[linear] = exact_extreme(input[linear], meta.scalar, MAXIMUM); \
}

SCALAR_EXTREME_KERNEL(scalar_min, false)
SCALAR_EXTREME_KERNEL(scalar_max, true)

kernel void clamp_fused(device const uint *input [[buffer(0)]],
                        device uint *output [[buffer(1)]],
                        constant ClampMeta &meta [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
    ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight);
    if (linear >= meta.elementCount) return;
    uint lowered = exact_extreme(input[linear], meta.lower, true);
    output[linear] = exact_extreme(lowered, meta.upper, false);
}

inline ulong reduction_offset(ulong outputLinear, ulong reducedLinear,
                              constant ReductionMeta &meta) {
    ulong outputRemaining = outputLinear;
    ulong reducedRemaining = reducedLinear;
    ulong offset = 0ul;
    for (int axis = int(meta.rank) - 1; axis >= 0; --axis) {
        ulong coordinate;
        if ((meta.axesMask & (1u << uint(axis))) != 0u) {
            coordinate = reducedRemaining % meta.dims[axis];
            reducedRemaining /= meta.dims[axis];
        } else {
            coordinate = outputRemaining % meta.dims[axis];
            outputRemaining /= meta.dims[axis];
        }
        offset += coordinate * meta.strides[axis];
    }
    return offset;
}

#define REDUCTION_KERNEL(NAME, MAXIMUM) \
kernel void NAME(device const uint *input [[buffer(0)]], \
                 device uint *output [[buffer(1)]], \
                 constant ReductionMeta &meta [[buffer(2)]], \
                 uint3 gid [[thread_position_in_grid]]) { \
    ulong outputLinear = linear_id(gid, meta.gridWidth, meta.gridHeight); \
    if (outputLinear >= meta.outputCount) return; \
    uint accumulator = input[reduction_offset(outputLinear, 0ul, meta)]; \
    for (ulong contributor = 1ul; contributor < meta.reducedCount; ++contributor) { \
        accumulator = exact_extreme(accumulator, input[reduction_offset(outputLinear, contributor, meta)], MAXIMUM); \
    } \
    output[outputLinear] = accumulator; \
}

REDUCTION_KERNEL(reduction_min, false)
REDUCTION_KERNEL(reduction_max, true)

#define SCAN_KERNEL(NAME, PRODUCT) \
kernel void NAME(device const uint *input [[buffer(0)]], \
                 device uint *output [[buffer(1)]], \
                 constant ScanMeta &meta [[buffer(2)]], \
                 uint3 gid [[thread_position_in_grid]]) { \
    ulong line = linear_id(gid, meta.gridWidth, meta.gridHeight); \
    if (line >= meta.lineCount) return; \
    ulong outer = line / meta.inner; \
    ulong innerIndex = line % meta.inner; \
    ulong base = outer * meta.axisLength * meta.inner + innerIndex; \
    uint accumulator = PRODUCT ? 0x3f800000u : 0x00000000u; \
    bool hasContributor = false; \
    for (ulong step = 0ul; step < meta.axisLength; ++step) { \
        ulong logical = meta.reverse != 0u ? (meta.axisLength - 1ul - step) : step; \
        ulong offset = base + logical * meta.inner; \
        uint value = input[offset]; \
        if (meta.exclusive != 0u) output[offset] = hasContributor ? accumulator : (PRODUCT ? 0x3f800000u : 0x00000000u); \
        if (!hasContributor) { \
            accumulator = value; \
            hasContributor = true; \
        } else { \
            float left = as_type<float>(accumulator); \
            float right = as_type<float>(value); \
            float result = PRODUCT ? (left * right) : (left + right); \
            accumulator = as_type<uint>(result); \
        } \
        if (meta.exclusive == 0u) output[offset] = accumulator; \
    } \
}

SCAN_KERNEL(scan_sum, false)
SCAN_KERNEL(scan_prod, true)
