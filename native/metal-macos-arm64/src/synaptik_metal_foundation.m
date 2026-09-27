#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import "synaptik_exact_kernels.h"
#import "synaptik_task0053_candidate_kernels.h"
#import "synaptik_task0059_data_kernels.h"
#import "synaptik_task0060_reduction_kernels.h"
#import "synaptik_task0061_matmul_kernels.h"
#import "synaptik_task0063_ordering_kernels.h"
#import "synaptik_task0064_convolution_pooling_kernels.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SYNAPTIK_EXPORT __attribute__((visibility("default")))
#define SYNAPTIK_MAX_RANK 16U
#define SYNAPTIK_MAX_SELECTOR_EXPANSION 16U
#define SYNAPTIK_MAX_NODE_INPUTS 16U
#define SYNAPTIK_MAX_NODE_OUTPUTS 5U
#define SYNAPTIK_MAX_ATTRIBUTE_WORDS 65U
#define SYNAPTIK_TASK0064_MAX_POOL_KERNEL_POSITIONS 65536U

enum {
    SYNAPTIK_METAL_STATUS_OK = 0,
    SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT = 1,
    SYNAPTIK_METAL_STATUS_NO_DEVICE = 2,
    SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE = 3,
    SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED = 4,
    SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS = 5,
    SYNAPTIK_METAL_STATUS_COPY_FAILED = 6,
    SYNAPTIK_METAL_STATUS_INTERNAL_ERROR = 7,
    SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE = 8,
    SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED = 9,
    SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE = 10,
    SYNAPTIK_METAL_STATUS_EXECUTION_FAILED = 11,
    SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED = 12,
    SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION = 13
};
typedef enum : uint32_t {
    SYNAPTIK_METAL_MPSGRAPH_NEG = 1U,
    SYNAPTIK_METAL_MPSGRAPH_ADD = 2U,
    SYNAPTIK_METAL_MPSGRAPH_SUB = 3U,
    SYNAPTIK_METAL_MPSGRAPH_MUL = 4U,
    SYNAPTIK_METAL_MPSGRAPH_DIV = 5U,
    SYNAPTIK_METAL_MPSGRAPH_RESHAPE = 6U,
    SYNAPTIK_METAL_MPSGRAPH_EXPAND = 7U,
    SYNAPTIK_METAL_MPSGRAPH_PERMUTE = 8U,
    SYNAPTIK_METAL_MPSGRAPH_EXPAND_DIMS = 9U,
    SYNAPTIK_METAL_MPSGRAPH_SQUEEZE = 10U,
    SYNAPTIK_METAL_MPSGRAPH_CONTIGUOUS = 11U,
    SYNAPTIK_METAL_MPSGRAPH_ABS = 12U,
    SYNAPTIK_METAL_MPSGRAPH_SUM = 13U,
    SYNAPTIK_METAL_MPSGRAPH_MEAN = 14U,
    SYNAPTIK_METAL_MPSGRAPH_MATMUL = 15U,
    SYNAPTIK_METAL_MPSGRAPH_GATHER = 16U,
    SYNAPTIK_METAL_MPSGRAPH_ONE_HOT = 17U,
    SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS = 18U,
    SYNAPTIK_METAL_MPSGRAPH_UNFOLD_AXIS = 19U,
    SYNAPTIK_METAL_CUSTOM_GT = 20U,
    SYNAPTIK_METAL_CUSTOM_GE = 21U,
    SYNAPTIK_METAL_CUSTOM_LT = 22U,
    SYNAPTIK_METAL_CUSTOM_LE = 23U,
    SYNAPTIK_METAL_CUSTOM_EQ = 24U,
    SYNAPTIK_METAL_CUSTOM_NE = 25U,
    SYNAPTIK_METAL_CUSTOM_TENSOR_MIN = 26U,
    SYNAPTIK_METAL_CUSTOM_TENSOR_MAX = 27U,
    SYNAPTIK_METAL_CUSTOM_SCALAR_MIN = 28U,
    SYNAPTIK_METAL_CUSTOM_SCALAR_MAX = 29U,
    SYNAPTIK_METAL_CUSTOM_CLAMP = 30U,
    SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN = 31U,
    SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX = 32U,
    SYNAPTIK_METAL_CUSTOM_CUM_SUM = 33U,
    SYNAPTIK_METAL_CUSTOM_CUM_PROD = 34U,
    SYNAPTIK_METAL_CUSTOM_CONV2D = 36U,
    SYNAPTIK_METAL_CUSTOM_CONV3D = 37U,
    SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW = 38U,
    SYNAPTIK_METAL_CUSTOM_CAST = 39U,
    SYNAPTIK_METAL_BOOL_IS_FINITE = 40U,
    SYNAPTIK_METAL_BOOL_IS_NAN = 41U,
    SYNAPTIK_METAL_BOOL_IS_INF = 42U,
    SYNAPTIK_METAL_BOOL_AND = 43U,
    SYNAPTIK_METAL_BOOL_OR = 44U,
    SYNAPTIK_METAL_BOOL_NOT = 45U,
    SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD = 46U,
    SYNAPTIK_METAL_MPSGRAPH_SCALAR_SUB = 47U,
    SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL = 48U,
    SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV = 49U,
    SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW = 50U,
    SYNAPTIK_METAL_BOOL_WHERE = 51U,
    SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL = 52U,
    SYNAPTIK_METAL_MPSGRAPH_LOG = 53U,
    SYNAPTIK_METAL_MPSGRAPH_LOG1P = 54U,
    SYNAPTIK_METAL_CUSTOM_EXP = 55U,
    SYNAPTIK_METAL_MPSGRAPH_EXPM1 = 56U,
    SYNAPTIK_METAL_MPSGRAPH_ERF = 57U,
    SYNAPTIK_METAL_MPSGRAPH_SQRT = 58U,
    SYNAPTIK_METAL_MPSGRAPH_RSQRT = 59U,
    SYNAPTIK_METAL_CUSTOM_FLOOR = 60U,
    SYNAPTIK_METAL_CUSTOM_CEIL = 61U,
    SYNAPTIK_METAL_CUSTOM_SIGN = 62U,
    SYNAPTIK_METAL_CUSTOM_RELU = 63U,
    SYNAPTIK_METAL_CUSTOM_SIGMOID = 64U,
    SYNAPTIK_METAL_MPSGRAPH_TANH = 65U,
    SYNAPTIK_METAL_MPSGRAPH_GELU = 66U,
    SYNAPTIK_METAL_MPSGRAPH_GELU_TANH = 67U,
    SYNAPTIK_METAL_MPSGRAPH_SILU = 68U,
    SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS = 69U,
    SYNAPTIK_METAL_MPSGRAPH_SCATTER_ADD = 70U,
    SYNAPTIK_METAL_CUSTOM_GATHER_ND = 71U,
    SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND = 72U,
    SYNAPTIK_METAL_CUSTOM_SELECT = 73U,
    SYNAPTIK_METAL_CUSTOM_PAD = 74U,
    SYNAPTIK_METAL_CUSTOM_SLICE = 75U,
    SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE = 76U,
    SYNAPTIK_METAL_CUSTOM_CONCAT = 77U,
    SYNAPTIK_METAL_CUSTOM_STACK = 78U,
    SYNAPTIK_METAL_CUSTOM_TILE = 79U,
    SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS = 80U,
    SYNAPTIK_METAL_CUSTOM_UNFOLD2D = 81U,
    SYNAPTIK_METAL_MPSGRAPH_FOLD2D = 82U,
    SYNAPTIK_METAL_CUSTOM_UNFOLD3D = 83U,
    SYNAPTIK_METAL_MPSGRAPH_FOLD3D = 84U,
    SYNAPTIK_METAL_MPSGRAPH_MSE = 85U,
    SYNAPTIK_METAL_CUSTOM_SORT = 94U,
    SYNAPTIK_METAL_CUSTOM_ARGSORT = 95U,
    SYNAPTIK_METAL_CUSTOM_TOP_K = 96U,
    SYNAPTIK_METAL_CUSTOM_MAX_POOL2D = 97U,
    SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D = 98U,
    SYNAPTIK_METAL_CUSTOM_MAX_POOL3D = 99U,
    SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D = 100U,
    SYNAPTIK_METAL_CUSTOM_PROD = 106U,
    SYNAPTIK_METAL_CUSTOM_ALL = 107U,
    SYNAPTIK_METAL_CUSTOM_ANY = 108U,
    SYNAPTIK_METAL_CUSTOM_ARG_MAX = 109U,
    SYNAPTIK_METAL_CUSTOM_ARG_MIN = 110U,
    SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP = 111U,
    SYNAPTIK_METAL_MPSGRAPH_VARIANCE = 112U,
    SYNAPTIK_METAL_MPSGRAPH_STANDARD_DEVIATION = 113U,
    SYNAPTIK_METAL_MPSGRAPH_L1_NORM = 114U,
    SYNAPTIK_METAL_MPSGRAPH_L2_NORM = 115U
} SynaptikMetalOperation;

typedef enum : uint32_t {
    SYNAPTIK_METAL_ROUTE_MPSGRAPH = 2U,
    SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM = 3U
} SynaptikMetalPreparedRoute;

typedef enum : uint32_t {
    SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE = 0U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_TARGET_SHAPE = 1U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_PERMUTATION = 2U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS = 3U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_REDUCTION = 4U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_DEPTH = 5U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_WINDOW_AXIS = 6U,
    SYNAPTIK_METAL_CUSTOM_ATTR_SCALAR_VALUE = 7U,
    SYNAPTIK_METAL_CUSTOM_ATTR_CLAMP_RANGE = 8U,
    SYNAPTIK_METAL_CUSTOM_ATTR_SCAN = 9U,
    SYNAPTIK_METAL_CUSTOM_ATTR_WINDOW_2D = 19U,
    SYNAPTIK_METAL_CUSTOM_ATTR_WINDOW_3D = 22U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_MSE = 25U,
    SYNAPTIK_METAL_CUSTOM_ATTR_SORT = 31U,
    SYNAPTIK_METAL_CUSTOM_ATTR_TOP_K = 32U,
    SYNAPTIK_METAL_CUSTOM_ATTR_ARG_EXTREMA = 36U,
    SYNAPTIK_METAL_CUSTOM_ATTR_CONV_2D = 40U,
    SYNAPTIK_METAL_CUSTOM_ATTR_CONV_3D = 41U,
} SynaptikMetalAttribute;

typedef enum : uint32_t {
    SYNAPTIK_METAL_REDUCTION_FULL = 1U,
    SYNAPTIK_METAL_REDUCTION_SINGLE_AXIS = 2U,
    SYNAPTIK_METAL_REDUCTION_MULTI_AXIS = 3U,
    SYNAPTIK_METAL_REDUCTION_SUM_TO_SHAPE = 4U
} SynaptikMetalReductionForm;

typedef enum : uint8_t {
    SYNAPTIK_METAL_VALUE_UNAVAILABLE = 0U,
    SYNAPTIK_METAL_VALUE_CANONICAL = 1U,
    SYNAPTIK_METAL_VALUE_AFFINE_VIEW = 2U,
    SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT = 3U
} SynaptikMetalValueState;

typedef enum : uint8_t {
    SYNAPTIK_METAL_TYPE_UNAVAILABLE = 0U,
    SYNAPTIK_METAL_TYPE_FLOAT32 = 1U,
    SYNAPTIK_METAL_TYPE_INT32 = 2U,
    SYNAPTIK_METAL_TYPE_BOOL = 3U,
    SYNAPTIK_METAL_TYPE_FLOAT64 = 4U,
    SYNAPTIK_METAL_TYPE_BFLOAT16 = 5U,
    SYNAPTIK_METAL_TYPE_INT64 = 6U
} SynaptikMetalValueType;

typedef struct {
    uint32_t operation;
    uint32_t attribute_kind;
    uint32_t first_input;
    uint32_t second_input;
    uint32_t output;
    uint32_t attribute_count;
    uint32_t axis;
    uint32_t auxiliary;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t inputs[SYNAPTIK_MAX_NODE_INPUTS];
    uint32_t outputs[SYNAPTIK_MAX_NODE_OUTPUTS];
    uint64_t attribute_values[SYNAPTIK_MAX_ATTRIBUTE_WORDS];
} SynaptikMetalDecodedNode;

static const uint8_t SYNAPTIK_MIN_INPUTS[116] = {
    0U, 1U, 2U, 2U, 2U, 2U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 2U, 2U, 1U, 3U, 1U,
    2U, 2U, 2U, 2U, 2U, 2U, 2U, 2U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 3U, 2U, 2U, 2U, 1U,
    1U, 1U, 1U, 2U, 2U, 1U, 1U, 1U, 1U, 1U, 1U, 3U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 2U, 3U, 2U, 3U, 1U, 1U, 1U, 2U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 2U, 2U, 2U, 5U, 5U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 2U, 0U, 5U, 5U, 6U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
};
static const uint8_t SYNAPTIK_MAX_INPUTS[116] = {
    0U, 1U, 2U, 2U, 2U, 2U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 2U, 2U, 1U, 3U, 1U,
    2U, 2U, 2U, 2U, 2U, 2U, 2U, 2U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 4U, 3U, 3U, 2U, 1U,
    1U, 1U, 1U, 2U, 2U, 1U, 1U, 1U, 1U, 1U, 1U, 3U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 2U, 3U, 2U, 3U, 1U, 1U, 1U, 2U, 255U, 255U, 1U,
    1U, 1U, 1U, 1U, 1U, 2U, 2U, 2U, 5U, 5U, 3U, 2U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 2U, 0U, 6U, 6U, 7U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
};
static const uint8_t SYNAPTIK_MIN_OUTPUTS[116] = {
    0U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 5U, 1U, 1U, 1U, 1U, 1U, 1U, 2U, 1U, 1U, 1U,
    1U, 3U, 1U, 2U, 2U, 3U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
};
static const uint8_t SYNAPTIK_MAX_OUTPUTS[116] = {
    0U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 2U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 5U, 1U, 1U, 1U, 1U, 1U, 1U, 2U, 1U, 1U, 1U,
    1U, 3U, 1U, 2U, 2U, 3U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
};
static const uint8_t SYNAPTIK_ATTRIBUTE_KINDS[116] = {
    0U, 0U, 0U, 0U, 0U, 0U, 1U, 1U, 2U, 3U, 3U, 0U, 0U, 4U, 4U, 0U, 3U, 5U, 12U, 6U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 7U, 7U, 8U, 4U, 4U, 9U, 9U, 39U, 40U, 41U, 0U, 10U,
    0U, 0U, 0U, 0U, 0U, 0U, 7U, 7U, 7U, 7U, 7U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 3U, 3U, 11U, 13U, 14U, 16U, 17U, 17U, 3U, 3U, 18U,
    6U, 19U, 21U, 22U, 24U, 25U, 26U, 27U, 29U, 30U, 28U, 28U, 3U, 3U, 31U, 31U, 32U, 19U, 19U, 22U,
    22U, 33U, 34U, 35U, 35U, 35U, 4U, 4U, 4U, 36U, 36U, 4U, 38U, 38U, 4U, 4U,
};

static uint32_t synaptik_read_le32(const uint8_t *bytes) {
    return (uint32_t)bytes[0]
            | ((uint32_t)bytes[1] << 8U)
            | ((uint32_t)bytes[2] << 16U)
            | ((uint32_t)bytes[3] << 24U);
}

static uint64_t synaptik_read_le64(const uint8_t *bytes) {
    return (uint64_t)synaptik_read_le32(bytes)
            | ((uint64_t)synaptik_read_le32(bytes + 4U) << 32U);
}

static BOOL synaptik_node_signature_is_valid(
        uint32_t operation, uint32_t attribute_kind,
        uint32_t input_count, uint32_t output_count) {
    if (operation == 0U || operation > 115U) return NO;
    uint32_t maximum_inputs = SYNAPTIK_MAX_INPUTS[operation] == 255U
            ? UINT32_MAX : SYNAPTIK_MAX_INPUTS[operation];
    BOOL alternate_attribute =
            ((operation == 75U || operation == 76U) && attribute_kind == 15U)
            || (operation == 81U && attribute_kind == 20U)
            || (operation == 83U && attribute_kind == 23U);
    return input_count >= SYNAPTIK_MIN_INPUTS[operation]
            && input_count <= maximum_inputs
            && output_count >= SYNAPTIK_MIN_OUTPUTS[operation]
            && output_count <= SYNAPTIK_MAX_OUTPUTS[operation]
            && !(operation == 90U && input_count == 2U)
            && (attribute_kind == SYNAPTIK_ATTRIBUTE_KINDS[operation]
                    || alternate_attribute);
}

static uint64_t synaptik_attribute_word(const uint8_t *words, uint32_t index) {
    return synaptik_read_le64(words + (uint64_t)index * 8U);
}

static BOOL synaptik_word_is_positive(uint64_t value) {
    return value > 0U && value <= INT64_MAX;
}

static BOOL synaptik_word_is_non_negative(uint64_t value) {
    return value <= INT64_MAX;
}

static BOOL synaptik_scalar_is_valid(const uint8_t *words, uint32_t count, uint32_t offset) {
    if (offset > count || count - offset < 2U) return NO;
    uint64_t type = synaptik_attribute_word(words, offset);
    uint64_t bits = synaptik_attribute_word(words, offset + 1U);
    if (type < 1U || type > 6U) return NO;
    if ((type == 1U || type == 2U) && bits > UINT32_MAX) return NO;
    if (type == 3U && bits > 1U) return NO;
    return type != 5U || bits <= UINT16_MAX;
}

static BOOL synaptik_shape_is_valid(
        const uint8_t *words, uint32_t count, uint32_t *cursor, BOOL positive_rank) {
    if (*cursor >= count) return NO;
    uint64_t rank = synaptik_attribute_word(words, (*cursor)++);
    if (rank > SYNAPTIK_MAX_RANK || (positive_rank && rank == 0U)
            || rank > count - *cursor) return NO;
    for (uint32_t axis = 0U; axis < (uint32_t)rank; axis++)
        if (!synaptik_word_is_positive(synaptik_attribute_word(words, (*cursor)++)))
            return NO;
    return YES;
}

static BOOL synaptik_axes_are_valid(
        const uint8_t *words, uint32_t offset, uint32_t count, uint32_t bound,
        BOOL complete) {
    if (bound > SYNAPTIK_MAX_RANK || count > SYNAPTIK_MAX_RANK
            || (complete && count != bound)) return NO;
    BOOL seen[SYNAPTIK_MAX_RANK] = {NO};
    for (uint32_t index = 0U; index < count; index++) {
        uint64_t axis = synaptik_attribute_word(words, offset + index);
        if (axis >= bound || seen[axis]) return NO;
        seen[axis] = YES;
    }
    return YES;
}

static BOOL synaptik_window_is_valid(
        const uint8_t *words, uint32_t count, uint32_t dimensions,
        BOOL padded, BOOL fold) {
    uint32_t cursor = 0U;
    if (fold && !synaptik_shape_is_valid(words, count, &cursor, NO)) return NO;
    uint32_t base_count = dimensions * 4U + 1U;
    uint32_t expected = cursor + base_count + (padded ? 2U : 0U);
    if (count != expected) return NO;
    for (uint32_t index = 0U; index < dimensions * 2U; index++)
        if (!synaptik_word_is_positive(synaptik_attribute_word(words, cursor + index)))
            return NO;
    for (uint32_t index = dimensions * 2U; index < dimensions * 3U; index++)
        if (!synaptik_word_is_non_negative(synaptik_attribute_word(words, cursor + index)))
            return NO;
    for (uint32_t index = dimensions * 3U; index < dimensions * 4U; index++)
        if (!synaptik_word_is_positive(synaptik_attribute_word(words, cursor + index)))
            return NO;
    if (synaptik_attribute_word(words, cursor + dimensions * 4U) > 1U) return NO;
    return !padded || synaptik_scalar_is_valid(words, count, cursor + base_count);
}

static BOOL synaptik_convolution_is_valid(
        const uint8_t *words, uint32_t count, uint32_t dimensions) {
    if (count != dimensions * 3U + 1U) return NO;
    for (uint32_t index = 0U; index < dimensions; index++)
        if (!synaptik_word_is_positive(synaptik_attribute_word(words, index))) return NO;
    for (uint32_t index = dimensions; index < dimensions * 2U; index++)
        if (!synaptik_word_is_non_negative(synaptik_attribute_word(words, index))) return NO;
    for (uint32_t index = dimensions * 2U; index <= dimensions * 3U; index++)
        if (!synaptik_word_is_positive(synaptik_attribute_word(words, index))) return NO;
    return YES;
}

static BOOL synaptik_attribute_is_valid(
        uint32_t operation, uint32_t kind, const uint8_t *words, uint32_t count) {
    if (kind == 0U) return count == 0U;
    if (kind == 1U) {
        uint32_t cursor = 0U;
        return synaptik_shape_is_valid(words, count, &cursor, YES) && cursor == count;
    }
    if (kind == 2U) {
        if (count < 2U) return NO;
        uint64_t axes = synaptik_attribute_word(words, 0U);
        return axes >= 1U && axes <= SYNAPTIK_MAX_RANK && count == axes + 1U
                && synaptik_axes_are_valid(words, 1U, (uint32_t)axes, (uint32_t)axes, YES);
    }
    if (kind == 3U || kind == 37U)
        return count == 1U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK;
    if (kind == 4U) {
        if (count < 3U) return NO;
        uint64_t form = synaptik_attribute_word(words, 0U);
        uint64_t keep = synaptik_attribute_word(words, 1U);
        uint64_t item_count = synaptik_attribute_word(words, 2U);
        if (form < 1U || form > 4U || keep > 1U || item_count > SYNAPTIK_MAX_RANK
                || count != item_count + 3U
                || (form == 1U && (item_count != 0U || keep != 0U))
                || (form == 2U && item_count != 1U)
                || (form == 4U && (operation != 13U || keep != 0U))) return NO;
        if (form == 4U) {
            for (uint32_t index = 0U; index < item_count; index++)
                if (!synaptik_word_is_positive(synaptik_attribute_word(words, 3U + index)))
                    return NO;
            return YES;
        }
        return form == 1U || synaptik_axes_are_valid(
                words, 3U, (uint32_t)item_count, SYNAPTIK_MAX_RANK, NO);
    }
    if (kind == 5U)
        return count == 1U && synaptik_word_is_positive(synaptik_attribute_word(words, 0U));
    if (kind == 6U)
        return count == 3U
                && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_word_is_positive(synaptik_attribute_word(words, 1U))
                && synaptik_word_is_positive(synaptik_attribute_word(words, 2U))
                && (operation != 19U
                        || synaptik_attribute_word(words, 1U) <= SYNAPTIK_MAX_SELECTOR_EXPANSION);
    if (kind == 7U)
        return count == 2U && synaptik_scalar_is_valid(words, count, 0U)
                && (operation > 34U || synaptik_attribute_word(words, 0U) == 1U);
    if (kind == 8U)
        return count == 4U && synaptik_scalar_is_valid(words, count, 0U)
                && synaptik_scalar_is_valid(words, count, 2U)
                && synaptik_attribute_word(words, 0U) == 1U
                && synaptik_attribute_word(words, 2U) == 1U;
    if (kind == 9U)
        return count == 3U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_attribute_word(words, 1U) <= 1U
                && synaptik_attribute_word(words, 2U) <= 1U;
    if (kind == 10U)
        return count == 1U && synaptik_attribute_word(words, 0U) >= 1U
                && synaptik_attribute_word(words, 0U) <= 6U;
    if (kind == 11U)
        return count == 1U && synaptik_attribute_word(words, 0U) <= SYNAPTIK_MAX_RANK;
    if (kind == 12U)
        return count == 2U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_attribute_word(words, 1U) >= 1U
                && synaptik_attribute_word(words, 1U) <= 5U
                && (operation != 18U || synaptik_attribute_word(words, 1U) == 1U);
    if (kind == 13U)
        return count == 2U && synaptik_attribute_word(words, 0U) <= UINT32_MAX
                && synaptik_attribute_word(words, 1U) >= 1U
                && synaptik_attribute_word(words, 1U) <= 5U;
    if (kind == 14U)
        return count == 2U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK;
    if (kind == 15U) {
        if (count < 2U) return NO;
        uint64_t target_rank = synaptik_attribute_word(words, 0U);
        if (target_rank > SYNAPTIK_MAX_RANK
                || 1U + target_rank >= count) return NO;
        for (uint32_t axis = 0U; axis < (uint32_t)target_rank; axis++)
            if (!synaptik_word_is_positive(
                        synaptik_attribute_word(words, 1U + axis))) return NO;
        uint32_t prefix_offset = 1U + (uint32_t)target_rank;
        if (synaptik_attribute_word(words, prefix_offset) != target_rank
                || count != 2U + target_rank * 2U) return NO;
        for (uint32_t axis = 0U; axis < (uint32_t)target_rank; axis++)
            if (!synaptik_word_is_non_negative(
                        synaptik_attribute_word(
                                words, prefix_offset + 1U + axis))) return NO;
        return YES;
    }
    if (kind == 16U) {
        if (count < 3U) return NO;
        uint64_t rank = synaptik_attribute_word(words, 0U);
        if (rank > SYNAPTIK_MAX_RANK || count != 1U + rank * 2U + 2U) return NO;
        for (uint32_t index = 1U; index <= rank * 2U; index++)
            if (!synaptik_word_is_non_negative(synaptik_attribute_word(words, index))) return NO;
        return synaptik_scalar_is_valid(words, count, 1U + (uint32_t)rank * 2U);
    }
    if (kind == 17U) {
        if (count < 1U) return NO;
        uint64_t item_count = synaptik_attribute_word(words, 0U);
        if (item_count > SYNAPTIK_MAX_RANK
                || count != 1U + item_count * 4U) return NO;
        BOOL seen[SYNAPTIK_MAX_RANK] = {NO};
        for (uint32_t index = 0U; index < item_count; index++) {
            if (!synaptik_word_is_positive(synaptik_attribute_word(
                            words, 1U + (uint32_t)item_count + index))) return NO;
            uint64_t axis = synaptik_attribute_word(
                    words, 1U + (uint32_t)item_count * 2U + index);
            if (axis >= SYNAPTIK_MAX_RANK || seen[axis]) return NO;
            seen[axis] = YES;
            if (synaptik_attribute_word(
                        words, 1U + (uint32_t)item_count * 3U + index) == 0U) return NO;
        }
        return YES;
    }
    if (kind == 18U) {
        if (count < 1U) return NO;
        uint64_t item_count = synaptik_attribute_word(words, 0U);
        if (item_count > SYNAPTIK_MAX_RANK || count != item_count + 1U) return NO;
        for (uint32_t index = 0U; index < item_count; index++)
            if (!synaptik_word_is_positive(synaptik_attribute_word(words, 1U + index)))
                return NO;
        return YES;
    }
    if (kind == 19U) return synaptik_window_is_valid(words, count, 2U, NO, NO);
    if (kind == 20U) return synaptik_window_is_valid(words, count, 2U, YES, NO);
    if (kind == 21U) return synaptik_window_is_valid(words, count, 2U, NO, YES);
    if (kind == 22U) return synaptik_window_is_valid(words, count, 3U, NO, NO);
    if (kind == 23U) return synaptik_window_is_valid(words, count, 3U, YES, NO);
    if (kind == 24U) return synaptik_window_is_valid(words, count, 3U, NO, YES);
    if (kind == 25U)
        return count == 1U && synaptik_attribute_word(words, 0U) >= 1U
                && synaptik_attribute_word(words, 0U) <= 3U;
    if (kind == 26U)
        return count == 2U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_attribute_word(words, 1U) >= 1U
                && synaptik_attribute_word(words, 1U) <= 3U;
    if (kind == 27U) {
        if (count < 3U || synaptik_attribute_word(words, 0U) >= SYNAPTIK_MAX_RANK
                || synaptik_attribute_word(words, 1U) < 1U
                || synaptik_attribute_word(words, 1U) > 3U) return NO;
        uint64_t present = synaptik_attribute_word(words, 2U);
        return present == 0U ? count == 3U
                : present == 1U && count == 5U
                        && synaptik_scalar_is_valid(words, count, 3U);
    }
    if (kind == 28U) {
        uint32_t cursor = 0U;
        return synaptik_shape_is_valid(words, count, &cursor, NO)
                && count == cursor + 2U && synaptik_scalar_is_valid(words, count, cursor);
    }
    if (kind == 29U)
        return count == 3U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_scalar_is_valid(words, count, 1U);
    if (kind == 30U)
        return count == 5U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_scalar_is_valid(words, count, 1U)
                && synaptik_scalar_is_valid(words, count, 3U);
    if (kind == 31U)
        return count == 2U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_attribute_word(words, 1U) <= 1U;
    if (kind == 32U)
        return count == 4U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_word_is_positive(synaptik_attribute_word(words, 1U))
                && synaptik_attribute_word(words, 2U) <= 1U
                && synaptik_attribute_word(words, 3U) <= 1U;
    if (kind == 33U) {
        if (count != 1U) return NO;
        uint64_t bits = synaptik_attribute_word(words, 0U);
        double probability;
        memcpy(&probability, &bits, sizeof(probability));
        return isfinite(probability) && probability >= 0.0 && probability < 1.0;
    }
    if (kind == 34U) return count == 2U;
    if (kind == 35U)
        return count == 1U && synaptik_attribute_word(words, 0U) >= 1U
                && synaptik_attribute_word(words, 0U) <= 2U;
    if (kind == 36U)
        return count == 3U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_attribute_word(words, 1U) <= 1U
                && synaptik_attribute_word(words, 2U) >= 1U
                && synaptik_attribute_word(words, 2U) <= 2U;
    if (kind == 38U) {
        if (count < 3U) return NO;
        uint64_t axes = synaptik_attribute_word(words, 0U);
        return axes <= SYNAPTIK_MAX_RANK && count == axes + 3U
                && synaptik_axes_are_valid(
                        words, 1U, (uint32_t)axes, SYNAPTIK_MAX_RANK, NO)
                && synaptik_attribute_word(words, 1U + (uint32_t)axes) <= 1U
                && synaptik_word_is_non_negative(
                        synaptik_attribute_word(words, 2U + (uint32_t)axes));
    }
    if (kind == 39U) {
        if (count < 2U) return NO;
        uint64_t present = synaptik_attribute_word(words, 0U);
        if (present == 0U)
            return count == 2U && synaptik_attribute_word(words, 1U) <= 1U;
        if (present != 1U || count != 4U || !synaptik_scalar_is_valid(words, count, 1U))
            return NO;
        uint64_t type = synaptik_attribute_word(words, 1U);
        return (type == 1U || type == 4U || type == 5U)
                && synaptik_attribute_word(words, 3U) <= 1U;
    }
    if (kind == 40U) return synaptik_convolution_is_valid(words, count, 2U);
    if (kind == 41U) return synaptik_convolution_is_valid(words, count, 3U);
    return NO;
}



@interface SynaptikMetalContextBox : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@end
@implementation SynaptikMetalContextBox @end

@interface SynaptikMetalBufferBox : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) uint64_t logicalByteSize;
@end
@implementation SynaptikMetalBufferBox @end

@interface SynaptikMetalIndexValidation : NSObject
@property(nonatomic) uint32_t operation;
@property(nonatomic) uint32_t nodeIndex;
@property(nonatomic) NSUInteger feedPosition;
@property(nonatomic) BOOL preflight;
@property(nonatomic) uint64_t elementCount;
@property(nonatomic) uint64_t bound;
@property(nonatomic) uint32_t axis;
@property(nonatomic) uint32_t indexType;
@property(nonatomic) uint32_t tupleDepth;
@property(nonatomic) uint64_t targetCount;
@property(nonatomic, strong, nullable) NSMutableData *coordinateExtents;
@property(nonatomic, strong, nullable) NSMutableData *dataStrides;
@property(nonatomic, strong, nullable) NSMutableData *prefixExtents;
@property(nonatomic, strong, nullable) NSMutableData *targetScratch;
@end
@implementation SynaptikMetalIndexValidation @end

typedef struct {
    uint64_t dims[16];
    uint64_t leftStrides[16];
    uint64_t rightStrides[16];
    uint64_t elementCount;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t rank;
    uint32_t reserved;
} SynaptikMetalBinaryMeta;

typedef struct {
    uint64_t elementCount;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t scalar;
    uint32_t reserved;
} SynaptikMetalPointMeta;

typedef struct {
    uint64_t dims[16];
    uint64_t conditionStrides[16];
    uint64_t trueStrides[16];
    uint64_t falseStrides[16];
    uint64_t elementCount;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t rank;
    uint32_t reserved;
} SynaptikMetalTernaryMeta;

typedef struct {
    uint64_t elementCount;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t lower;
    uint32_t upper;
} SynaptikMetalClampMeta;

typedef struct {
    uint64_t dims[16];
    uint64_t strides[16];
    uint64_t outputCount;
    uint64_t reducedCount;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t rank;
    uint32_t axesMask;
} SynaptikMetalReductionMeta;

typedef struct {
    uint64_t elementCount;
    uint64_t axisLength;
    uint64_t inner;
    uint64_t lineCount;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t exclusive;
    uint32_t reverse;
} SynaptikMetalScanMeta;

typedef struct {
    uint64_t inputDims[16];
    uint64_t outputDims[16];
    uint64_t auxiliaryDims[16];
    uint64_t inputStrides[16];
    uint64_t outputStrides[16];
    uint64_t auxiliaryStrides[16];
    uint64_t inputOffset;
    uint64_t outputOffset;
    uint64_t auxiliaryOffset;
    uint64_t inputExtents[16];
    uint64_t inputPrefixes[16];
    uint64_t attrs[SYNAPTIK_MAX_ATTRIBUTE_WORDS];
    uint64_t elementCount;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t operation;
    uint32_t inputRank;
    uint32_t outputRank;
    uint32_t auxiliaryRank;
    uint32_t width;
    uint32_t sourceType;
    uint32_t targetType;
    uint32_t inputCount;
    uint32_t attributeKind;
    uint32_t attributeCount;
    uint32_t indexType;
    uint32_t reserved;
} SynaptikMetalDataMeta;

typedef struct {
    uint32_t elementCount;
    uint32_t outputCount;
    uint32_t axisLength;
    uint32_t inner;
    uint32_t k;
    uint32_t valueType;
    uint32_t width;
    uint32_t direction;
    uint32_t sorted;
    uint32_t tieLast;
} SynaptikMetalOrderingMeta;

@interface SynaptikMetalProgramStep : NSObject
@property(nonatomic) BOOL custom;
@property(nonatomic) uint32_t operation;
@property(nonatomic) uint32_t stage;
@property(nonatomic) uint32_t firstInput;
@property(nonatomic) uint32_t secondInput;
@property(nonatomic) uint32_t auxiliaryInput;
@property(nonatomic) uint32_t output;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *outputValues;
@property(nonatomic, strong, nullable) id<MTLComputePipelineState> pipeline;
@property(nonatomic, strong, nullable) id<MTLBuffer> metadata;
@property(nonatomic) MTLSize grid;
@property(nonatomic) NSUInteger threadsPerThreadgroup;
@property(nonatomic, strong, nullable) MPSGraphExecutable *graphExecutable;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *feedValues;
@property(nonatomic, copy, nullable) NSArray<MPSShape *> *feedShapes;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *feedDataTypes;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *feedPermutation;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *targetPermutation;
@property(nonatomic, strong, nullable) MPSShape *targetShape;
@property(nonatomic, strong, nullable) id nestedExecutable;
@property(nonatomic) MPSDataType targetDataType;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *inputValues;
@property(nonatomic, strong, nullable) SynaptikMetalIndexValidation *indexValidation;
@end
@implementation SynaptikMetalProgramStep @end

@interface SynaptikMetalExecutableBox : NSObject
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) SynaptikMetalContextBox *context;
@property(nonatomic, copy) NSArray<MPSShape *> *feedShapes;
@property(nonatomic, copy) NSArray<MPSShape *> *targetShapes;
@property(nonatomic, copy) NSArray<NSNumber *> *feedBytes;
@property(nonatomic, copy) NSArray<NSNumber *> *targetBytes;
@property(nonatomic, copy) NSArray<NSNumber *> *feedPermutation;
@property(nonatomic, copy) NSArray<NSNumber *> *targetPermutation;
@property(nonatomic, copy) NSArray<NSNumber *> *feedDataTypes;
@property(nonatomic, copy) NSArray<NSNumber *> *targetDataTypes;
@property(nonatomic, copy) NSArray<SynaptikMetalIndexValidation *> *indexValidations;
@property(nonatomic, copy) NSArray<NSNumber *> *canonicalBoolInputIndices;
@property(nonatomic, copy) NSArray<MPSShape *> *inputShapes;
@property(nonatomic, copy) NSArray<NSArray<NSNumber *> *> *inputStrides;
@property(nonatomic, copy) NSArray<NSNumber *> *inputOffsets;
@property(nonatomic) BOOL customProgram;
@property(nonatomic) uint32_t valueCount;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *valueBytes;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *targetValueIndices;
@property(nonatomic, copy, nullable) NSArray<SynaptikMetalProgramStep *> *programSteps;
@end
@implementation SynaptikMetalExecutableBox @end

@interface SynaptikMetalNegKernelPipelineBox : NSObject
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property(nonatomic, strong) SynaptikMetalContextBox *context;
@property(nonatomic) uint64_t elementCount;
@property(nonatomic) uint64_t requiredBytes;
@property(nonatomic) NSUInteger threadsPerThreadgroup;
@end
@implementation SynaptikMetalNegKernelPipelineBox @end

SYNAPTIK_EXPORT uint32_t synaptik_metal_foundation_abi_version(void) { return 5U; }

SYNAPTIK_EXPORT int32_t synaptik_metal_context_create(void **out_context) {
    if (out_context == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_context = NULL;
    @try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return SYNAPTIK_METAL_STATUS_NO_DEVICE;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) return SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE;
        SynaptikMetalContextBox *box = [SynaptikMetalContextBox new];
        if (box == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        box.device = device; box.commandQueue = queue;
        *out_context = (__bridge_retained void *)box;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_context_release(void *context) {
    if (context == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { __unused id consumed = (__bridge_transfer id)context;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_create(
        void *context, uint64_t logical_byte_size, void **out_buffer) {
    if (out_buffer == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_buffer = NULL;
    if (context == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try {
        SynaptikMetalContextBox *ctx = (__bridge SynaptikMetalContextBox *)context;
        uint64_t physical = logical_byte_size == 0U ? 1U : logical_byte_size;
        if (physical > (uint64_t)NSUIntegerMax) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        id<MTLBuffer> buffer = [ctx.device newBufferWithLength:(NSUInteger)physical
                options:MTLResourceStorageModeShared];
        if (buffer == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        SynaptikMetalBufferBox *box = [SynaptikMetalBufferBox new];
        if (box == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        box.buffer = buffer; box.logicalByteSize = logical_byte_size;
        *out_buffer = (__bridge_retained void *)box;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_release(void *buffer) {
    if (buffer == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { __unused id consumed = (__bridge_transfer id)buffer;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

static int32_t validate_copy(SynaptikMetalBufferBox *box, uint64_t offset,
        const void *bytes, uint64_t count) {
    if (box == nil || (count != 0U && bytes == NULL)) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    if (offset > box.logicalByteSize || count > box.logicalByteSize - offset)
        return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
    return SYNAPTIK_METAL_STATUS_OK;
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_upload(
        void *buffer, uint64_t offset, const void *source, uint64_t count) {
    if (buffer == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try {
        SynaptikMetalBufferBox *box = (__bridge SynaptikMetalBufferBox *)buffer;
        int32_t status = validate_copy(box, offset, source, count);
        if (status != 0 || count == 0U) return status;
        void *contents = box.buffer.contents;
        if (contents == NULL) return SYNAPTIK_METAL_STATUS_COPY_FAILED;
        memcpy((uint8_t *)contents + (size_t)offset, source, (size_t)count);
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_buffer_download(
        void *buffer, uint64_t offset, void *destination, uint64_t count) {
    if (buffer == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try {
        SynaptikMetalBufferBox *box = (__bridge SynaptikMetalBufferBox *)buffer;
        int32_t status = validate_copy(box, offset, destination, count);
        if (status != 0 || count == 0U) return status;
        void *contents = box.buffer.contents;
        if (contents == NULL) return SYNAPTIK_METAL_STATUS_COPY_FAILED;
        memcpy(destination, (uint8_t *)contents + (size_t)offset, (size_t)count);
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

static NSArray<NSNumber *> *permutation(
        NSArray<MPSGraphTensor *> *reported, NSArray<MPSGraphTensor *> *stable) {
    if (reported == nil || stable == nil || reported.count != stable.count) return nil;
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:reported.count];
    NSMutableIndexSet *seen = [NSMutableIndexSet indexSet];
    for (MPSGraphTensor *tensor in reported) {
        NSUInteger match = NSNotFound;
        for (NSUInteger index = 0; index < stable.count; index++)
            if (tensor == stable[index]) { if (match != NSNotFound) return nil; match = index; }
        if (match == NSNotFound || [seen containsIndex:match]) return nil;
        [seen addIndex:match]; [result addObject:@(match)];
    }
    return [result copy];
}

static BOOL shape_broadcasts_exactly_to(
        MPSShape *left, MPSShape *right, MPSShape *output) {
    NSUInteger output_rank = MAX(left.count, right.count);
    if (output.count != output_rank) return NO;
    NSUInteger left_padding = output_rank - left.count;
    NSUInteger right_padding = output_rank - right.count;
    for (NSUInteger axis = 0; axis < output_rank; axis++) {
        uint64_t left_dimension = axis < left_padding
                ? 1U : left[axis - left_padding].unsignedLongLongValue;
        uint64_t right_dimension = axis < right_padding
                ? 1U : right[axis - right_padding].unsignedLongLongValue;
        if (left_dimension != right_dimension
                && left_dimension != 1U && right_dimension != 1U) return NO;
        if (output[axis].unsignedLongLongValue
                != MAX(left_dimension, right_dimension)) return NO;
    }
    return YES;
}

static BOOL node_values_are_zero_from(
        SynaptikMetalDecodedNode node, uint32_t first) {
    for (uint32_t index = first; index < SYNAPTIK_MAX_RANK; index++)
        if (node.attribute_values[index] != 0U) return NO;
    return YES;
}

static BOOL node_has_no_attributes(SynaptikMetalDecodedNode node) {
    return node.attribute_kind == SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE
            && node.attribute_count == 0U
            && node.axis == UINT32_MAX
            && node.auxiliary == 0U
            && node_values_are_zero_from(node, 0U);
}

static BOOL node_target_matches(
        SynaptikMetalDecodedNode node, MPSShape *output) {
    if (node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_TARGET_SHAPE
            || node.attribute_count != output.count
            || node.attribute_count == 0U
            || node.attribute_count > SYNAPTIK_MAX_RANK
            || node.axis != UINT32_MAX
            || node.auxiliary != 0U
            || !node_values_are_zero_from(node, node.attribute_count))
        return NO;
    for (uint32_t axis = 0; axis < node.attribute_count; axis++)
        if (node.attribute_values[axis] == 0U
                || node.attribute_values[axis] > (uint64_t)NSUIntegerMax
                || node.attribute_values[axis]
                        != output[axis].unsignedLongLongValue)
            return NO;
    return YES;
}

static BOOL shape_expands_exactly_to(MPSShape *input, MPSShape *output) {
    if (input.count > output.count) return NO;
    NSUInteger padding = output.count - input.count;
    for (NSUInteger axis = 0; axis < input.count; axis++) {
        uint64_t source = input[axis].unsignedLongLongValue;
        uint64_t target = output[axis + padding].unsignedLongLongValue;
        if (source != target && source != 1U) return NO;
    }
    return YES;
}

static uint64_t shape_element_count(MPSShape *shape) {
    uint64_t result = 1U;
    for (NSNumber *dimension in shape)
        result *= dimension.unsignedLongLongValue;
    return result;
}

static BOOL task0059_shapes_equal(MPSShape *left, MPSShape *right) {
    return left != nil && right != nil && [left isEqualToArray:right];
}

static BOOL task0059_shape_matches(
        MPSShape *shape, const uint64_t *dimensions, NSUInteger rank) {
    if (shape == nil || dimensions == NULL || shape.count != rank) return NO;
    for (NSUInteger axis = 0U; axis < rank; axis++) {
        if (shape[axis].unsignedLongLongValue != dimensions[axis]) return NO;
    }
    return YES;
}

static BOOL task0059_window_extent(
        uint64_t input,
        uint64_t kernel,
        uint64_t padding,
        uint64_t stride,
        uint64_t dilation,
        BOOL ceil_mode,
        uint64_t *result) {
    if (kernel == 0U || stride == 0U || dilation == 0U || result == NULL
            || kernel - 1U > (UINT64_MAX - 1U) / dilation)
        return NO;
    uint64_t effective = dilation * (kernel - 1U) + 1U;
    if (padding > (UINT64_MAX - input) / 2U) return NO;
    uint64_t padded = input + padding * 2U;
    if (padded < effective) return NO;
    uint64_t numerator = padded - effective;
    *result = numerator / stride
            + ((ceil_mode && numerator % stride != 0U) ? 1U : 0U) + 1U;
    return *result != 0U;
}

static BOOL task0059_mpsgraph_window_origins_fit(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes) {
    NSUInteger dimensions;
    BOOL fold;
    switch ((SynaptikMetalOperation)node.operation) {
        case SYNAPTIK_METAL_CUSTOM_UNFOLD2D:
            dimensions = 2U;
            fold = NO;
            break;
        case SYNAPTIK_METAL_CUSTOM_UNFOLD3D:
            dimensions = 3U;
            fold = NO;
            break;
        case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
            dimensions = 2U;
            fold = YES;
            break;
        case SYNAPTIK_METAL_MPSGRAPH_FOLD3D:
            dimensions = 3U;
            fold = YES;
            break;
        default:
            return YES;
    }
    MPSShape *spatial_shape = shapes[fold ? node.output : node.first_input];
    NSUInteger offset = fold ? dimensions + 3U : 0U;
    NSUInteger first_bounded_axis = dimensions == 3U ? 1U : 0U;
    for (NSUInteger spatial = first_bounded_axis;
            spatial < dimensions; spatial++) {
        uint64_t stride = node.attribute_values[offset + dimensions + spatial];
        uint64_t extent = 0U;
        if (!task0059_window_extent(
                    spatial_shape[spatial + 2U].unsignedLongLongValue,
                    node.attribute_values[offset + spatial],
                    node.attribute_values[offset + dimensions * 2U + spatial],
                    stride,
                    node.attribute_values[offset + dimensions * 3U + spatial],
                    node.attribute_values[offset + dimensions * 4U] != 0U,
                    &extent)
                || extent - 1U > UINT32_MAX / stride)
            return NO;
    }
    return YES;
}

static BOOL task0059_crop_region_matches(
        SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *region) {
    if (node.attribute_kind != 15U || node.attribute_count < 2U
            || input == nil || region == nil) return NO;
    uint64_t target_rank_word = node.attribute_values[0];
    if (target_rank_word > SYNAPTIK_MAX_RANK
            || target_rank_word != input.count
            || 1U + target_rank_word >= node.attribute_count) return NO;
    NSUInteger rank = (NSUInteger)target_rank_word;
    NSUInteger prefix_offset = 1U + rank;
    uint64_t prefix_rank_word = node.attribute_values[prefix_offset];
    if (prefix_rank_word != rank
            || node.attribute_count != 2U + rank * 2U
            || region.count != rank) return NO;
    for (NSUInteger axis = 0U; axis < rank; axis++) {
        uint64_t target = node.attribute_values[1U + axis];
        uint64_t prefix = node.attribute_values[prefix_offset + 1U + axis];
        uint64_t extent = input[axis].unsignedLongLongValue;
        if (target == 0U || prefix > extent || target > extent - prefix
                || region[axis].unsignedLongLongValue != target) return NO;
    }
    return YES;
}

static BOOL synaptik_storage_layout_supported(
        uint32_t value,
        const uint32_t *ranks,
        const uint64_t *dimensions,
        const uint64_t *strides,
        const uint64_t *offsets,
        const uint64_t *spans,
        const uint8_t *present,
        uint32_t minimum_rank) {
    if (ranks == NULL || dimensions == NULL || strides == NULL || offsets == NULL
            || spans == NULL || present == NULL || !present[value]
            || ranks[value] < minimum_rank || ranks[value] > SYNAPTIK_MAX_RANK)
        return NO;
    uint32_t order[SYNAPTIK_MAX_RANK] = {0};
    uint32_t count = 0U;
    for (uint32_t axis = 0U; axis < ranks[value]; axis++) {
        uint64_t dimension =
                dimensions[(size_t)value * SYNAPTIK_MAX_RANK + axis];
        uint64_t stride = strides[(size_t)value * SYNAPTIK_MAX_RANK + axis];
        if (dimension == 0U || stride == 0U || stride > (uint64_t)INT64_MAX)
            return NO;
        if (dimension > 1U) {
            uint32_t insertion = count;
            while (insertion > 0U
                    && strides[(size_t)value * SYNAPTIK_MAX_RANK
                                    + order[insertion - 1U]] > stride) {
                order[insertion] = order[insertion - 1U];
                insertion--;
            }
            order[insertion] = axis;
            count++;
        }
    }
    uint64_t covered = 1U;
    for (uint32_t index = 0U; index < count; index++) {
        uint32_t axis = order[index];
        uint64_t dimension =
                dimensions[(size_t)value * SYNAPTIK_MAX_RANK + axis];
        uint64_t stride = strides[(size_t)value * SYNAPTIK_MAX_RANK + axis];
        if (stride < covered || dimension - 1U > ((uint64_t)INT64_MAX - covered) / stride)
            return NO;
        covered += (dimension - 1U) * stride;
    }
    return offsets[value] <= (uint64_t)INT64_MAX - covered
            && spans[value] == offsets[value] + covered;
}

static BOOL task0059_layout_add_product(
        uint64_t *value, uint64_t left, uint64_t right) {
    if (value == NULL || (left != 0U && right > (uint64_t)INT64_MAX / left))
        return NO;
    uint64_t product = left * right;
    if (product > (uint64_t)INT64_MAX - *value) return NO;
    *value += product;
    return YES;
}

static BOOL task0059_validate_layout(
        SynaptikMetalDecodedNode node,
        const uint32_t *ranks,
        const uint64_t *dimensions,
        const uint64_t *strides,
        const uint64_t *offsets,
        const uint64_t *spans,
        const uint8_t *present,
        const uint8_t *view) {
    if (ranks == NULL || dimensions == NULL || strides == NULL || offsets == NULL
            || spans == NULL || present == NULL || view == NULL) return NO;
    if (node.operation != SYNAPTIK_METAL_CUSTOM_SELECT
            && node.operation != SYNAPTIK_METAL_CUSTOM_SLICE) return YES;
    uint32_t input = node.first_input;
    uint32_t output = node.output;
    uint32_t input_rank = ranks[input];
    uint32_t output_rank = ranks[output];
    if (!synaptik_storage_layout_supported(
                input, ranks, dimensions, strides, offsets, spans, present,
                node.operation == SYNAPTIK_METAL_CUSTOM_SELECT ? 2U : 1U)
            || !synaptik_storage_layout_supported(
                    output, ranks, dimensions, strides, offsets, spans, present, 1U)
            || !view[output])
        return NO;
    if (node.operation == SYNAPTIK_METAL_CUSTOM_SELECT) {
        if (input_rank <= 1U || output_rank + 1U != input_rank
                || node.attribute_count != 2U) return NO;
        uint64_t axis_word = node.attribute_values[0];
        if (axis_word >= input_rank) return NO;
        uint32_t axis = (uint32_t)axis_word;
        uint64_t expected_offset = offsets[input];
        uint64_t selected_stride =
                strides[(size_t)input * SYNAPTIK_MAX_RANK + axis];
        if (!task0059_layout_add_product(
                    &expected_offset, node.attribute_values[1], selected_stride)
                || offsets[output] != expected_offset)
            return NO;
        for (uint32_t source = 0U, target = 0U; source < input_rank; source++) {
            if (source == axis) continue;
            if (strides[(size_t)output * SYNAPTIK_MAX_RANK + target]
                    != strides[(size_t)input * SYNAPTIK_MAX_RANK + source])
                return NO;
            target++;
        }
        return YES;
    }
    if (input_rank == 0U || output_rank != input_rank) return NO;
    uint32_t count;
    uint32_t starts_offset;
    uint32_t axes_offset = 0U;
    uint32_t steps_offset = 0U;
    BOOL crop = node.attribute_kind == 15U;
    if (crop) {
        if (node.attribute_count < 2U
                || node.attribute_values[0] != input_rank) return NO;
        uint32_t prefix_rank_offset = 1U + input_rank;
        if (prefix_rank_offset >= node.attribute_count
                || node.attribute_values[prefix_rank_offset] != input_rank)
            return NO;
        count = input_rank;
        starts_offset = prefix_rank_offset + 1U;
    } else {
        if (node.attribute_kind != 17U || node.attribute_count == 0U
                || node.attribute_values[0] > input_rank) return NO;
        count = (uint32_t)node.attribute_values[0];
        starts_offset = 1U;
        axes_offset = 1U + count * 2U;
        steps_offset = 1U + count * 3U;
    }
    if (!crop) {
        for (uint32_t item = 0U; item < count; item++) {
            uint64_t step = node.attribute_values[steps_offset + item];
            if (step == 0U || step > (uint64_t)INT64_MAX) return NO;
        }
    }
    uint64_t expected_offset = offsets[input];
    uint64_t expected[SYNAPTIK_MAX_RANK] = {0};
    for (uint32_t axis = 0U; axis < input_rank; axis++)
        expected[axis] = strides[(size_t)input * SYNAPTIK_MAX_RANK + axis];
    for (uint32_t item = 0U; item < count; item++) {
        uint32_t axis = crop ? item : (uint32_t)node.attribute_values[axes_offset + item];
        uint64_t start = node.attribute_values[starts_offset + item];
        uint64_t step = crop ? 1U : node.attribute_values[steps_offset + item];
        if (axis >= input_rank
                || !task0059_layout_add_product(&expected_offset, start, expected[axis])
                || (expected[axis] != 0U
                        && step > (uint64_t)INT64_MAX / expected[axis]))
            return NO;
        expected[axis] *= step;
    }
    if (offsets[output] != expected_offset) return NO;
    for (uint32_t axis = 0U; axis < input_rank; axis++)
        if (strides[(size_t)output * SYNAPTIK_MAX_RANK + axis] != expected[axis])
            return NO;
    return YES;
}

static BOOL task0059_validate_shape(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes) {
    MPSShape *input = shapes[node.first_input];
    MPSShape *output = shapes[node.output];
    switch ((SynaptikMetalOperation)node.operation) {
        case SYNAPTIK_METAL_CUSTOM_CAST:
            return node.attribute_kind == 10U && node.attribute_count == 1U
                    && task0059_shapes_equal(input, output);
        case SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS: {
            if (node.attribute_kind != 3U || node.attribute_count != 1U
                    || node.second_input >= shapes.count) return NO;
            MPSShape *indices = shapes[node.second_input];
            if (input.count == 0U || input.count != indices.count
                    || !task0059_shapes_equal(indices, output)
                    || node.axis >= input.count) return NO;
            for (NSUInteger axis = 0U; axis < input.count; axis++) {
                if (axis != node.axis
                        && input[axis].unsignedLongLongValue
                                != indices[axis].unsignedLongLongValue)
                    return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_CUSTOM_GATHER_ND: {
            if (node.attribute_kind != 11U || node.attribute_count != 1U
                    || node.second_input >= shapes.count) return NO;
            MPSShape *indices = shapes[node.second_input];
            uint64_t batch_word = node.attribute_values[0];
            if (indices.count == 0U || batch_word >= indices.count
                    || batch_word > input.count) return NO;
            NSUInteger batch = (NSUInteger)batch_word;
            for (NSUInteger axis = 0U; axis < batch; axis++) {
                if (input[axis].unsignedLongLongValue
                        != indices[axis].unsignedLongLongValue) return NO;
            }
            uint64_t tuple_word = indices.lastObject.unsignedLongLongValue;
            if (tuple_word == 0U || tuple_word > input.count - batch) return NO;
            NSUInteger tuple = (NSUInteger)tuple_word;
            NSUInteger expected_rank =
                    indices.count - 1U + input.count - batch - tuple;
            if (expected_rank > SYNAPTIK_MAX_RANK || output.count != expected_rank)
                return NO;
            NSUInteger cursor = 0U;
            for (NSUInteger axis = 0U; axis + 1U < indices.count; axis++, cursor++) {
                if (output[cursor].unsignedLongLongValue
                        != indices[axis].unsignedLongLongValue) return NO;
            }
            for (NSUInteger axis = batch + tuple; axis < input.count; axis++, cursor++) {
                if (output[cursor].unsignedLongLongValue
                        != input[axis].unsignedLongLongValue) return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_CUSTOM_SELECT: {
            if (node.attribute_kind != 14U || node.attribute_count != 2U
                    || input.count <= 1U || output.count == 0U) return NO;
            uint64_t axis_word = node.attribute_values[0];
            uint64_t index = node.attribute_values[1];
            if (axis_word >= input.count
                    || index >= input[(NSUInteger)axis_word].unsignedLongLongValue
                    || output.count + 1U != input.count) return NO;
            NSUInteger cursor = 0U;
            for (NSUInteger axis = 0U; axis < input.count; axis++) {
                if (axis != axis_word) {
                    if (output[cursor].unsignedLongLongValue
                            != input[axis].unsignedLongLongValue) return NO;
                    cursor++;
                }
            }
            return YES;
        }
        case SYNAPTIK_METAL_CUSTOM_PAD: {
            if (node.attribute_kind != 16U || node.attribute_count < 3U
                    || node.attribute_values[0] != input.count
                    || output.count != input.count) return NO;
            NSUInteger rank = input.count;
            if (node.attribute_count != 1U + rank * 2U + 2U) return NO;
            for (NSUInteger axis = 0U; axis < rank; axis++) {
                uint64_t before = node.attribute_values[1U + axis];
                uint64_t after = node.attribute_values[1U + rank + axis];
                uint64_t extent = input[axis].unsignedLongLongValue;
                if (before > UINT64_MAX - extent
                        || after > UINT64_MAX - extent - before
                        || output[axis].unsignedLongLongValue != extent + before + after)
                    return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_CUSTOM_SLICE: {
            if (input.count == 0U || output.count == 0U) return NO;
            if (node.attribute_kind == 15U)
                return task0059_crop_region_matches(node, input, output);
            if (node.attribute_kind != 17U || node.attribute_count == 0U
                    || input.count != output.count) return NO;
            NSUInteger count = (NSUInteger)node.attribute_values[0];
            if (count > input.count || node.attribute_count != 1U + count * 4U)
                return NO;
            uint64_t expected[16] = {0};
            for (NSUInteger axis = 0U; axis < input.count; axis++)
                expected[axis] = input[axis].unsignedLongLongValue;
            for (NSUInteger item = 0U; item < count; item++) {
                int64_t start = (int64_t)node.attribute_values[1U + item];
                uint64_t length = node.attribute_values[1U + count + item];
                NSUInteger axis =
                        (NSUInteger)node.attribute_values[1U + count * 2U + item];
                int64_t step =
                        (int64_t)node.attribute_values[1U + count * 3U + item];
                if (axis >= input.count || length == 0U || step <= 0
                        || start < 0 || length - 1U > (uint64_t)INT64_MAX
                        || length - 1U
                                > (uint64_t)(INT64_MAX - start) / (uint64_t)step)
                    return NO;
                int64_t last = start + (int64_t)(length - 1U) * step;
                if (last < 0
                        || (uint64_t)start >= input[axis].unsignedLongLongValue
                        || (uint64_t)last >= input[axis].unsignedLongLongValue)
                    return NO;
                expected[axis] = length;
            }
            return task0059_shape_matches(output, expected, input.count);
        }
        case SYNAPTIK_METAL_CUSTOM_CONCAT:
        case SYNAPTIK_METAL_CUSTOM_STACK: {
            if (node.attribute_kind != 3U || node.attribute_count != 1U
                    || node.input_count == 0U || node.input_count > 16U) return NO;
            NSUInteger rank = input.count;
            NSUInteger axis = node.axis;
            if (node.operation == SYNAPTIK_METAL_CUSTOM_CONCAT) {
                if (rank == 0U || axis >= rank || output.count != rank) return NO;
                uint64_t total = 0U;
                for (uint32_t item = 0U; item < node.input_count; item++) {
                    MPSShape *part = shapes[node.inputs[item]];
                    if (part.count != rank) return NO;
                    for (NSUInteger dimension = 0U; dimension < rank; dimension++) {
                        if (dimension != axis
                                && part[dimension].unsignedLongLongValue
                                        != input[dimension].unsignedLongLongValue) return NO;
                    }
                    uint64_t extent = part[axis].unsignedLongLongValue;
                    if (extent > UINT64_MAX - total) return NO;
                    total += extent;
                }
                if (output[axis].unsignedLongLongValue != total) return NO;
                for (NSUInteger dimension = 0U; dimension < rank; dimension++) {
                    if (dimension != axis
                            && output[dimension].unsignedLongLongValue
                                    != input[dimension].unsignedLongLongValue) return NO;
                }
                return YES;
            }
            if (rank >= SYNAPTIK_MAX_RANK || axis > rank
                    || output.count != rank + 1U
                    || output[axis].unsignedLongLongValue != node.input_count) return NO;
            for (uint32_t item = 0U; item < node.input_count; item++) {
                if (!task0059_shapes_equal(input, shapes[node.inputs[item]])) return NO;
            }
            for (NSUInteger dimension = 0U; dimension < rank; dimension++) {
                NSUInteger target = dimension < axis ? dimension : dimension + 1U;
                if (output[target].unsignedLongLongValue
                        != input[dimension].unsignedLongLongValue) return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_CUSTOM_TILE: {
            if (node.attribute_kind != 18U || node.attribute_count != input.count + 1U
                    || node.attribute_values[0] != input.count
                    || output.count != input.count) return NO;
            for (NSUInteger axis = 0U; axis < input.count; axis++) {
                uint64_t repeat = node.attribute_values[1U + axis];
                uint64_t extent = input[axis].unsignedLongLongValue;
                if (repeat == 0U || extent > UINT64_MAX / repeat
                        || output[axis].unsignedLongLongValue != extent * repeat)
                    return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_CUSTOM_UNFOLD2D:
        case SYNAPTIK_METAL_CUSTOM_UNFOLD3D: {
            NSUInteger dimensions =
                    node.operation == SYNAPTIK_METAL_CUSTOM_UNFOLD2D ? 2U : 3U;
            uint32_t direct_kind = dimensions == 2U ? 19U : 22U;
            uint32_t padded_kind = dimensions == 2U ? 20U : 23U;
            if ((node.attribute_kind != direct_kind && node.attribute_kind != padded_kind)
                    || input.count != dimensions + 2U || output.count != 3U) return NO;
            uint64_t positions = 1U;
            uint64_t kernel_volume = 1U;
            for (NSUInteger spatial = 0U; spatial < dimensions; spatial++) {
                uint64_t kernel = node.attribute_values[spatial];
                uint64_t stride = node.attribute_values[dimensions + spatial];
                uint64_t padding = node.attribute_values[dimensions * 2U + spatial];
                uint64_t dilation = node.attribute_values[dimensions * 3U + spatial];
                uint64_t extent = 0U;
                if (!task0059_window_extent(
                            input[spatial + 2U].unsignedLongLongValue,
                            kernel, padding, stride, dilation,
                            node.attribute_values[dimensions * 4U] != 0U,
                            &extent)
                        || kernel_volume > UINT64_MAX / kernel
                        || positions > UINT64_MAX / extent) return NO;
                kernel_volume *= kernel;
                positions *= extent;
            }
            uint64_t channels = input[1].unsignedLongLongValue;
            return output[0].unsignedLongLongValue
                            == input[0].unsignedLongLongValue
                    && channels <= UINT64_MAX / kernel_volume
                    && output[1].unsignedLongLongValue == channels * kernel_volume
                    && output[2].unsignedLongLongValue == positions;
        }
        case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ADD: {
            if (node.attribute_kind != 3U || node.attribute_count != 1U
                    || node.input_count != 3U || node.axis >= input.count
                    || !task0059_shapes_equal(input, output)) return NO;
            MPSShape *indices = shapes[node.second_input];
            MPSShape *updates = shapes[node.auxiliary];
            if (!task0059_shapes_equal(indices, updates)
                    || indices.count != input.count) return NO;
            for (NSUInteger axis = 0U; axis < input.count; axis++) {
                if (axis != node.axis
                        && input[axis].unsignedLongLongValue
                                != indices[axis].unsignedLongLongValue) return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND: {
            if (node.attribute_kind != 13U || node.attribute_count != 2U
                    || node.input_count != 3U
                    || !task0059_shapes_equal(input, output)) return NO;
            MPSShape *indices = shapes[node.second_input];
            MPSShape *updates = shapes[node.auxiliary];
            NSUInteger batch = (NSUInteger)node.attribute_values[0];
            if (indices.count == 0U || batch >= indices.count
                    || batch > input.count) return NO;
            for (NSUInteger axis = 0U; axis < batch; axis++) {
                if (indices[axis].unsignedLongLongValue
                        != input[axis].unsignedLongLongValue) return NO;
            }
            NSUInteger tuple =
                    (NSUInteger)indices.lastObject.unsignedLongLongValue;
            if (tuple == 0U || tuple > input.count - batch) return NO;
            NSUInteger expected_rank =
                    indices.count - 1U + input.count - batch - tuple;
            if (updates.count != expected_rank) return NO;
            NSUInteger cursor = 0U;
            for (NSUInteger axis = 0U; axis + 1U < indices.count; axis++, cursor++) {
                if (updates[cursor].unsignedLongLongValue
                        != indices[axis].unsignedLongLongValue) return NO;
            }
            for (NSUInteger axis = batch + tuple;
                    axis < input.count; axis++, cursor++) {
                if (updates[cursor].unsignedLongLongValue
                        != input[axis].unsignedLongLongValue) return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE: {
            if (node.input_count != 2U || !task0059_shapes_equal(input, output))
                return NO;
            MPSShape *updates = shapes[node.second_input];
            if (node.attribute_kind == 15U)
                return task0059_crop_region_matches(node, input, updates);
            if (node.attribute_kind != 17U) return NO;
            NSUInteger count = (NSUInteger)node.attribute_values[0];
            if (count > input.count || updates.count != input.count) return NO;
            uint64_t expected[16] = {0};
            for (NSUInteger axis = 0U; axis < input.count; axis++)
                expected[axis] = input[axis].unsignedLongLongValue;
            for (NSUInteger item = 0U; item < count; item++) {
                NSUInteger axis = (NSUInteger)
                        node.attribute_values[1U + count * 2U + item];
                int64_t start = (int64_t)node.attribute_values[1U + item];
                uint64_t length =
                        node.attribute_values[1U + count + item];
                int64_t step = (int64_t)
                        node.attribute_values[1U + count * 3U + item];
                if (axis >= input.count || start < 0 || length == 0U || step == 0)
                    return NO;
                uint64_t extent = input[axis].unsignedLongLongValue;
                if ((uint64_t)start >= extent) return NO;
                __int128 last = (__int128)start
                        + (__int128)(length - 1U) * (__int128)step;
                if (last < 0 || last >= (__int128)extent) return NO;
                expected[axis] = length;
            }
            return task0059_shape_matches(updates, expected, input.count);
        }
        case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS: {
            if (node.attribute_kind != 6U || node.attribute_count != 3U
                    || output.count + 1U != input.count || node.axis >= output.count)
                return NO;
            uint64_t size = node.attribute_values[0];
            uint64_t step = node.attribute_values[1];
            if (size == 0U || step == 0U
                    || input.lastObject.unsignedLongLongValue != size
                    || output[node.axis].unsignedLongLongValue < size) return NO;
            uint64_t positions =
                    (output[node.axis].unsignedLongLongValue - size) / step + 1U;
            for (NSUInteger axis = 0U; axis < output.count; axis++) {
                uint64_t expected = axis == node.axis
                        ? positions : output[axis].unsignedLongLongValue;
                if (input[axis].unsignedLongLongValue != expected) return NO;
            }
            return YES;
        }
        case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
        case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: {
            NSUInteger dimensions =
                    node.operation == SYNAPTIK_METAL_MPSGRAPH_FOLD2D ? 2U : 3U;
            NSUInteger target_rank = dimensions + 2U;
            if (node.attribute_values[0] != target_rank
                    || output.count != target_rank || input.count != 3U) return NO;
            for (NSUInteger axis = 0U; axis < target_rank; axis++) {
                if (node.attribute_values[1U + axis]
                        != output[axis].unsignedLongLongValue) return NO;
            }
            NSUInteger offset = 1U + target_rank;
            uint64_t kernel_volume = 1U;
            uint64_t positions = 1U;
            for (NSUInteger spatial = 0U; spatial < dimensions; spatial++) {
                uint64_t kernel = node.attribute_values[offset + spatial];
                uint64_t extent = 0U;
                if (!task0059_window_extent(
                            output[spatial + 2U].unsignedLongLongValue,
                            kernel,
                            node.attribute_values[offset + dimensions * 2U + spatial],
                            node.attribute_values[offset + dimensions + spatial],
                            node.attribute_values[offset + dimensions * 3U + spatial],
                            node.attribute_values[offset + dimensions * 4U] != 0U,
                            &extent)
                        || kernel_volume > UINT64_MAX / kernel
                        || positions > UINT64_MAX / extent) return NO;
                kernel_volume *= kernel;
                positions *= extent;
            }
            uint64_t channels = output[1].unsignedLongLongValue;
            return input[0].unsignedLongLongValue
                            == output[0].unsignedLongLongValue
                    && channels <= UINT64_MAX / kernel_volume
                    && input[1].unsignedLongLongValue == channels * kernel_volume
                    && input[2].unsignedLongLongValue == positions;
        }
        default:
            return NO;
    }
}

static int compare_u64(const void *left, const void *right) {
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static BOOL scatter_target_linear(
        SynaptikMetalIndexValidation *validation,
        uint64_t ordinal,
        int32_t index,
        uint64_t *out_target) {
    if (validation.coordinateExtents == nil
            || validation.dataStrides == nil
            || out_target == NULL
            || index < 0)
        return NO;
    const uint64_t *extents = validation.coordinateExtents.bytes;
    const uint64_t *strides = validation.dataStrides.bytes;
    NSUInteger rank = validation.coordinateExtents.length / sizeof(uint64_t);
    if (rank == 0U
            || rank != validation.dataStrides.length / sizeof(uint64_t)
            || validation.axis >= rank)
        return NO;
    uint64_t remaining = ordinal;
    uint64_t target = 0U;
    for (NSUInteger dimension = rank; dimension-- > 0U;) {
        uint64_t extent = extents[dimension];
        if (extent == 0U) return NO;
        uint64_t coordinate = remaining % extent;
        remaining /= extent;
        if (dimension == validation.axis) coordinate = (uint64_t)index;
        if (coordinate > UINT64_MAX / strides[dimension]) return NO;
        uint64_t contribution = coordinate * strides[dimension];
        if (target > UINT64_MAX - contribution) return NO;
        target += contribution;
    }
    if (remaining != 0U) return NO;
    *out_target = target;
    return YES;
}

static BOOL scatter_nd_target_linear(
        SynaptikMetalIndexValidation *validation,
        const void *contents,
        uint32_t index_type,
        uint64_t tuple,
        uint64_t *out_target) {
    if (validation.prefixExtents == nil
            || validation.dataStrides == nil
            || contents == NULL
            || out_target == NULL)
        return NO;
    const uint64_t *prefix = validation.prefixExtents.bytes;
    const uint64_t *strides = validation.dataStrides.bytes;
    NSUInteger prefix_rank = validation.prefixExtents.length / sizeof(uint64_t);
    NSUInteger data_rank = validation.dataStrides.length / sizeof(uint64_t);
    NSUInteger batch = validation.axis;
    NSUInteger depth = validation.tupleDepth;
    if (depth == 0U || prefix_rank < batch || batch + depth > data_rank)
        return NO;
    uint64_t remaining = tuple;
    uint64_t target = 0U;
    for (NSUInteger dimension = prefix_rank; dimension-- > 0U;) {
        uint64_t extent = prefix[dimension];
        if (extent == 0U) return NO;
        uint64_t coordinate = remaining % extent;
        remaining /= extent;
        if (dimension < batch) {
            if (coordinate > UINT64_MAX / strides[dimension]) return NO;
            uint64_t contribution = coordinate * strides[dimension];
            if (target > UINT64_MAX - contribution) return NO;
            target += contribution;
        }
    }
    if (remaining != 0U) return NO;
    for (NSUInteger component = 0U; component < depth; component++) {
        uint64_t ordinal = tuple * depth + component;
        int64_t value = index_type == SYNAPTIK_METAL_TYPE_INT32
                ? (int64_t)((const int32_t *)contents)[ordinal]
                : ((const int64_t *)contents)[ordinal];
        uint64_t coordinate = (uint64_t)value;
        NSUInteger dimension = batch + component;
        if (coordinate > UINT64_MAX / strides[dimension]) return NO;
        uint64_t contribution = coordinate * strides[dimension];
        if (target > UINT64_MAX - contribution) return NO;
        target += contribution;
    }
    *out_target = target;
    return YES;
}

static int32_t validate_index_buffer(
        SynaptikMetalIndexValidation *validation,
        SynaptikMetalBufferBox *buffer) {
    if (validation == nil || buffer == nil) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    uint32_t index_type = validation.indexType == 0U
            ? SYNAPTIK_METAL_TYPE_INT32 : validation.indexType;
    uint64_t width = index_type == SYNAPTIK_METAL_TYPE_INT32
            ? sizeof(int32_t)
            : index_type == SYNAPTIK_METAL_TYPE_INT64 ? sizeof(int64_t) : 0U;
    if (width == 0U || validation.elementCount > UINT64_MAX / width)
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    uint64_t required = validation.elementCount * width;
    if (buffer.logicalByteSize < required)
        return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    const void *contents = buffer.buffer.contents;
    if (contents == NULL) return SYNAPTIK_METAL_STATUS_COPY_FAILED;
    BOOL nd = validation.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
            || validation.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND;
    const uint64_t *bounds = nd
            ? validation.coordinateExtents.bytes : NULL;
    if (nd && (validation.tupleDepth == 0U || bounds == NULL
            || validation.coordinateExtents.length
                    != (NSUInteger)validation.tupleDepth * sizeof(uint64_t)))
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    BOOL scatter_elements =
            validation.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS;
    BOOL scatter_nd =
            validation.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
            && validation.targetScratch != nil;
    BOOL scatter = scatter_elements || scatter_nd;
    uint64_t *targets = NULL;
    if (scatter) {
        if (validation.targetScratch == nil
                || validation.targetCount
                        > (uint64_t)NSUIntegerMax / sizeof(uint64_t)
                || validation.targetScratch.length
                        != (NSUInteger)validation.targetCount * sizeof(uint64_t))
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        targets = validation.targetScratch.mutableBytes;
        if (targets == NULL) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
    for (uint64_t ordinal = 0U; ordinal < validation.elementCount; ordinal++) {
        int64_t value = index_type == SYNAPTIK_METAL_TYPE_INT32
                ? (int64_t)((const int32_t *)contents)[ordinal]
                : ((const int64_t *)contents)[ordinal];
        uint64_t bound = nd
                ? bounds[ordinal % validation.tupleDepth]
                : validation.bound;
        if (value < 0 || (uint64_t)value >= bound)
            return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
        if (scatter_elements && !scatter_target_linear(
                    validation, ordinal, (int32_t)value, &targets[ordinal]))
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
    if (scatter_nd) {
        for (uint64_t tuple = 0U; tuple < validation.targetCount; tuple++) {
            if (!scatter_nd_target_linear(
                        validation, contents, index_type, tuple, &targets[tuple]))
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        }
    }
    if (scatter) {
        qsort(
                targets,
                (size_t)validation.targetCount,
                sizeof(uint64_t),
                compare_u64);
        for (uint64_t ordinal = 1U;
                ordinal < validation.targetCount; ordinal++) {
            if (targets[ordinal - 1U] == targets[ordinal])
                return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
        }
    }
    return SYNAPTIK_METAL_STATUS_OK;
}

static BOOL node_permutation_matches(
        SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *output) {
    if (node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_PERMUTATION
            || node.attribute_count != input.count
            || output.count != input.count
            || node.attribute_count == 0U
            || node.attribute_count > SYNAPTIK_MAX_RANK
            || node.axis != UINT32_MAX
            || node.auxiliary != 0U
            || !node_values_are_zero_from(node, node.attribute_count))
        return NO;
    uint32_t seen = 0U;
    for (uint32_t axis = 0; axis < node.attribute_count; axis++) {
        uint64_t source = node.attribute_values[axis];
        if (source >= node.attribute_count
                || (seen & (1U << (uint32_t)source)) != 0U
                || output[axis].unsignedLongLongValue
                        != input[(NSUInteger)source].unsignedLongLongValue)
            return NO;
        seen |= 1U << (uint32_t)source;
    }
    return YES;
}

static BOOL node_axis_header_is_valid(SynaptikMetalDecodedNode node) {
    return node.attribute_kind == SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS
            && node.attribute_count == 1U
            && node.axis < SYNAPTIK_MAX_RANK
            && node.auxiliary == 0U
            && node_values_are_zero_from(node, 0U);
}

static BOOL node_expand_dims_matches(
        SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *output) {
    if (!node_axis_header_is_valid(node)
            || output.count != input.count + 1U
            || node.axis > input.count)
        return NO;
    for (NSUInteger axis = 0; axis < output.count; axis++) {
        uint64_t expected = axis == node.axis
                ? 1U
                : input[axis < node.axis ? axis : axis - 1U].unsignedLongLongValue;
        if (output[axis].unsignedLongLongValue != expected) return NO;
    }
    return YES;
}

static BOOL node_squeeze_matches(
        SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *output) {
    if (!node_axis_header_is_valid(node)
            || input.count != output.count + 1U
            || node.axis >= input.count
            || input[node.axis].unsignedLongLongValue != 1U)
        return NO;
    for (NSUInteger source = 0U, target = 0U; source < input.count; source++)
        if (source != node.axis
                && input[source].unsignedLongLongValue
                        != output[target++].unsignedLongLongValue)
            return NO;
    return YES;
}

static BOOL node_gather_matches(
        SynaptikMetalDecodedNode node,
        MPSShape *data,
        MPSShape *indices,
        MPSShape *output) {
    if (!node_axis_header_is_valid(node)
            || node.axis >= data.count
            || output.count != data.count - 1U + indices.count)
        return NO;
    for (NSUInteger axis = 0U; axis < node.axis; axis++)
        if (output[axis].unsignedLongLongValue
                != data[axis].unsignedLongLongValue)
            return NO;
    for (NSUInteger axis = 0U; axis < indices.count; axis++)
        if (output[node.axis + axis].unsignedLongLongValue
                != indices[axis].unsignedLongLongValue)
            return NO;
    for (NSUInteger axis = node.axis + 1U; axis < data.count; axis++)
        if (output[indices.count + axis - 1U].unsignedLongLongValue
                != data[axis].unsignedLongLongValue)
            return NO;
    return YES;
}

static BOOL node_scatter_elements_matches(
        SynaptikMetalDecodedNode node,
        MPSShape *data,
        MPSShape *indices,
        MPSShape *updates,
        MPSShape *output) {
    if (node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS
            || node.attribute_count != 1U
            || node.axis >= data.count
            || !node_values_are_zero_from(node, 0U)
            || data.count == 0U
            || indices.count != data.count
            || updates.count != data.count
            || output.count != data.count)
        return NO;
    for (NSUInteger axis = 0U; axis < data.count; axis++) {
        uint64_t data_dimension = data[axis].unsignedLongLongValue;
        uint64_t index_dimension = indices[axis].unsignedLongLongValue;
        if (updates[axis].unsignedLongLongValue != index_dimension
                || output[axis].unsignedLongLongValue != data_dimension
                || (axis != node.axis && index_dimension != data_dimension))
            return NO;
    }
    return YES;
}

static BOOL node_one_hot_matches(
        SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *output) {
    if (node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_DEPTH
            || node.attribute_count != 1U
            || node.axis != UINT32_MAX
            || node.auxiliary != 0U
            || node.attribute_values[0] == 0U
            || node.attribute_values[0] > (uint64_t)NSUIntegerMax
            || !node_values_are_zero_from(node, 1U)
            || output.count != input.count + 1U)
        return NO;
    for (NSUInteger axis = 0U; axis < input.count; axis++)
        if (output[axis].unsignedLongLongValue
                != input[axis].unsignedLongLongValue)
            return NO;
    return output[input.count].unsignedLongLongValue == node.attribute_values[0];
}

static BOOL node_unfold_axis_matches(
        SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *output) {
    if (node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_WINDOW_AXIS
            || node.attribute_count != 3U
            || node.second_input != UINT32_MAX
            || node.auxiliary != 0U
            || input.count == 0U
            || input.count >= SYNAPTIK_MAX_RANK
            || output.count != input.count + 1U
            || node.axis >= input.count
            || node.attribute_values[0] == 0U
            || node.attribute_values[0] > SYNAPTIK_MAX_SELECTOR_EXPANSION
            || node.attribute_values[1] == 0U
            || node.attribute_values[1] > (uint64_t)NSIntegerMax
            || !node_values_are_zero_from(node, 2U))
        return NO;
    uint64_t size = node.attribute_values[0];
    uint64_t step = node.attribute_values[1];
    uint64_t selected = input[node.axis].unsignedLongLongValue;
    if (size > selected) return NO;
    uint64_t positions = (selected - size) / step + 1U;
    uint64_t repeated_span = positions - 1U;
    if (repeated_span > UINT64_MAX / step) return NO;
    repeated_span *= step;
    if (size - 1U > UINT64_MAX - repeated_span
            || size - 1U + repeated_span > UINT64_MAX - 1U)
        return NO;
    uint64_t last_end = size - 1U + repeated_span + 1U;
    if (last_end > selected || last_end > (uint64_t)NSIntegerMax) return NO;
    for (NSUInteger axis = 0U; axis < input.count; axis++) {
        uint64_t dimension = input[axis].unsignedLongLongValue;
        uint64_t expected = axis == node.axis ? positions : dimension;
        if (dimension > (uint64_t)NSIntegerMax
                || output[axis].unsignedLongLongValue != expected)
            return NO;
    }
    return output[input.count].unsignedLongLongValue == size;
}

static BOOL require_value_type(
        uint8_t *types, uint32_t value, SynaptikMetalValueType required) {
    if (types[value] != SYNAPTIK_METAL_TYPE_UNAVAILABLE
            && types[value] != required)
        return NO;
    types[value] = required;
    return YES;
}

static uint64_t value_type_width(uint8_t type) {
    switch ((SynaptikMetalValueType)type) {
        case SYNAPTIK_METAL_TYPE_FLOAT32:
        case SYNAPTIK_METAL_TYPE_INT32:
            return 4U;
        case SYNAPTIK_METAL_TYPE_FLOAT64:
        case SYNAPTIK_METAL_TYPE_INT64:
            return 8U;
        case SYNAPTIK_METAL_TYPE_BFLOAT16:
            return 2U;
        case SYNAPTIK_METAL_TYPE_BOOL:
            return 1U;
        default:
            return 0U;
    }
}

static MPSDataType value_mps_data_type(uint8_t type) {
    switch ((SynaptikMetalValueType)type) {
        case SYNAPTIK_METAL_TYPE_FLOAT32:
            return MPSDataTypeFloat32;
        case SYNAPTIK_METAL_TYPE_INT32:
            return MPSDataTypeInt32;
        case SYNAPTIK_METAL_TYPE_BOOL:
            return MPSDataTypeBool;
        case SYNAPTIK_METAL_TYPE_BFLOAT16:
            return MPSDataTypeBFloat16;
        case SYNAPTIK_METAL_TYPE_INT64:
            return MPSDataTypeInt64;
        default:
            return MPSDataTypeInvalid;
    }
}

static NSUInteger feed_position(
        uint32_t value, uint32_t feed_count, const uint32_t *feed_indices) {
    for (uint32_t feed = 0U; feed < feed_count; feed++)
        if (feed_indices[feed] == value) return (NSUInteger)feed;
    return NSNotFound;
}

static NSArray<NSNumber *> *node_attribute_array(
        SynaptikMetalDecodedNode node) {
    NSMutableArray<NSNumber *> *result =
            [NSMutableArray arrayWithCapacity:node.attribute_count];
    for (uint32_t index = 0; index < node.attribute_count; index++)
        [result addObject:@((NSUInteger)node.attribute_values[index])];
    return [result copy];
}

static NSArray<NSNumber *> *node_reduction_axes(
        SynaptikMetalDecodedNode node,
        MPSShape *input,
        MPSShape *output,
        BOOL *keep_dimensions) {
    if (node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_REDUCTION
            || node.second_input != UINT32_MAX
            || node.attribute_count > SYNAPTIK_MAX_RANK
            || node.auxiliary > 1U
            || !node_values_are_zero_from(node, node.attribute_count))
        return nil;
    NSMutableArray<NSNumber *> *axes = [NSMutableArray array];
    if (node.axis == SYNAPTIK_METAL_REDUCTION_SUM_TO_SHAPE) {
        if (node.operation != SYNAPTIK_METAL_MPSGRAPH_SUM
                || node.auxiliary != 0U
                || node.attribute_count != output.count
                || output.count > input.count)
            return nil;
        NSUInteger padding = input.count - output.count;
        uint64_t count = 1U;
        for (NSUInteger axis = 0; axis < input.count; axis++) {
            uint64_t source = input[axis].unsignedLongLongValue;
            if (axis < padding) {
                if (count > UINT64_MAX / source) return nil;
                count *= source;
                [axes addObject:@(axis)];
            } else {
                uint64_t target = node.attribute_values[axis - padding];
                if (target == 0U || target != output[axis - padding].unsignedLongLongValue
                        || (target != 1U && target != source))
                    return nil;
                if (target == 1U && source != 1U) {
                    if (count > UINT64_MAX / source) return nil;
                    count *= source;
                    [axes addObject:@(axis)];
                }
            }
        }
        *keep_dimensions = NO;
        return count > 0U ? [axes copy] : nil;
    }

    if (node.axis == SYNAPTIK_METAL_REDUCTION_FULL) {
        if (node.attribute_count != 0U || node.auxiliary != 0U || output.count != 0U)
            return nil;
        uint64_t count = 1U;
        for (NSUInteger axis = 0; axis < input.count; axis++) {
            uint64_t dimension = input[axis].unsignedLongLongValue;
            if (count > UINT64_MAX / dimension) return nil;
            count *= dimension;
            [axes addObject:@(axis)];
        }
        *keep_dimensions = NO;
        return count > 0U ? [axes copy] : nil;
    }

    if ((node.axis == SYNAPTIK_METAL_REDUCTION_SINGLE_AXIS
                    && node.attribute_count != 1U)
            || (node.axis != SYNAPTIK_METAL_REDUCTION_SINGLE_AXIS
                    && node.axis != SYNAPTIK_METAL_REDUCTION_MULTI_AXIS))
        return nil;
    NSMutableIndexSet *seen = [NSMutableIndexSet indexSet];
    uint64_t count = 1U;
    for (uint32_t index = 0; index < node.attribute_count; index++) {
        uint64_t axis = node.attribute_values[index];
        if (axis >= input.count || [seen containsIndex:(NSUInteger)axis]) return nil;
        [seen addIndex:(NSUInteger)axis];
        uint64_t dimension = input[(NSUInteger)axis].unsignedLongLongValue;
        if (count > UINT64_MAX / dimension) return nil;
        count *= dimension;
        [axes addObject:@((NSUInteger)axis)];
    }
    BOOL keep = node.auxiliary == 1U;
    NSMutableArray<NSNumber *> *expected = [NSMutableArray array];
    for (NSUInteger axis = 0; axis < input.count; axis++) {
        if ([seen containsIndex:axis]) {
            if (keep) [expected addObject:@1];
        } else {
            [expected addObject:input[axis]];
        }
    }
    if (count == 0U || ![expected isEqualToArray:output]) return nil;
    *keep_dimensions = keep;
    return [axes copy];
}
static NSArray<NSNumber *> *node_statistical_axes(
        SynaptikMetalDecodedNode node,
        MPSShape *input,
        MPSShape *output,
        BOOL *keep_dimensions,
        uint64_t *correction) {
    if (node.attribute_kind != 38U
            || node.second_input != UINT32_MAX
            || node.attribute_count < 3U
            || node.attribute_values[0] > SYNAPTIK_MAX_RANK)
        return nil;
    uint32_t count = (uint32_t)node.attribute_values[0];
    if (node.attribute_count != count + 3U
            || node.attribute_values[count + 1U] > 1U)
        return nil;
    NSMutableIndexSet *seen = [NSMutableIndexSet indexSet];
    NSMutableArray<NSNumber *> *axes = [NSMutableArray arrayWithCapacity:count];
    if (seen == nil || axes == nil) return nil;
    uint64_t selected = 1U;
    for (uint32_t index = 0U; index < count; index++) {
        uint64_t axis = node.attribute_values[index + 1U];
        uint64_t extent = axis < input.count
                ? input[(NSUInteger)axis].unsignedLongLongValue : 0U;
        if (axis >= input.count || [seen containsIndex:(NSUInteger)axis]
                || extent == 0U || selected > UINT64_MAX / extent)
            return nil;
        [seen addIndex:(NSUInteger)axis];
        [axes addObject:@((NSUInteger)axis)];
        selected *= extent;
    }
    uint64_t requested = node.attribute_values[count + 2U];
    if (selected <= requested) return nil;
    BOOL keep = node.attribute_values[count + 1U] != 0U;
    NSMutableArray<NSNumber *> *expected = [NSMutableArray array];
    if (expected == nil) return nil;
    for (NSUInteger axis = 0U; axis < input.count; axis++) {
        if ([seen containsIndex:axis]) {
            if (keep) [expected addObject:@1];
        } else {
            [expected addObject:input[axis]];
        }
    }
    if (![expected isEqualToArray:output]) return nil;
    *keep_dimensions = keep;
    *correction = requested;
    return [axes copy];
}


static BOOL custom_grid(uint64_t count, MTLSize *grid, uint64_t *width, uint64_t *height) {
    if (count == 0U || grid == NULL || width == NULL || height == NULL) return NO;
    uint64_t w = MIN(count, (uint64_t)UINT32_MAX);
    uint64_t remaining = count / w + (count % w != 0U);
    uint64_t h = MIN(remaining, (uint64_t)UINT32_MAX);
    uint64_t depth = remaining / h + (remaining % h != 0U);
    if (depth == 0U || depth > UINT32_MAX) return NO;
    *width = w;
    *height = h;
    *grid = MTLSizeMake((NSUInteger)w, (NSUInteger)h, (NSUInteger)depth);
    return YES;
}

static id<MTLBuffer> custom_metadata(
        id<MTLDevice> device, const void *bytes, NSUInteger length) {
    if (device == nil || bytes == NULL || length == 0U) return nil;
    return [device newBufferWithBytes:bytes
            length:length options:MTLResourceStorageModeShared];
}

static BOOL operation_uses_custom_kernel(uint32_t operation) {
    return (operation >= SYNAPTIK_METAL_CUSTOM_GT
                    && operation <= SYNAPTIK_METAL_CUSTOM_CUM_PROD)
            || (operation >= SYNAPTIK_METAL_BOOL_IS_FINITE
                    && operation <= SYNAPTIK_METAL_BOOL_NOT)
            || operation == SYNAPTIK_METAL_CUSTOM_CONV2D
            || operation == SYNAPTIK_METAL_CUSTOM_CONV3D
            || operation == SYNAPTIK_METAL_BOOL_WHERE
            || (operation >= SYNAPTIK_METAL_CUSTOM_FLOOR
                    && operation <= SYNAPTIK_METAL_CUSTOM_RELU)
            || operation == SYNAPTIK_METAL_CUSTOM_EXP
            || operation == SYNAPTIK_METAL_CUSTOM_SIGMOID
            || operation == SYNAPTIK_METAL_CUSTOM_CAST
            || operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
            || operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
            || operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
            || (operation >= SYNAPTIK_METAL_CUSTOM_SELECT
                    && operation <= SYNAPTIK_METAL_CUSTOM_TILE)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_FOLD3D)
            || (operation >= SYNAPTIK_METAL_CUSTOM_SORT
                    && operation <= SYNAPTIK_METAL_CUSTOM_TOP_K)
            || (operation >= SYNAPTIK_METAL_CUSTOM_MAX_POOL2D
                    && operation <= SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D)
            || (operation >= SYNAPTIK_METAL_CUSTOM_PROD
                    && operation <= SYNAPTIK_METAL_CUSTOM_ANY)
            || (operation >= SYNAPTIK_METAL_CUSTOM_ARG_MAX
                    && operation <= SYNAPTIK_METAL_CUSTOM_ARG_MIN);
}
static BOOL matmul_uses_custom_kernel(
        SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes,
        const uint8_t *value_types) {
    if (node.operation != SYNAPTIK_METAL_MPSGRAPH_MATMUL
            || shapes == nil || value_types == NULL
            || node.second_input == UINT32_MAX)
        return NO;
    return value_types[node.first_input] != SYNAPTIK_METAL_TYPE_FLOAT32
            || value_types[node.second_input] != SYNAPTIK_METAL_TYPE_FLOAT32
            || value_types[node.output] != SYNAPTIK_METAL_TYPE_FLOAT32
            || shapes[node.first_input].count != 2U
            || shapes[node.second_input].count != 2U
            || shapes[node.output].count != 2U;
}

static BOOL node_uses_custom_kernel(
        SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes,
        const uint8_t *value_types) {
    return operation_uses_custom_kernel(node.operation)
            || matmul_uses_custom_kernel(node, shapes, value_types);
}
static BOOL matmul_shapes_match(
        MPSShape *left, MPSShape *right, MPSShape *output) {
    if (left == nil || right == nil || output == nil
            || left.count == 0U || right.count == 0U)
        return NO;
    NSUInteger left_batch = left.count > 1U ? left.count - 2U : 0U;
    NSUInteger right_batch = right.count > 1U ? right.count - 2U : 0U;
    NSUInteger batch_rank = MAX(left_batch, right_batch);
    NSUInteger expected_rank = batch_rank
            + (left.count > 1U ? 1U : 0U)
            + (right.count > 1U ? 1U : 0U);
    if (output.count != expected_rank
            || left[left.count - 1U].unsignedLongLongValue
                    != right[right.count == 1U ? 0U : right.count - 2U]
                            .unsignedLongLongValue)
        return NO;
    for (NSUInteger axis = 0U; axis < batch_rank; axis++) {
        NSInteger left_axis = (NSInteger)axis - (NSInteger)(batch_rank - left_batch);
        NSInteger right_axis = (NSInteger)axis - (NSInteger)(batch_rank - right_batch);
        uint64_t left_extent = left_axis < 0
                ? 1U : left[(NSUInteger)left_axis].unsignedLongLongValue;
        uint64_t right_extent = right_axis < 0
                ? 1U : right[(NSUInteger)right_axis].unsignedLongLongValue;
        if ((left_extent != right_extent && left_extent != 1U && right_extent != 1U)
                || output[axis].unsignedLongLongValue != MAX(left_extent, right_extent))
            return NO;
    }
    NSUInteger output_axis = batch_rank;
    if (left.count > 1U
            && output[output_axis++].unsignedLongLongValue
                    != left[left.count - 2U].unsignedLongLongValue)
        return NO;
    return right.count == 1U
            || output[output_axis].unsignedLongLongValue
                    == right[right.count - 1U].unsignedLongLongValue;
}

static BOOL node_is_last_two_transpose(
        SynaptikMetalDecodedNode node,
        MPSShape *input,
        MPSShape *output,
        uint8_t input_state,
        const uint64_t *value_strides,
        const uint64_t *layout_offsets,
        const uint64_t *layout_spans,
        const uint8_t *declared_states) {
    if (input_state != SYNAPTIK_METAL_VALUE_CANONICAL
            || input == nil || output == nil
            || input.count < 2U || output.count != input.count
            || node.attribute_count != input.count
            || value_strides == NULL || layout_offsets == NULL
            || layout_spans == NULL || declared_states == NULL
            || declared_states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
            || declared_states[node.output] != SYNAPTIK_METAL_VALUE_AFFINE_VIEW
            || layout_offsets[node.output] != layout_offsets[node.first_input]
            || layout_spans[node.output] != layout_spans[node.first_input])
        return NO;
    for (NSUInteger axis = 0U; axis < input.count; axis++) {
        NSUInteger source = node.attribute_values[axis];
        BOOL expected_axis = axis < input.count - 2U
                ? source == axis
                : source == (axis == input.count - 2U ? input.count - 1U
                                                      : input.count - 2U);
        if (!expected_axis
                || value_strides[(size_t)node.output * SYNAPTIK_MAX_RANK + axis]
                        != value_strides[
                                (size_t)node.first_input * SYNAPTIK_MAX_RANK + source])
            return NO;
    }
    return YES;
}


static BOOL node_is_task0064_singleton_height(
        SynaptikMetalDecodedNode node,
        MPSShape *input,
        MPSShape *output,
        uint8_t input_state,
        const uint64_t *value_strides,
        const uint64_t *layout_offsets,
        const uint64_t *layout_spans,
        const uint8_t *declared_states) {
    if (node.operation != SYNAPTIK_METAL_MPSGRAPH_EXPAND_DIMS
            || node.axis != 2U
            || input_state != SYNAPTIK_METAL_VALUE_CANONICAL
            || input == nil || input.count != 3U
            || output == nil || output.count != 4U
            || output[2].unsignedLongLongValue != 1U
            || value_strides == NULL || layout_offsets == NULL
            || layout_spans == NULL || declared_states == NULL
            || declared_states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
            || declared_states[node.output] != SYNAPTIK_METAL_VALUE_AFFINE_VIEW
            || layout_offsets[node.output] != layout_offsets[node.first_input]
            || layout_spans[node.output] != layout_spans[node.first_input])
        return NO;
    uint64_t input_base = (uint64_t)node.first_input * SYNAPTIK_MAX_RANK;
    uint64_t output_base = (uint64_t)node.output * SYNAPTIK_MAX_RANK;
    return value_strides[output_base] == value_strides[input_base]
            && value_strides[output_base + 1U] == value_strides[input_base + 1U]
            && value_strides[output_base + 2U]
                    == value_strides[input_base + 2U]
                            * input[2].unsignedLongLongValue
            && value_strides[output_base + 3U] == value_strides[input_base + 2U];
}


static BOOL operation_has_direct_mpsgraph(uint32_t operation) {
    return (operation >= SYNAPTIK_METAL_MPSGRAPH_NEG
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_UNFOLD_AXIS)
            || operation == SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW
            || (operation >= SYNAPTIK_METAL_BOOL_IS_FINITE
                    && operation <= SYNAPTIK_METAL_BOOL_NOT)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_LOG1P)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_EXPM1
                    && operation <= SYNAPTIK_METAL_CUSTOM_RELU)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_TANH
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_SILU)
            || operation == SYNAPTIK_METAL_BOOL_WHERE
            || operation == SYNAPTIK_METAL_CUSTOM_CAST
            || (operation >= SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_FOLD3D)
            || operation == SYNAPTIK_METAL_MPSGRAPH_MSE
            || (operation >= SYNAPTIK_METAL_CUSTOM_PROD
                    && operation <= SYNAPTIK_METAL_CUSTOM_ANY)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_L2_NORM);
}

static NSString *custom_function(
        SynaptikMetalDecodedNode node, const uint8_t *value_types, uint32_t stage) {
    if (node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL) {
        uint8_t left = value_types[node.first_input];
        uint8_t right = value_types[node.second_input];
        if (left == SYNAPTIK_METAL_TYPE_INT32 && right == SYNAPTIK_METAL_TYPE_INT32)
            return @"matmul_i32_i32_i32_0061";
        if (left == SYNAPTIK_METAL_TYPE_INT32 && right == SYNAPTIK_METAL_TYPE_INT64)
            return @"matmul_i32_i64_i64_0061";
        if (left == SYNAPTIK_METAL_TYPE_INT64 && right == SYNAPTIK_METAL_TYPE_INT32)
            return @"matmul_i64_i32_i64_0061";
        if (left == SYNAPTIK_METAL_TYPE_INT64 && right == SYNAPTIK_METAL_TYPE_INT64)
            return @"matmul_i64_i64_i64_0061";
        if (left == SYNAPTIK_METAL_TYPE_FLOAT32 && right == SYNAPTIK_METAL_TYPE_FLOAT32)
            return @"matmul_f32_f32_f32_0061";
        if (left == SYNAPTIK_METAL_TYPE_BFLOAT16 && right == SYNAPTIK_METAL_TYPE_FLOAT32)
            return @"matmul_bf16_f32_f32_0061";
        if (left == SYNAPTIK_METAL_TYPE_FLOAT32 && right == SYNAPTIK_METAL_TYPE_BFLOAT16)
            return @"matmul_f32_bf16_f32_0061";
        return nil;
    }
    if (node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D) return @"conv2d_0064";
    if (node.operation == SYNAPTIK_METAL_CUSTOM_CONV3D) return @"conv3d_0064";
    if (node.operation == SYNAPTIK_METAL_CUSTOM_MAX_POOL2D) return @"max_pool2d_0064";
    if (node.operation == SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D)
        return @"average_pool2d_0064";
    if (node.operation == SYNAPTIK_METAL_CUSTOM_MAX_POOL3D) return @"max_pool3d_0064";
    if (node.operation == SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D)
        return @"average_pool3d_0064";
    switch ((SynaptikMetalOperation)node.operation) {
        case SYNAPTIK_METAL_CUSTOM_GT: return @"cmp_gt";
        case SYNAPTIK_METAL_CUSTOM_GE: return @"cmp_ge";
        case SYNAPTIK_METAL_CUSTOM_LT: return @"cmp_lt";
        case SYNAPTIK_METAL_CUSTOM_LE: return @"cmp_le";
        case SYNAPTIK_METAL_CUSTOM_EQ: return @"cmp_eq";
        case SYNAPTIK_METAL_CUSTOM_NE: return @"cmp_ne";
        case SYNAPTIK_METAL_CUSTOM_TENSOR_MIN: return @"tensor_min";
        case SYNAPTIK_METAL_CUSTOM_TENSOR_MAX: return @"tensor_max";
        case SYNAPTIK_METAL_CUSTOM_SCALAR_MIN: return @"scalar_min";
        case SYNAPTIK_METAL_CUSTOM_SCALAR_MAX: return @"scalar_max";
        case SYNAPTIK_METAL_CUSTOM_CLAMP: return @"clamp_fused";
        case SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN: return @"reduction_min";
        case SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX: return @"reduction_max";
        case SYNAPTIK_METAL_CUSTOM_CUM_SUM: return @"scan_sum";
        case SYNAPTIK_METAL_CUSTOM_CUM_PROD: return @"scan_prod";
        case SYNAPTIK_METAL_BOOL_IS_FINITE: return @"bool_is_finite";
        case SYNAPTIK_METAL_BOOL_IS_NAN: return @"bool_is_nan";
        case SYNAPTIK_METAL_BOOL_IS_INF: return @"bool_is_inf";
        case SYNAPTIK_METAL_BOOL_AND: return @"bool_and";
        case SYNAPTIK_METAL_BOOL_OR: return @"bool_or";
        case SYNAPTIK_METAL_BOOL_NOT: return @"bool_not";
        case SYNAPTIK_METAL_BOOL_WHERE: return @"bool_where";
        case SYNAPTIK_METAL_CUSTOM_EXP: return @"task0053_candidate_exp";
        case SYNAPTIK_METAL_CUSTOM_SIGMOID: return @"task0053_candidate_sigmoid";
        case SYNAPTIK_METAL_CUSTOM_FLOOR: return @"exact_floor";
        case SYNAPTIK_METAL_CUSTOM_CEIL: return @"exact_ceil";
        case SYNAPTIK_METAL_CUSTOM_SIGN: return @"exact_sign";
        case SYNAPTIK_METAL_CUSTOM_RELU: return @"exact_relu";
        case SYNAPTIK_METAL_CUSTOM_CAST: return @"exact_cast_0059";
        case SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS:
        case SYNAPTIK_METAL_CUSTOM_GATHER_ND: return @"exact_move_binary_0059";
        case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND:
            return stage == 0U ? @"exact_scatter_copy_0060" : @"exact_scatter_nd_0060";
        case SYNAPTIK_METAL_CUSTOM_SORT: return @"exact_sort_0063";
        case SYNAPTIK_METAL_CUSTOM_ARGSORT: return @"exact_argsort_0063";
        case SYNAPTIK_METAL_CUSTOM_TOP_K: return @"exact_top_k_0063";
        case SYNAPTIK_METAL_CUSTOM_ARG_MAX: return @"exact_arg_max_0063";
        case SYNAPTIK_METAL_CUSTOM_ARG_MIN: return @"exact_arg_min_0063";
        case SYNAPTIK_METAL_CUSTOM_SELECT:
        case SYNAPTIK_METAL_CUSTOM_PAD:
        case SYNAPTIK_METAL_CUSTOM_SLICE:
        case SYNAPTIK_METAL_CUSTOM_TILE:
        case SYNAPTIK_METAL_CUSTOM_UNFOLD2D:
        case SYNAPTIK_METAL_CUSTOM_UNFOLD3D: return @"exact_move_unary_0059";
        case SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE: return @"exact_slice_update_0060";
        case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS:
        case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
        case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: return @"exact_fold_0060";
        case SYNAPTIK_METAL_CUSTOM_PROD:
        case SYNAPTIK_METAL_CUSTOM_ALL:
        case SYNAPTIK_METAL_CUSTOM_ANY: return @"exact_reduce_0060";
        case SYNAPTIK_METAL_CUSTOM_CONCAT:
        case SYNAPTIK_METAL_CUSTOM_STACK: return @"exact_compose_0059";
        default: return nil;
    }
}

static BOOL custom_broadcast_strides(
        MPSShape *source, MPSShape *output, uint64_t strides[16]) {
    if (source == nil || output == nil || strides == NULL
            || source.count > output.count || output.count > SYNAPTIK_MAX_RANK)
        return NO;
    uint64_t contiguous[16] = {0};
    uint64_t stride = 1U;
    for (NSUInteger reverse = 0U; reverse < source.count; reverse++) {
        NSUInteger axis = source.count - 1U - reverse;
        uint64_t extent = source[axis].unsignedLongLongValue;
        if (extent == 0U || stride > UINT64_MAX / extent) return NO;
        contiguous[axis] = stride;
        stride *= extent;
    }
    NSUInteger padding = output.count - source.count;
    for (NSUInteger axis = 0U; axis < output.count; axis++) {
        if (axis < padding) {
            strides[axis] = 0U;
            continue;
        }
        NSUInteger source_axis = axis - padding;
        uint64_t source_extent = source[source_axis].unsignedLongLongValue;
        uint64_t output_extent = output[axis].unsignedLongLongValue;
        if (source_extent != 1U && source_extent != output_extent) return NO;
        strides[axis] = source_extent == 1U && output_extent != 1U
                ? 0U : contiguous[source_axis];
    }
    return YES;
}

static BOOL custom_physical_layout(
        MPSShape *shape,
        uint32_t value,
        const uint64_t *value_strides,
        const uint64_t *layout_offsets,
        const uint8_t *value_states,
        uint64_t strides[16],
        uint64_t *offset) {
    if (shape == nil || strides == NULL || offset == NULL
            || shape.count > SYNAPTIK_MAX_RANK)
        return NO;
    if (value_strides != NULL && layout_offsets != NULL && value_states != NULL
            && (value_states[value] == SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT
                    || value_states[value] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW)) {
        for (NSUInteger axis = 0U; axis < shape.count; axis++)
            strides[axis] =
                    value_strides[(size_t)value * SYNAPTIK_MAX_RANK + axis];
        *offset = layout_offsets[value];
        return YES;
    }
    uint64_t stride = 1U;
    for (NSUInteger reverse = 0U; reverse < shape.count; reverse++) {
        NSUInteger axis = shape.count - 1U - reverse;
        uint64_t extent = shape[axis].unsignedLongLongValue;
        if (extent == 0U || stride > UINT64_MAX / extent) return NO;
        strides[axis] = stride;
        stride *= extent;
    }
    *offset = 0U;
    return YES;
}


static SynaptikMetalProgramStep *make_custom_step(
        SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes,
        const uint8_t *value_types,
        const uint64_t *value_strides,
        const uint64_t *layout_offsets,
        const uint8_t *value_states,
        const uint32_t *local_transpose_sources,
        const uint32_t *local_singleton_height_sources,
        id<MTLDevice> device,
        id<MTLLibrary> library,
        uint32_t stage) {
    NSString *name = custom_function(node, value_types, stage);
    id<MTLFunction> function = name == nil ? nil : [library newFunctionWithName:name];
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = function == nil ? nil
            : [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil || error != nil) return nil;
    SynaptikMetalProgramStep *step = [SynaptikMetalProgramStep new];
    if (step == nil) return nil;
    step.custom = YES;
    step.operation = node.operation;
    step.stage = stage;
    step.firstInput = node.first_input;
    step.secondInput = node.second_input;
    step.auxiliaryInput = node.auxiliary;
    step.output = node.output;
    NSMutableArray<NSNumber *> *output_values =
            [NSMutableArray arrayWithCapacity:node.output_count];
    if (output_values == nil) return nil;
    for (uint32_t output_index = 0U;
            output_index < node.output_count;
            output_index++)
        [output_values addObject:@(node.outputs[output_index])];
    step.outputValues = [output_values copy];
    if (node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
            && local_transpose_sources != NULL) {
        if (local_transpose_sources[node.first_input] != UINT32_MAX)
            step.firstInput = local_transpose_sources[node.first_input];
        if (local_transpose_sources[node.second_input] != UINT32_MAX)
            step.secondInput = local_transpose_sources[node.second_input];
    }
    if (local_singleton_height_sources != NULL
            && (node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                    || node.operation == SYNAPTIK_METAL_CUSTOM_MAX_POOL2D
                    || node.operation == SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D)) {
        if (local_singleton_height_sources[node.first_input] != UINT32_MAX)
            step.firstInput = local_singleton_height_sources[node.first_input];
        if (node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                && local_singleton_height_sources[node.second_input] != UINT32_MAX)
            step.secondInput = local_singleton_height_sources[node.second_input];
    }
    step.pipeline = pipeline;
    step.threadsPerThreadgroup = MAX(
            (NSUInteger)1U,
            MIN(pipeline.maxTotalThreadsPerThreadgroup, pipeline.threadExecutionWidth));

    MPSShape *input = shapes[node.first_input];
    MPSShape *output = shapes[node.output];
    BOOL task0063 = (node.operation >= SYNAPTIK_METAL_CUSTOM_SORT
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_TOP_K)
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_ARG_MAX
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_ARG_MIN);
    BOOL task0059 = node.operation == SYNAPTIK_METAL_CUSTOM_CAST
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
            || node.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
            || node.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_SELECT
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_TILE)
            || (node.operation >= SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS
                    && node.operation <= SYNAPTIK_METAL_MPSGRAPH_FOLD3D)
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_PROD
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_ANY);
    BOOL task0064 = node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
            || node.operation == SYNAPTIK_METAL_CUSTOM_CONV3D
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_MAX_POOL2D
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D);
    if (task0063) {
        SynaptikMetalOrderingMeta meta = {0};
        uint64_t element_count = shape_element_count(input);
        uint64_t output_count = shape_element_count(output);
        NSUInteger axis = (NSUInteger)node.attribute_values[0];
        uint64_t axis_length = input[axis].unsignedLongLongValue;
        uint64_t inner = 1U;
        for (NSUInteger dimension = axis + 1U;
                dimension < input.count;
                dimension++) {
            uint64_t extent = input[dimension].unsignedLongLongValue;
            if (inner > UINT32_MAX / extent) return nil;
            inner *= extent;
        }
        uint64_t width = value_type_width(value_types[node.first_input]);
        if (element_count == 0U || element_count > UINT32_MAX
                || output_count == 0U || output_count > UINT32_MAX
                || axis_length == 0U || axis_length > UINT32_MAX
                || inner == 0U || inner > UINT32_MAX
                || width == 0U || width > UINT32_MAX)
            return nil;
        meta.elementCount = (uint32_t)element_count;
        meta.outputCount = (uint32_t)output_count;
        meta.axisLength = (uint32_t)axis_length;
        meta.inner = (uint32_t)inner;
        meta.valueType = value_types[node.first_input];
        meta.width = (uint32_t)width;
        if (node.operation == SYNAPTIK_METAL_CUSTOM_SORT
                || node.operation == SYNAPTIK_METAL_CUSTOM_ARGSORT) {
            meta.direction = (uint32_t)node.attribute_values[1];
        } else if (node.operation == SYNAPTIK_METAL_CUSTOM_TOP_K) {
            meta.k = (uint32_t)node.attribute_values[1];
            meta.direction = (uint32_t)node.attribute_values[2];
            meta.sorted = (uint32_t)node.attribute_values[3];
        } else {
            meta.tieLast = node.attribute_values[2] == 2U ? 1U : 0U;
        }
        uint32_t threads = node.operation == SYNAPTIK_METAL_CUSTOM_ARG_MAX
                        || node.operation == SYNAPTIK_METAL_CUSTOM_ARG_MIN
                ? meta.outputCount : meta.elementCount;
        step.grid = MTLSizeMake((NSUInteger)threads, 1U, 1U);
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (task0059 || task0064) {
        SynaptikMetalDataMeta meta = {0};
        meta.elementCount = shape_element_count(output);
        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND && stage == 1U)
            meta.elementCount = shape_element_count(shapes[node.auxiliary]);
        meta.operation = node.operation;
        meta.inputRank = (uint32_t)input.count;
        meta.outputRank = (uint32_t)output.count;
        meta.width = (uint32_t)value_type_width(value_types[node.first_input]);
        meta.sourceType = value_types[node.first_input];
        meta.targetType = value_types[node.output];
        meta.inputCount = node.input_count;
        meta.attributeKind = node.attribute_kind;
        meta.attributeCount = node.attribute_count;
        if (node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                || node.operation == SYNAPTIK_METAL_CUSTOM_CONV3D) {
            meta.reserved = node.input_count == 3U
                    ? value_types[node.auxiliary] : SYNAPTIK_METAL_TYPE_FLOAT32;
        }
        for (NSUInteger axis = 0U; axis < input.count; axis++)
            meta.inputDims[axis] = input[axis].unsignedLongLongValue;
        for (NSUInteger axis = 0U; axis < output.count; axis++)
            meta.outputDims[axis] = output[axis].unsignedLongLongValue;
        if (!custom_physical_layout(
                    input, node.first_input, value_strides, layout_offsets,
                    value_states, meta.inputStrides, &meta.inputOffset)
                || !custom_physical_layout(
                    output, node.output, value_strides, layout_offsets,
                    value_states, meta.outputStrides, &meta.outputOffset))
            return nil;
        for (uint32_t word = 0U; word < node.attribute_count; word++)
            meta.attrs[word] = node.attribute_values[word];
        if (node.attribute_kind == SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS
                && node.attribute_count == 1U)
            meta.attrs[0] = node.axis;
        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS)
            meta.reserved = node.axis;
        if (node.operation >= SYNAPTIK_METAL_CUSTOM_PROD
                && node.operation <= SYNAPTIK_METAL_CUSTOM_ANY) {
            uint32_t axes_mask = 0U;
            if (node.axis == SYNAPTIK_METAL_REDUCTION_FULL) {
                axes_mask = input.count == 32U
                        ? UINT32_MAX : ((UINT32_C(1) << input.count) - 1U);
            } else {
                for (uint32_t index = 0U; index < node.attribute_count; index++)
                    axes_mask |= UINT32_C(1)
                            << (uint32_t)node.attribute_values[index];
            }
            meta.attrs[0] = axes_mask;
            meta.inputExtents[0] = shape_element_count(input);
        }
        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL)
            meta.inputExtents[0] =
                    input[input.count - 1U].unsignedLongLongValue;
        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND && stage == 1U) {
            MPSShape *updates = shapes[node.auxiliary];
            meta.reserved = (uint32_t)updates.count;
            for (NSUInteger axis = 0U; axis < updates.count; axis++)
                meta.inputExtents[axis] = updates[axis].unsignedLongLongValue;
        }
        if (node.second_input != UINT32_MAX) {
            MPSShape *auxiliary = shapes[node.second_input];
            meta.auxiliaryRank = (uint32_t)auxiliary.count;
            meta.indexType = value_types[node.second_input];
            for (NSUInteger axis = 0U; axis < auxiliary.count; axis++)
                meta.auxiliaryDims[axis] =
                        auxiliary[axis].unsignedLongLongValue;
            if (!custom_physical_layout(
                        auxiliary, node.second_input, value_strides, layout_offsets,
                        value_states, meta.auxiliaryStrides, &meta.auxiliaryOffset))
                return nil;
        }
        NSMutableArray<NSNumber *> *input_values =
                [NSMutableArray arrayWithCapacity:node.input_count];
        if (input_values == nil) return nil;
        uint64_t prefix = 0U;
        for (uint32_t item = 0U; item < node.input_count; item++) {
            [input_values addObject:@(node.inputs[item])];
            if (node.operation == SYNAPTIK_METAL_CUSTOM_CONCAT) {
                NSUInteger axis = node.axis;
                uint64_t extent =
                        shapes[node.inputs[item]][axis].unsignedLongLongValue;
                meta.inputPrefixes[item] = prefix;
                meta.inputExtents[item] = extent;
                prefix += extent;
            }
        }
        step.inputValues = [input_values copy];
        if (task0064) {
            if (meta.width == 0U || meta.elementCount == 0U
                    || meta.elementCount > UINT32_MAX)
                return nil;
            meta.gridWidth = meta.elementCount;
            meta.gridHeight = 1U;
            step.grid = MTLSizeMake((NSUInteger)meta.elementCount, 1U, 1U);
            step.metadata = custom_metadata(device, &meta, sizeof(meta));
        } else {
            MTLSize grid = MTLSizeMake(0U, 0U, 0U);
            if (meta.width == 0U
                    || !custom_grid(
                            meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
                return nil;
            step.grid = grid;
            step.metadata = custom_metadata(device, &meta, sizeof(meta));
        }
    } else if ((node.operation >= SYNAPTIK_METAL_CUSTOM_GT
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_TENSOR_MAX)
            || node.operation == SYNAPTIK_METAL_BOOL_AND
            || node.operation == SYNAPTIK_METAL_BOOL_OR) {
        SynaptikMetalBinaryMeta meta = {0};
        meta.rank = (uint32_t)output.count;
        meta.elementCount = shape_element_count(output);
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        uint64_t left_stride = 1U;
        uint64_t right_stride = 1U;
        MPSShape *right = shapes[node.second_input];
        NSUInteger left_padding = output.count - input.count;
        NSUInteger right_padding = output.count - right.count;
        uint64_t left_contiguous[16] = {0};
        uint64_t right_contiguous[16] = {0};
        for (NSUInteger reverse = 0U; reverse < input.count; reverse++) {
            NSUInteger axis = input.count - 1U - reverse;
            left_contiguous[axis] = left_stride;
            left_stride *= input[axis].unsignedLongLongValue;
        }
        for (NSUInteger reverse = 0U; reverse < right.count; reverse++) {
            NSUInteger axis = right.count - 1U - reverse;
            right_contiguous[axis] = right_stride;
            right_stride *= right[axis].unsignedLongLongValue;
        }
        for (NSUInteger axis = 0U; axis < output.count; axis++) {
            uint64_t extent = output[axis].unsignedLongLongValue;
            meta.dims[axis] = extent;
            if (axis >= left_padding) {
                NSUInteger source = axis - left_padding;
                meta.leftStrides[axis] =
                        input[source].unsignedLongLongValue == 1U && extent != 1U
                        ? 0U : left_contiguous[source];
            }
            if (axis >= right_padding) {
                NSUInteger source = axis - right_padding;
                meta.rightStrides[axis] =
                        right[source].unsignedLongLongValue == 1U && extent != 1U
                        ? 0U : right_contiguous[source];
            }
        }
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (node.operation == SYNAPTIK_METAL_BOOL_WHERE) {
        SynaptikMetalTernaryMeta meta = {0};
        meta.rank = (uint32_t)output.count;
        meta.elementCount = shape_element_count(output);
        for (NSUInteger axis = 0U; axis < output.count; axis++)
            meta.dims[axis] = output[axis].unsignedLongLongValue;
        if (!custom_broadcast_strides(
                        input, output, meta.conditionStrides)
                || !custom_broadcast_strides(
                        shapes[node.second_input], output, meta.trueStrides)
                || !custom_broadcast_strides(
                        shapes[node.auxiliary], output, meta.falseStrides))
            return nil;
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (node.operation == SYNAPTIK_METAL_BOOL_IS_FINITE
            || node.operation == SYNAPTIK_METAL_BOOL_IS_NAN
            || node.operation == SYNAPTIK_METAL_BOOL_IS_INF
            || node.operation == SYNAPTIK_METAL_BOOL_NOT
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_FLOOR
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_RELU)
            || node.operation == SYNAPTIK_METAL_CUSTOM_EXP
            || node.operation == SYNAPTIK_METAL_CUSTOM_SIGMOID) {
        SynaptikMetalPointMeta meta = {0};
        meta.elementCount = shape_element_count(output);
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (node.operation == SYNAPTIK_METAL_CUSTOM_SCALAR_MIN
            || node.operation == SYNAPTIK_METAL_CUSTOM_SCALAR_MAX) {
        SynaptikMetalPointMeta meta = {0};
        meta.elementCount = shape_element_count(output);
        meta.scalar = (uint32_t)node.attribute_values[0];
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (node.operation == SYNAPTIK_METAL_CUSTOM_CLAMP) {
        SynaptikMetalClampMeta meta = {0};
        meta.elementCount = shape_element_count(output);
        meta.lower = (uint32_t)node.attribute_values[0];
        meta.upper = (uint32_t)node.attribute_values[1];
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (node.operation == SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN
            || node.operation == SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX) {
        SynaptikMetalReductionMeta meta = {0};
        meta.rank = (uint32_t)input.count;
        if (node.axis == SYNAPTIK_METAL_REDUCTION_FULL) {
            meta.axesMask = input.count == 32U
                    ? UINT32_MAX : ((UINT32_C(1) << input.count) - 1U);
        } else {
            for (uint32_t index = 0U; index < node.attribute_count; index++)
                meta.axesMask |= UINT32_C(1) << (uint32_t)node.attribute_values[index];
        }
        uint64_t stride = 1U;
        meta.outputCount = 1U;
        meta.reducedCount = 1U;
        for (NSUInteger reverse = 0U; reverse < input.count; reverse++) {
            NSUInteger axis = input.count - 1U - reverse;
            uint64_t extent = input[axis].unsignedLongLongValue;
            meta.dims[axis] = extent;
            meta.strides[axis] = stride;
            stride *= extent;
            if ((meta.axesMask & (UINT32_C(1) << axis)) != 0U)
                meta.reducedCount *= extent;
            else
                meta.outputCount *= extent;
        }
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.outputCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else {
        SynaptikMetalScanMeta meta = {0};
        meta.elementCount = shape_element_count(input);
        meta.axisLength = input[node.axis].unsignedLongLongValue;
        meta.inner = 1U;
        for (NSUInteger axis = node.axis + 1U; axis < input.count; axis++)
            meta.inner *= input[axis].unsignedLongLongValue;
        meta.lineCount = meta.elementCount / meta.axisLength;
        meta.exclusive = (uint32_t)node.attribute_values[0];
        meta.reverse = (uint32_t)node.attribute_values[1];
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.lineCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    }
    if (step.metadata == nil) return nil;
    return step;
}

SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_executable_run(
        void *executable, uint32_t input_count, void *const *input_buffers,
        uint32_t output_count, void *const *output_buffers);


static MPSGraphTensor *exact_float32_scalar(MPSGraph *graph, uint32_t bits) {
    if (graph == nil) return nil;
    NSData *data = [NSData dataWithBytes:&bits length:sizeof(bits)];
    if (data == nil) return nil;
    return [graph constantWithData:data
            shape:@[@1]
            dataType:MPSDataTypeFloat32];
}

static MPSGraphTensor *exact_int32_scalar(MPSGraph *graph, int32_t value) {
    if (graph == nil) return nil;
    NSData *data = [NSData dataWithBytes:&value length:sizeof(value)];
    if (data == nil) return nil;
    return [graph constantWithData:data shape:@[@1] dataType:MPSDataTypeInt32];
}

static MPSGraphTensor *exact_int64_scalar(MPSGraph *graph, int64_t value) {
    if (graph == nil) return nil;
    NSData *data = [NSData dataWithBytes:&value length:sizeof(value)];
    if (data == nil) return nil;
    return [graph constantWithData:data shape:@[@1] dataType:MPSDataTypeInt64];
}

static double task0059_scalar_double(
        SynaptikMetalDecodedNode node, uint32_t offset) {
    uint32_t type = (uint32_t)node.attribute_values[offset];
    uint64_t bits = node.attribute_values[offset + 1U];
    if (type == SYNAPTIK_METAL_TYPE_FLOAT32) {
        uint32_t raw = (uint32_t)bits;
        float value = 0.0f;
        memcpy(&value, &raw, sizeof(value));
        return (double)value;
    }
    if (type == SYNAPTIK_METAL_TYPE_FLOAT64) {
        double value = 0.0;
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
    if (type == SYNAPTIK_METAL_TYPE_BFLOAT16) {
        uint32_t raw = ((uint32_t)bits) << 16U;
        float value = 0.0f;
        memcpy(&value, &raw, sizeof(value));
        return (double)value;
    }
    if (type == SYNAPTIK_METAL_TYPE_INT32) return (double)(int32_t)bits;
    if (type == SYNAPTIK_METAL_TYPE_INT64) return (double)(int64_t)bits;
    return bits == 0U ? 0.0 : 1.0;
}

static MPSGraphImToColOpDescriptor *task0059_im2col_descriptor(
        SynaptikMetalDecodedNode node,
        uint32_t offset,
        BOOL include_padding,
        MPSShape *spatial_shape) {
    if (spatial_shape == nil || spatial_shape.count != 4U) return nil;
    uint64_t kH = node.attribute_values[offset];
    uint64_t kW = node.attribute_values[offset + 1U];
    uint64_t sH = node.attribute_values[offset + 2U];
    uint64_t sW = node.attribute_values[offset + 3U];
    uint64_t pH = include_padding ? node.attribute_values[offset + 4U] : 0U;
    uint64_t pW = include_padding ? node.attribute_values[offset + 5U] : 0U;
    uint64_t dH = node.attribute_values[offset + 6U];
    uint64_t dW = node.attribute_values[offset + 7U];
    BOOL ceil_mode = node.attribute_values[offset + 8U] != 0U;
    uint64_t hOut = 0U, wOut = 0U;
    uint64_t inputH = spatial_shape[2].unsignedLongLongValue;
    uint64_t inputW = spatial_shape[3].unsignedLongLongValue;
    if (!task0059_window_extent(inputH, kH, pH, sH, dH, ceil_mode, &hOut)
            || !task0059_window_extent(inputW, kW, pW, sW, dW, ceil_mode, &wOut))
        return nil;
    uint64_t effectiveH = dH * (kH - 1U) + 1U;
    uint64_t effectiveW = dW * (kW - 1U) + 1U;
    if ((hOut - 1U) > (UINT64_MAX - effectiveH) / sH
            || (wOut - 1U) > (UINT64_MAX - effectiveW) / sW)
        return nil;
    uint64_t requiredH = (hOut - 1U) * sH + effectiveH;
    uint64_t requiredW = (wOut - 1U) * sW + effectiveW;
    uint64_t paddedH = inputH + pH * 2U;
    uint64_t paddedW = inputW + pW * 2U;
    uint64_t extraH = requiredH > paddedH ? requiredH - paddedH : 0U;
    uint64_t extraW = requiredW > paddedW ? requiredW - paddedW : 0U;
    if (pH > UINT64_MAX - extraH || pW > UINT64_MAX - extraW) return nil;
    return [MPSGraphImToColOpDescriptor
            descriptorWithKernelWidth:(NSUInteger)kW
            kernelHeight:(NSUInteger)kH
            strideInX:(NSUInteger)sW
            strideInY:(NSUInteger)sH
            dilationRateInX:(NSUInteger)dW
            dilationRateInY:(NSUInteger)dH
            paddingLeft:(NSUInteger)pW
            paddingRight:(NSUInteger)(pW + extraW)
            paddingTop:(NSUInteger)pH
            paddingBottom:(NSUInteger)(pH + extraH)
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
}

static BOOL task0063_shape_is_uint32_bounded(MPSShape *shape, BOOL allow_scalar) {
    if (shape == nil || shape.count > SYNAPTIK_MAX_RANK
            || (!allow_scalar && shape.count == 0U))
        return NO;
    uint64_t elements = 1U;
    for (NSNumber *encoded in shape) {
        uint64_t dimension = encoded.unsignedLongLongValue;
        if (dimension == 0U || dimension > UINT32_MAX
                || elements > UINT32_MAX / dimension)
            return NO;
        elements *= dimension;
    }
    return elements <= UINT32_MAX;
}

static BOOL task0064_positive_uint32(uint64_t value) {
    return value > 0U && value <= UINT32_MAX;
}

static BOOL task0064_window_extent(
        uint64_t input,
        uint64_t kernel,
        uint64_t stride,
        uint64_t padding,
        uint64_t dilation,
        BOOL ceil_mode,
        uint64_t *result) {
    if (result == NULL || !task0064_positive_uint32(input)
            || !task0064_positive_uint32(kernel)
            || !task0064_positive_uint32(stride)
            || padding > UINT32_MAX
            || !task0064_positive_uint32(dilation)
            || kernel - 1U > (UINT32_MAX - 1U) / dilation)
        return NO;
    uint64_t effective = dilation * (kernel - 1U) + 1U;
    if (padding > (UINT32_MAX - input) / 2U) return NO;
    uint64_t padded = input + 2U * padding;
    if (padded < effective) return NO;
    uint64_t numerator = padded - effective;
    uint64_t extent = numerator / stride
            + (ceil_mode && numerator % stride != 0U ? 1U : 0U) + 1U;
    if (!task0064_positive_uint32(extent)) return NO;
    if (extent - 1U > UINT32_MAX / stride) return NO;
    *result = extent;
    return YES;
}

static BOOL task0064_validate_node(
        SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes,
        const uint8_t *states,
        const uint8_t *local_singleton_height) {
    BOOL convolution = node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
            || node.operation == SYNAPTIK_METAL_CUSTOM_CONV3D;
    uint32_t spatial =
            node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                    || node.operation == SYNAPTIK_METAL_CUSTOM_MAX_POOL2D
                    || node.operation == SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D
            ? 2U : 3U;
    uint32_t rank = spatial + 2U;
    MPSShape *input = shapes[node.first_input];
    MPSShape *output = shapes[node.output];
    BOOL two_dimensional = spatial == 2U;
    BOOL input_state_valid = states[node.first_input] == SYNAPTIK_METAL_VALUE_CANONICAL
            || (two_dimensional
                    && states[node.first_input] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                    && local_singleton_height[node.first_input] != 0U);
    if (!input_state_valid || input.count != rank || output.count != rank
            || states[node.output] != SYNAPTIK_METAL_VALUE_UNAVAILABLE
            || !task0063_shape_is_uint32_bounded(input, NO)
            || !task0063_shape_is_uint32_bounded(output, NO)
            || input[0].unsignedLongLongValue != output[0].unsignedLongLongValue)
        return NO;
    if (convolution) {
        if (node.input_count < 2U || node.input_count > 3U
                || node.second_input == UINT32_MAX
                || node.attribute_kind
                        != (spatial == 2U
                                ? SYNAPTIK_METAL_CUSTOM_ATTR_CONV_2D
                                : SYNAPTIK_METAL_CUSTOM_ATTR_CONV_3D)
                || node.attribute_count != spatial * 3U + 1U)
            return NO;
        MPSShape *weight = shapes[node.second_input];
        BOOL weight_state_valid =
                states[node.second_input] == SYNAPTIK_METAL_VALUE_CANONICAL
                || (two_dimensional
                        && states[node.second_input] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                        && local_singleton_height[node.second_input] != 0U);
        uint64_t groups = node.attribute_values[spatial * 3U];
        if (!weight_state_valid || weight.count != rank
                || !task0063_shape_is_uint32_bounded(weight, NO)
                || !task0064_positive_uint32(groups)
                || input[1].unsignedLongLongValue % groups != 0U
                || weight[0].unsignedLongLongValue % groups != 0U
                || weight[1].unsignedLongLongValue
                        != input[1].unsignedLongLongValue / groups
                || output[1].unsignedLongLongValue
                        != weight[0].unsignedLongLongValue)
            return NO;
        if (node.input_count == 3U) {
            MPSShape *bias = shapes[node.auxiliary];
            if (states[node.auxiliary] != SYNAPTIK_METAL_VALUE_CANONICAL
                    || bias.count != 1U
                    || !task0063_shape_is_uint32_bounded(bias, NO)
                    || bias[0].unsignedLongLongValue
                            != weight[0].unsignedLongLongValue)
                return NO;
        }
        uint64_t contributors = weight[1].unsignedLongLongValue;
        for (uint32_t axis = 0U; axis < spatial; axis++) {
            uint64_t expected = 0U;
            if (!task0064_window_extent(
                        input[axis + 2U].unsignedLongLongValue,
                        weight[axis + 2U].unsignedLongLongValue,
                        node.attribute_values[axis],
                        node.attribute_values[spatial + axis],
                        node.attribute_values[spatial * 2U + axis],
                        NO,
                        &expected)
                    || output[axis + 2U].unsignedLongLongValue != expected
                    || contributors > UINT32_MAX
                            / weight[axis + 2U].unsignedLongLongValue)
                return NO;
            contributors *= weight[axis + 2U].unsignedLongLongValue;
        }
        return contributors <= UINT32_MAX;
    }
    if (node.input_count != 1U || node.second_input != UINT32_MAX
            || node.attribute_kind
                    != (spatial == 2U
                            ? SYNAPTIK_METAL_CUSTOM_ATTR_WINDOW_2D
                            : SYNAPTIK_METAL_CUSTOM_ATTR_WINDOW_3D)
            || node.attribute_count != spatial * 4U + 1U
            || node.attribute_values[spatial * 4U] > 1U
            || input[1].unsignedLongLongValue != output[1].unsignedLongLongValue)
        return NO;
    uint64_t divisor = 1U;
    for (uint32_t axis = 0U; axis < spatial; axis++) {
        uint64_t expected = 0U;
        uint64_t kernel = node.attribute_values[axis];
        if (!task0064_window_extent(
                    input[axis + 2U].unsignedLongLongValue,
                    kernel,
                    node.attribute_values[spatial + axis],
                    node.attribute_values[spatial * 2U + axis],
                    node.attribute_values[spatial * 3U + axis],
                    node.attribute_values[spatial * 4U] != 0U,
                    &expected)
                || output[axis + 2U].unsignedLongLongValue != expected
                || !task0064_positive_uint32(kernel)
                || divisor > SYNAPTIK_TASK0064_MAX_POOL_KERNEL_POSITIONS / kernel)
            return NO;
        divisor *= kernel;
    }
    return YES;
}


static int32_t synaptik_metal_create_decoded(
        void *context, uint32_t route, uint32_t value_count,
        const uint32_t *value_ranks, const uint64_t *value_dimensions,
        const uint64_t *value_strides, const uint64_t *layout_offsets,
        const uint64_t *layout_spans, const uint8_t *declared_states,
        const uint8_t *declared_types,
        uint32_t node_count, const SynaptikMetalDecodedNode *nodes,
        uint32_t feed_count, const uint32_t *feed_indices,
        uint32_t target_count, const uint32_t *target_indices, void **out_executable) {
    if (out_executable == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_executable = NULL;
    BOOL has_layouts = value_strides != NULL && layout_offsets != NULL
            && layout_spans != NULL && declared_states != NULL;
    BOOL lacks_layouts = value_strides == NULL && layout_offsets == NULL
            && layout_spans == NULL && declared_states == NULL;
    if (context == NULL
            || (route != SYNAPTIK_METAL_ROUTE_MPSGRAPH
                    && route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
            || value_count == 0U || node_count == 0U
            || feed_count == 0U || target_count == 0U
            || value_ranks == NULL || value_dimensions == NULL || declared_types == NULL
            || (!has_layouts && !lacks_layouts)
            || nodes == NULL || feed_indices == NULL || target_indices == NULL)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    if ((uint64_t)value_count > SIZE_MAX / SYNAPTIK_MAX_RANK)
        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
    @try { @autoreleasepool {
        NSMutableArray<MPSShape *> *shapes = [NSMutableArray arrayWithCapacity:value_count];
        NSMutableArray<NSNumber *> *element_counts =
                [NSMutableArray arrayWithCapacity:value_count];
        for (uint32_t value = 0; value < value_count; value++) {
            uint32_t rank = value_ranks[value];
            if (rank > SYNAPTIK_MAX_RANK)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            uint64_t elements = 1U;
            NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:rank];
            for (uint32_t axis = 0; axis < SYNAPTIK_MAX_RANK; axis++) {
                uint64_t dimension = value_dimensions[(size_t)value * SYNAPTIK_MAX_RANK + axis];
                if (axis < rank) {
                    if (dimension == 0U || dimension > (uint64_t)NSUIntegerMax
                            || elements > UINT64_MAX / dimension)
                        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                    elements *= dimension;
                    [shape addObject:@((NSUInteger)dimension)];
                } else if (dimension != 0U) {
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                }
            }
            if (elements > (uint64_t)NSUIntegerMax)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            [shapes addObject:[shape copy]];
            [element_counts addObject:@(elements)];
        }

        NSMutableData *state_data = [NSMutableData dataWithLength:value_count];
        NSMutableData *type_data = [NSMutableData dataWithLength:value_count];
        NSMutableData *used_data = [NSMutableData dataWithLength:value_count];
        NSMutableData *produced_data = [NSMutableData dataWithLength:value_count];
        NSMutableData *targeted_data = [NSMutableData dataWithLength:value_count];
        NSMutableData *local_transpose_data = [NSMutableData dataWithLength:value_count];
        NSMutableData *local_transpose_source_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * sizeof(uint32_t)];
        NSMutableData *local_singleton_height_data =
                [NSMutableData dataWithLength:value_count];
        NSMutableData *local_singleton_height_source_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * sizeof(uint32_t)];
        NSMutableArray<SynaptikMetalIndexValidation *> *index_validations =
                [NSMutableArray array];
        if (state_data == nil || type_data == nil || used_data == nil
                || produced_data == nil || targeted_data == nil
                || local_transpose_data == nil || local_transpose_source_data == nil
                || local_singleton_height_data == nil
                || local_singleton_height_source_data == nil
                || index_validations == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        uint8_t *states = state_data.mutableBytes;
        uint8_t *value_types = type_data.mutableBytes;
        uint8_t *used = used_data.mutableBytes;
        uint8_t *produced = produced_data.mutableBytes;
        uint8_t *targeted = targeted_data.mutableBytes;
        uint8_t *local_transpose = local_transpose_data.mutableBytes;
        uint32_t *local_transpose_sources = local_transpose_source_data.mutableBytes;
        uint8_t *local_singleton_height = local_singleton_height_data.mutableBytes;
        uint32_t *local_singleton_height_sources =
                local_singleton_height_source_data.mutableBytes;
        for (uint32_t value = 0U; value < value_count; value++) {
            local_transpose_sources[value] = UINT32_MAX;
            local_singleton_height_sources[value] = UINT32_MAX;
        }
        memcpy(value_types, declared_types, value_count);
        BOOL contains_matmul = NO;
        BOOL contains_custom = NO;
        for (uint32_t feed = 0; feed < feed_count; feed++) {
            uint32_t value = feed_indices[feed];
            if (value >= value_count
                    || states[value] != SYNAPTIK_METAL_VALUE_UNAVAILABLE
                    || (has_layouts
                            && declared_states[value] != SYNAPTIK_METAL_VALUE_CANONICAL
                            && declared_states[value]
                                    != SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            states[value] = has_layouts
                    ? declared_states[value] : SYNAPTIK_METAL_VALUE_CANONICAL;
            used[value] = 1U;
        }
        for (uint32_t node_index = 0; node_index < node_count; node_index++) {
            SynaptikMetalDecodedNode node = nodes[node_index];
            if (node.first_input >= value_count
                    || node.output >= value_count
                    || node.output_count == 0U
                    || node.output_count > SYNAPTIK_MAX_NODE_OUTPUTS
                    || states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            for (uint32_t output = 0U; output < node.output_count; output++) {
                uint32_t value = node.outputs[output];
                if (value >= value_count
                        || states[value] != SYNAPTIK_METAL_VALUE_UNAVAILABLE)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            BOOL task0064_two_dimensional_consumer =
                    node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                    || node.operation == SYNAPTIK_METAL_CUSTOM_MAX_POOL2D
                    || node.operation == SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D;
            for (uint32_t input = 0U; input < node.input_count; input++) {
                uint32_t value = node.inputs[input];
                if (value >= value_count
                        || states[value] == SYNAPTIK_METAL_VALUE_UNAVAILABLE)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                used[value] = 1U;
                if (local_transpose[value] != 0U
                        && node.operation != SYNAPTIK_METAL_MPSGRAPH_MATMUL)
                    local_transpose[value] = 3U;
                if (local_singleton_height[value] != 0U
                        && !task0064_two_dimensional_consumer)
                    local_singleton_height[value] = 3U;
            }
            BOOL custom_node = node_uses_custom_kernel(node, shapes, value_types);
            if (route == SYNAPTIK_METAL_ROUTE_MPSGRAPH
                    && (!operation_has_direct_mpsgraph(node.operation)
                            || matmul_uses_custom_kernel(node, shapes, value_types)))
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION;
            if (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM && custom_node)
                contains_custom = YES;
            BOOL affine_view = NO;
            switch ((SynaptikMetalOperation)node.operation) {
                case SYNAPTIK_METAL_CUSTOM_EXP:
                case SYNAPTIK_METAL_CUSTOM_SIGMOID:
                    return SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION;
                case SYNAPTIK_METAL_MPSGRAPH_NEG:
                case SYNAPTIK_METAL_MPSGRAPH_ABS:
                case SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL:
                case SYNAPTIK_METAL_MPSGRAPH_LOG:
                case SYNAPTIK_METAL_MPSGRAPH_LOG1P:
                case SYNAPTIK_METAL_MPSGRAPH_EXPM1:
                case SYNAPTIK_METAL_MPSGRAPH_ERF:
                case SYNAPTIK_METAL_MPSGRAPH_SQRT:
                case SYNAPTIK_METAL_MPSGRAPH_RSQRT:
                case SYNAPTIK_METAL_MPSGRAPH_TANH:
                case SYNAPTIK_METAL_MPSGRAPH_GELU:
                case SYNAPTIK_METAL_MPSGRAPH_GELU_TANH:
                case SYNAPTIK_METAL_MPSGRAPH_SILU:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_FLOOR:
                case SYNAPTIK_METAL_CUSTOM_CEIL:
                case SYNAPTIK_METAL_CUSTOM_SIGN:
                case SYNAPTIK_METAL_CUSTOM_RELU:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || shapes[node.first_input].count == 0U
                            || shapes[node.output].count == 0U
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_BOOL_IS_FINITE:
                case SYNAPTIK_METAL_BOOL_IS_NAN:
                case SYNAPTIK_METAL_BOOL_IS_INF:
                case SYNAPTIK_METAL_BOOL_NOT:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || [shapes[node.first_input] count] == 0U
                            || [shapes[node.output] count] == 0U
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SUB:
                case SYNAPTIK_METAL_MPSGRAPH_MUL:
                case SYNAPTIK_METAL_MPSGRAPH_DIV:
                case SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW:
                case SYNAPTIK_METAL_BOOL_AND:
                case SYNAPTIK_METAL_BOOL_OR:
                case SYNAPTIK_METAL_CUSTOM_GT:
                case SYNAPTIK_METAL_CUSTOM_GE:
                case SYNAPTIK_METAL_CUSTOM_LT:
                case SYNAPTIK_METAL_CUSTOM_LE:
                case SYNAPTIK_METAL_CUSTOM_EQ:
                case SYNAPTIK_METAL_CUSTOM_NE:
                case SYNAPTIK_METAL_CUSTOM_TENSOR_MIN:
                case SYNAPTIK_METAL_CUSTOM_TENSOR_MAX:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input >= value_count
                            || states[node.second_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || !node_has_no_attributes(node)
                            || ((node.operation == SYNAPTIK_METAL_BOOL_AND
                                            || node.operation == SYNAPTIK_METAL_BOOL_OR)
                                    && ([shapes[node.first_input] count] == 0U
                                            || [shapes[node.second_input] count] == 0U
                                            || [shapes[node.output] count] == 0U))
                            || !shape_broadcasts_exactly_to(
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    used[node.second_input] = 1U;
                    break;
                case SYNAPTIK_METAL_BOOL_WHERE:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input >= value_count
                            || node.auxiliary >= value_count
                            || states[node.second_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || states[node.auxiliary] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE
                            || node.attribute_count != 0U
                            || node.axis != UINT32_MAX
                            || !node_values_are_zero_from(node, 0U)
                            || [shapes[node.first_input] count] == 0U
                            || [shapes[node.second_input] count] == 0U
                            || [shapes[node.auxiliary] count] == 0U
                            || [shapes[node.output] count] == 0U
                            || !shape_broadcasts_exactly_to(
                                    shapes[node.second_input],
                                    shapes[node.auxiliary],
                                    shapes[node.output])
                            || !shape_broadcasts_exactly_to(
                                    shapes[node.first_input],
                                    shapes[node.output],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    used[node.second_input] = 1U;
                    used[node.auxiliary] = 1U;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_CONTIGUOUS:
                    if (node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_RESHAPE:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_target_matches(node, shapes[node.output])
                            || shape_element_count(shapes[node.first_input])
                                    != shape_element_count(shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_target_matches(node, shapes[node.output])
                            || !shape_expands_exactly_to(
                                    shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_PERMUTE:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_permutation_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (node_is_last_two_transpose(
                                node, shapes[node.first_input], shapes[node.output],
                                states[node.first_input], value_strides, layout_offsets,
                                layout_spans, declared_states)) {
                        local_transpose[node.output] = 1U;
                        local_transpose_sources[node.output] = node.first_input;
                    }
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND_DIMS:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_expand_dims_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (node_is_task0064_singleton_height(
                                node, shapes[node.first_input], shapes[node.output],
                                states[node.first_input], value_strides, layout_offsets,
                                layout_spans, declared_states)) {
                        local_singleton_height[node.output] = 1U;
                        local_singleton_height_sources[node.output] = node.first_input;
                    }
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SQUEEZE:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_squeeze_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SUM:
                case SYNAPTIK_METAL_MPSGRAPH_MEAN:
                case SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN:
                case SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX: {
                    BOOL keep_dimensions = NO;
                    NSArray<NSNumber *> *axes = node_reduction_axes(
                            node, shapes[node.first_input], shapes[node.output],
                            &keep_dimensions);
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || value_ranks[node.first_input] == 0U
                            || axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_PROD:
                case SYNAPTIK_METAL_CUSTOM_ALL:
                case SYNAPTIK_METAL_CUSTOM_ANY:
                case SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP:
                case SYNAPTIK_METAL_MPSGRAPH_L1_NORM:
                case SYNAPTIK_METAL_MPSGRAPH_L2_NORM: {
                    BOOL keep_dimensions = NO;
                    NSArray<NSNumber *> *axes = node_reduction_axes(
                            node, shapes[node.first_input], shapes[node.output],
                            &keep_dimensions);
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_VARIANCE:
                case SYNAPTIK_METAL_MPSGRAPH_STANDARD_DEVIATION: {
                    BOOL keep_dimensions = NO;
                    uint64_t correction = 0U;
                    NSArray<NSNumber *> *axes = node_statistical_axes(
                            node, shapes[node.first_input], shapes[node.output],
                            &keep_dimensions, &correction);
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SCALAR_MIN:
                case SYNAPTIK_METAL_CUSTOM_SCALAR_MAX:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_SUB:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_SCALAR_VALUE
                            || node.attribute_count != 1U
                            || node.axis != UINT32_MAX
                            || node.auxiliary != 0U
                            || (node.attribute_values[0] & ~UINT64_C(0xffffffff)) != 0U
                            || !node_values_are_zero_from(node, 1U)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_CLAMP:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_CLAMP_RANGE
                            || node.attribute_count != 2U
                            || node.axis != UINT32_MAX
                            || node.auxiliary != 0U
                            || (node.attribute_values[0] & ~UINT64_C(0xffffffff)) != 0U
                            || (node.attribute_values[1] & ~UINT64_C(0xffffffff)) != 0U
                            || !node_values_are_zero_from(node, 2U)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_CUM_SUM:
                case SYNAPTIK_METAL_CUSTOM_CUM_PROD:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_SCAN
                            || node.attribute_count != 2U
                            || node.axis >= shapes[node.first_input].count
                            || node.auxiliary != 0U
                            || node.attribute_values[0] > 1U
                            || node.attribute_values[1] > 1U
                            || !node_values_are_zero_from(node, 2U)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_SORT:
                case SYNAPTIK_METAL_CUSTOM_ARGSORT: {
                    MPSShape *input_shape = shapes[node.first_input];
                    MPSShape *output_shape = shapes[node.output];
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.output_count != 1U
                            || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_SORT
                            || node.attribute_count != 2U
                            || node.attribute_values[0] >= input_shape.count
                            || node.attribute_values[1] > 1U
                            || !task0063_shape_is_uint32_bounded(input_shape, NO)
                            || !task0063_shape_is_uint32_bounded(output_shape, NO)
                            || ![input_shape isEqualToArray:output_shape])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_TOP_K: {
                    MPSShape *input_shape = shapes[node.first_input];
                    MPSShape *value_shape = shapes[node.outputs[0]];
                    MPSShape *index_shape = shapes[node.outputs[1]];
                    uint64_t axis = node.attribute_values[0];
                    uint64_t k = node.attribute_values[1];
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.output_count != 2U
                            || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_TOP_K
                            || node.attribute_count != 4U
                            || axis >= input_shape.count
                            || k == 0U || k > UINT32_MAX
                            || k > input_shape[(NSUInteger)axis].unsignedLongLongValue
                            || node.attribute_values[2] > 1U
                            || node.attribute_values[3] > 1U
                            || !task0063_shape_is_uint32_bounded(input_shape, NO)
                            || !task0063_shape_is_uint32_bounded(value_shape, NO)
                            || ![value_shape isEqualToArray:index_shape]
                            || value_shape.count != input_shape.count)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    for (NSUInteger dimension = 0U;
                            dimension < input_shape.count;
                            dimension++) {
                        uint64_t expected = dimension == axis
                                ? k : input_shape[dimension].unsignedLongLongValue;
                        if (value_shape[dimension].unsignedLongLongValue != expected)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    }
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_ARG_MAX:
                case SYNAPTIK_METAL_CUSTOM_ARG_MIN: {
                    MPSShape *input_shape = shapes[node.first_input];
                    MPSShape *output_shape = shapes[node.output];
                    uint64_t axis = node.attribute_values[0];
                    BOOL keep = node.attribute_values[1] != 0U;
                    NSUInteger expected_rank = keep
                            ? input_shape.count : input_shape.count - 1U;
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.output_count != 1U
                            || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_ARG_EXTREMA
                            || node.attribute_count != 3U
                            || axis >= input_shape.count
                            || node.attribute_values[1] > 1U
                            || node.attribute_values[2] < 1U
                            || node.attribute_values[2] > 2U
                            || !task0063_shape_is_uint32_bounded(input_shape, NO)
                            || !task0063_shape_is_uint32_bounded(output_shape, YES)
                            || output_shape.count != expected_rank)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    for (NSUInteger source = 0U, destination = 0U;
                            source < input_shape.count;
                            source++) {
                        if (source == axis) {
                            if (keep
                                    && output_shape[destination++].unsignedLongLongValue != 1U)
                                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        } else if (output_shape[destination++].unsignedLongLongValue
                                != input_shape[source].unsignedLongLongValue) {
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        }
                    }
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_CONV2D:
                case SYNAPTIK_METAL_CUSTOM_CONV3D:
                case SYNAPTIK_METAL_CUSTOM_MAX_POOL2D:
                case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D:
                case SYNAPTIK_METAL_CUSTOM_MAX_POOL3D:
                case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D:
                    if (!task0064_validate_node(
                                node, shapes, states, local_singleton_height))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (task0064_two_dimensional_consumer) {
                        if (local_singleton_height[node.first_input] == 1U)
                            local_singleton_height[node.first_input] = 2U;
                        if (node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                                && local_singleton_height[node.second_input] == 1U)
                            local_singleton_height[node.second_input] = 2U;
                    }
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_MATMUL: {
                    if (node.second_input >= value_count
                            || !node_has_no_attributes(node))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    BOOL left_valid =
                            states[node.first_input] == SYNAPTIK_METAL_VALUE_CANONICAL
                            || (states[node.first_input] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                                    && local_transpose[node.first_input] != 0U);
                    BOOL right_valid =
                            states[node.second_input] == SYNAPTIK_METAL_VALUE_CANONICAL
                            || (states[node.second_input] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                                    && local_transpose[node.second_input] != 0U);
                    if (!left_valid || !right_valid
                            || !matmul_shapes_match(
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    used[node.second_input] = 1U;
                    if (local_transpose[node.first_input] == 1U)
                        local_transpose[node.first_input] = 2U;
                    if (local_transpose[node.second_input] == 1U)
                        local_transpose[node.second_input] = 2U;
                    if (!custom_node) contains_matmul = YES;
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_GATHER: {
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input >= value_count
                            || value_ranks[node.second_input] == 0U
                            || states[node.second_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || !node_gather_matches(
                                    node,
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    NSUInteger position =
                            feed_position(node.second_input, feed_count, feed_indices);
                    if (position == NSNotFound)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    SynaptikMetalIndexValidation *validation =
                            [SynaptikMetalIndexValidation new];
                    if (validation == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    validation.operation = node.operation;
                    validation.nodeIndex = node_index;
                    validation.feedPosition = position;
                    validation.elementCount =
                            shape_element_count(shapes[node.second_input]);
                    validation.bound = [shapes[node.first_input][node.axis]
                            unsignedLongLongValue];
                    validation.axis = node.axis;
                    [index_validations addObject:validation];
                    used[node.second_input] = 1U;
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS: {
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input >= value_count
                            || node.auxiliary >= value_count
                            || value_ranks[node.second_input] == 0U
                            || value_ranks[node.auxiliary] == 0U
                            || states[node.second_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || states[node.auxiliary] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || !node_scatter_elements_matches(
                                    node,
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.auxiliary],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    NSUInteger position =
                            feed_position(node.second_input, feed_count, feed_indices);
                    if (position == NSNotFound)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    uint64_t element_count =
                            shape_element_count(shapes[node.second_input]);
                    NSUInteger rank = shapes[node.second_input].count;
                    if (rank == 0U
                            || rank > SYNAPTIK_MAX_RANK
                            || element_count > (uint64_t)NSUIntegerMax / sizeof(uint64_t))
                        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                    NSMutableData *extents =
                            [NSMutableData dataWithLength:rank * sizeof(uint64_t)];
                    NSMutableData *strides =
                            [NSMutableData dataWithLength:rank * sizeof(uint64_t)];
                    NSMutableData *targets = [NSMutableData dataWithLength:
                            (NSUInteger)element_count * sizeof(uint64_t)];
                    SynaptikMetalIndexValidation *validation =
                            [SynaptikMetalIndexValidation new];
                    if (extents == nil || strides == nil || targets == nil || validation == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    uint64_t *extent_cells = extents.mutableBytes;
                    uint64_t *stride_cells = strides.mutableBytes;
                    uint64_t stride = 1U;
                    for (NSUInteger dimension = rank; dimension-- > 0U;) {
                        uint64_t extent =
                                shapes[node.second_input][dimension].unsignedLongLongValue;
                        uint64_t data_extent =
                                shapes[node.first_input][dimension].unsignedLongLongValue;
                        if (extent == 0U || data_extent == 0U
                                || stride > UINT64_MAX / data_extent)
                            return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                        extent_cells[dimension] = extent;
                        stride_cells[dimension] = stride;
                        stride *= data_extent;
                    }
                    validation.targetCount = element_count;
                    validation.operation = node.operation;
                    validation.nodeIndex = node_index;
                    validation.feedPosition = position;
                    validation.elementCount = element_count;
                    validation.bound = [shapes[node.first_input][node.axis]
                            unsignedLongLongValue];
                    validation.axis = node.axis;
                    validation.coordinateExtents = extents;
                    validation.dataStrides = strides;
                    validation.targetScratch = targets;
                    [index_validations addObject:validation];
                    used[node.second_input] = 1U;
                    used[node.auxiliary] = 1U;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_CAST:
                case SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS:
                case SYNAPTIK_METAL_CUSTOM_GATHER_ND:
                case SYNAPTIK_METAL_CUSTOM_SELECT:
                case SYNAPTIK_METAL_CUSTOM_PAD:
                case SYNAPTIK_METAL_CUSTOM_SLICE:
                case SYNAPTIK_METAL_CUSTOM_CONCAT:
                case SYNAPTIK_METAL_CUSTOM_STACK:
                case SYNAPTIK_METAL_CUSTOM_TILE:
                case SYNAPTIK_METAL_CUSTOM_UNFOLD2D:
                case SYNAPTIK_METAL_CUSTOM_UNFOLD3D:
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND:
                case SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: {
                    BOOL storage_layout_operation =
                            node.operation == SYNAPTIK_METAL_CUSTOM_SELECT
                            || node.operation == SYNAPTIK_METAL_CUSTOM_SLICE;
                    for (uint32_t input = 0U; input < node.input_count; input++) {
                        uint8_t input_state = states[node.inputs[input]];
                        if ((!storage_layout_operation
                                        && input_state != SYNAPTIK_METAL_VALUE_CANONICAL)
                                || (storage_layout_operation
                                        && input_state
                                                == SYNAPTIK_METAL_VALUE_UNAVAILABLE))
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    }
                    if (!task0059_validate_shape(node, shapes))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (route == SYNAPTIK_METAL_ROUTE_MPSGRAPH
                            && !task0059_mpsgraph_window_origins_fit(node, shapes))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM) {
                        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS
                                && node.attribute_values[1] < node.attribute_values[0])
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_FOLD2D
                                || node.operation == SYNAPTIK_METAL_MPSGRAPH_FOLD3D) {
                            NSUInteger dimensions =
                                    node.operation == SYNAPTIK_METAL_MPSGRAPH_FOLD2D
                                            ? 2U : 3U;
                            NSUInteger offset = dimensions + 3U;
                            for (NSUInteger spatial = 0U;
                                    spatial < dimensions; spatial++) {
                                uint64_t kernel = node.attribute_values[offset + spatial];
                                uint64_t stride =
                                        node.attribute_values[offset + dimensions + spatial];
                                uint64_t dilation = node.attribute_values[
                                        offset + dimensions * 3U + spatial];
                                if (kernel == 0U || dilation == 0U
                                        || kernel - 1U > (UINT64_MAX - 1U) / dilation
                                        || stride < dilation * (kernel - 1U) + 1U)
                                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                            }
                        }
                    }
                    if (node.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
                            || node.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
                            || node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND) {
                        NSUInteger position = route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                                ? node.second_input
                                : feed_position(
                                        node.second_input, feed_count, feed_indices);
                        if (position == NSNotFound)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        SynaptikMetalIndexValidation *validation =
                                [SynaptikMetalIndexValidation new];
                        if (validation == nil)
                            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                        validation.operation = node.operation;
                        validation.nodeIndex = node_index;
                        validation.feedPosition = position;
                        validation.preflight =
                                feed_position(
                                        node.second_input, feed_count, feed_indices)
                                != NSNotFound;
                        validation.elementCount =
                                shape_element_count(shapes[node.second_input]);
                        validation.indexType = declared_types[node.second_input];
                        if (node.operation
                                == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS) {
                            validation.bound =
                                    shapes[node.first_input][node.axis]
                                            .unsignedLongLongValue;
                            validation.axis = node.axis;
                        } else {
                            NSUInteger batch =
                                    (NSUInteger)node.attribute_values[0];
                            NSUInteger tuple = (NSUInteger)shapes[node.second_input]
                                    .lastObject.unsignedLongLongValue;
                            NSMutableData *bounds = [NSMutableData
                                    dataWithLength:tuple * sizeof(uint64_t)];
                            if (bounds == nil)
                                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                            uint64_t *cells = bounds.mutableBytes;
                            for (NSUInteger component = 0U;
                                    component < tuple; component++) {
                                cells[component] =
                                        shapes[node.first_input][batch + component]
                                                .unsignedLongLongValue;
                            }
                            validation.coordinateExtents = bounds;
                            validation.tupleDepth = (uint32_t)tuple;
                            validation.axis = UINT32_MAX;
                            if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
                                    && route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM) {
                                if (node.attribute_values[1] != 1U
                                        || validation.elementCount % tuple != 0U)
                                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                                NSUInteger prefix_rank =
                                        shapes[node.second_input].count - 1U;
                                NSUInteger data_rank = shapes[node.first_input].count;
                                uint64_t target_count =
                                        validation.elementCount / (uint64_t)tuple;
                                if (target_count
                                        > (uint64_t)NSUIntegerMax / sizeof(uint64_t))
                                    return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                                NSMutableData *prefix = [NSMutableData
                                        dataWithLength:prefix_rank * sizeof(uint64_t)];
                                NSMutableData *strides = [NSMutableData
                                        dataWithLength:data_rank * sizeof(uint64_t)];
                                NSMutableData *targets = [NSMutableData dataWithLength:
                                        (NSUInteger)target_count * sizeof(uint64_t)];
                                if (prefix == nil || strides == nil || targets == nil)
                                    return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                                uint64_t *prefix_cells = prefix.mutableBytes;
                                for (NSUInteger dimension = 0U;
                                        dimension < prefix_rank; dimension++)
                                    prefix_cells[dimension] =
                                            shapes[node.second_input][dimension]
                                                    .unsignedLongLongValue;
                                uint64_t *stride_cells = strides.mutableBytes;
                                uint64_t stride = 1U;
                                for (NSUInteger dimension = data_rank;
                                        dimension-- > 0U;) {
                                    uint64_t extent = shapes[node.first_input][dimension]
                                            .unsignedLongLongValue;
                                    if (extent == 0U || stride > UINT64_MAX / extent)
                                        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                                    stride_cells[dimension] = stride;
                                    stride *= extent;
                                }
                                validation.axis = (uint32_t)batch;
                                validation.targetCount = target_count;
                                validation.prefixExtents = prefix;
                                validation.dataStrides = strides;
                                validation.targetScratch = targets;
                            }
                        }
                        [index_validations addObject:validation];
                    }
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_MSE: {
                    uint64_t reduction = node.attribute_values[0];
                    BOOL output_shape_valid = reduction == 1U
                            ? [shapes[node.first_input]
                                    isEqualToArray:shapes[node.output]]
                            : shapes[node.output].count == 0U;
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input >= value_count
                            || states[node.second_input]
                                    != SYNAPTIK_METAL_VALUE_CANONICAL
                            || shapes[node.first_input].count == 0U
                            || ![shapes[node.first_input]
                                    isEqualToArray:shapes[node.second_input]]
                            || node.attribute_kind
                                    != SYNAPTIK_METAL_MPSGRAPH_ATTR_MSE
                            || node.attribute_count != 1U
                            || reduction < 1U
                            || reduction > 3U
                            || node.axis != UINT32_MAX
                            || node.auxiliary != 0U
                            || !node_values_are_zero_from(node, 1U)
                            || !output_shape_valid)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    used[node.second_input] = 1U;
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_UNFOLD_AXIS:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || !node_unfold_axis_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ONE_HOT: {
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_one_hot_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    NSUInteger position =
                            feed_position(node.first_input, feed_count, feed_indices);
                    if (position == NSNotFound)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    SynaptikMetalIndexValidation *validation =
                            [SynaptikMetalIndexValidation new];
                    if (validation == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    validation.operation = node.operation;
                    validation.nodeIndex = node_index;
                    validation.feedPosition = position;
                    validation.elementCount =
                            shape_element_count(shapes[node.first_input]);
                    validation.bound = node.attribute_values[0];
                    validation.axis = UINT32_MAX;
                    [index_validations addObject:validation];
                    break;
                }
                default:
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            BOOL type_valid = NO;
            switch ((SynaptikMetalOperation)node.operation) {
                case SYNAPTIK_METAL_MPSGRAPH_NEG:
                case SYNAPTIK_METAL_MPSGRAPH_ABS:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_SUB:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW:
                case SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL:
                case SYNAPTIK_METAL_MPSGRAPH_LOG:
                case SYNAPTIK_METAL_MPSGRAPH_LOG1P:
                case SYNAPTIK_METAL_MPSGRAPH_EXPM1:
                case SYNAPTIK_METAL_MPSGRAPH_ERF:
                case SYNAPTIK_METAL_MPSGRAPH_SQRT:
                case SYNAPTIK_METAL_MPSGRAPH_RSQRT:
                case SYNAPTIK_METAL_CUSTOM_FLOOR:
                case SYNAPTIK_METAL_CUSTOM_CEIL:
                case SYNAPTIK_METAL_CUSTOM_SIGN:
                case SYNAPTIK_METAL_CUSTOM_RELU:
                case SYNAPTIK_METAL_MPSGRAPH_TANH:
                case SYNAPTIK_METAL_MPSGRAPH_GELU:
                case SYNAPTIK_METAL_MPSGRAPH_GELU_TANH:
                case SYNAPTIK_METAL_MPSGRAPH_SILU:
                case SYNAPTIK_METAL_CUSTOM_EXP:
                case SYNAPTIK_METAL_CUSTOM_SIGMOID:
                case SYNAPTIK_METAL_MPSGRAPH_CONTIGUOUS:
                case SYNAPTIK_METAL_MPSGRAPH_RESHAPE:
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND:
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND_DIMS:
                case SYNAPTIK_METAL_MPSGRAPH_SQUEEZE:
                case SYNAPTIK_METAL_MPSGRAPH_SUM:
                case SYNAPTIK_METAL_MPSGRAPH_MEAN:
                case SYNAPTIK_METAL_CUSTOM_SCALAR_MIN:
                case SYNAPTIK_METAL_CUSTOM_SCALAR_MAX:
                case SYNAPTIK_METAL_CUSTOM_CLAMP:
                case SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN:
                case SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX:
                case SYNAPTIK_METAL_CUSTOM_CUM_SUM:
                case SYNAPTIK_METAL_CUSTOM_CUM_PROD:
                case SYNAPTIK_METAL_MPSGRAPH_UNFOLD_AXIS:
                case SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP:
                case SYNAPTIK_METAL_MPSGRAPH_VARIANCE:
                case SYNAPTIK_METAL_MPSGRAPH_STANDARD_DEVIATION:
                case SYNAPTIK_METAL_MPSGRAPH_L1_NORM:
                case SYNAPTIK_METAL_MPSGRAPH_L2_NORM:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_PERMUTE: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = type == declared_types[node.output]
                            && (type == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || type == SYNAPTIK_METAL_TYPE_BFLOAT16
                                    || type == SYNAPTIK_METAL_TYPE_INT32
                                    || type == SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SORT:
                    type_valid = declared_types[node.first_input]
                            == declared_types[node.output];
                    break;
                case SYNAPTIK_METAL_CUSTOM_ARGSORT:
                    type_valid = declared_types[node.output]
                            == SYNAPTIK_METAL_TYPE_INT64;
                    break;
                case SYNAPTIK_METAL_CUSTOM_TOP_K:
                    type_valid = declared_types[node.first_input]
                                    == declared_types[node.outputs[0]]
                            && declared_types[node.outputs[1]]
                                    == SYNAPTIK_METAL_TYPE_INT64;
                    break;
                case SYNAPTIK_METAL_CUSTOM_ARG_MAX:
                case SYNAPTIK_METAL_CUSTOM_ARG_MIN:
                    type_valid = declared_types[node.first_input]
                                    != SYNAPTIK_METAL_TYPE_BOOL
                            && declared_types[node.output]
                                    == SYNAPTIK_METAL_TYPE_INT64;
                    break;
                case SYNAPTIK_METAL_CUSTOM_PROD: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = type == declared_types[node.output]
                            && (type == SYNAPTIK_METAL_TYPE_INT32
                                    || type == SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_ALL:
                case SYNAPTIK_METAL_CUSTOM_ANY:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_BOOL)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SUB:
                case SYNAPTIK_METAL_MPSGRAPH_MUL:
                case SYNAPTIK_METAL_MPSGRAPH_DIV:
                case SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW:
                case SYNAPTIK_METAL_CUSTOM_TENSOR_MIN:
                case SYNAPTIK_METAL_CUSTOM_TENSOR_MAX:
                case SYNAPTIK_METAL_MPSGRAPH_MSE:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.second_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                case SYNAPTIK_METAL_CUSTOM_CONV2D:
                case SYNAPTIK_METAL_CUSTOM_CONV3D: {
                    uint8_t output_type = declared_types[node.output];
                    BOOL any_float32 = NO;
                    BOOL inputs_valid = YES;
                    for (uint32_t input = 0U; input < node.input_count; input++) {
                        uint8_t type = declared_types[node.inputs[input]];
                        any_float32 |= type == SYNAPTIK_METAL_TYPE_FLOAT32;
                        inputs_valid &= type == SYNAPTIK_METAL_TYPE_FLOAT32
                                || type == SYNAPTIK_METAL_TYPE_BFLOAT16;
                    }
                    type_valid = inputs_valid && any_float32
                            && output_type == SYNAPTIK_METAL_TYPE_FLOAT32;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_MAX_POOL2D:
                case SYNAPTIK_METAL_CUSTOM_MAX_POOL3D: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = type == declared_types[node.output]
                            && (type == SYNAPTIK_METAL_TYPE_FLOAT64
                                    || type == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || type == SYNAPTIK_METAL_TYPE_BFLOAT16);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D:
                case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D:
                    type_valid = declared_types[node.first_input]
                                    == SYNAPTIK_METAL_TYPE_FLOAT32
                            && declared_types[node.output]
                                    == SYNAPTIK_METAL_TYPE_FLOAT32;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_MATMUL: {
                    uint8_t left = declared_types[node.first_input];
                    uint8_t right = declared_types[node.second_input];
                    uint8_t output = declared_types[node.output];
                    BOOL integral =
                            (left == SYNAPTIK_METAL_TYPE_INT32
                                    || left == SYNAPTIK_METAL_TYPE_INT64)
                            && (right == SYNAPTIK_METAL_TYPE_INT32
                                    || right == SYNAPTIK_METAL_TYPE_INT64);
                    uint8_t promoted = left == SYNAPTIK_METAL_TYPE_INT64
                                    || right == SYNAPTIK_METAL_TYPE_INT64
                            ? SYNAPTIK_METAL_TYPE_INT64 : SYNAPTIK_METAL_TYPE_INT32;
                    type_valid = integral
                            ? output == promoted
                            : output == SYNAPTIK_METAL_TYPE_FLOAT32
                                    && ((left == SYNAPTIK_METAL_TYPE_FLOAT32
                                                    && right
                                                            == SYNAPTIK_METAL_TYPE_FLOAT32)
                                            || (left == SYNAPTIK_METAL_TYPE_BFLOAT16
                                                    && right
                                                            == SYNAPTIK_METAL_TYPE_FLOAT32)
                                            || (left == SYNAPTIK_METAL_TYPE_FLOAT32
                                                    && right
                                                            == SYNAPTIK_METAL_TYPE_BFLOAT16));
                    break;
                }
                case SYNAPTIK_METAL_BOOL_IS_FINITE:
                case SYNAPTIK_METAL_BOOL_IS_NAN:
                case SYNAPTIK_METAL_BOOL_IS_INF:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                case SYNAPTIK_METAL_BOOL_NOT:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_BOOL)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                case SYNAPTIK_METAL_BOOL_AND:
                case SYNAPTIK_METAL_BOOL_OR:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_BOOL)
                            && require_value_type(
                                    value_types, node.second_input,
                                    SYNAPTIK_METAL_TYPE_BOOL)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                case SYNAPTIK_METAL_BOOL_WHERE:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_BOOL)
                            && require_value_type(
                                    value_types, node.second_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.auxiliary,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                case SYNAPTIK_METAL_CUSTOM_GT:
                case SYNAPTIK_METAL_CUSTOM_GE:
                case SYNAPTIK_METAL_CUSTOM_LT:
                case SYNAPTIK_METAL_CUSTOM_LE:
                case SYNAPTIK_METAL_CUSTOM_EQ:
                case SYNAPTIK_METAL_CUSTOM_NE:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.second_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_GATHER:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.second_input,
                                    SYNAPTIK_METAL_TYPE_INT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.second_input,
                                    SYNAPTIK_METAL_TYPE_INT32)
                            && require_value_type(
                                    value_types, node.auxiliary,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ONE_HOT:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_INT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                case SYNAPTIK_METAL_CUSTOM_CAST: {
                    uint8_t source = declared_types[node.first_input];
                    uint8_t target = declared_types[node.output];
                    BOOL approved = source == target
                            || source == SYNAPTIK_METAL_TYPE_BOOL
                            || target == SYNAPTIK_METAL_TYPE_BOOL
                            || ((source == SYNAPTIK_METAL_TYPE_INT32
                                            && target == SYNAPTIK_METAL_TYPE_INT64)
                                    || (source == SYNAPTIK_METAL_TYPE_INT64
                                            && target == SYNAPTIK_METAL_TYPE_INT32))
                            || (source == SYNAPTIK_METAL_TYPE_BFLOAT16
                                    && target == SYNAPTIK_METAL_TYPE_FLOAT32);
                    type_valid = approved
                            && node.attribute_values[0] == target;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS:
                case SYNAPTIK_METAL_CUSTOM_GATHER_ND: {
                    uint8_t index_type = declared_types[node.second_input];
                    type_valid = declared_types[node.first_input]
                                    == declared_types[node.output]
                            && (index_type == SYNAPTIK_METAL_TYPE_INT32
                                    || index_type == SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SELECT:
                case SYNAPTIK_METAL_CUSTOM_SLICE:
                case SYNAPTIK_METAL_CUSTOM_TILE:
                    type_valid = declared_types[node.first_input]
                            == declared_types[node.output];
                    break;
                case SYNAPTIK_METAL_CUSTOM_PAD:
                    type_valid = declared_types[node.first_input]
                                    == declared_types[node.output]
                            && node.attribute_values[
                                    1U + shapes[node.first_input].count * 2U]
                                    == declared_types[node.first_input];
                    break;
                case SYNAPTIK_METAL_CUSTOM_CONCAT:
                case SYNAPTIK_METAL_CUSTOM_STACK:
                    type_valid = YES;
                    for (uint32_t input = 0U; input < node.input_count; input++) {
                        if (declared_types[node.inputs[input]]
                                != declared_types[node.output]) type_valid = NO;
                    }
                    break;
                case SYNAPTIK_METAL_CUSTOM_UNFOLD2D:
                case SYNAPTIK_METAL_CUSTOM_UNFOLD3D: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = type == declared_types[node.output]
                            && (type == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || type == SYNAPTIK_METAL_TYPE_FLOAT64
                                    || type == SYNAPTIK_METAL_TYPE_BFLOAT16);
                    NSUInteger dimensions = node.operation
                                    == SYNAPTIK_METAL_CUSTOM_UNFOLD2D
                            ? 2U : 3U;
                    uint32_t padded_kind = dimensions == 2U ? 20U : 23U;
                    if (node.attribute_kind == padded_kind
                            && node.attribute_values[dimensions * 4U + 1U] != type)
                        type_valid = NO;
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND: {
                    uint8_t data_type = declared_types[node.first_input];
                    uint8_t index_type = declared_types[node.second_input];
                    BOOL reduction_type_valid =
                            data_type != SYNAPTIK_METAL_TYPE_BOOL
                            || (node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
                                    && node.attribute_values[1] == 1U);
                    type_valid = reduction_type_valid
                            && data_type == declared_types[node.auxiliary]
                            && data_type == declared_types[node.output]
                            && (index_type == SYNAPTIK_METAL_TYPE_INT32
                                    || index_type == SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE:
                    type_valid = declared_types[node.first_input]
                                    == declared_types[node.second_input]
                            && declared_types[node.first_input]
                                    == declared_types[node.output];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: {
                    uint8_t type = declared_types[node.first_input];
                    BOOL mps_type = type == SYNAPTIK_METAL_TYPE_FLOAT32
                            || type == SYNAPTIK_METAL_TYPE_FLOAT64
                            || type == SYNAPTIK_METAL_TYPE_BFLOAT16;
                    type_valid = type == declared_types[node.output] && mps_type;
                    break;
                }
                default:
                    break;
            }
            if (!type_valid) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            uint8_t output_state =
                    node.operation == SYNAPTIK_METAL_CUSTOM_SELECT
                                    || node.operation == SYNAPTIK_METAL_CUSTOM_SLICE
                            ? SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT
                            : (affine_view
                                    ? SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                                    : SYNAPTIK_METAL_VALUE_CANONICAL);
            for (uint32_t output = 0U; output < node.output_count; output++) {
                uint32_t value = node.outputs[output];
                if (has_layouts && declared_states[value] != output_state)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                states[value] = output_state;
                used[value] = 1U;
                produced[value] = 1U;
            }
            used[node.first_input] = 1U;
        }
        if (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM && !contains_custom)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        for (uint32_t target = 0; target < target_count; target++) {
            uint32_t value = target_indices[target];
            if (value >= value_count || produced[value] == 0U
                    || states[value] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                    || targeted[value] != 0U)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            if (local_transpose[value] != 0U) local_transpose[value] = 3U;
            if (local_singleton_height[value] != 0U)
                local_singleton_height[value] = 3U;
            targeted[value] = 1U;
        }
        NSMutableArray<NSNumber *> *bytes = [NSMutableArray arrayWithCapacity:value_count];
        for (uint32_t value = 0; value < value_count; value++) {
            if (used[value] == 0U || value_types[value] == SYNAPTIK_METAL_TYPE_UNAVAILABLE)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            uint64_t width = value_type_width(value_types[value]);
            uint64_t elements = has_layouts
                            && states[value] == SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT
                    ? layout_spans[value]
                    : element_counts[value].unsignedLongLongValue;
            if (width == 0U || elements > UINT64_MAX / width
                    || elements * width > (uint64_t)NSUIntegerMax)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            [bytes addObject:@((NSUInteger)(elements * width))];
        }
        NSMutableArray<NSArray<NSNumber *> *> *physical_strides =
                [NSMutableArray arrayWithCapacity:value_count];
        NSMutableArray<NSNumber *> *physical_offsets =
                [NSMutableArray arrayWithCapacity:value_count];
        if (physical_strides == nil || physical_offsets == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        for (uint32_t value = 0U; value < value_count; value++) {
            uint64_t stride_cells[SYNAPTIK_MAX_RANK] = {0U};
            uint64_t offset = 0U;
            if (!custom_physical_layout(
                        shapes[value], value, value_strides, layout_offsets,
                        states, stride_cells, &offset))
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            NSMutableArray<NSNumber *> *stride_values =
                    [NSMutableArray arrayWithCapacity:value_ranks[value]];
            if (stride_values == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            for (uint32_t axis = 0U; axis < value_ranks[value]; axis++)
                [stride_values addObject:@(stride_cells[axis])];
            [physical_strides addObject:[stride_values copy]];
            [physical_offsets addObject:@(offset)];
        }
        if (contains_custom) {
            SynaptikMetalContextBox *ctx = (__bridge SynaptikMetalContextBox *)context;
            MTLCompileOptions *options = [MTLCompileOptions new];
            if (options == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            options.mathMode = MTLMathModeSafe;
            NSError *library_error = nil;
            BOOL task0053_domain_approved = NO;
            NSString *kernel_source = [[[[[SynaptikExactKernelSource
                    stringByAppendingString:SynaptikTask0059DataKernelSource]
                    stringByAppendingString:SynaptikTask0060ReductionKernelSource]
                    stringByAppendingString:SynaptikTask0061MatmulKernelSource]
                    stringByAppendingString:SynaptikTask0063OrderingKernelSource]
                    stringByAppendingString:SynaptikTask0064ConvolutionPoolingKernelSource];
            if (task0053_domain_approved) {
                kernel_source = [kernel_source
                        stringByAppendingString:SynaptikTask0053CandidateKernelSource];
            }
            id<MTLLibrary> library = [ctx.device
                    newLibraryWithSource:kernel_source options:options error:&library_error];
            if (library == nil || library_error != nil)
                return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            NSMutableArray<SynaptikMetalProgramStep *> *steps =
                    [NSMutableArray arrayWithCapacity:node_count];
            if (steps == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            for (uint32_t node_index = 0U; node_index < node_count; node_index++) {
                SynaptikMetalDecodedNode node = nodes[node_index];
                if (local_transpose[node.output] == 2U
                        || local_singleton_height[node.output] == 2U) continue;
                BOOL custom_step = node_uses_custom_kernel(node, shapes, value_types);
                if (custom_step) {
                    uint32_t final_stage =
                            node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND ? 1U : 0U;
                    for (uint32_t stage = 0U; stage <= final_stage; stage++) {
                        SynaptikMetalProgramStep *step = make_custom_step(
                                node, shapes, value_types, value_strides,
                                layout_offsets, states, local_transpose_sources,
                                local_singleton_height_sources,
                                ctx.device, library, stage);
                        if (step == nil)
                            return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
                        if (stage == 0U) {
                            for (SynaptikMetalIndexValidation *validation
                                    in index_validations) {
                                if (validation.nodeIndex == node_index) {
                                    step.indexValidation = validation;
                                    break;
                                }
                            }
                        }
                        [steps addObject:step];
                    }
                    continue;
                }

                uint32_t input_count = 1U;
                if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS)
                    input_count = 3U;
                else if (node.operation == SYNAPTIK_METAL_MPSGRAPH_ADD
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_SUB
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_MUL
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_DIV
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_GATHER
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_MSE)
                    input_count = 2U;
                BOOL nested_transposed_matmul =
                        node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
                        && (local_transpose_sources[node.first_input] != UINT32_MAX
                                || local_transpose_sources[node.second_input] != UINT32_MAX);
                uint32_t source_values[3] = {
                    node.first_input, node.second_input, node.auxiliary
                };
                uint32_t compact_ranks[6] = {0U, 0U, 0U, 0U, 0U, 0U};
                uint64_t compact_dimensions[6U * SYNAPTIK_MAX_RANK] = {0U};
                uint64_t compact_strides[6U * SYNAPTIK_MAX_RANK] = {0U};
                uint64_t compact_offsets[6] = {0U, 0U, 0U, 0U, 0U, 0U};
                uint64_t compact_spans[6] = {0U, 0U, 0U, 0U, 0U, 0U};
                uint8_t compact_states[6] = {0U, 0U, 0U, 0U, 0U, 0U};
                uint8_t compact_types[6] = {0U, 0U, 0U, 0U, 0U, 0U};
                uint32_t compact_feeds[3] = {0U, 1U, 2U};
                NSMutableArray<NSNumber *> *step_feeds =
                        [NSMutableArray arrayWithCapacity:input_count];
                if (step_feeds == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                for (uint32_t input_index = 0U;
                        input_index < input_count;
                        input_index++) {
                    uint32_t logical_source = source_values[input_index];
                    uint32_t feed_source = nested_transposed_matmul
                                    && local_transpose_sources[logical_source] != UINT32_MAX
                            ? local_transpose_sources[logical_source]
                            : logical_source;
                    compact_ranks[input_index] = value_ranks[feed_source];
                    compact_types[input_index] = declared_types[feed_source];
                    memcpy(
                            &compact_dimensions[
                                    (size_t)input_index * SYNAPTIK_MAX_RANK],
                            &value_dimensions[
                                    (size_t)feed_source * SYNAPTIK_MAX_RANK],
                            SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                    if (nested_transposed_matmul) {
                        memcpy(
                                &compact_strides[
                                        (size_t)input_index * SYNAPTIK_MAX_RANK],
                                &value_strides[
                                        (size_t)feed_source * SYNAPTIK_MAX_RANK],
                                SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                        compact_offsets[input_index] = layout_offsets[feed_source];
                        compact_spans[input_index] = layout_spans[feed_source];
                        compact_states[input_index] = declared_states[feed_source];
                    }
                    [step_feeds addObject:@(feed_source)];
                }
                SynaptikMetalDecodedNode compact_nodes[3] = {0};
                uint32_t compact_node_count = 0U;
                uint32_t compact_value_count = 0U;
                uint32_t compact_target = 0U;
                if (nested_transposed_matmul) {
                    uint32_t matmul_inputs[2] = {0U, 1U};
                    uint32_t next_value = input_count;
                    for (uint32_t operand = 0U; operand < 2U; operand++) {
                        uint32_t logical_source = source_values[operand];
                        if (local_transpose_sources[logical_source] == UINT32_MAX) continue;
                        uint32_t producer_index = UINT32_MAX;
                        for (uint32_t prior = 0U; prior < node_index; prior++) {
                            if (nodes[prior].output == logical_source) {
                                producer_index = prior;
                                break;
                            }
                        }
                        if (producer_index == UINT32_MAX
                                || nodes[producer_index].operation
                                        != SYNAPTIK_METAL_MPSGRAPH_PERMUTE)
                            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                        SynaptikMetalDecodedNode transpose = nodes[producer_index];
                        transpose.first_input = operand;
                        transpose.output = next_value;
                        transpose.outputs[0] = next_value;
                        transpose.inputs[0] = operand;
                        compact_nodes[compact_node_count++] = transpose;
                        compact_ranks[next_value] = value_ranks[logical_source];
                        compact_types[next_value] = declared_types[logical_source];
                        memcpy(
                                &compact_dimensions[
                                        (size_t)next_value * SYNAPTIK_MAX_RANK],
                                &value_dimensions[
                                        (size_t)logical_source * SYNAPTIK_MAX_RANK],
                                SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                        memcpy(
                                &compact_strides[
                                        (size_t)next_value * SYNAPTIK_MAX_RANK],
                                &value_strides[
                                        (size_t)logical_source * SYNAPTIK_MAX_RANK],
                                SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                        compact_offsets[next_value] = layout_offsets[logical_source];
                        compact_spans[next_value] = layout_spans[logical_source];
                        compact_states[next_value] = declared_states[logical_source];
                        matmul_inputs[operand] = next_value++;
                    }
                    SynaptikMetalDecodedNode compact = node;
                    compact.first_input = matmul_inputs[0];
                    compact.second_input = matmul_inputs[1];
                    compact.output = next_value;
                    compact.outputs[0] = next_value;
                    compact.inputs[0] = matmul_inputs[0];
                    compact.inputs[1] = matmul_inputs[1];
                    compact_nodes[compact_node_count++] = compact;
                    compact_ranks[next_value] = value_ranks[node.output];
                    compact_types[next_value] = declared_types[node.output];
                    memcpy(
                            &compact_dimensions[
                                    (size_t)next_value * SYNAPTIK_MAX_RANK],
                            &value_dimensions[
                                    (size_t)node.output * SYNAPTIK_MAX_RANK],
                            SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                    memcpy(
                            &compact_strides[
                                    (size_t)next_value * SYNAPTIK_MAX_RANK],
                            &value_strides[
                                    (size_t)node.output * SYNAPTIK_MAX_RANK],
                            SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                    compact_offsets[next_value] = layout_offsets[node.output];
                    compact_spans[next_value] = layout_spans[node.output];
                    compact_states[next_value] = declared_states[node.output];
                    compact_target = next_value;
                    compact_value_count = next_value + 1U;
                } else {
                    compact_ranks[input_count] = value_ranks[node.output];
                    compact_types[input_count] = declared_types[node.output];
                    memcpy(
                            &compact_dimensions[
                                    (size_t)input_count * SYNAPTIK_MAX_RANK],
                            &value_dimensions[
                                    (size_t)node.output * SYNAPTIK_MAX_RANK],
                            SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                    SynaptikMetalDecodedNode compact = node;
                    compact.first_input = 0U;
                    compact.second_input = input_count >= 2U ? 1U : UINT32_MAX;
                    compact.output = input_count;
                    compact.outputs[0] = input_count;
                    compact.input_count = input_count;
                    for (uint32_t input_index = 0U;
                            input_index < input_count;
                            input_index++)
                        compact.inputs[input_index] = input_index;
                    if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS)
                        compact.auxiliary = 2U;
                    compact_nodes[0] = compact;
                    compact_node_count = 1U;
                    compact_target = input_count;
                    compact_value_count = input_count + 1U;
                }
                void *nested_handle = NULL;
                int32_t nested_status = synaptik_metal_create_decoded(
                        context,
                        SYNAPTIK_METAL_ROUTE_MPSGRAPH,
                        compact_value_count,
                        compact_ranks,
                        compact_dimensions,
                        nested_transposed_matmul ? compact_strides : NULL,
                        nested_transposed_matmul ? compact_offsets : NULL,
                        nested_transposed_matmul ? compact_spans : NULL,
                        nested_transposed_matmul ? compact_states : NULL,
                        compact_types,
                        compact_node_count,
                        compact_nodes,
                        input_count,
                        compact_feeds,
                        1U,
                        &compact_target,
                        &nested_handle);
                if (nested_status != SYNAPTIK_METAL_STATUS_OK || nested_handle == NULL) {
                    if (nested_handle != NULL) {
                        __unused id consumed = (__bridge_transfer id)nested_handle;
                    }
                    return nested_status == SYNAPTIK_METAL_STATUS_OK
                            ? SYNAPTIK_METAL_STATUS_INTERNAL_ERROR : nested_status;
                }
                SynaptikMetalProgramStep *step = [SynaptikMetalProgramStep new];
                if (step == nil) {
                    __unused id consumed = (__bridge_transfer id)nested_handle;
                    return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                }
                step.custom = NO;
                step.operation = node.operation;
                step.output = node.output;
                for (SynaptikMetalIndexValidation *validation in index_validations) {
                    if (validation.nodeIndex == node_index) {
                        step.indexValidation = validation;
                        break;
                    }
                }
                step.feedValues = [step_feeds copy];
                step.nestedExecutable = (__bridge_transfer id)nested_handle;
                [steps addObject:step];
            }
            NSMutableArray<NSNumber *> *target_bytes =
                    [NSMutableArray arrayWithCapacity:target_count];
            NSMutableArray<NSNumber *> *target_values =
                    [NSMutableArray arrayWithCapacity:target_count];
            for (uint32_t target = 0U; target < target_count; target++) {
                [target_bytes addObject:bytes[target_indices[target]]];
                [target_values addObject:@(target_indices[target])];
            }
            SynaptikMetalExecutableBox *box = [SynaptikMetalExecutableBox new];
            if (box == nil || target_bytes == nil || target_values == nil)
                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            box.customProgram = YES;
            box.valueCount = value_count;
            box.valueBytes = [bytes copy];
            box.targetBytes = [target_bytes copy];
            box.targetValueIndices = [target_values copy];
            box.programSteps = [steps copy];
            box.indexValidations = [index_validations copy];
            NSMutableArray<NSNumber *> *canonical_bool_inputs =
                    [NSMutableArray array];
            if (canonical_bool_inputs == nil)
                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            for (uint32_t feed = 0U; feed < feed_count; feed++) {
                uint32_t value = feed_indices[feed];
                if (declared_types[value] == SYNAPTIK_METAL_TYPE_BOOL)
                    [canonical_bool_inputs addObject:@(value)];
            }
            box.canonicalBoolInputIndices = [canonical_bool_inputs copy];
            box.inputShapes = [shapes copy];
            box.inputStrides = [physical_strides copy];
            box.inputOffsets = [physical_offsets copy];
            box.context = ctx;
            *out_executable = (__bridge_retained void *)box;
            return SYNAPTIK_METAL_STATUS_OK;
        }
        for (uint32_t value = 0U; value < value_count; value++) {
            if (used[value] != 0U
                    && value_mps_data_type(value_types[value]) == MPSDataTypeInvalid)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }


        MPSGraph *graph = [MPSGraph new];
        if (graph == nil) return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
        graph.options = MPSGraphOptionsNone;
        NSMutableArray *table = [NSMutableArray arrayWithCapacity:value_count];
        for (uint32_t value = 0; value < value_count; value++) [table addObject:NSNull.null];
        NSMutableArray<MPSGraphTensor *> *feeds = [NSMutableArray arrayWithCapacity:feed_count];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *types =
                [NSMutableDictionary dictionaryWithCapacity:feed_count];
        for (uint32_t feed = 0; feed < feed_count; feed++) {
            uint32_t value = feed_indices[feed];
            MPSDataType data_type = value_mps_data_type(value_types[value]);
            if (data_type == MPSDataTypeInvalid)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            MPSGraphTensor *tensor = [graph placeholderWithShape:shapes[value]
                    dataType:data_type name:nil];
            MPSGraphShapedType *type = [[MPSGraphShapedType alloc]
                    initWithShape:shapes[value] dataType:data_type];
            if (tensor == nil || type == nil)
                return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
            table[value] = tensor;
            [feeds addObject:tensor];
            types[tensor] = type;
        }
        for (uint32_t node_index = 0; node_index < node_count; node_index++) {
            SynaptikMetalDecodedNode node = nodes[node_index];
            MPSGraphTensor *first = (MPSGraphTensor *)table[node.first_input];
            MPSGraphTensor *second = node.second_input == UINT32_MAX
                    ? nil : (MPSGraphTensor *)table[node.second_input];
            MPSGraphTensor *auxiliary = node.input_count >= 3U
                    ? (MPSGraphTensor *)table[node.auxiliary] : nil;
            MPSGraphTensor *output = nil;
            switch ((SynaptikMetalOperation)node.operation) {
                case SYNAPTIK_METAL_MPSGRAPH_NEG:
                    output = [graph negativeWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ABS:
                    output = [graph absoluteWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL: {
                    MPSGraphTensor *one = exact_float32_scalar(
                            graph, UINT32_C(0x3f800000));
                    output = [graph divisionWithPrimaryTensor:one
                            secondaryTensor:first name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_LOG:
                    output = [graph logarithmWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_LOG1P: {
                    MPSGraphTensor *one = exact_float32_scalar(
                            graph, UINT32_C(0x3f800000));
                    MPSGraphTensor *sum = [graph additionWithPrimaryTensor:one
                            secondaryTensor:first name:nil];
                    output = [graph logarithmWithTensor:sum name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_EXPM1: {
                    MPSGraphTensor *one = exact_float32_scalar(
                            graph, UINT32_C(0x3f800000));
                    MPSGraphTensor *exponent = [graph exponentWithTensor:first name:nil];
                    output = [graph subtractionWithPrimaryTensor:exponent
                            secondaryTensor:one name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_ERF:
                    output = [graph erfWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SQRT:
                    output = [graph squareRootWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_RSQRT:
                    output = [graph reciprocalSquareRootWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_FLOOR:
                    output = [graph floorWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_CEIL:
                    output = [graph ceilWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_SIGN:
                    output = [graph signWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_RELU:
                    output = [graph reLUWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_TANH:
                    output = [graph tanhWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_GELU: {
                    MPSGraphTensor *half = exact_float32_scalar(
                            graph, UINT32_C(0x3f000000));
                    MPSGraphTensor *one = exact_float32_scalar(
                            graph, UINT32_C(0x3f800000));
                    MPSGraphTensor *two = exact_float32_scalar(
                            graph, UINT32_C(0x40000000));
                    MPSGraphTensor *rootTwo = [graph squareRootWithTensor:two name:nil];
                    MPSGraphTensor *normalized = [graph divisionWithPrimaryTensor:first
                            secondaryTensor:rootTwo name:nil];
                    MPSGraphTensor *erf = [graph erfWithTensor:normalized name:nil];
                    MPSGraphTensor *shifted = [graph additionWithPrimaryTensor:one
                            secondaryTensor:erf name:nil];
                    MPSGraphTensor *weighted = [graph multiplicationWithPrimaryTensor:first
                            secondaryTensor:shifted name:nil];
                    output = [graph multiplicationWithPrimaryTensor:half
                            secondaryTensor:weighted name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_GELU_TANH: {
                    MPSGraphTensor *half = exact_float32_scalar(
                            graph, UINT32_C(0x3f000000));
                    MPSGraphTensor *one = exact_float32_scalar(
                            graph, UINT32_C(0x3f800000));
                    MPSGraphTensor *two = exact_float32_scalar(
                            graph, UINT32_C(0x40000000));
                    MPSGraphTensor *three = exact_float32_scalar(
                            graph, UINT32_C(0x40400000));
                    MPSGraphTensor *pi = exact_float32_scalar(
                            graph, UINT32_C(0x40490fdb));
                    MPSGraphTensor *coefficient = exact_float32_scalar(
                            graph, UINT32_C(0x3d372713));
                    MPSGraphTensor *ratio = [graph divisionWithPrimaryTensor:two
                            secondaryTensor:pi name:nil];
                    MPSGraphTensor *root = [graph squareRootWithTensor:ratio name:nil];
                    MPSGraphTensor *cube = [graph powerWithPrimaryTensor:first
                            secondaryTensor:three name:nil];
                    MPSGraphTensor *cubic = [graph multiplicationWithPrimaryTensor:coefficient
                            secondaryTensor:cube name:nil];
                    MPSGraphTensor *polynomial = [graph additionWithPrimaryTensor:first
                            secondaryTensor:cubic name:nil];
                    MPSGraphTensor *argument = [graph multiplicationWithPrimaryTensor:root
                            secondaryTensor:polynomial name:nil];
                    MPSGraphTensor *tanh = [graph tanhWithTensor:argument name:nil];
                    MPSGraphTensor *shifted = [graph additionWithPrimaryTensor:one
                            secondaryTensor:tanh name:nil];
                    MPSGraphTensor *weighted = [graph multiplicationWithPrimaryTensor:first
                            secondaryTensor:shifted name:nil];
                    output = [graph multiplicationWithPrimaryTensor:half
                            secondaryTensor:weighted name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_SILU: {
                    MPSGraphTensor *sigmoid = [graph sigmoidWithTensor:first name:nil];
                    output = [graph multiplicationWithPrimaryTensor:first
                            secondaryTensor:sigmoid name:nil];
                    break;
                }
                case SYNAPTIK_METAL_BOOL_IS_FINITE:
                    output = [graph isFiniteWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_BOOL_IS_NAN:
                    output = [graph isNaNWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_BOOL_IS_INF:
                    output = [graph isInfiniteWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_BOOL_NOT:
                    output = [graph notWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ADD:
                    output = [graph additionWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SUB:
                    output = [graph subtractionWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_MUL:
                    output = [graph multiplicationWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_DIV:
                    output = [graph divisionWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW:
                    output = [graph powerWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_MSE: {
                    MPSGraphTensor *difference =
                            [graph subtractionWithPrimaryTensor:first
                                    secondaryTensor:second name:nil];
                    MPSGraphTensor *square =
                            [graph multiplicationWithPrimaryTensor:difference
                                    secondaryTensor:difference name:nil];
                    uint64_t reduction = node.attribute_values[0];
                    if (reduction == 1U) {
                        output = square;
                    } else {
                        NSMutableArray<NSNumber *> *axes =
                                [NSMutableArray arrayWithCapacity:
                                        shapes[node.first_input].count];
                        for (NSUInteger axis = 0U;
                                axis < shapes[node.first_input].count; axis++) {
                            [axes addObject:@(axis)];
                        }
                        output = reduction == 2U
                                ? [graph reductionSumWithTensor:square
                                        axes:axes name:nil]
                                : [graph meanOfTensor:square axes:axes name:nil];
                        if (output != nil
                                && ![output.shape
                                        isEqualToArray:shapes[node.output]]) {
                            output = [graph reshapeTensor:output
                                    withShape:shapes[node.output] name:nil];
                        }
                    }
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_SUB:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV:
                case SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW: {
                    MPSGraphTensor *scalar = exact_float32_scalar(
                            graph, (uint32_t)node.attribute_values[0]);
                    switch ((SynaptikMetalOperation)node.operation) {
                        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD:
                            output = [graph additionWithPrimaryTensor:first
                                    secondaryTensor:scalar name:nil];
                            break;
                        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_SUB:
                            output = [graph subtractionWithPrimaryTensor:first
                                    secondaryTensor:scalar name:nil];
                            break;
                        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL:
                            output = [graph multiplicationWithPrimaryTensor:first
                                    secondaryTensor:scalar name:nil];
                            break;
                        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV:
                            output = [graph divisionWithPrimaryTensor:first
                                    secondaryTensor:scalar name:nil];
                            break;
                        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW:
                            output = [graph powerWithPrimaryTensor:first
                                    secondaryTensor:scalar name:nil];
                            break;
                        default:
                            break;
                    }
                    break;
                }
                case SYNAPTIK_METAL_BOOL_AND:
                    output = [graph logicalANDWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_BOOL_OR:
                    output = [graph logicalORWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_BOOL_WHERE:
                    output = [graph selectWithPredicateTensor:first
                            truePredicateTensor:second
                            falsePredicateTensor:auxiliary
                            name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_MATMUL:
                    output = [graph matrixMultiplicationWithPrimaryTensor:first
                            secondaryTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_CONTIGUOUS:
                    output = [graph reshapeTensor:first
                            withShape:shapes[node.output] name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_RESHAPE:
                    output = [graph reshapeTensor:first
                            withShape:node_attribute_array(node) name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND:
                    output = [graph broadcastTensor:first
                            toShape:node_attribute_array(node) name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_PERMUTE:
                    output = [graph transposeTensor:first
                            permutation:node_attribute_array(node) name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND_DIMS:
                    output = [graph expandDimsOfTensor:first
                            axis:(NSInteger)node.axis name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SQUEEZE:
                    output = [graph squeezeTensor:first
                            axis:(NSInteger)node.axis name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SUM:
                case SYNAPTIK_METAL_MPSGRAPH_MEAN: {
                    BOOL keep_dimensions = NO;
                    NSArray<NSNumber *> *axes = node_reduction_axes(
                            node, shapes[node.first_input], shapes[node.output],
                            &keep_dimensions);
                    if (axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (axes.count == 0U) {
                        output = [graph reshapeTensor:first
                                withShape:shapes[node.output] name:nil];
                    } else if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SUM) {
                        output = [graph reductionSumWithTensor:first axes:axes name:nil];
                    } else {
                        output = [graph meanOfTensor:first axes:axes name:nil];
                    }
                    if (output != nil
                            && ![output.shape isEqualToArray:shapes[node.output]]) {
                        output = [graph reshapeTensor:output
                                withShape:shapes[node.output] name:nil];
                    }
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_PROD:
                case SYNAPTIK_METAL_CUSTOM_ALL:
                case SYNAPTIK_METAL_CUSTOM_ANY:
                case SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP:
                case SYNAPTIK_METAL_MPSGRAPH_L1_NORM:
                case SYNAPTIK_METAL_MPSGRAPH_L2_NORM: {
                    BOOL keep_dimensions = NO;
                    NSArray<NSNumber *> *axes = node_reduction_axes(
                            node, shapes[node.first_input], shapes[node.output],
                            &keep_dimensions);
                    if (axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (axes.count == 0U) {
                        output = [graph reshapeTensor:first
                                withShape:shapes[node.output] name:nil];
                        break;
                    }
                    switch ((SynaptikMetalOperation)node.operation) {
                        case SYNAPTIK_METAL_CUSTOM_PROD:
                            output = [graph reductionProductWithTensor:first
                                    axes:axes name:nil];
                            break;
                        case SYNAPTIK_METAL_CUSTOM_ALL:
                            output = [graph reductionAndWithTensor:first
                                    axes:axes name:nil];
                            break;
                        case SYNAPTIK_METAL_CUSTOM_ANY:
                            output = [graph reductionOrWithTensor:first
                                    axes:axes name:nil];
                            break;
                        case SYNAPTIK_METAL_MPSGRAPH_L1_NORM: {
                            MPSGraphTensor *absolute =
                                    [graph absoluteWithTensor:first name:nil];
                            output = [graph reductionSumWithTensor:absolute
                                    axes:axes name:nil];
                            break;
                        }
                        case SYNAPTIK_METAL_MPSGRAPH_L2_NORM: {
                            MPSGraphTensor *square =
                                    [graph squareWithTensor:first name:nil];
                            MPSGraphTensor *sum = [graph reductionSumWithTensor:square
                                    axes:axes name:nil];
                            output = [graph squareRootWithTensor:sum name:nil];
                            break;
                        }
                        case SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP: {
                            MPSGraphTensor *maximum =
                                    [graph reductionMaximumWithTensor:first
                                            axes:axes name:nil];
                            NSMutableArray<NSNumber *> *broadcast_shape =
                                    [NSMutableArray arrayWithCapacity:
                                            shapes[node.first_input].count];
                            if (broadcast_shape == nil)
                                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                            for (NSUInteger axis = 0U;
                                    axis < shapes[node.first_input].count; axis++) {
                                [broadcast_shape addObject:
                                        [axes containsObject:@(axis)]
                                                ? @1 : shapes[node.first_input][axis]];
                            }
                            MPSGraphTensor *broadcast_maximum =
                                    [graph reshapeTensor:maximum
                                            withShape:broadcast_shape name:nil];
                            MPSGraphTensor *shifted =
                                    [graph subtractionWithPrimaryTensor:first
                                            secondaryTensor:broadcast_maximum name:nil];
                            MPSGraphTensor *exponent =
                                    [graph exponentWithTensor:shifted name:nil];
                            MPSGraphTensor *sum =
                                    [graph reductionSumWithTensor:exponent
                                            axes:axes name:nil];
                            MPSGraphTensor *logarithm =
                                    [graph logarithmWithTensor:sum name:nil];
                            output = [graph additionWithPrimaryTensor:logarithm
                                    secondaryTensor:maximum name:nil];
                            break;
                        }
                        default:
                            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                    }
                    if (output != nil
                            && ![output.shape isEqualToArray:shapes[node.output]]) {
                        output = [graph reshapeTensor:output
                                withShape:shapes[node.output] name:nil];
                    }
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_VARIANCE:
                case SYNAPTIK_METAL_MPSGRAPH_STANDARD_DEVIATION: {
                    BOOL keep_dimensions = NO;
                    uint64_t correction = 0U;
                    NSArray<NSNumber *> *axes = node_statistical_axes(
                            node, shapes[node.first_input], shapes[node.output],
                            &keep_dimensions, &correction);
                    if (axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (axes.count == 0U) {
                        output = [graph reshapeTensor:first
                                withShape:shapes[node.output] name:nil];
                        break;
                    }
                    output = [graph varianceOfTensor:first axes:axes name:nil];
                    if (correction != 0U) {
                        uint64_t selected = 1U;
                        for (NSNumber *axis in axes)
                            selected *= shapes[node.first_input][axis.unsignedIntegerValue]
                                    .unsignedLongLongValue;
                        float factor = (float)((double)selected
                                / (double)(selected - correction));
                        MPSGraphTensor *scale =
                                [graph constantWithScalar:(double)factor
                                        shape:@[] dataType:MPSDataTypeFloat32];
                        output = [graph multiplicationWithPrimaryTensor:output
                                secondaryTensor:scale name:nil];
                    }
                    if (node.operation
                            == SYNAPTIK_METAL_MPSGRAPH_STANDARD_DEVIATION)
                        output = [graph squareRootWithTensor:output name:nil];
                    if (output != nil
                            && ![output.shape isEqualToArray:shapes[node.output]]) {
                        output = [graph reshapeTensor:output
                                withShape:shapes[node.output] name:nil];
                    }
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_CAST:
                    output = [graph castTensor:first
                            toType:value_mps_data_type(value_types[node.output])
                            name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS:
                    output = [graph gatherAlongAxis:(NSInteger)node.axis
                            withUpdatesTensor:first indicesTensor:second name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ADD:
                    output = [graph scatterAlongAxis:(NSInteger)node.axis
                            withDataTensor:first
                            updatesTensor:auxiliary
                            indicesTensor:second
                            mode:MPSGraphScatterModeAdd
                            name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_GATHER_ND:
                    output = [graph gatherNDWithUpdatesTensor:first
                            indicesTensor:second
                            batchDimensions:(NSUInteger)node.attribute_values[0]
                            name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND: {
                    uint64_t reduction = node.attribute_values[1];
                    MPSGraphScatterMode mode = reduction == 1U
                            ? MPSGraphScatterModeSet
                            : reduction == 2U ? MPSGraphScatterModeAdd
                            : reduction == 3U ? MPSGraphScatterModeMul
                            : reduction == 4U ? MPSGraphScatterModeMax
                            : MPSGraphScatterModeMin;
                    output = [graph scatterNDWithDataTensor:first
                            updatesTensor:auxiliary
                            indicesTensor:second
                            batchDimensions:(NSUInteger)node.attribute_values[0]
                            mode:mode
                            name:nil];
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SELECT: {
                    NSUInteger axis = (NSUInteger)node.attribute_values[0];
                    MPSGraphTensor *slice = [graph sliceTensor:first
                            dimension:axis
                            start:(NSInteger)node.attribute_values[1]
                            length:1U
                            name:nil];
                    output = [graph squeezeTensor:slice axis:(NSInteger)axis name:nil];
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_PAD: {
                    NSUInteger rank = (NSUInteger)node.attribute_values[0];
                    NSMutableArray<NSNumber *> *left =
                            [NSMutableArray arrayWithCapacity:rank];
                    NSMutableArray<NSNumber *> *right =
                            [NSMutableArray arrayWithCapacity:rank];
                    if (left == nil || right == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    for (NSUInteger axis = 0U; axis < rank; axis++) {
                        [left addObject:@((NSUInteger)node.attribute_values[1U + axis])];
                        [right addObject:@((NSUInteger)
                                node.attribute_values[1U + rank + axis])];
                    }
                    output = [graph padTensor:first
                            withPaddingMode:MPSGraphPaddingModeConstant
                            leftPadding:left
                            rightPadding:right
                            constantValue:task0059_scalar_double(
                                    node, 1U + (uint32_t)rank * 2U)
                            name:nil];
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SLICE:
                case SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE: {
                    NSUInteger rank = shapes[node.first_input].count;
                    NSMutableArray<NSNumber *> *starts =
                            [NSMutableArray arrayWithCapacity:rank];
                    NSMutableArray<NSNumber *> *ends =
                            [NSMutableArray arrayWithCapacity:rank];
                    NSMutableArray<NSNumber *> *strides =
                            [NSMutableArray arrayWithCapacity:rank];
                    if (starts == nil || ends == nil || strides == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    for (NSUInteger axis = 0U; axis < rank; axis++) {
                        [starts addObject:@0];
                        [ends addObject:@((NSInteger)
                                shapes[node.first_input][axis].unsignedLongLongValue)];
                        [strides addObject:@1];
                    }
                    if (node.attribute_kind == 15U) {
                        NSUInteger target_rank =
                                (NSUInteger)node.attribute_values[0];
                        NSUInteger prefix_offset = 1U + target_rank;
                        for (NSUInteger axis = 0U; axis < rank; axis++) {
                            uint64_t start =
                                    node.attribute_values[prefix_offset + 1U + axis];
                            uint64_t length = node.attribute_values[1U + axis];
                            starts[axis] = @(start);
                            ends[axis] = @(start + length);
                        }
                    } else {
                        NSUInteger count = (NSUInteger)node.attribute_values[0];
                        for (NSUInteger item = 0U; item < count; item++) {
                            NSUInteger axis = (NSUInteger)
                                    node.attribute_values[1U + count * 2U + item];
                            int64_t start =
                                    (int64_t)node.attribute_values[1U + item];
                            int64_t length =
                                    (int64_t)node.attribute_values[1U + count + item];
                            int64_t stride =
                                    (int64_t)node.attribute_values[1U + count * 3U + item];
                            int64_t last =
                                    start + (length - 1) * stride;
                            starts[axis] = @(start);
                            ends[axis] = @(last + (stride > 0 ? 1 : -1));
                            strides[axis] = @(stride);
                        }
                    }
                    output = node.operation == SYNAPTIK_METAL_CUSTOM_SLICE
                            ? [graph sliceTensor:first
                                    starts:starts ends:ends strides:strides name:nil]
                            : [graph sliceUpdateDataTensor:first
                                    updateTensor:second
                                    starts:starts ends:ends strides:strides name:nil];
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_CONCAT:
                case SYNAPTIK_METAL_CUSTOM_STACK: {
                    NSMutableArray<MPSGraphTensor *> *parts =
                            [NSMutableArray arrayWithCapacity:node.input_count];
                    if (parts == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    for (uint32_t item = 0U; item < node.input_count; item++)
                        [parts addObject:(MPSGraphTensor *)table[node.inputs[item]]];
                    output = node.operation == SYNAPTIK_METAL_CUSTOM_CONCAT
                            ? [graph concatTensors:parts
                                    dimension:(NSInteger)node.axis name:nil]
                            : [graph stackTensors:parts
                                    axis:(NSInteger)node.axis name:nil];
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_TILE: {
                    NSUInteger rank = (NSUInteger)node.attribute_values[0];
                    NSMutableArray<NSNumber *> *multipliers =
                            [NSMutableArray arrayWithCapacity:rank];
                    if (multipliers == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    for (NSUInteger axis = 0U; axis < rank; axis++)
                        [multipliers addObject:@((NSUInteger)
                                node.attribute_values[1U + axis])];
                    output = [graph tileTensor:first
                            withMultiplier:multipliers name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS: {
                    NSUInteger input_rank = shapes[node.first_input].count;
                    NSUInteger target_rank = shapes[node.output].count;
                    NSUInteger size = (NSUInteger)node.attribute_values[0];
                    NSUInteger step = (NSUInteger)node.attribute_values[1];
                    MPSGraphTensor *accumulator = nil;
                    for (NSUInteger offset = 0U; offset < size; offset++) {
                        MPSGraphTensor *slice = [graph sliceTensor:first
                                dimension:input_rank - 1U
                                start:(NSInteger)offset length:1U name:nil];
                        MPSGraphTensor *updates = [graph squeezeTensor:slice
                                axis:(NSInteger)(input_rank - 1U) name:nil];
                        MPSGraphTensor *coordinate = [graph coordinateAlongAxis:
                                (NSInteger)node.axis
                                withShape:shapes[node.first_input] name:nil];
                        coordinate = [graph sliceTensor:coordinate
                                dimension:input_rank - 1U
                                start:(NSInteger)offset length:1U name:nil];
                        coordinate = [graph squeezeTensor:coordinate
                                axis:(NSInteger)(input_rank - 1U) name:nil];
                        MPSGraphTensor *stride_scalar =
                                exact_int32_scalar(graph, (int32_t)step);
                        MPSGraphTensor *offset_scalar =
                                exact_int32_scalar(graph, (int32_t)offset);
                        MPSGraphTensor *indices = [graph multiplicationWithPrimaryTensor:coordinate
                                secondaryTensor:stride_scalar name:nil];
                        indices = [graph additionWithPrimaryTensor:indices
                                secondaryTensor:offset_scalar name:nil];
                        accumulator = accumulator == nil
                                ? [graph scatterAlongAxis:(NSInteger)node.axis
                                        withUpdatesTensor:updates
                                        indicesTensor:indices
                                        shape:shapes[node.output]
                                        mode:MPSGraphScatterModeAdd
                                        name:nil]
                                : [graph scatterAlongAxis:(NSInteger)node.axis
                                        withDataTensor:accumulator
                                        updatesTensor:updates
                                        indicesTensor:indices
                                        mode:MPSGraphScatterModeAdd
                                        name:nil];
                    }
                    output = target_rank == 0U ? nil : accumulator;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_UNFOLD2D: {
                    MPSGraphImToColOpDescriptor *descriptor =
                            task0059_im2col_descriptor(
                                    node, 0U, YES, shapes[node.first_input]);
                    if (descriptor == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    MPSGraphTensor *columns = [graph imToColWithSourceTensor:first
                            descriptor:descriptor name:nil];
                    output = [graph reshapeTensor:columns
                            withShape:shapes[node.output] name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_FOLD2D: {
                    uint32_t offset = 1U + (uint32_t)node.attribute_values[0];
                    MPSShape *target = shapes[node.output];
                    MPSGraphTensor *columns = first;
                    MPSGraphImToColOpDescriptor *descriptor =
                            task0059_im2col_descriptor(node, offset, YES, target);
                    if (descriptor == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    output = [graph colToImWithSourceTensor:columns
                            outputShape:target descriptor:descriptor name:nil];
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_UNFOLD3D: {
                    uint64_t kD = node.attribute_values[0];
                    uint64_t kH = node.attribute_values[1];
                    uint64_t kW = node.attribute_values[2];
                    uint64_t sD = node.attribute_values[3];
                    uint64_t sH = node.attribute_values[4];
                    uint64_t sW = node.attribute_values[5];
                    uint64_t pD = node.attribute_values[6];
                    uint64_t pH = node.attribute_values[7];
                    uint64_t pW = node.attribute_values[8];
                    uint64_t dD = node.attribute_values[9];
                    uint64_t dH = node.attribute_values[10];
                    uint64_t dW = node.attribute_values[11];
                    BOOL ceil_mode = node.attribute_values[12] != 0U;
                    uint64_t dOut = 0U, hOut = 0U, wOut = 0U;
                    if (!task0059_window_extent(
                                shapes[node.first_input][2].unsignedLongLongValue,
                                kD, pD, sD, dD, ceil_mode, &dOut)
                            || !task0059_window_extent(
                                shapes[node.first_input][3].unsignedLongLongValue,
                                kH, pH, sH, dH, ceil_mode, &hOut)
                            || !task0059_window_extent(
                                shapes[node.first_input][4].unsignedLongLongValue,
                                kW, pW, sW, dW, ceil_mode, &wOut))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    uint64_t effectiveD = dD * (kD - 1U) + 1U;
                    uint64_t effectiveH = dH * (kH - 1U) + 1U;
                    uint64_t effectiveW = dW * (kW - 1U) + 1U;
                    uint64_t paddedD =
                            shapes[node.first_input][2].unsignedLongLongValue + pD * 2U;
                    uint64_t paddedH =
                            shapes[node.first_input][3].unsignedLongLongValue + pH * 2U;
                    uint64_t paddedW =
                            shapes[node.first_input][4].unsignedLongLongValue + pW * 2U;
                    uint64_t requiredD = (dOut - 1U) * sD + effectiveD;
                    uint64_t requiredH = (hOut - 1U) * sH + effectiveH;
                    uint64_t requiredW = (wOut - 1U) * sW + effectiveW;
                    uint64_t extraD = requiredD > paddedD ? requiredD - paddedD : 0U;
                    uint64_t extraH = requiredH > paddedH ? requiredH - paddedH : 0U;
                    uint64_t extraW = requiredW > paddedW ? requiredW - paddedW : 0U;
                    MPSGraphTensor *padded = first;
                    if (pD != 0U || pH != 0U || pW != 0U
                            || extraD != 0U || extraH != 0U || extraW != 0U) {
                        double constant = node.attribute_kind == 23U
                                ? task0059_scalar_double(node, 13U) : 0.0;
                        padded = [graph padTensor:first
                                withPaddingMode:MPSGraphPaddingModeConstant
                                leftPadding:@[@0, @0, @(pD), @(pH), @(pW)]
                                rightPadding:@[
                                    @0, @0, @(pD + extraD),
                                    @(pH + extraH), @(pW + extraW)
                                ]
                                constantValue:constant
                                name:nil];
                    }
                    paddedD += extraD;
                    paddedH += extraH;
                    paddedW += extraW;
                    MPSGraphImToColOpDescriptor *descriptor =
                            [MPSGraphImToColOpDescriptor
                                    descriptorWithKernelWidth:(NSUInteger)kW
                                    kernelHeight:(NSUInteger)kH
                                    strideInX:(NSUInteger)sW
                                    strideInY:(NSUInteger)sH
                                    dilationRateInX:(NSUInteger)dW
                                    dilationRateInY:(NSUInteger)dH
                                    paddingLeft:0U paddingRight:0U
                                    paddingTop:0U paddingBottom:0U
                                    dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
                    NSMutableArray<MPSGraphTensor *> *depth_windows =
                            [NSMutableArray arrayWithCapacity:(NSUInteger)kD];
                    if (descriptor == nil || depth_windows == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    uint64_t n = shapes[node.first_input][0].unsignedLongLongValue;
                    uint64_t c = shapes[node.first_input][1].unsignedLongLongValue;
                    uint64_t plane_positions = hOut * wOut;
                    uint64_t positions = dOut * plane_positions;
                    for (uint64_t kd = 0U; kd < kD; kd++) {
                        uint64_t depth_start = kd * dD;
                        MPSGraphTensor *depth_slice = [graph sliceTensor:padded
                                starts:@[@0, @0, @(depth_start), @0, @0]
                                ends:@[
                                    @(n), @(c),
                                    @(depth_start + (dOut - 1U) * sD + 1U),
                                    @(paddedH), @(paddedW)
                                ]
                                strides:@[@1, @1, @(sD), @1, @1]
                                name:nil];
                        MPSGraphTensor *depth_major = [graph transposeTensor:depth_slice
                                permutation:@[@0, @2, @1, @3, @4] name:nil];
                        depth_major = [graph reshapeTensor:depth_major
                                withShape:@[
                                    @(n * dOut), @(c), @(paddedH), @(paddedW)
                                ]
                                name:nil];
                        MPSGraphTensor *columns = [graph imToColWithSourceTensor:depth_major
                                descriptor:descriptor name:nil];
                        columns = [graph reshapeTensor:columns
                                withShape:@[
                                    @(n), @(dOut), @(c * kH * kW),
                                    @(plane_positions)
                                ]
                                name:nil];
                        columns = [graph transposeTensor:columns
                                permutation:@[@0, @2, @1, @3] name:nil];
                        columns = [graph reshapeTensor:columns
                                withShape:@[
                                    @(n), @(c), @(kH * kW), @(positions)
                                ]
                                name:nil];
                        [depth_windows addObject:columns];
                    }
                    MPSGraphTensor *stacked = [graph stackTensors:depth_windows
                            axis:2 name:nil];
                    output = [graph reshapeTensor:stacked
                            withShape:shapes[node.output] name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: {
                    uint32_t offset = 1U + (uint32_t)node.attribute_values[0];
                    uint64_t kD = node.attribute_values[offset];
                    uint64_t kH = node.attribute_values[offset + 1U];
                    uint64_t kW = node.attribute_values[offset + 2U];
                    uint64_t sD = node.attribute_values[offset + 3U];
                    uint64_t sH = node.attribute_values[offset + 4U];
                    uint64_t sW = node.attribute_values[offset + 5U];
                    uint64_t pD = node.attribute_values[offset + 6U];
                    uint64_t pH = node.attribute_values[offset + 7U];
                    uint64_t pW = node.attribute_values[offset + 8U];
                    uint64_t dD = node.attribute_values[offset + 9U];
                    uint64_t dH = node.attribute_values[offset + 10U];
                    uint64_t dW = node.attribute_values[offset + 11U];
                    BOOL ceil_mode = node.attribute_values[offset + 12U] != 0U;
                    MPSShape *target = shapes[node.output];
                    uint64_t dOut = 0U, hOut = 0U, wOut = 0U;
                    if (!task0059_window_extent(
                                target[2].unsignedLongLongValue,
                                kD, pD, sD, dD, ceil_mode, &dOut)
                            || !task0059_window_extent(
                                target[3].unsignedLongLongValue,
                                kH, pH, sH, dH, ceil_mode, &hOut)
                            || !task0059_window_extent(
                                target[4].unsignedLongLongValue,
                                kW, pW, sW, dW, ceil_mode, &wOut))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    uint64_t n = target[0].unsignedLongLongValue;
                    uint64_t c = target[1].unsignedLongLongValue;
                    uint64_t plane_positions = hOut * wOut;
                    uint64_t positions = dOut * plane_positions;
                    MPSGraphTensor *columns = [graph reshapeTensor:first
                            withShape:@[
                                @(n), @(c), @(kD), @(kH * kW), @(positions)
                            ]
                            name:nil];
                    uint64_t effectiveH = dH * (kH - 1U) + 1U;
                    uint64_t effectiveW = dW * (kW - 1U) + 1U;
                    if ((hOut - 1U) > (UINT64_MAX - effectiveH) / sH
                            || (wOut - 1U) > (UINT64_MAX - effectiveW) / sW)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    uint64_t requiredH = (hOut - 1U) * sH + effectiveH;
                    uint64_t requiredW = (wOut - 1U) * sW + effectiveW;
                    uint64_t paddedH =
                            target[3].unsignedLongLongValue + pH * 2U;
                    uint64_t paddedW =
                            target[4].unsignedLongLongValue + pW * 2U;
                    uint64_t extraH =
                            requiredH > paddedH ? requiredH - paddedH : 0U;
                    uint64_t extraW =
                            requiredW > paddedW ? requiredW - paddedW : 0U;
                    MPSGraphImToColOpDescriptor *descriptor =
                            [MPSGraphImToColOpDescriptor
                                    descriptorWithKernelWidth:(NSUInteger)kW
                                    kernelHeight:(NSUInteger)kH
                                    strideInX:(NSUInteger)sW
                                    strideInY:(NSUInteger)sH
                                    dilationRateInX:(NSUInteger)dW
                                    dilationRateInY:(NSUInteger)dH
                                    paddingLeft:(NSUInteger)pW
                                    paddingRight:(NSUInteger)(pW + extraW)
                                    paddingTop:(NSUInteger)pH
                                    paddingBottom:(NSUInteger)(pH + extraH)
                                    dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
                    MPSGraphTensor *accumulator = nil;
                    for (uint64_t kd = 0U; kd < kD; kd++) {
                        MPSGraphTensor *kernel_slice = [graph sliceTensor:columns
                                dimension:2U start:(NSInteger)kd length:1U name:nil];
                        kernel_slice = [graph squeezeTensor:kernel_slice axis:2 name:nil];
                        kernel_slice = [graph reshapeTensor:kernel_slice
                                withShape:@[
                                    @(n), @(c * kH * kW), @(dOut),
                                    @(plane_positions)
                                ]
                                name:nil];
                        kernel_slice = [graph transposeTensor:kernel_slice
                                permutation:@[@0, @2, @1, @3] name:nil];
                        kernel_slice = [graph reshapeTensor:kernel_slice
                                withShape:@[
                                    @(n * dOut), @(c * kH * kW),
                                    @(plane_positions)
                                ]
                                name:nil];
                        MPSGraphTensor *planes = [graph colToImWithSourceTensor:kernel_slice
                                outputShape:@[
                                    @(n * dOut), @(c), target[3], target[4]
                                ]
                                descriptor:descriptor
                                name:nil];
                        planes = [graph reshapeTensor:planes
                                withShape:@[
                                    @(n), @(dOut), @(c), target[3], target[4]
                                ]
                                name:nil];
                        planes = [graph transposeTensor:planes
                                permutation:@[@0, @2, @1, @3, @4] name:nil];
                        uint64_t base = kd * dD;
                        uint64_t target_depth = target[2].unsignedLongLongValue;
                        uint64_t upper = pD + target_depth;
                        if (base >= upper) continue;
                        uint64_t first_depth = 0U;
                        if (base < pD) {
                            uint64_t delta = pD - base;
                            first_depth = delta / sD
                                    + (delta % sD == 0U ? 0U : 1U);
                        }
                        uint64_t last_depth = (upper - 1U - base) / sD;
                        if (first_depth >= dOut) continue;
                        if (last_depth >= dOut) last_depth = dOut - 1U;
                        if (last_depth < first_depth
                                || first_depth > (uint64_t)NSIntegerMax
                                || last_depth - first_depth + 1U
                                        > (uint64_t)NSUIntegerMax)
                            continue;
                        uint64_t depth_count =
                                last_depth - first_depth + 1U;
                        planes = [graph sliceTensor:planes
                                dimension:2U
                                start:(NSInteger)first_depth
                                length:(NSUInteger)depth_count
                                name:nil];
                        MPSGraphTensor *coordinate = [graph coordinateAlongAxis:2
                                withShape:planes.shape name:nil];
                        coordinate = [graph castTensor:coordinate
                                toType:MPSDataTypeInt64 name:nil];
                        MPSGraphTensor *stride_scalar =
                                exact_int64_scalar(graph, (int64_t)sD);
                        uint64_t first_position = base + first_depth * sD - pD;
                        MPSGraphTensor *offset_scalar =
                                exact_int64_scalar(graph, (int64_t)first_position);
                        MPSGraphTensor *indices = [graph multiplicationWithPrimaryTensor:coordinate
                                secondaryTensor:stride_scalar name:nil];
                        indices = [graph additionWithPrimaryTensor:indices
                                secondaryTensor:offset_scalar name:nil];
                        if (accumulator == nil)
                            accumulator = [graph constantWithScalar:0.0
                                    shape:target
                                    dataType:value_mps_data_type(
                                            value_types[node.output])];
                        accumulator = [graph scatterAlongAxis:2
                                withDataTensor:accumulator
                                updatesTensor:planes
                                indicesTensor:indices
                                mode:MPSGraphScatterModeAdd
                                name:nil];
                    }
                    output = accumulator == nil
                            ? [graph constantWithScalar:0.0
                                    shape:target
                                    dataType:value_mps_data_type(
                                            value_types[node.output])]
                            : accumulator;
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_UNFOLD_AXIS: {
                    NSUInteger rank = shapes[node.first_input].count;
                    uint64_t size = node.attribute_values[0];
                    uint64_t step = node.attribute_values[1];
                    uint64_t selected =
                            shapes[node.first_input][node.axis].unsignedLongLongValue;
                    uint64_t positions = (selected - size) / step + 1U;
                    NSMutableArray<MPSGraphTensor *> *windows =
                            [NSMutableArray arrayWithCapacity:(NSUInteger)size];
                    if (windows == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    for (uint64_t window_axis = 0U; window_axis < size; window_axis++) {
                        NSMutableArray<NSNumber *> *starts =
                                [NSMutableArray arrayWithCapacity:rank];
                        NSMutableArray<NSNumber *> *ends =
                                [NSMutableArray arrayWithCapacity:rank];
                        NSMutableArray<NSNumber *> *strides =
                                [NSMutableArray arrayWithCapacity:rank];
                        if (starts == nil || ends == nil || strides == nil)
                            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                        for (NSUInteger axis = 0U; axis < rank; axis++) {
                            uint64_t start = axis == node.axis ? window_axis : 0U;
                            uint64_t end = axis == node.axis
                                    ? window_axis + (positions - 1U) * step + 1U
                                    : shapes[node.first_input][axis].unsignedLongLongValue;
                            uint64_t stride = axis == node.axis ? step : 1U;
                            [starts addObject:@((NSInteger)start)];
                            [ends addObject:@((NSInteger)end)];
                            [strides addObject:@((NSInteger)stride)];
                        }
                        MPSGraphTensor *slice = [graph sliceTensor:first
                                starts:starts
                                ends:ends
                                strides:strides
                                name:nil];
                        MPSGraphTensor *window = slice == nil ? nil
                                : [graph expandDimsOfTensor:slice axis:(NSInteger)rank name:nil];
                        if (window == nil)
                            return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
                        [windows addObject:window];
                    }
                    output = [graph concatTensors:windows
                            dimension:(NSInteger)rank name:nil];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_GATHER:
                    output = [graph gatherWithUpdatesTensor:first
                            indicesTensor:second
                            axis:(NSInteger)node.axis
                            batchDimensions:0U
                            name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS:
                    output = [graph scatterAlongAxis:(NSInteger)node.axis
                            withDataTensor:first
                            updatesTensor:auxiliary
                            indicesTensor:second
                            mode:MPSGraphScatterModeSet
                            name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ONE_HOT:
                    output = [graph oneHotWithIndicesTensor:first
                            depth:(NSUInteger)node.attribute_values[0]
                            dataType:MPSDataTypeBool
                            onValue:1.0
                            offValue:0.0
                            name:nil];
                    break;
                default:
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (output == nil
                    || ![output.shape isEqualToArray:shapes[node.output]])
                return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
            table[node.output] = output;
        }
        NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray arrayWithCapacity:target_count];
        for (uint32_t target = 0; target < target_count; target++) {
            id tensor = table[target_indices[target]];
            if (tensor == NSNull.null) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            [targets addObject:(MPSGraphTensor *)tensor];
        }
        SynaptikMetalContextBox *ctx = (__bridge SynaptikMetalContextBox *)context;
        MPSGraphDevice *device = [MPSGraphDevice deviceWithMTLDevice:ctx.device];
        MPSGraphCompilationDescriptor *descriptor = [MPSGraphCompilationDescriptor new];
        if (device == nil || descriptor == nil)
            return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
        if (@available(macOS 26.0, *)) {
            descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone;
            if (descriptor.reducedPrecisionFastMath
                    != MPSGraphReducedPrecisionFastMathNone)
                return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
        } else if (contains_matmul) {
            return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
        }
        MPSGraphExecutable *compiled = [graph compileWithDevice:device feeds:types
                targetTensors:targets targetOperations:nil compilationDescriptor:descriptor];
        if (compiled == nil) return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
        compiled.options = MPSGraphOptionsNone;
        NSArray<NSNumber *> *feed_permutation = permutation(compiled.feedTensors, feeds);
        NSArray<NSNumber *> *target_permutation = permutation(compiled.targetTensors, targets);
        if (feed_permutation == nil || target_permutation == nil)
            return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
        NSMutableArray *feed_shapes = [NSMutableArray arrayWithCapacity:feed_count];
        NSMutableArray *feed_bytes = [NSMutableArray arrayWithCapacity:feed_count];
        NSMutableArray *feed_data_types = [NSMutableArray arrayWithCapacity:feed_count];
        for (uint32_t feed = 0; feed < feed_count; feed++) {
            uint32_t value = feed_indices[feed];
            [feed_shapes addObject:shapes[value]];
            [feed_bytes addObject:bytes[value]];
            [feed_data_types addObject:@((NSUInteger)value_mps_data_type(value_types[value]))];
        }
        NSMutableArray<NSArray<NSNumber *> *> *feed_strides =
                [NSMutableArray arrayWithCapacity:feed_count];
        NSMutableArray<NSNumber *> *feed_offsets =
                [NSMutableArray arrayWithCapacity:feed_count];
        if (feed_strides == nil || feed_offsets == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        for (uint32_t feed = 0U; feed < feed_count; feed++) {
            uint32_t value = feed_indices[feed];
            [feed_strides addObject:physical_strides[value]];
            [feed_offsets addObject:physical_offsets[value]];
        }
        NSMutableArray *target_shapes = [NSMutableArray arrayWithCapacity:target_count];
        NSMutableArray *target_bytes = [NSMutableArray arrayWithCapacity:target_count];
        NSMutableArray *target_data_types = [NSMutableArray arrayWithCapacity:target_count];
        for (uint32_t target = 0; target < target_count; target++) {
            uint32_t value = target_indices[target];
            [target_shapes addObject:shapes[value]];
            [target_bytes addObject:bytes[value]];
            [target_data_types addObject:@((NSUInteger)value_mps_data_type(value_types[value]))];
        }
        SynaptikMetalExecutableBox *box = [SynaptikMetalExecutableBox new];
        if (box == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        box.executable = compiled;
        box.context = ctx;
        box.feedShapes = feed_shapes;
        box.targetShapes = target_shapes;
        box.feedBytes = feed_bytes;
        box.targetBytes = target_bytes;
        box.feedPermutation = feed_permutation;
        box.targetPermutation = target_permutation;
        box.feedDataTypes = feed_data_types;
        box.targetDataTypes = target_data_types;
        box.indexValidations = index_validations;
        NSMutableArray<NSNumber *> *canonical_bool_inputs =
                [NSMutableArray array];
        if (canonical_bool_inputs == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        for (uint32_t feed = 0U; feed < feed_count; feed++)
            if (declared_types[feed_indices[feed]] == SYNAPTIK_METAL_TYPE_BOOL)
                [canonical_bool_inputs addObject:@(feed)];
        box.canonicalBoolInputIndices = [canonical_bool_inputs copy];
        box.inputShapes = [feed_shapes copy];
        box.inputStrides = [feed_strides copy];
        box.inputOffsets = [feed_offsets copy];
        *out_executable = (__bridge_retained void *)box;
        return SYNAPTIK_METAL_STATUS_OK;
    } } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_executable_create(
        void *context, const uint8_t *program, uint32_t program_bytes,
        void **out_executable) {
    if (out_executable == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_executable = NULL;
    if (context == NULL || program == NULL || program_bytes < 64U
            || program_bytes > (uint32_t)INT32_MAX)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { @autoreleasepool {
        if (synaptik_read_le32(program) != UINT32_C(0x35314d53)
                || synaptik_read_le32(program + 4U) != 15U
                || synaptik_read_le32(program + 8U) != program_bytes)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        uint32_t route = synaptik_read_le32(program + 40U);
        if (route != SYNAPTIK_METAL_ROUTE_MPSGRAPH
                && route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        uint32_t stride_count = synaptik_read_le32(program + 44U);
        for (uint32_t offset = 48U; offset < 64U; offset += 4U)
            if (synaptik_read_le32(program + offset) != 0U)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;

        uint32_t value_count = synaptik_read_le32(program + 12U);
        uint32_t node_count = synaptik_read_le32(program + 16U);
        uint32_t feed_count = synaptik_read_le32(program + 20U);
        uint32_t target_count = synaptik_read_le32(program + 24U);
        uint32_t dimension_count = synaptik_read_le32(program + 28U);
        uint32_t reference_count = synaptik_read_le32(program + 32U);
        uint32_t attribute_count = synaptik_read_le32(program + 36U);
        if (value_count == 0U || node_count == 0U || target_count == 0U)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;

        uint64_t values_offset = 64U;
        uint64_t nodes_offset = values_offset + (uint64_t)value_count * 40U;
        uint64_t dimensions_offset = nodes_offset + (uint64_t)node_count * 32U;
        uint64_t strides_offset =
                dimensions_offset + (uint64_t)dimension_count * 8U;
        uint64_t references_offset =
                strides_offset + (uint64_t)stride_count * 8U;
        uint64_t references_end =
                references_offset + (uint64_t)reference_count * 4U;
        uint64_t attributes_offset = (references_end + 7U) & ~UINT64_C(7);
        uint64_t image_end = attributes_offset + (uint64_t)attribute_count * 8U;
        if (image_end != program_bytes
                || references_end > attributes_offset
                || attributes_offset - references_end > 4U)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        for (uint64_t offset = references_end; offset < attributes_offset; offset++)
            if (program[offset] != 0U)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;

        NSMutableData *rank_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * sizeof(uint32_t)];
        NSMutableData *dimension_data = [NSMutableData dataWithLength:
                (NSUInteger)value_count * SYNAPTIK_MAX_RANK * sizeof(uint64_t)];
        NSMutableData *stride_data = [NSMutableData dataWithLength:
                (NSUInteger)value_count * SYNAPTIK_MAX_RANK * sizeof(uint64_t)];
        NSMutableData *layout_offset_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * sizeof(uint64_t)];
        NSMutableData *layout_span_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * sizeof(uint64_t)];
        NSMutableData *layout_state_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * 3U];
        NSMutableData *type_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count];
        NSMutableData *node_data = [NSMutableData dataWithLength:
                (NSUInteger)node_count * sizeof(SynaptikMetalDecodedNode)];
        NSMutableData *feed_data =
                [NSMutableData dataWithLength:(NSUInteger)feed_count * sizeof(uint32_t)];
        NSMutableData *target_data =
                [NSMutableData dataWithLength:(NSUInteger)target_count * sizeof(uint32_t)];
        NSMutableData *topology_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * 4U];
        if (rank_data == nil || dimension_data == nil || stride_data == nil
                || layout_offset_data == nil || layout_span_data == nil
                || layout_state_data == nil || type_data == nil || node_data == nil
                || feed_data == nil || target_data == nil || topology_data == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        uint32_t *value_ranks = rank_data.mutableBytes;
        uint64_t *value_dimensions = dimension_data.mutableBytes;
        uint64_t *value_strides = stride_data.mutableBytes;
        uint64_t *layout_offsets = layout_offset_data.mutableBytes;
        uint64_t *layout_spans = layout_span_data.mutableBytes;
        uint8_t *layout_present = layout_state_data.mutableBytes;
        uint8_t *layout_view = layout_present + value_count;
        uint8_t *declared_states = layout_view + value_count;
        uint8_t *declared_types = type_data.mutableBytes;
        SynaptikMetalDecodedNode *nodes = node_data.mutableBytes;
        uint32_t *feed_indices = feed_data.mutableBytes;
        uint32_t *target_indices = target_data.mutableBytes;
        uint8_t *available = topology_data.mutableBytes;
        uint8_t *produced = available + value_count;
        uint8_t *consumed = produced + value_count;
        uint8_t *targeted = consumed + value_count;

        uint32_t dimension_cursor = 0U;
        uint32_t stride_cursor = 0U;
        for (uint32_t value = 0U; value < value_count; value++) {
            const uint8_t *descriptor = program + values_offset + (uint64_t)value * 40U;
            uint32_t type = synaptik_read_le32(descriptor);
            uint32_t rank = synaptik_read_le32(descriptor + 4U);
            uint32_t dimension_offset = synaptik_read_le32(descriptor + 8U);
            uint32_t stride_offset = synaptik_read_le32(descriptor + 12U);
            uint32_t flags = synaptik_read_le32(descriptor + 16U);
            uint32_t kind = synaptik_read_le32(descriptor + 20U);
            uint64_t storage_offset = synaptik_read_le64(descriptor + 24U);
            uint64_t referenced_span = synaptik_read_le64(descriptor + 32U);
            BOOL present = (flags & 2U) != 0U;
            BOOL view = (flags & 4U) != 0U;
            BOOL dense = (flags & 8U) != 0U;
            if (type < SYNAPTIK_METAL_TYPE_FLOAT32 || type > SYNAPTIK_METAL_TYPE_INT64
                    || rank > SYNAPTIK_MAX_RANK || dimension_offset != dimension_cursor
                    || dimension_cursor > dimension_count
                    || rank > dimension_count - dimension_cursor || (flags & ~15U) != 0U
                    || storage_offset > (uint64_t)INT64_MAX
                    || referenced_span > (uint64_t)INT64_MAX)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            value_ranks[value] = rank;
            declared_types[value] = (uint8_t)type;
            for (uint32_t axis = 0U; axis < rank; axis++) {
                uint64_t dimension = synaptik_read_le64(
                        program + dimensions_offset + (uint64_t)(dimension_cursor + axis) * 8U);
                if (dimension == 0U || dimension > (uint64_t)INT64_MAX)
                    return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                value_dimensions[(size_t)value * SYNAPTIK_MAX_RANK + axis] = dimension;
            }
            if (!present) {
                if (view || dense || stride_offset != UINT32_MAX || kind != 0U
                        || storage_offset != 0U || referenced_span != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            } else {
                if (stride_offset != stride_cursor || stride_cursor > stride_count
                        || rank > stride_count - stride_cursor
                        || kind < 1U || kind > 4U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                BOOL canonical = YES;
                BOOL broadcast = NO;
                uint64_t canonical_stride = 1U;
                uint64_t span = storage_offset;
                for (uint32_t reverse = rank; reverse > 0U; reverse--) {
                    uint32_t axis = reverse - 1U;
                    uint64_t stride = synaptik_read_le64(
                            program + strides_offset
                                    + (uint64_t)(stride_cursor + axis) * 8U);
                    uint64_t dimension =
                            value_dimensions[(size_t)value * SYNAPTIK_MAX_RANK + axis];
                    if (stride > (uint64_t)INT64_MAX)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    value_strides[(size_t)value * SYNAPTIK_MAX_RANK + axis] = stride;
                    canonical &= stride == canonical_stride;
                    broadcast |= dimension > 1U && stride == 0U;
                    if (dimension - 1U != 0U
                            && stride > ((uint64_t)INT64_MAX - span) / (dimension - 1U))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    span += (dimension - 1U) * stride;
                    if (axis > 0U) {
                        if (dimension > (uint64_t)INT64_MAX / canonical_stride)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        canonical_stride *= dimension;
                    }
                }
                if (span == (uint64_t)INT64_MAX) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                span++;
                uint32_t expected_kind =
                        canonical ? (storage_offset == 0U ? 1U : 2U)
                                  : (broadcast ? 4U : 3U);
                if (kind != expected_kind || referenced_span != span
                        || (kind == 4U && !view) || (dense && !view))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                layout_present[value] = 1U;
                layout_view[value] = view ? 1U : 0U;
                layout_offsets[value] = storage_offset;
                layout_spans[value] = referenced_span;
                declared_states[value] = dense
                        ? SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                        : (kind == 1U && !view
                                ? SYNAPTIK_METAL_VALUE_CANONICAL
                                : SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT);
                stride_cursor += rank;
            }
            dimension_cursor += rank;
        }
        if (dimension_cursor != dimension_count || stride_cursor != stride_count)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;

        if ((uint64_t)feed_count + target_count > reference_count)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        uint32_t reference_cursor = 0U;
        for (uint32_t feed = 0U; feed < feed_count; feed++) {
            feed_indices[feed] = synaptik_read_le32(
                    program + references_offset + (uint64_t)reference_cursor++ * 4U);
            if (feed_indices[feed] >= value_count || available[feed_indices[feed]])
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            available[feed_indices[feed]] = 1U;
        }
        for (uint32_t target = 0U; target < target_count; target++) {
            target_indices[target] = synaptik_read_le32(
                    program + references_offset + (uint64_t)reference_cursor++ * 4U);
            if (target_indices[target] >= value_count)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t feed = 0U; feed < feed_count; feed++) {
            uint32_t value = feed_indices[feed];
            if (declared_states[value] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                    || !synaptik_storage_layout_supported(
                            value, value_ranks, value_dimensions, value_strides,
                            layout_offsets, layout_spans, layout_present, 0U))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }

        uint32_t attribute_cursor = 0U;
        BOOL contains_unsupported = NO;
        for (uint32_t node_index = 0U; node_index < node_count; node_index++) {
            const uint8_t *descriptor =
                    program + nodes_offset + (uint64_t)node_index * 32U;
            uint32_t operation = synaptik_read_le32(descriptor);
            uint32_t attribute_kind = synaptik_read_le32(descriptor + 4U);
            uint32_t input_offset = synaptik_read_le32(descriptor + 8U);
            uint32_t input_count = synaptik_read_le32(descriptor + 12U);
            uint32_t output_offset = synaptik_read_le32(descriptor + 16U);
            uint32_t output_count = synaptik_read_le32(descriptor + 20U);
            uint32_t attribute_offset = synaptik_read_le32(descriptor + 24U);
            uint32_t attribute_word_count = synaptik_read_le32(descriptor + 28U);
            if (!synaptik_node_signature_is_valid(
                            operation, attribute_kind, input_count, output_count)
                    || input_offset != reference_cursor
                    || input_count > reference_count - reference_cursor
                    || input_count > SYNAPTIK_MAX_NODE_INPUTS
                    || output_count > SYNAPTIK_MAX_NODE_OUTPUTS
                    || output_offset != reference_cursor + input_count
                    || output_count > reference_count - output_offset
                    || attribute_offset != attribute_cursor
                    || attribute_word_count > attribute_count - attribute_cursor
                    || attribute_word_count > SYNAPTIK_MAX_ATTRIBUTE_WORDS)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            const uint8_t *words =
                    program + attributes_offset + (uint64_t)attribute_offset * 8U;
            if (!synaptik_attribute_is_valid(
                        operation, attribute_kind, words, attribute_word_count))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            if (operation >= SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW
                    && synaptik_attribute_word(words, 0U) != SYNAPTIK_METAL_TYPE_FLOAT32)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            for (uint32_t reference = input_offset; reference < output_offset; reference++) {
                uint32_t value = synaptik_read_le32(
                        program + references_offset + (uint64_t)reference * 4U);
                if (value >= value_count || !available[value])
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                consumed[value] = 1U;
            }
            for (uint32_t reference = output_offset;
                    reference < output_offset + output_count; reference++) {
                uint32_t value = synaptik_read_le32(
                        program + references_offset + (uint64_t)reference * 4U);
                if (value >= value_count || available[value])
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                available[value] = 1U;
                produced[value] = 1U;
                if (operation == SYNAPTIK_METAL_CUSTOM_TOP_K)
                    consumed[value] = 1U;
            }
            reference_cursor = output_offset + output_count;
            if (operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD
                    || operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_SUB
                    || operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL
                    || operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV
                    || operation == SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t input_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                if ((input_flags & 1U) != 0U || (output_flags & 1U) != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                if (value_ranks[input_value] == 0U || value_ranks[output_value] == 0U)
                    return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            }
            if (operation == SYNAPTIK_METAL_MPSGRAPH_MSE) {
                uint32_t left_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t right_value = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(input_offset + 1U) * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t left_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)left_value * 40U + 16U);
                uint32_t right_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)right_value * 40U + 16U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                BOOL input_gradient =
                        (left_flags & 1U) != 0U || (right_flags & 1U) != 0U;
                if (((output_flags & 1U) != 0U) != input_gradient)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (operation == SYNAPTIK_METAL_MPSGRAPH_PERMUTE) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t input_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                if ((input_flags & 1U) != (output_flags & 1U))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if ((operation >= SYNAPTIK_METAL_CUSTOM_SORT
                            && operation <= SYNAPTIK_METAL_CUSTOM_TOP_K)
                    || (operation >= SYNAPTIK_METAL_CUSTOM_ARG_MAX
                            && operation <= SYNAPTIK_METAL_CUSTOM_ARG_MIN)) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t primary_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                BOOL input_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U) & 1U) != 0U;
                BOOL primary_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)primary_value * 40U + 16U) & 1U) != 0U;
                BOOL propagates_gradient =
                        operation == SYNAPTIK_METAL_CUSTOM_SORT
                        || operation == SYNAPTIK_METAL_CUSTOM_TOP_K;
                if (propagates_gradient
                                ? input_gradient != primary_gradient
                                : primary_gradient)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                if (operation == SYNAPTIK_METAL_CUSTOM_TOP_K) {
                    uint32_t index_value = synaptik_read_le32(
                            program + references_offset
                                    + (uint64_t)(output_offset + 1U) * 4U);
                    if ((synaptik_read_le32(
                                    program + values_offset
                                            + (uint64_t)index_value * 40U + 16U)
                                    & 1U) != 0U)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                }
            }
            if (operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                    || operation == SYNAPTIK_METAL_CUSTOM_CONV3D) {
                BOOL any_float32 = NO;
                BOOL any_bfloat16 = NO;
                BOOL any_gradient = NO;
                for (uint32_t input = 0U; input < input_count; input++) {
                    uint32_t value = synaptik_read_le32(
                            program + references_offset
                                    + (uint64_t)(input_offset + input) * 4U);
                    uint8_t type = declared_types[value];
                    if (type != SYNAPTIK_METAL_TYPE_FLOAT32
                            && type != SYNAPTIK_METAL_TYPE_BFLOAT16)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    any_float32 |= type == SYNAPTIK_METAL_TYPE_FLOAT32;
                    any_bfloat16 |= type == SYNAPTIK_METAL_TYPE_BFLOAT16;
                    any_gradient |= (synaptik_read_le32(
                            program + values_offset + (uint64_t)value * 40U + 16U)
                            & 1U) != 0U;
                }
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                BOOL output_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U)
                        & 1U) != 0U;
                if (!any_float32
                        || declared_types[output_value] != SYNAPTIK_METAL_TYPE_FLOAT32
                        || (any_bfloat16
                                ? any_gradient || output_gradient
                                : output_gradient != any_gradient))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (operation >= SYNAPTIK_METAL_CUSTOM_MAX_POOL2D
                    && operation <= SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint8_t input_type = declared_types[input_value];
                uint8_t output_type = declared_types[output_value];
                BOOL average = operation == SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D
                        || operation == SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D;
                BOOL type_valid = average
                        ? input_type == SYNAPTIK_METAL_TYPE_FLOAT32
                                && output_type == SYNAPTIK_METAL_TYPE_FLOAT32
                        : input_type == output_type
                                && (input_type == SYNAPTIK_METAL_TYPE_FLOAT64
                                        || input_type == SYNAPTIK_METAL_TYPE_FLOAT32
                                        || input_type == SYNAPTIK_METAL_TYPE_BFLOAT16);
                BOOL input_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U)
                        & 1U) != 0U;
                BOOL output_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U)
                        & 1U) != 0U;
                if (!type_valid || input_gradient != output_gradient)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            BOOL task0059_no_gradient = operation == SYNAPTIK_METAL_CUSTOM_CAST
                    || operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
                    || operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
                    || (operation >= SYNAPTIK_METAL_CUSTOM_SELECT
                            && operation <= SYNAPTIK_METAL_CUSTOM_SLICE)
                    || (operation >= SYNAPTIK_METAL_CUSTOM_CONCAT
                            && operation <= SYNAPTIK_METAL_CUSTOM_TILE)
                    || operation == SYNAPTIK_METAL_CUSTOM_UNFOLD2D
                    || operation == SYNAPTIK_METAL_CUSTOM_UNFOLD3D
                    || operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
                    || operation == SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS
                    || operation == SYNAPTIK_METAL_MPSGRAPH_FOLD2D
                    || operation == SYNAPTIK_METAL_MPSGRAPH_FOLD3D
                    || (operation >= SYNAPTIK_METAL_CUSTOM_PROD
                            && operation <= SYNAPTIK_METAL_CUSTOM_ANY);
            if (task0059_no_gradient) {
                for (uint32_t reference = input_offset;
                        reference < output_offset + output_count; reference++) {
                    uint32_t value = synaptik_read_le32(
                            program + references_offset + (uint64_t)reference * 4U);
                    uint32_t flags = synaptik_read_le32(
                            program + values_offset + (uint64_t)value * 40U + 16U);
                    if ((flags & 1U) != 0U)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                }
            }
            if (operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL) {
                uint32_t left_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t right_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)(input_offset + 1U) * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t left_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)left_value * 40U + 16U);
                uint32_t right_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)right_value * 40U + 16U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                uint8_t left_type = declared_types[left_value];
                uint8_t right_type = declared_types[right_value];
                BOOL integral = (left_type == SYNAPTIK_METAL_TYPE_INT32
                                || left_type == SYNAPTIK_METAL_TYPE_INT64)
                        && (right_type == SYNAPTIK_METAL_TYPE_INT32
                                || right_type == SYNAPTIK_METAL_TYPE_INT64);
                BOOL mixed = (left_type == SYNAPTIK_METAL_TYPE_BFLOAT16
                                && right_type == SYNAPTIK_METAL_TYPE_FLOAT32)
                        || (left_type == SYNAPTIK_METAL_TYPE_FLOAT32
                                && right_type == SYNAPTIK_METAL_TYPE_BFLOAT16);
                BOOL any_input_gradient =
                        (left_flags & 1U) != 0U || (right_flags & 1U) != 0U;
                BOOL output_gradient = (output_flags & 1U) != 0U;
                if ((integral || mixed) && (any_input_gradient || output_gradient))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                if (!integral && !mixed
                        && output_gradient != any_input_gradient)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }

            if (!operation_has_direct_mpsgraph(operation)
                    && !operation_uses_custom_kernel(operation)) {
                contains_unsupported = YES;
                attribute_cursor += attribute_word_count;
                continue;
            }
            SynaptikMetalDecodedNode node = {0};
            node.operation = operation;
            node.attribute_kind = attribute_kind;
            node.first_input = synaptik_read_le32(
                    program + references_offset + (uint64_t)input_offset * 4U);
            node.second_input = input_count >= 2U
                    ? synaptik_read_le32(
                            program + references_offset + (uint64_t)(input_offset + 1U) * 4U)
                    : UINT32_MAX;
            node.auxiliary = input_count >= 3U
                    ? synaptik_read_le32(
                            program + references_offset + (uint64_t)(input_offset + 2U) * 4U)
                    : 0U;
            node.input_count = input_count;
            node.output_count = output_count;
            for (uint32_t input = 0U; input < input_count; input++) {
                node.inputs[input] = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(input_offset + input) * 4U);
            }
            node.output = synaptik_read_le32(
                    program + references_offset + (uint64_t)output_offset * 4U);
            for (uint32_t output = 0U; output < output_count; output++) {
                node.outputs[output] = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(output_offset + output) * 4U);
            }
            node.axis = UINT32_MAX;
            if (attribute_kind == 0U) {
                if (attribute_word_count != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            } else if (attribute_kind == 1U || attribute_kind == 2U) {
                if (attribute_word_count < 2U
                        || synaptik_read_le64(words) != attribute_word_count - 1U
                        || attribute_word_count - 1U > SYNAPTIK_MAX_RANK)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = attribute_word_count - 1U;
                for (uint32_t word = 0U; word < node.attribute_count; word++)
                    node.attribute_values[word] = synaptik_read_le64(words + (uint64_t)(word + 1U) * 8U);
            } else if (attribute_kind == 3U) {
                if (attribute_word_count != 1U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                uint64_t axis = synaptik_read_le64(words);
                if (axis > UINT32_MAX) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = 1U;
                node.axis = (uint32_t)axis;
            } else if (attribute_kind == 4U) {
                if (attribute_word_count < 3U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                uint64_t form = synaptik_read_le64(words);
                uint64_t keep = synaptik_read_le64(words + 8U);
                uint64_t count = synaptik_read_le64(words + 16U);
                if (form < 1U || form > 4U || keep > 1U || count > SYNAPTIK_MAX_RANK
                        || count != attribute_word_count - 3U
                        || (form == 1U && (count != 0U || keep != 0U))
                        || (form == 2U && count != 1U)
                        || (form == 4U && (operation != 13U || keep != 0U)))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = (uint32_t)count;
                node.axis = (uint32_t)form;
                node.auxiliary = (uint32_t)keep;
                for (uint32_t word = 0U; word < node.attribute_count; word++)
                    node.attribute_values[word] = synaptik_read_le64(words + (uint64_t)(word + 3U) * 8U);
            } else if (attribute_kind == 5U) {
                if (attribute_word_count != 1U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = 1U;
                node.attribute_values[0] = synaptik_read_le64(words);
            } else if (attribute_kind == 6U) {
                if (attribute_word_count != 3U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                uint64_t axis = synaptik_read_le64(words);
                if (axis > UINT32_MAX) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = 3U;
                node.axis = (uint32_t)axis;
                node.attribute_values[0] = synaptik_read_le64(words + 8U);
                node.attribute_values[1] = synaptik_read_le64(words + 16U);
            } else if (attribute_kind == 7U) {
                if (attribute_word_count != 2U || synaptik_read_le64(words) != 1U
                        || (synaptik_read_le64(words + 8U) & ~UINT64_C(0xffffffff)) != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = 1U;
                node.attribute_values[0] = synaptik_read_le64(words + 8U);
            } else if (attribute_kind == 8U) {
                if (attribute_word_count != 4U
                        || synaptik_read_le64(words) != 1U
                        || synaptik_read_le64(words + 16U) != 1U
                        || (synaptik_read_le64(words + 8U) & ~UINT64_C(0xffffffff)) != 0U
                        || (synaptik_read_le64(words + 24U) & ~UINT64_C(0xffffffff)) != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = 2U;
                node.attribute_values[0] = synaptik_read_le64(words + 8U);
                node.attribute_values[1] = synaptik_read_le64(words + 24U);
            } else if (attribute_kind == 9U) {
                if (attribute_word_count != 3U
                        || synaptik_read_le64(words) > UINT32_MAX
                        || synaptik_read_le64(words + 8U) > 1U
                        || synaptik_read_le64(words + 16U) > 1U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = 2U;
                node.axis = (uint32_t)synaptik_read_le64(words);
                node.attribute_values[0] = synaptik_read_le64(words + 8U);
                node.attribute_values[1] = synaptik_read_le64(words + 16U);
            } else if (attribute_kind == 12U) {
                if (attribute_word_count != 2U
                        || synaptik_read_le64(words) > UINT32_MAX
                        || synaptik_read_le64(words + 8U) != 1U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_kind = SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS;
                node.attribute_count = 1U;
                node.axis = (uint32_t)synaptik_read_le64(words);
            } else if (attribute_kind >= 10U && attribute_kind <= 41U) {
                node.attribute_count = attribute_word_count;
                for (uint32_t word = 0U; word < attribute_word_count; word++) {
                    node.attribute_values[word] =
                            synaptik_read_le64(words + (uint64_t)word * 8U);
                }
            } else {
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (!task0059_validate_layout(
                        node, value_ranks, value_dimensions, value_strides,
                        layout_offsets, layout_spans, layout_present, layout_view))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            nodes[node_index] = node;
            attribute_cursor += attribute_word_count;
        }
        if (reference_cursor != reference_count || attribute_cursor != attribute_count)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        for (uint32_t target = 0U; target < target_count; target++) {
            uint32_t value = target_indices[target];
            if (!produced[value] || targeted[value])
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            targeted[value] = 1U;
            consumed[value] = 1U;
        }
        for (uint32_t value = 0U; value < value_count; value++)
            if (!available[value] || !consumed[value])
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        if (contains_unsupported) return SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION;
        return synaptik_metal_create_decoded(
                context, route, value_count, value_ranks, value_dimensions,
                value_strides, layout_offsets, layout_spans, declared_states,
                declared_types, node_count, nodes, feed_count, feed_indices,
                target_count, target_indices, out_executable);
    } } @catch (__unused NSException *exception) {
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_executable_release(void *executable) {
    if (executable == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { __unused id consumed = (__bridge_transfer id)executable;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

static int32_t validate_canonical_bool_inputs(
        SynaptikMetalExecutableBox *box,
        uint32_t input_count,
        void *const *input_buffers,
        NSArray<NSNumber *> *required_bytes) {
    if (box.canonicalBoolInputIndices == nil || required_bytes == nil
            || box.inputShapes == nil || box.inputStrides == nil
            || box.inputOffsets == nil)
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    for (NSNumber *encoded_index in box.canonicalBoolInputIndices) {
        NSUInteger index = encoded_index.unsignedIntegerValue;
        if (index >= input_count || index >= required_bytes.count
                || index >= box.inputShapes.count || index >= box.inputStrides.count
                || index >= box.inputOffsets.count || input_buffers[index] == NULL)
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        SynaptikMetalBufferBox *buffer =
                (__bridge SynaptikMetalBufferBox *)input_buffers[index];
        NSUInteger count = required_bytes[index].unsignedIntegerValue;
        if (buffer.logicalByteSize < count)
            return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
        const uint8_t *values = buffer.buffer.contents;
        if (values == NULL) return SYNAPTIK_METAL_STATUS_COPY_FAILED;
        MPSShape *shape = box.inputShapes[index];
        NSArray<NSNumber *> *strides = box.inputStrides[index];
        if (shape == nil || strides == nil || shape.count != strides.count)
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        uint64_t logical_count = shape_element_count(shape);
        for (uint64_t ordinal = 0U; ordinal < logical_count; ordinal++) {
            uint64_t remaining = ordinal;
            uint64_t storage = box.inputOffsets[index].unsignedLongLongValue;
            for (NSUInteger reverse = 0U; reverse < shape.count; reverse++) {
                NSUInteger axis = shape.count - 1U - reverse;
                uint64_t extent = shape[axis].unsignedLongLongValue;
                uint64_t coordinate = remaining % extent;
                remaining /= extent;
                storage += coordinate * strides[axis].unsignedLongLongValue;
            }
            if (storage >= count) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            if (values[storage] > 1U)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }
    }
    return SYNAPTIK_METAL_STATUS_OK;
}

static int32_t run_custom_program_box(
        SynaptikMetalExecutableBox *box,
        uint32_t input_count,
        void *const *input_buffers,
        uint32_t output_count,
        void *const *output_buffers) {
    if (input_count != box.valueCount
            || output_count != box.targetValueIndices.count
            || input_count != box.valueBytes.count)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    /*
     * Public ABI safety validation: runtime handles remain untrusted even though Java repeats these
     * checks before its downcall. Validate in-place without allocating a hot value-table mirror.
     */
    for (uint32_t value = 0U; value < input_count; value++) {
        if (input_buffers[value] == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        SynaptikMetalBufferBox *buffer =
                (__bridge SynaptikMetalBufferBox *)input_buffers[value];
        if (buffer.buffer.device != box.context.device
                || buffer.logicalByteSize < box.valueBytes[value].unsignedLongLongValue)
            return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
        for (uint32_t previous = 0U; previous < value; previous++) {
            SynaptikMetalBufferBox *other =
                    (__bridge SynaptikMetalBufferBox *)input_buffers[previous];
            if (other.buffer == buffer.buffer)
                return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
        }
    }
    for (uint32_t output = 0U; output < output_count; output++) {
        NSUInteger value = box.targetValueIndices[output].unsignedIntegerValue;
        if (value >= input_count || output_buffers[output] != input_buffers[value])
            return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    }
    int32_t bool_status = validate_canonical_bool_inputs(
            box, input_count, input_buffers, box.valueBytes);
    if (bool_status != SYNAPTIK_METAL_STATUS_OK) return bool_status;
    for (SynaptikMetalIndexValidation *validation in box.indexValidations) {
        if (!validation.preflight) continue;
        if (validation.feedPosition >= input_count)
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        SynaptikMetalBufferBox *index_buffer =
                (__bridge SynaptikMetalBufferBox *)
                        input_buffers[validation.feedPosition];
        int32_t index_status = validate_index_buffer(validation, index_buffer);
        if (index_status != SYNAPTIK_METAL_STATUS_OK) return index_status;
    }
    NSUInteger cursor = 0U;
    while (cursor < box.programSteps.count) {
        SynaptikMetalProgramStep *step = box.programSteps[cursor];
        if (step.indexValidation != nil) {
            if (step.indexValidation.feedPosition >= input_count)
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            SynaptikMetalBufferBox *index_buffer =
                    (__bridge SynaptikMetalBufferBox *)
                            input_buffers[step.indexValidation.feedPosition];
            int32_t index_status =
                    validate_index_buffer(step.indexValidation, index_buffer);
            if (index_status != SYNAPTIK_METAL_STATUS_OK) return index_status;
        }
        if (!step.custom) {
            NSUInteger nested_input_count = step.feedValues.count;
            if (nested_input_count == 0U || nested_input_count > 3U
                    || step.output >= input_count || step.nestedExecutable == nil)
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            void *nested_inputs[3] = {NULL, NULL, NULL};
            for (NSUInteger input = 0U; input < nested_input_count; input++) {
                NSUInteger value = step.feedValues[input].unsignedIntegerValue;
                if (value >= input_count) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                nested_inputs[input] = input_buffers[value];
            }
            void *nested_output = input_buffers[step.output];
            int32_t nested_status = synaptik_metal_mpsgraph_executable_run(
                    (__bridge void *)step.nestedExecutable,
                    (uint32_t)nested_input_count,
                    nested_inputs,
                    1U,
                    &nested_output);
            if (nested_status != SYNAPTIK_METAL_STATUS_OK) return nested_status;
            cursor++;
            continue;
        }
        id<MTLCommandBuffer> command_buffer = [box.context.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
                command_buffer == nil ? nil : [command_buffer computeCommandEncoder];
        if (encoder == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        do {
            step = box.programSteps[cursor];
            if (step.pipeline == nil || step.metadata == nil
                    || step.firstInput >= input_count || step.output >= input_count)
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            [encoder setComputePipelineState:step.pipeline];
            SynaptikMetalBufferBox *first =
                    (__bridge SynaptikMetalBufferBox *)input_buffers[step.firstInput];
            SynaptikMetalBufferBox *output =
                    (__bridge SynaptikMetalBufferBox *)input_buffers[step.output];
            [encoder setBuffer:first.buffer offset:0U atIndex:0U];
            BOOL where = step.operation == SYNAPTIK_METAL_BOOL_WHERE;
            BOOL scatter = step.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
                    && step.stage == 1U;
            BOOL task0063 = (step.operation >= SYNAPTIK_METAL_CUSTOM_SORT
                            && step.operation <= SYNAPTIK_METAL_CUSTOM_TOP_K)
                    || (step.operation >= SYNAPTIK_METAL_CUSTOM_ARG_MAX
                            && step.operation <= SYNAPTIK_METAL_CUSTOM_ARG_MIN);
            BOOL compose = step.operation == SYNAPTIK_METAL_CUSTOM_CONCAT
                    || step.operation == SYNAPTIK_METAL_CUSTOM_STACK;
            BOOL binary = (step.operation >= SYNAPTIK_METAL_CUSTOM_GT
                            && step.operation <= SYNAPTIK_METAL_CUSTOM_TENSOR_MAX)
                    || step.operation == SYNAPTIK_METAL_BOOL_AND
                    || step.operation == SYNAPTIK_METAL_BOOL_OR
                    || step.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
                    || step.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
                    || step.operation == SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE
                    || step.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL;
            BOOL task0064_conv = step.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                    || step.operation == SYNAPTIK_METAL_CUSTOM_CONV3D;
            if (task0064_conv) {
                if (step.secondInput >= input_count)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                SynaptikMetalBufferBox *weight =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[step.secondInput];
                NSUInteger bias_value = step.auxiliaryInput;
                if (bias_value >= input_count) bias_value = step.firstInput;
                SynaptikMetalBufferBox *bias =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[bias_value];
                [encoder setBuffer:weight.buffer offset:0U atIndex:1U];
                [encoder setBuffer:bias.buffer offset:0U atIndex:2U];
                [encoder setBuffer:output.buffer offset:0U atIndex:3U];
                [encoder setBuffer:step.metadata offset:0U atIndex:4U];
            } else
            if (task0063) {
                NSUInteger required_outputs =
                        step.operation == SYNAPTIK_METAL_CUSTOM_TOP_K ? 2U : 1U;
                if (step.outputValues.count != required_outputs
                        || step.outputValues[0].unsignedIntegerValue != step.output)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                [encoder setBuffer:output.buffer offset:0U atIndex:1U];
                if (required_outputs == 2U) {
                    NSUInteger index_value =
                            step.outputValues[1].unsignedIntegerValue;
                    if (index_value >= input_count)
                        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                    SynaptikMetalBufferBox *indices =
                            (__bridge SynaptikMetalBufferBox *)input_buffers[index_value];
                    [encoder setBuffer:indices.buffer offset:0U atIndex:2U];
                    [encoder setBuffer:step.metadata offset:0U atIndex:3U];
                } else {
                    [encoder setBuffer:step.metadata offset:0U atIndex:2U];
                }
            } else if (compose) {
                if (step.inputValues.count == 0U || step.inputValues.count > 16U)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                for (NSUInteger slot = 0U; slot < 16U; slot++) {
                    NSUInteger source_slot =
                            MIN(slot, step.inputValues.count - 1U);
                    NSUInteger value =
                            step.inputValues[source_slot].unsignedIntegerValue;
                    if (value >= input_count)
                        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                    SynaptikMetalBufferBox *part =
                            (__bridge SynaptikMetalBufferBox *)input_buffers[value];
                    [encoder setBuffer:part.buffer offset:0U atIndex:slot];
                }
                [encoder setBuffer:output.buffer offset:0U atIndex:16U];
                [encoder setBuffer:step.metadata offset:0U atIndex:17U];
            } else if (scatter) {
                if (step.secondInput >= input_count || step.auxiliaryInput >= input_count)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                SynaptikMetalBufferBox *indices =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[step.secondInput];
                SynaptikMetalBufferBox *updates =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[step.auxiliaryInput];
                [encoder setBuffer:indices.buffer offset:0U atIndex:1U];
                [encoder setBuffer:updates.buffer offset:0U atIndex:2U];
                [encoder setBuffer:output.buffer offset:0U atIndex:3U];
                [encoder setBuffer:step.metadata offset:0U atIndex:4U];
            } else if (where) {
                if (step.secondInput >= input_count || step.auxiliaryInput >= input_count)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                SynaptikMetalBufferBox *second =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[step.secondInput];
                SynaptikMetalBufferBox *auxiliary =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[step.auxiliaryInput];
                [encoder setBuffer:second.buffer offset:0U atIndex:1U];
                [encoder setBuffer:auxiliary.buffer offset:0U atIndex:2U];
                [encoder setBuffer:output.buffer offset:0U atIndex:3U];
                [encoder setBuffer:step.metadata offset:0U atIndex:4U];
            } else if (binary) {
                if (step.secondInput >= input_count)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                SynaptikMetalBufferBox *second =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[step.secondInput];
                [encoder setBuffer:second.buffer offset:0U atIndex:1U];
                [encoder setBuffer:output.buffer offset:0U atIndex:2U];
                [encoder setBuffer:step.metadata offset:0U atIndex:3U];
            } else {
                [encoder setBuffer:output.buffer offset:0U atIndex:1U];
                [encoder setBuffer:step.metadata offset:0U atIndex:2U];
            }
            [encoder dispatchThreads:step.grid threadsPerThreadgroup:
                    MTLSizeMake(step.threadsPerThreadgroup, 1U, 1U)];
            cursor++;
        } while (cursor < box.programSteps.count
                && box.programSteps[cursor].custom
                && box.programSteps[cursor].indexValidation == nil);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted
                || command_buffer.error != nil)
            return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
    }
    return SYNAPTIK_METAL_STATUS_OK;
}

SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_executable_run(
        void *executable, uint32_t input_count, void *const *input_buffers,
        uint32_t output_count, void *const *output_buffers) {
    if (executable == NULL || input_count == 0U || output_count == 0U
            || input_buffers == NULL || output_buffers == NULL)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { @autoreleasepool {
        SynaptikMetalExecutableBox *box = (__bridge SynaptikMetalExecutableBox *)executable;
        if (box.customProgram)
            return run_custom_program_box(
                    box, input_count, input_buffers, output_count, output_buffers);
        if (input_count != box.feedShapes.count || output_count != box.targetShapes.count)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        NSMutableArray<SynaptikMetalBufferBox *> *input_boxes =
                [NSMutableArray arrayWithCapacity:input_count];
        NSMutableArray<SynaptikMetalBufferBox *> *output_boxes =
                [NSMutableArray arrayWithCapacity:output_count];
        if (input_boxes == nil || output_boxes == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        for (uint32_t input = 0; input < input_count; input++) {
            if (input_buffers[input] == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            SynaptikMetalBufferBox *buffer =
                    (__bridge SynaptikMetalBufferBox *)input_buffers[input];
            if (buffer.buffer.device != box.context.device
                    || buffer.logicalByteSize < box.feedBytes[input].unsignedLongLongValue)
                return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            [input_boxes addObject:buffer];
        }
        for (uint32_t output = 0; output < output_count; output++) {
            if (output_buffers[output] == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            for (uint32_t input = 0; input < input_count; input++)
                if (input_buffers[input] == output_buffers[output])
                    return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            SynaptikMetalBufferBox *buffer =
                    (__bridge SynaptikMetalBufferBox *)output_buffers[output];
            if (buffer.buffer.device != box.context.device
                    || buffer.logicalByteSize < box.targetBytes[output].unsignedLongLongValue)
                return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            [output_boxes addObject:buffer];
        }
        int32_t bool_status = validate_canonical_bool_inputs(
                box, input_count, input_buffers, box.feedBytes);
        if (bool_status != SYNAPTIK_METAL_STATUS_OK) return bool_status;
        for (SynaptikMetalIndexValidation *validation in box.indexValidations) {
            if (validation.feedPosition >= input_boxes.count)
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            int32_t index_status = validate_index_buffer(
                    validation, input_boxes[validation.feedPosition]);
            if (index_status != SYNAPTIK_METAL_STATUS_OK) return index_status;
        }
        NSMutableArray *stable_inputs = [NSMutableArray arrayWithCapacity:input_count];
        NSMutableArray *stable_outputs = [NSMutableArray arrayWithCapacity:output_count];
        if (stable_inputs == nil || stable_outputs == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        for (uint32_t input = 0; input < input_count; input++) {
            MPSDataType data_type =
                    (MPSDataType)box.feedDataTypes[input].unsignedIntegerValue;
            MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:input_boxes[input].buffer
                    shape:box.feedShapes[input]
                    dataType:data_type];
            if (data == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            [stable_inputs addObject:data];
        }
        for (uint32_t output = 0; output < output_count; output++) {
            MPSDataType data_type =
                    (MPSDataType)box.targetDataTypes[output].unsignedIntegerValue;
            MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:output_boxes[output].buffer
                    shape:box.targetShapes[output]
                    dataType:data_type];
            if (data == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            [stable_outputs addObject:data];
        }
        NSMutableArray *inputs = [NSMutableArray arrayWithCapacity:input_count];
        for (NSNumber *index in box.feedPermutation)
            [inputs addObject:stable_inputs[index.unsignedIntegerValue]];
        NSMutableArray *outputs = [NSMutableArray arrayWithCapacity:output_count];
        for (NSNumber *index in box.targetPermutation)
            [outputs addObject:stable_outputs[index.unsignedIntegerValue]];
        MPSGraphExecutableExecutionDescriptor *descriptor = [MPSGraphExecutableExecutionDescriptor new];
        if (inputs == nil || outputs == nil || descriptor == nil)
            return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        __block NSError *completion_error = nil;
        descriptor.waitUntilCompleted = YES;
        descriptor.completionHandler = ^(__unused NSArray<MPSGraphTensorData *> *results,
                NSError *error) { completion_error = error; };
        NSArray *results = [box.executable runWithMTLCommandQueue:box.context.commandQueue
                inputsArray:inputs resultsArray:outputs executionDescriptor:descriptor];
        if (completion_error != nil || results == nil || results.count != output_count)
            return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        for (id result in results) if (result == nil || result == NSNull.null)
            return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
        return SYNAPTIK_METAL_STATUS_OK;
    } } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_neg_kernel_pipeline_create(
        void *context, uint64_t element_count, void **out_pipeline) {
    if (out_pipeline == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_pipeline = NULL;
    if (context == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    if (element_count == 0U || element_count > UINT32_MAX
            || element_count > UINT64_MAX / sizeof(float)
            || element_count > (uint64_t)NSUIntegerMax
            || element_count * sizeof(float) > (uint64_t)NSUIntegerMax)
        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
    @try {
        @autoreleasepool {
            SynaptikMetalContextBox *ctx = (__bridge SynaptikMetalContextBox *)context;
            if (![ctx.device supportsFamily:MTLGPUFamilyApple4])
                return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            static NSString *const source =
                    @"#include <metal_stdlib>\n"
                     "using namespace metal;\n"
                     "kernel void synaptik_neg_f32(\n"
                     "        device const float *input [[buffer(0)]],\n"
                     "        device float *output [[buffer(1)]],\n"
                     "        uint index [[thread_position_in_grid]]) {\n"
                     "    output[index] = -input[index];\n"
                     "}\n";
            NSError *library_error = nil;
            id<MTLLibrary> library = [ctx.device newLibraryWithSource:source
                    options:nil error:&library_error];
            if (library == nil || library_error != nil)
                return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            id<MTLFunction> function = [library newFunctionWithName:@"synaptik_neg_f32"];
            if (function == nil) return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            NSError *pipeline_error = nil;
            id<MTLComputePipelineState> pipeline =
                    [ctx.device newComputePipelineStateWithFunction:function error:&pipeline_error];
            if (pipeline == nil || pipeline_error != nil)
                return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            NSUInteger execution_width = pipeline.threadExecutionWidth;
            NSUInteger maximum_width = pipeline.maxTotalThreadsPerThreadgroup;
            if (execution_width == 0U || maximum_width == 0U)
                return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            NSUInteger group_width = MIN(execution_width, maximum_width);
            MTLSize grid = MTLSizeMake((NSUInteger)element_count, 1U, 1U);
            MTLSize group = MTLSizeMake(group_width, 1U, 1U);
            if (grid.width != element_count || grid.height != 1U || grid.depth != 1U)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            if (group.width == 0U || group.width > maximum_width
                    || group.height != 1U || group.depth != 1U)
                return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            SynaptikMetalNegKernelPipelineBox *box =
                    [SynaptikMetalNegKernelPipelineBox new];
            if (box == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            box.pipeline = pipeline;
            box.context = ctx;
            box.elementCount = element_count;
            box.requiredBytes = element_count * sizeof(float);
            box.threadsPerThreadgroup = group_width;
            *out_pipeline = (__bridge_retained void *)box;
            return SYNAPTIK_METAL_STATUS_OK;
        }
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_neg_kernel_pipeline_release(void *pipeline) {
    if (pipeline == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { __unused id consumed = (__bridge_transfer id)pipeline;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}

SYNAPTIK_EXPORT int32_t synaptik_metal_neg_kernel_pipeline_run(
        void *pipeline, void *input_buffer, void *output_buffer) {
    if (pipeline == NULL || input_buffer == NULL || output_buffer == NULL)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    if (input_buffer == output_buffer) return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    @try {
        @autoreleasepool {
            SynaptikMetalNegKernelPipelineBox *box =
                    (__bridge SynaptikMetalNegKernelPipelineBox *)pipeline;
            SynaptikMetalBufferBox *input = (__bridge SynaptikMetalBufferBox *)input_buffer;
            SynaptikMetalBufferBox *output = (__bridge SynaptikMetalBufferBox *)output_buffer;
            if (input.buffer.device != box.context.device
                    || output.buffer.device != box.context.device
                    || input.logicalByteSize < box.requiredBytes
                    || output.logicalByteSize < box.requiredBytes)
                return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            if (box.elementCount == 0U || box.elementCount > UINT32_MAX
                    || box.threadsPerThreadgroup == 0U
                    || box.threadsPerThreadgroup > box.pipeline.maxTotalThreadsPerThreadgroup)
                return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            id<MTLCommandBuffer> command = [box.context.commandQueue commandBuffer];
            if (command == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (encoder == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            [encoder setComputePipelineState:box.pipeline];
            [encoder setBuffer:input.buffer offset:0U atIndex:0U];
            [encoder setBuffer:output.buffer offset:0U atIndex:1U];
            [encoder dispatchThreads:MTLSizeMake((NSUInteger)box.elementCount, 1U, 1U)
                    threadsPerThreadgroup:MTLSizeMake(box.threadsPerThreadgroup, 1U, 1U)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted || command.error != nil)
                return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            return SYNAPTIK_METAL_STATUS_OK;
        }
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
}
