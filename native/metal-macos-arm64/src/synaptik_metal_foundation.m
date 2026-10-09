#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import "synaptik_exact_kernels.h"
#import "SynaptikPointwiseFusionKernelSource.h"
#import "synaptik_task0059_data_kernels.h"
#import "synaptik_task0060_reduction_kernels.h"
#import "synaptik_task0061_matmul_kernels.h"
#import "synaptik_task0063_ordering_kernels.h"
#import "synaptik_task0064_convolution_pooling_kernels.h"
#import "synaptik_task0065_rng_dropout_kernels.h"
#import "synaptik_task0066_dtype_layout_kernels.h"
#import "synaptik_task0069_aggregate_kernels.h"
#import "synaptik_task0071_anchor_epilogue_kernels.h"
#import "synaptik_low_precision_kernels.h"
#import <CommonCrypto/CommonDigest.h>

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SYNAPTIK_EXPORT __attribute__((visibility("default")))
#define SYNAPTIK_MAX_RANK 16U
#define SYNAPTIK_MAX_NODE_INPUTS 16U
#define SYNAPTIK_MAX_NODE_OUTPUTS 5U
#define SYNAPTIK_MAX_ATTRIBUTE_WORDS 65U
#define SYNAPTIK_TASK0064_MAX_POOL_KERNEL_POSITIONS 65536U

#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
typedef void (*SynaptikMetalTestDispatchObserver)(
        uint32_t step_ordinal,
        uint32_t step_kind,
        const uint8_t manifest_digest[CC_SHA256_DIGEST_LENGTH],
        uint64_t grid_width,
        uint64_t grid_height,
        uint64_t grid_depth,
        uint64_t threadgroup_width,
        uint64_t metadata_bytes,
        uint64_t element_count,
        uint64_t point_grid_width,
        uint64_t point_grid_height,
        uint32_t scalar,
        uint32_t reserved);
extern void synaptik_metal_test_observe_dispatch(
        uint32_t step_ordinal,
        uint32_t step_kind,
        const uint8_t manifest_digest[CC_SHA256_DIGEST_LENGTH],
        uint64_t grid_width,
        uint64_t grid_height,
        uint64_t grid_depth,
        uint64_t threadgroup_width,
        uint64_t metadata_bytes,
        uint64_t element_count,
        uint64_t point_grid_width,
        uint64_t point_grid_height,
        uint32_t scalar,
        uint32_t reserved);
static SynaptikMetalTestDispatchObserver synaptik_metal_test_dispatch_observer =
        synaptik_metal_test_observe_dispatch;
#endif

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
    SYNAPTIK_METAL_CUSTOM_RESHAPE = 6U,
    SYNAPTIK_METAL_CUSTOM_EXPAND = 7U,
    SYNAPTIK_METAL_CUSTOM_PERMUTE = 8U,
    SYNAPTIK_METAL_CUSTOM_EXPAND_DIMS = 9U,
    SYNAPTIK_METAL_CUSTOM_SQUEEZE = 10U,
    SYNAPTIK_METAL_CUSTOM_CONTIGUOUS = 11U,
    SYNAPTIK_METAL_MPSGRAPH_ABS = 12U,
    SYNAPTIK_METAL_MPSGRAPH_SUM = 13U,
    SYNAPTIK_METAL_MPSGRAPH_MEAN = 14U,
    SYNAPTIK_METAL_MPSGRAPH_MATMUL = 15U,
    SYNAPTIK_METAL_CUSTOM_GATHER = 16U,
    SYNAPTIK_METAL_CUSTOM_ONE_HOT = 17U,
    SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS = 18U,
    SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS = 19U,
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
    SYNAPTIK_METAL_MPSGRAPH_EXP = 55U,
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
    SYNAPTIK_METAL_CUSTOM_SCATTER_ADD = 70U,
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
    SYNAPTIK_METAL_CUSTOM_DROPOUT = 101U,
    SYNAPTIK_METAL_CUSTOM_INITIAL_STATE = 102U,
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
    SYNAPTIK_METAL_CUSTOM_ATTR_DROPOUT = 33U,
    SYNAPTIK_METAL_CUSTOM_ATTR_GRAPH_RNG_STATE = 34U,
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
    SYNAPTIK_METAL_TYPE_INT64 = 6U,
    SYNAPTIK_METAL_TYPE_FLOAT16 = 7U
} SynaptikMetalValueType;
static BOOL task0066_is_carrier(uint8_t type) {
    return type >= SYNAPTIK_METAL_TYPE_FLOAT32
            && type <= SYNAPTIK_METAL_TYPE_FLOAT16;
}

static BOOL task0066_is_floating(uint8_t type) {
    return type == SYNAPTIK_METAL_TYPE_FLOAT64
            || type == SYNAPTIK_METAL_TYPE_FLOAT32
            || type == SYNAPTIK_METAL_TYPE_BFLOAT16
            || type == SYNAPTIK_METAL_TYPE_FLOAT16;
}

static uint8_t task0066_promote_floating(uint8_t left, uint8_t right) {
    if (!task0066_is_floating(left) || !task0066_is_floating(right))
        return SYNAPTIK_METAL_TYPE_UNAVAILABLE;
    if (left == SYNAPTIK_METAL_TYPE_FLOAT64
            || right == SYNAPTIK_METAL_TYPE_FLOAT64)
        return SYNAPTIK_METAL_TYPE_FLOAT64;
    if (left == SYNAPTIK_METAL_TYPE_FLOAT32
            || right == SYNAPTIK_METAL_TYPE_FLOAT32
            || left != right)
        return SYNAPTIK_METAL_TYPE_FLOAT32;
    return left;
}


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

static BOOL synaptik_sha256_string(NSString *source, uint8_t digest[CC_SHA256_DIGEST_LENGTH]) {
    if (source == nil || digest == NULL) return NO;
    NSData *bytes = [source dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    if (bytes == nil || bytes.length > UINT32_MAX) return NO;
    CC_SHA256(bytes.bytes, (CC_LONG)bytes.length, digest);
    return YES;
}

static uint8_t synaptik_hex_nibble(char digit) {
    if (digit >= '0' && digit <= '9') return (uint8_t)(digit - '0');
    if (digit >= 'a' && digit <= 'f') return (uint8_t)(digit - 'a' + 10);
    return UINT8_MAX;
}

static BOOL synaptik_digest_matches_hex(
        const uint8_t digest[CC_SHA256_DIGEST_LENGTH],
        const char expected[CC_SHA256_DIGEST_LENGTH * 2U + 1U]) {
    if (digest == NULL || expected == NULL
            || strlen(expected) != CC_SHA256_DIGEST_LENGTH * 2U)
        return NO;
    for (uint32_t index = 0U; index < CC_SHA256_DIGEST_LENGTH; index++) {
        uint8_t high = synaptik_hex_nibble(expected[index * 2U]);
        uint8_t low = synaptik_hex_nibble(expected[index * 2U + 1U]);
        if (high == UINT8_MAX || low == UINT8_MAX
                || digest[index] != (uint8_t)((high << 4U) | low))
            return NO;
    }
    return YES;
}


static BOOL synaptik_source_digest_matches(NSString *source, const char *expected) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    return synaptik_sha256_string(source, digest)
            && synaptik_digest_matches_hex(digest, expected);
}

static NSString *synaptik_authenticated_fixed_source(void) {
    if (!synaptik_source_digest_matches(
                SynaptikExactKernelSource,
                "650af9b349f1559250471f2c53c8195aebe2d77e642dfb4d11d89683fb14c1c7")
            || !synaptik_source_digest_matches(
                SynaptikTask0059DataKernelSource,
                "4316a3ff46d640ca8813afd070c1d3746bd832b28afc847f556925e5d3bf44ac")
            || !synaptik_source_digest_matches(
                SynaptikTask0060ReductionKernelSource,
                "aa5e6b524926b64058aecc90c0b93f5e571ccd81e1e362459252d8a2b5b65c7f")
            || !synaptik_source_digest_matches(
                SynaptikTask0061MatmulKernelSource,
                "4cd012516868ca5c8eead1ebdf4b63de13f702d9526894ef5da00548dbd0732c")
            || !synaptik_source_digest_matches(
                SynaptikTask0063OrderingKernelSource,
                "a54fa5431ed23e88ea7b2a118c631492218e14fe1ce932de9ca8652f2b2bf65e")
            || !synaptik_source_digest_matches(
                SynaptikTask0064ConvolutionPoolingKernelSource,
                "282cb823f95499e63f66674221c44bfbf77c9fa51230995eb3518ea130537cbe")
            || !synaptik_source_digest_matches(
                SynaptikTask0069AggregateKernelSource,
                "6791792854f011172b98c9998fe67717c83ab13eb771d8b967cc64cf6f9548ca")
            || !synaptik_source_digest_matches(
                SynaptikTask0065RngDropoutKernelSource,
                "8217948f0c28d3df56c5f2edb5408306b588bf2c5e0b1f90a23463e295ac6b99")
            || !synaptik_source_digest_matches(
                SynaptikTask0066DtypeLayoutKernelSource,
                "630b6401079438a773aca080e0ee039d33006ef861264bb3f14549e69a50960c")
            || !synaptik_source_digest_matches(
                SynaptikTask0071AnchorEpilogueKernelSource,
                "e165612540f551ce547c145dde4fa48e46ff06e652b4729272b8bfabc75f0663"))
        return nil;
    NSString *fixed = [[[[[[SynaptikExactKernelSource
            stringByAppendingString:SynaptikTask0059DataKernelSource]
            stringByAppendingString:SynaptikTask0060ReductionKernelSource]
            stringByAppendingString:SynaptikTask0061MatmulKernelSource]
            stringByAppendingString:SynaptikTask0063OrderingKernelSource]
            stringByAppendingString:SynaptikTask0064ConvolutionPoolingKernelSource]
            stringByAppendingString:SynaptikTask0069AggregateKernelSource];
    fixed = [[[fixed stringByAppendingString:SynaptikTask0065RngDropoutKernelSource]
            stringByAppendingString:SynaptikTask0066DtypeLayoutKernelSource]
            stringByAppendingString:SynaptikTask0071AnchorEpilogueKernelSource];
    return [fixed lengthOfBytesUsingEncoding:NSUTF8StringEncoding] == 84603U
            && synaptik_source_digest_matches(
                    fixed,
                    "eb9a4bc2e9156bd971c59a3f8c1416df61627fe1c8e5655ccff28ecdf28b39f5")
            ? fixed : nil;
}

static NSString *synaptik_authenticated_low_precision_source(void) {
    return [SynaptikLowPrecisionKernelSource
                            lengthOfBytesUsingEncoding:NSUTF8StringEncoding] == 35407U
                    && synaptik_source_digest_matches(
                            SynaptikLowPrecisionKernelSource,
                            "9505b41b20a20a5f7ff2c9f056957039a83f4d81a5fa31568629fbb954e173ab")
            ? SynaptikLowPrecisionKernelSource : nil;
}

static BOOL synaptik_assembled_source_is_authentic(
        NSString *fixed,
        NSString *generated,
        NSString *assembled) {
    if (fixed == nil || generated == nil || assembled == nil) return NO;
    uint8_t generated_digest[CC_SHA256_DIGEST_LENGTH];
    uint8_t assembled_digest[CC_SHA256_DIGEST_LENGTH];
    if (!synaptik_sha256_string(generated, generated_digest)
            || !synaptik_sha256_string(assembled, assembled_digest))
        return NO;
    NSData *fixed_bytes =
            [fixed dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    NSData *generated_bytes =
            [generated dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    if (fixed_bytes == nil || generated_bytes == nil
            || fixed_bytes.length > NSUIntegerMax - generated_bytes.length)
        return NO;
    NSMutableData *independent =
            [NSMutableData dataWithCapacity:fixed_bytes.length + generated_bytes.length];
    if (independent == nil) return NO;
    [independent appendData:fixed_bytes];
    [independent appendData:generated_bytes];
    uint8_t independent_digest[CC_SHA256_DIGEST_LENGTH];
    if (independent.length > UINT32_MAX) return NO;
    CC_SHA256(independent.bytes, (CC_LONG)independent.length, independent_digest);
    return memcmp(
            assembled_digest,
            independent_digest,
            CC_SHA256_DIGEST_LENGTH) == 0;
}

typedef struct {
    uint32_t step;
    uint32_t argument;
    uint32_t access;
    uint32_t slot;
    uint32_t value;
    uint32_t reserved;
} SynaptikPointwiseBindingRecord;

typedef struct {
    uint32_t step_count;
    const SynaptikPointwiseStepRecord *steps;
    uint32_t member_count;
    const uint32_t *members;
    uint32_t binding_count;
    const SynaptikPointwiseBindingRecord *bindings;
    uint32_t materialized_count;
    const uint32_t *materialized_values;
    const uint32_t *program_to_slot;
    uint32_t instruction_count;
    const SynaptikPointwiseInstructionRecord *instructions;
    uint32_t generated_unit_count;
    uint32_t expected_generated_bytes;
    uint32_t expected_total_bytes;
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
    const uint8_t *manifest_digest;
#endif
} SynaptikPointwisePlan;

static BOOL synaptik_pointwise_node_is_eligible(
        const uint8_t *program,
        uint64_t values_offset,
        const SynaptikMetalDecodedNode *nodes,
        uint32_t node,
        const uint32_t *ranks,
        const uint64_t *dimensions,
        const uint8_t *states,
        const uint8_t *types) {
    SynaptikMetalDecodedNode record = nodes[node];
    if (record.operation < SYNAPTIK_METAL_CUSTOM_FLOOR
            || record.operation > SYNAPTIK_METAL_CUSTOM_RELU
            || record.input_count != 1U || record.output_count != 1U
            || record.attribute_kind != 0U || record.attribute_count != 0U)
        return NO;
    uint32_t input = record.first_input;
    uint32_t output = record.output;
    if (ranks[input] < 1U || ranks[input] > SYNAPTIK_MAX_RANK
            || ranks[output] < 1U || ranks[output] > SYNAPTIK_MAX_RANK)
        return NO;
    uint64_t element_count = 1U;
    for (uint32_t axis = 0U; axis < ranks[input]; axis++) {
        uint64_t dimension =
                dimensions[(size_t)input * SYNAPTIK_MAX_RANK + axis];
        if (dimension == 0U || element_count > UINT32_MAX / dimension) return NO;
        element_count *= dimension;
    }
    if (element_count < 1U || element_count > UINT32_MAX) return NO;
    uint32_t input_flags =
            synaptik_read_le32(program + values_offset + (uint64_t)input * 40U + 16U);
    uint32_t output_flags =
            synaptik_read_le32(program + values_offset + (uint64_t)output * 40U + 16U);
    return types[input] == SYNAPTIK_METAL_TYPE_FLOAT32
            && types[output] == SYNAPTIK_METAL_TYPE_FLOAT32
            && states[input] == SYNAPTIK_METAL_VALUE_CANONICAL
            && states[output] == SYNAPTIK_METAL_VALUE_CANONICAL
            && (input_flags & 1U) == (output_flags & 1U)
            && ranks[input] == ranks[output]
            && memcmp(
                    dimensions + (size_t)input * SYNAPTIK_MAX_RANK,
                    dimensions + (size_t)output * SYNAPTIK_MAX_RANK,
                    (size_t)ranks[input] * sizeof(uint64_t)) == 0;
}

static BOOL synaptik_anchor_same_shape(
        uint32_t left,
        uint32_t right,
        const uint32_t *ranks,
        const uint64_t *dimensions) {
    return ranks[left] == ranks[right]
            && memcmp(
                    dimensions + (size_t)left * SYNAPTIK_MAX_RANK,
                    dimensions + (size_t)right * SYNAPTIK_MAX_RANK,
                    (size_t)ranks[left] * sizeof(uint64_t)) == 0;
}

static BOOL synaptik_anchor_broadcasts_to(
        uint32_t source,
        uint32_t target,
        const uint32_t *ranks,
        const uint64_t *dimensions) {
    if (ranks[source] > ranks[target]) return NO;
    uint32_t shift = ranks[target] - ranks[source];
    for (uint32_t axis = 0U; axis < ranks[source]; axis++) {
        uint64_t source_extent =
                dimensions[(size_t)source * SYNAPTIK_MAX_RANK + axis];
        uint64_t target_extent =
                dimensions[(size_t)target * SYNAPTIK_MAX_RANK + shift + axis];
        if (source_extent != 1U && source_extent != target_extent) return NO;
    }
    return YES;
}
static BOOL synaptik_matmul_dimensions_match(
        uint32_t left_rank,
        const uint64_t *left,
        uint32_t right_rank,
        const uint64_t *right,
        uint32_t output_rank,
        const uint64_t *output) {
    if (left == NULL || right == NULL || output == NULL
            || left_rank == 0U || right_rank == 0U
            || left_rank > SYNAPTIK_MAX_RANK || right_rank > SYNAPTIK_MAX_RANK
            || output_rank > SYNAPTIK_MAX_RANK)
        return NO;
    uint32_t left_batch = left_rank > 1U ? left_rank - 2U : 0U;
    uint32_t right_batch = right_rank > 1U ? right_rank - 2U : 0U;
    uint32_t batch_rank = MAX(left_batch, right_batch);
    uint32_t expected_rank = batch_rank
            + (left_rank > 1U ? 1U : 0U)
            + (right_rank > 1U ? 1U : 0U);
    if (output_rank != expected_rank
            || left[left_rank - 1U]
                    != right[right_rank == 1U ? 0U : right_rank - 2U])
        return NO;
    for (uint32_t axis = 0U; axis < batch_rank; axis++) {
        int32_t left_axis =
                (int32_t)axis - (int32_t)(batch_rank - left_batch);
        int32_t right_axis =
                (int32_t)axis - (int32_t)(batch_rank - right_batch);
        uint64_t left_extent =
                left_axis < 0 ? 1U : left[(uint32_t)left_axis];
        uint64_t right_extent =
                right_axis < 0 ? 1U : right[(uint32_t)right_axis];
        if ((left_extent != right_extent
                        && left_extent != 1U && right_extent != 1U)
                || output[axis] != MAX(left_extent, right_extent))
            return NO;
    }
    uint32_t output_axis = batch_rank;
    if (left_rank > 1U
            && output[output_axis++] != left[left_rank - 2U])
        return NO;
    return right_rank == 1U
            || output[output_axis] == right[right_rank - 1U];
}


static uint32_t synaptik_anchor_consumer_count(
        uint32_t value,
        uint32_t node_count,
        const SynaptikMetalDecodedNode *nodes) {
    uint32_t count = 0U;
    for (uint32_t node = 0U; node < node_count; node++)
        for (uint32_t input = 0U; input < nodes[node].input_count; input++)
            if (nodes[node].inputs[input] == value) count++;
    return count;
}

static BOOL synaptik_validate_anchor_step(
        const uint8_t *program,
        uint64_t values_offset,
        uint32_t node_count,
        const SynaptikMetalDecodedNode *nodes,
        const uint32_t *ranks,
        const uint64_t *dimensions,
        const uint8_t *states,
        const uint8_t *types,
        const uint8_t *targeted,
        const uint32_t *program_to_slot,
        uint32_t ordinal,
        SynaptikPointwiseStepRecord step,
        const uint32_t *members,
        const SynaptikPointwiseBindingRecord *bindings,
        const SynaptikPointwiseInstructionRecord *instructions) {
    if (step.flags < 1U || step.flags > 2U
            || step.member_count < 2U || step.member_count > 4U
            || step.instruction_count != step.member_count - 1U
            || step.function_bytes != 0U)
        return NO;
    SynaptikMetalDecodedNode anchor = nodes[members[step.member_start]];
    BOOL matmul = step.flags == 1U;
    if ((matmul && (anchor.operation != SYNAPTIK_METAL_MPSGRAPH_MATMUL
                    || anchor.input_count != 2U || anchor.output_count != 1U
                    || anchor.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE
                    || anchor.attribute_count != 0U))
            || (!matmul && (anchor.operation != SYNAPTIK_METAL_CUSTOM_CONV2D
                    || anchor.input_count < 2U || anchor.input_count > 3U
                    || anchor.output_count != 1U
                    || anchor.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_CONV_2D)))
        return NO;
    uint32_t carrier = anchor.output;
    if (types[carrier] != SYNAPTIK_METAL_TYPE_FLOAT32
            || states[carrier] != SYNAPTIK_METAL_VALUE_CANONICAL
            || targeted[carrier]
            || synaptik_anchor_consumer_count(carrier, node_count, nodes) != 1U
            || program_to_slot[carrier] != UINT32_MAX)
        return NO;
    BOOL anchor_gradient =
            (synaptik_read_le32(program + values_offset + (uint64_t)carrier * 40U + 16U)
                    & 1U) != 0U;
    if (matmul) {
        uint32_t left = anchor.first_input;
        uint32_t right = anchor.second_input;
        BOOL mixed = types[left] != types[right];
        BOOL carrier_types =
                (types[left] == SYNAPTIK_METAL_TYPE_FLOAT32
                        && types[right] == SYNAPTIK_METAL_TYPE_FLOAT32)
                || (types[left] == SYNAPTIK_METAL_TYPE_FLOAT32
                        && types[right] == SYNAPTIK_METAL_TYPE_BFLOAT16)
                || (types[left] == SYNAPTIK_METAL_TYPE_BFLOAT16
                        && types[right] == SYNAPTIK_METAL_TYPE_FLOAT32);
        BOOL left_gradient =
                (synaptik_read_le32(program + values_offset
                        + (uint64_t)left * 40U + 16U) & 1U) != 0U;
        BOOL right_gradient =
                (synaptik_read_le32(program + values_offset
                        + (uint64_t)right * 40U + 16U) & 1U) != 0U;
        if (!carrier_types
                || states[left] != SYNAPTIK_METAL_VALUE_CANONICAL
                || states[right] != SYNAPTIK_METAL_VALUE_CANONICAL
                || !synaptik_matmul_dimensions_match(
                        ranks[left],
                        &dimensions[(size_t)left * SYNAPTIK_MAX_RANK],
                        ranks[right],
                        &dimensions[(size_t)right * SYNAPTIK_MAX_RANK],
                        ranks[carrier],
                        &dimensions[(size_t)carrier * SYNAPTIK_MAX_RANK])
                || (mixed
                        ? left_gradient || right_gradient || anchor_gradient
                        : anchor_gradient != (left_gradient || right_gradient)))
            return NO;
    } else {
        BOOL any_float32 = NO;
        BOOL any_bfloat16 = NO;
        BOOL any_gradient = anchor_gradient;
        for (uint32_t input = 0U; input < anchor.input_count; input++) {
            uint32_t value = anchor.inputs[input];
            if ((types[value] != SYNAPTIK_METAL_TYPE_FLOAT32
                            && types[value] != SYNAPTIK_METAL_TYPE_BFLOAT16)
                    || states[value] != SYNAPTIK_METAL_VALUE_CANONICAL)
                return NO;
            any_float32 |= types[value] == SYNAPTIK_METAL_TYPE_FLOAT32;
            any_bfloat16 |= types[value] == SYNAPTIK_METAL_TYPE_BFLOAT16;
            any_gradient |= (synaptik_read_le32(
                    program + values_offset + (uint64_t)value * 40U + 16U) & 1U) != 0U;
        }
        if (!any_float32 || (any_bfloat16 && any_gradient)) return NO;
    }

    BOOL saw_scalar = NO;
    BOOL saw_add = NO;
    BOOL saw_terminal = NO;
    uint32_t external = UINT32_MAX;
    for (uint32_t relative = 1U; relative < step.member_count; relative++) {
        SynaptikMetalDecodedNode node =
                nodes[members[step.member_start + relative]];
        SynaptikPointwiseInstructionRecord instruction =
                instructions[step.instruction_start + relative - 1U];
        if (instruction.step != ordinal
                || instruction.relative_node != relative
                || instruction.semantic_site != 2U
                || instruction.input2 != UINT32_MAX
                || instruction.output != node.output
                || saw_terminal)
            return NO;
        uint32_t expected_opcode = 0U;
        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL) {
            if (!matmul || saw_scalar || saw_add || relative != 1U
                    || node.input_count != 1U || node.output_count != 1U
                    || node.first_input != carrier
                    || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_SCALAR_VALUE
                    || node.attribute_count != 1U
                    || instruction.input_count != 1U
                    || instruction.input0 != carrier
                    || instruction.input1 != UINT32_MAX
                    || instruction.immediate_count != 1U
                    || instruction.immediate_type0 != 1U
                    || instruction.immediate_type1 != 0U
                    || instruction.raw0 != node.attribute_values[0]
                    || instruction.raw1 != 0U)
                return NO;
            saw_scalar = YES;
            expected_opcode = 1U;
        } else if (node.operation == SYNAPTIK_METAL_MPSGRAPH_ADD) {
            if (saw_add || node.input_count != 2U || node.output_count != 1U
                    || node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE
                    || node.attribute_count != 0U
                    || ((node.first_input == carrier) == (node.second_input == carrier))
                    || instruction.input_count != 2U
                    || instruction.input0 != node.first_input
                    || instruction.input1 != node.second_input
                    || instruction.immediate_count != 0U
                    || instruction.immediate_type0 != 0U
                    || instruction.immediate_type1 != 0U
                    || instruction.raw0 > 1U)
                return NO;
            external = node.first_input == carrier
                    ? node.second_input : node.first_input;
            BOOL external_on_left = node.first_input == external;
            if (instruction.raw0 != (external_on_left ? 1U : 0U)
                    || instruction.raw1 != external
                    || external == anchor.first_input
                    || external == anchor.second_input
                    || (anchor.input_count == 3U && external == anchor.auxiliary)
                    || types[external] != SYNAPTIK_METAL_TYPE_FLOAT32
                    || states[external] != SYNAPTIK_METAL_VALUE_CANONICAL
                    || !synaptik_anchor_broadcasts_to(
                            external, carrier, ranks, dimensions))
                return NO;
            BOOL carrier_grad =
                    (synaptik_read_le32(program + values_offset
                            + (uint64_t)carrier * 40U + 16U) & 1U) != 0U;
            BOOL external_grad =
                    (synaptik_read_le32(program + values_offset
                            + (uint64_t)external * 40U + 16U) & 1U) != 0U;
            BOOL output_grad =
                    (synaptik_read_le32(program + values_offset
                            + (uint64_t)node.output * 40U + 16U) & 1U) != 0U;
            if (carrier_grad != external_grad || carrier_grad != output_grad)
                return NO;
            saw_add = YES;
            expected_opcode = 2U;
        } else if (node.operation == SYNAPTIK_METAL_CUSTOM_RELU) {
            if (node.input_count != 1U || node.output_count != 1U
                    || node.first_input != carrier
                    || node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE
                    || node.attribute_count != 0U
                    || instruction.input_count != 1U
                    || instruction.input0 != carrier
                    || instruction.input1 != UINT32_MAX
                    || instruction.immediate_count != 0U
                    || instruction.immediate_type0 != 0U
                    || instruction.immediate_type1 != 0U
                    || instruction.raw0 != 0U || instruction.raw1 != 0U)
                return NO;
            saw_terminal = YES;
            expected_opcode = 3U;
        } else if (node.operation == SYNAPTIK_METAL_CUSTOM_CLAMP) {
            if (node.input_count != 1U || node.output_count != 1U
                    || node.first_input != carrier
                    || node.attribute_kind != SYNAPTIK_METAL_CUSTOM_ATTR_CLAMP_RANGE
                    || node.attribute_count != 2U
                    || instruction.input_count != 1U
                    || instruction.input0 != carrier
                    || instruction.input1 != UINT32_MAX
                    || instruction.immediate_count != 2U
                    || instruction.immediate_type0 != 1U
                    || instruction.immediate_type1 != 1U
                    || instruction.raw0 != node.attribute_values[0]
                    || instruction.raw1 != node.attribute_values[1])
                return NO;
            saw_terminal = YES;
            expected_opcode = 4U;
        } else {
            return NO;
        }
        if (instruction.opcode != expected_opcode
                || types[node.output] != SYNAPTIK_METAL_TYPE_FLOAT32
                || states[node.output] != SYNAPTIK_METAL_VALUE_CANONICAL
                || !synaptik_anchor_same_shape(
                        carrier, node.output, ranks, dimensions))
            return NO;
        BOOL carrier_grad =
                (synaptik_read_le32(program + values_offset
                        + (uint64_t)carrier * 40U + 16U) & 1U) != 0U;
        BOOL output_grad =
                (synaptik_read_le32(program + values_offset
                        + (uint64_t)node.output * 40U + 16U) & 1U) != 0U;
        if ((node.operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL
                    || node.operation == SYNAPTIK_METAL_CUSTOM_CLAMP)
                    ? carrier_grad || output_grad
                    : carrier_grad != output_grad)
            return NO;
        if (relative + 1U < step.member_count
                && (targeted[node.output]
                        || synaptik_anchor_consumer_count(
                                node.output, node_count, nodes) != 1U
                        || program_to_slot[node.output] != UINT32_MAX))
            return NO;
        carrier = node.output;
    }
    if (program_to_slot[carrier] == UINT32_MAX) return NO;
    uint32_t expected_bindings =
            anchor.input_count + (external == UINT32_MAX ? 0U : 1U) + 1U;
    if (step.binding_count != expected_bindings) return NO;
    for (uint32_t argument = 0U; argument < expected_bindings; argument++) {
        uint32_t expected_value;
        uint32_t expected_access = 1U;
        if (argument < anchor.input_count) {
            expected_value = anchor.inputs[argument];
        } else if (external != UINT32_MAX && argument == anchor.input_count) {
            expected_value = external;
        } else {
            expected_value = carrier;
            expected_access = 2U;
        }
        SynaptikPointwiseBindingRecord binding =
                bindings[step.binding_start + argument];
        if (binding.step != ordinal || binding.argument != argument
                || binding.access != expected_access
                || binding.value != expected_value)
            return NO;
    }
    return YES;
}

static BOOL synaptik_validate_pointwise_cap_stop(
        const uint8_t *program,
        uint64_t values_offset,
        uint32_t value_count,
        uint32_t node_count,
        const SynaptikMetalDecodedNode *nodes,
        const uint32_t *ranks,
        const uint64_t *dimensions,
        const uint8_t *states,
        const uint8_t *types,
        uint32_t target_count,
        const uint32_t *targets,
        const SynaptikPointwisePlan *fusion,
        BOOL contains_low_precision,
        uint32_t rejected_node,
        uint32_t cap_reason) {
    if (program == NULL || nodes == NULL || ranks == NULL || dimensions == NULL
            || states == NULL || types == NULL || targets == NULL || fusion == NULL)
        return NO;
    if (contains_low_precision) {
        // A low-containing partition keeps the serialized extension, but every node is
        // one fixed step. Topology validation makes every value a feed or node output;
        // singleton fixed steps therefore materialize every value. Recompute this shape
        // rather than trusting its digest or counts. create_decoded checks each step kind
        // against the corresponding node's native custom-kernel classification.
        if (fusion->step_count != node_count || fusion->member_count != node_count
                || fusion->materialized_count != value_count
                || fusion->instruction_count != 0U || fusion->generated_unit_count != 0U
                || fusion->expected_generated_bytes != 0U
                || rejected_node != UINT32_MAX || cap_reason != 0U)
            return NO;
        for (uint32_t value = 0U; value < value_count; value++)
            if (fusion->materialized_values[value] != value
                    || fusion->program_to_slot[value] != value)
                return NO;
        for (uint32_t ordinal = 0U; ordinal < node_count; ordinal++) {
            SynaptikPointwiseStepRecord step = fusion->steps[ordinal];
            if ((step.kind != 1U && step.kind != 2U)
                    || step.member_start != ordinal || step.member_count != 1U
                    || fusion->members[ordinal] != ordinal
                    || step.instruction_start != 0U || step.instruction_count != 0U
                    || step.function_bytes != 0U || step.flags != 0U)
                return NO;
        }
        return YES;
    }
    NSMutableData *consumer_data =
            [NSMutableData dataWithLength:(NSUInteger)value_count * sizeof(uint32_t)];
    NSMutableData *target_data = [NSMutableData dataWithLength:value_count];
    NSMutableData *node_step_data =
            [NSMutableData dataWithLength:(NSUInteger)node_count * sizeof(uint32_t)];
    if (consumer_data == nil || target_data == nil || node_step_data == nil) return NO;
    uint32_t *consumers = consumer_data.mutableBytes;
    uint8_t *is_target = target_data.mutableBytes;
    uint32_t *node_step = node_step_data.mutableBytes;
    memset(node_step, 0xff, (size_t)node_count * sizeof(uint32_t));
    for (uint32_t node = 0U; node < node_count; node++) {
        SynaptikMetalDecodedNode record = nodes[node];
        for (uint32_t input = 0U; input < record.input_count; input++) {
            uint32_t value = record.inputs[input];
            if (consumers[value] == UINT32_MAX) return NO;
            consumers[value]++;
        }
    }
    for (uint32_t target = 0U; target < target_count; target++)
        is_target[targets[target]] = 1U;
    for (uint32_t step = 0U; step < fusion->step_count; step++) {
        SynaptikPointwiseStepRecord record = fusion->steps[step];
        for (uint32_t relative = 0U; relative < record.member_count; relative++) {
            uint32_t member = fusion->members[record.member_start + relative];
            if (member >= node_count || node_step[member] != UINT32_MAX) return NO;
            node_step[member] = step;
        }
    }

    uint32_t expected_units = 0U;
    uint32_t expected_instructions = 0U;
    uint32_t expected_generated_bytes = 0U;
    uint32_t expected_rejected = UINT32_MAX;
    uint32_t expected_reason = 0U;
    uint32_t projected_node = 0U;
    uint32_t projected_step = 0U;
    BOOL stopped = NO;
    uint32_t position = 0U;
    while (position < node_count) {
        uint32_t position_step = node_step[position];
        if (position_step >= fusion->step_count) return NO;
        if (fusion->steps[position_step].kind == SYNAPTIK_ANCHOR_EPILOGUE_STEP) {
            position++;
            continue;
        }
        if (!synaptik_pointwise_node_is_eligible(
                    program, values_offset, nodes, position, ranks, dimensions, states, types)) {
            position++;
            continue;
        }
        uint32_t end = position + 1U;
        while (end < node_count
                && synaptik_pointwise_node_is_eligible(
                        program, values_offset, nodes, end, ranks, dimensions, states, types)
                && nodes[end - 1U].output == nodes[end].first_input
                && consumers[nodes[end - 1U].output] == 1U
                && is_target[nodes[end - 1U].output] == 0U)
            end++;
        uint32_t cursor = position;
        while (end - cursor >= 2U) {
            uint32_t remaining = end - cursor;
            uint32_t length = remaining == 9U ? 7U : remaining > 8U ? 8U : remaining;
            while (projected_node < cursor) {
                if (projected_step >= fusion->step_count
                        || node_step[projected_node] != projected_step)
                    return NO;
                SynaptikPointwiseStepRecord preceding = fusion->steps[projected_step];
                if (preceding.member_count == 0U
                        || preceding.member_start > fusion->member_count
                        || preceding.member_count
                                > fusion->member_count - preceding.member_start
                        || preceding.member_count > cursor - projected_node)
                    return NO;
                for (uint32_t relative = 0U;
                        relative < preceding.member_count;
                        relative++)
                    if (fusion->members[preceding.member_start + relative]
                            != projected_node + relative)
                        return NO;
                projected_node += preceding.member_count;
                projected_step++;
            }
            if (projected_node != cursor) return NO;
            SynaptikPointwiseInstructionRecord candidate[8] = {0};
            for (uint32_t relative = 0U; relative < length; relative++)
                candidate[relative].opcode = nodes[cursor + relative].operation - 59U;
            uint32_t function_bytes =
                    SynaptikPointwiseFunctionBytes(projected_step, candidate, length);
            uint32_t separator = expected_units == 0U
                    ? SYNAPTIK_POINTWISE_GENERATED_PREAMBLE_BYTES : 0U;
            SynaptikPointwiseCapProjection projection = {
                .units = (uint64_t)expected_units + 1U,
                .instructions = (uint64_t)expected_instructions + length,
                .function_bytes = function_bytes == 0U
                        ? UINT64_MAX : function_bytes,
                .generated_bytes = (uint64_t)expected_generated_bytes
                        + separator + function_bytes,
            };
            uint32_t reason = !stopped
                    ? SynaptikPointwiseCapReason(
                            SynaptikPointwiseProductionCapLimits(), projection)
                    : 0U;
            if (!stopped && reason != 0U) {
                stopped = YES;
                expected_rejected = cursor;
                expected_reason = reason;
            }
            if (!stopped) {
                uint32_t step = node_step[cursor];
                if (step != projected_step || step >= fusion->step_count)
                    return NO;
                SynaptikPointwiseStepRecord actual = fusion->steps[step];
                if (actual.kind != SYNAPTIK_POINTWISE_GENERATED_STEP
                        || actual.member_count != length
                        || actual.function_bytes != function_bytes)
                    return NO;
                expected_units++;
                expected_instructions += length;
                expected_generated_bytes += separator + function_bytes;
                projected_node = cursor + length;
                projected_step++;
            } else {
                for (uint32_t relative = 0U; relative < length; relative++) {
                    uint32_t step = node_step[cursor + relative];
                    if (step >= fusion->step_count
                            || fusion->steps[step].kind
                                    == SYNAPTIK_POINTWISE_GENERATED_STEP)
                        return NO;
                }
            }
            cursor += length;
        }
        position = end;
    }
    uint32_t actual_pointwise_instructions = 0U;
    for (uint32_t step = 0U; step < fusion->step_count; step++)
        if (fusion->steps[step].kind == SYNAPTIK_POINTWISE_GENERATED_STEP)
            actual_pointwise_instructions += fusion->steps[step].instruction_count;
    return expected_units == fusion->generated_unit_count
            && expected_instructions == actual_pointwise_instructions
            && expected_generated_bytes == fusion->expected_generated_bytes
            && expected_rejected == rejected_node
            && expected_reason == cap_reason;
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
    if (type < 1U || type > 7U) return NO;
    if ((type == 1U || type == 2U) && bits > UINT32_MAX) return NO;
    if (type == 3U && bits > 1U) return NO;
    return (type != 5U && type != 7U) || bits <= UINT16_MAX;
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
        return synaptik_shape_is_valid(words, count, &cursor, NO) && cursor == count;
    }
    if (kind == 2U) {
        if (count < 1U) return NO;
        uint64_t axes = synaptik_attribute_word(words, 0U);
        return axes <= SYNAPTIK_MAX_RANK && count == axes + 1U
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
                && synaptik_word_is_positive(synaptik_attribute_word(words, 2U));
    if (kind == 7U) {
        uint64_t type = synaptik_attribute_word(words, 0U);
        return count == 2U && synaptik_scalar_is_valid(words, count, 0U)
                && (operation > 34U || type == 1U || type == 5U || type == 7U);
    }
    if (kind == 8U) {
        uint64_t lower_type = synaptik_attribute_word(words, 0U);
        return count == 4U && synaptik_scalar_is_valid(words, count, 0U)
                && synaptik_scalar_is_valid(words, count, 2U)
                && lower_type == synaptik_attribute_word(words, 2U)
                && (lower_type == 1U || lower_type == 5U || lower_type == 7U);
    }
    if (kind == 9U)
        return count == 3U && synaptik_attribute_word(words, 0U) < SYNAPTIK_MAX_RANK
                && synaptik_attribute_word(words, 1U) <= 1U
                && synaptik_attribute_word(words, 2U) <= 1U;
    if (kind == 10U)
        return count == 1U && synaptik_attribute_word(words, 0U) >= 1U
                && synaptik_attribute_word(words, 0U) <= 7U;
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
@property(nonatomic) uint32_t indexValue;
@property(nonatomic) uint64_t elementCount;
@property(nonatomic) uint64_t bound;
@property(nonatomic) uint32_t axis;
@property(nonatomic) uint32_t indexType;
@property(nonatomic) uint32_t tupleDepth;
@property(nonatomic) uint64_t targetCount;
@property(nonatomic) uint64_t indexSpan;
@property(nonatomic) uint64_t indexOffset;
@property(nonatomic, strong, nullable) NSMutableData *indexExtents;
@property(nonatomic, strong, nullable) NSMutableData *indexStrides;
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
    uint64_t leftOffset;
    uint64_t rightOffset;
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
_Static_assert(sizeof(SynaptikMetalPointMeta) == 32U, "PointMeta size");
_Static_assert(_Alignof(SynaptikMetalPointMeta) == 8U, "PointMeta alignment");
_Static_assert(offsetof(SynaptikMetalPointMeta, elementCount) == 0U, "PointMeta elementCount");
_Static_assert(offsetof(SynaptikMetalPointMeta, gridWidth) == 8U, "PointMeta gridWidth");
_Static_assert(offsetof(SynaptikMetalPointMeta, gridHeight) == 16U, "PointMeta gridHeight");
_Static_assert(offsetof(SynaptikMetalPointMeta, scalar) == 24U, "PointMeta scalar");
_Static_assert(offsetof(SynaptikMetalPointMeta, reserved) == 28U, "PointMeta reserved");

typedef struct {
    uint64_t dims[16];
    uint64_t conditionStrides[16];
    uint64_t trueStrides[16];
    uint64_t falseStrides[16];
    uint64_t conditionOffset;
    uint64_t trueOffset;
    uint64_t falseOffset;
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
    uint64_t roleDims[SYNAPTIK_MAX_NODE_INPUTS][16];
    uint64_t roleStrides[SYNAPTIK_MAX_NODE_INPUTS][16];
    uint64_t roleOffsets[SYNAPTIK_MAX_NODE_INPUTS];
    uint32_t roleRanks[SYNAPTIK_MAX_NODE_INPUTS];
} SynaptikMetalDataMeta;

typedef struct {
    SynaptikMetalDataMeta data;
    uint64_t addStrides[16];
    uint64_t addOffset;
    uint32_t scalarBits;
    uint32_t lowerBits;
    uint32_t upperBits;
    uint32_t flags;
    uint32_t addRank;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} SynaptikMetalAnchorEpilogueMeta;
_Static_assert(
        sizeof(SynaptikMetalAnchorEpilogueMeta) == 6096U,
        "AnchorEpilogueMeta size");
_Static_assert(
        _Alignof(SynaptikMetalAnchorEpilogueMeta) == 8U,
        "AnchorEpilogueMeta alignment");

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

typedef struct {
    uint64_t first;
    uint64_t second;
    uint32_t complementBits;
    uint32_t count;
} SynaptikMetalRngDropoutMeta;

@interface SynaptikMetalProgramStep : NSObject
@property(nonatomic) BOOL custom;
@property(nonatomic) uint32_t operation;
@property(nonatomic) uint32_t stage;
@property(nonatomic) uint32_t firstInput;
@property(nonatomic) uint32_t secondInput;
@property(nonatomic) uint32_t auxiliaryInput;
@property(nonatomic) uint32_t output;
@property(nonatomic, copy, nullable) NSArray<NSNumber *> *outputValues;
@property(nonatomic) BOOL anchorEpilogue;
@property(nonatomic) BOOL hasExternalAdd;
@property(nonatomic) uint32_t epilogueInput;
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
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
@property(nonatomic) uint32_t planOrdinal;
@property(nonatomic) uint32_t planKind;
#endif
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
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
@property(nonatomic, copy, nullable) NSData *manifestDigest;
#endif
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

SYNAPTIK_EXPORT uint32_t synaptik_metal_foundation_abi_version(void) { return 7U; }

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
    uint32_t minimum_input_rank =
            node.operation == SYNAPTIK_METAL_CUSTOM_SELECT ? 1U : 0U;
    if (!present[input] || !present[output]
            || input_rank < minimum_input_rank
            || !view[output])
        return NO;
    for (uint32_t axis = 0U; axis < input_rank; axis++)
        if (dimensions[(size_t)input * SYNAPTIK_MAX_RANK + axis] == 0U) return NO;
    for (uint32_t axis = 0U; axis < output_rank; axis++)
        if (dimensions[(size_t)output * SYNAPTIK_MAX_RANK + axis] == 0U) return NO;
    if (node.operation == SYNAPTIK_METAL_CUSTOM_SELECT) {
        if (input_rank == 0U || output_rank + 1U != input_rank
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
    if (output_rank != input_rank) return NO;
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

static BOOL task0066_positive_physical_span(
        uint32_t value,
        const uint32_t *ranks,
        const uint64_t *dimensions,
        const uint64_t *strides,
        const uint64_t *offsets,
        uint64_t *spans) {
    uint32_t rank = ranks[value];
    uint32_t order[SYNAPTIK_MAX_RANK] = {0U};
    uint32_t count = 0U;
    for (uint32_t axis = 0U; axis < rank; axis++) {
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
        if (stride < covered
                || dimension - 1U > ((uint64_t)INT64_MAX - covered) / stride)
            return NO;
        covered += (dimension - 1U) * stride;
    }
    if (offsets[value] > (uint64_t)INT64_MAX - covered) return NO;
    spans[value] = offsets[value] + covered;
    return YES;
}

static BOOL task0066_derive_physical_view(
        SynaptikMetalDecodedNode node,
        const uint32_t *ranks,
        const uint64_t *dimensions,
        uint64_t *strides,
        uint64_t *offsets,
        uint64_t *spans) {
    if (node.operation != SYNAPTIK_METAL_CUSTOM_SELECT
            && node.operation != SYNAPTIK_METAL_CUSTOM_SLICE)
        return YES;
    uint32_t input = node.first_input;
    uint32_t output = node.output;
    uint32_t input_rank = ranks[input];
    uint32_t output_rank = ranks[output];
    uint64_t expected_offset = offsets[input];
    if (node.operation == SYNAPTIK_METAL_CUSTOM_SELECT) {
        if (node.attribute_count != 2U
                || node.attribute_values[0] >= input_rank
                || output_rank + 1U != input_rank)
            return NO;
        uint32_t selected_axis = (uint32_t)node.attribute_values[0];
        uint64_t selected_stride =
                strides[(size_t)input * SYNAPTIK_MAX_RANK + selected_axis];
        if (!task0059_layout_add_product(
                    &expected_offset, node.attribute_values[1], selected_stride))
            return NO;
        for (uint32_t source = 0U, target = 0U; source < input_rank; source++) {
            if (source == selected_axis) continue;
            strides[(size_t)output * SYNAPTIK_MAX_RANK + target++] =
                    strides[(size_t)input * SYNAPTIK_MAX_RANK + source];
        }
    } else {
        if (output_rank != input_rank) return NO;
        for (uint32_t axis = 0U; axis < input_rank; axis++)
            strides[(size_t)output * SYNAPTIK_MAX_RANK + axis] =
                    strides[(size_t)input * SYNAPTIK_MAX_RANK + axis];
        BOOL crop = node.attribute_kind == 15U;
        uint32_t count;
        uint32_t starts_offset;
        uint32_t axes_offset = 0U;
        uint32_t steps_offset = 0U;
        if (crop) {
            if (node.attribute_count < 2U
                    || node.attribute_values[0] != input_rank)
                return NO;
            uint32_t prefix_rank_offset = 1U + input_rank;
            if (prefix_rank_offset >= node.attribute_count
                    || node.attribute_values[prefix_rank_offset] != input_rank)
                return NO;
            count = input_rank;
            starts_offset = prefix_rank_offset + 1U;
        } else {
            if (node.attribute_kind != 17U || node.attribute_count == 0U
                    || node.attribute_values[0] > input_rank)
                return NO;
            count = (uint32_t)node.attribute_values[0];
            starts_offset = 1U;
            axes_offset = 1U + count * 2U;
            steps_offset = 1U + count * 3U;
        }
        for (uint32_t item = 0U; item < count; item++) {
            uint32_t axis =
                    crop ? item : (uint32_t)node.attribute_values[axes_offset + item];
            uint64_t start = node.attribute_values[starts_offset + item];
            uint64_t step = crop ? 1U : node.attribute_values[steps_offset + item];
            if (axis >= input_rank || step == 0U || step > (uint64_t)INT64_MAX)
                return NO;
            uint64_t input_stride =
                    strides[(size_t)input * SYNAPTIK_MAX_RANK + axis];
            if (!task0059_layout_add_product(
                        &expected_offset, start, input_stride)
                    || input_stride > (uint64_t)INT64_MAX / step)
                return NO;
            strides[(size_t)output * SYNAPTIK_MAX_RANK + axis] =
                    input_stride * step;
        }
    }
    offsets[output] = expected_offset;
    return task0066_positive_physical_span(
            output, ranks, dimensions, strides, offsets, spans);
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
                    || input.count == 0U) return NO;
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
            if (input.count != output.count) return NO;
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
        uint64_t index,
        uint64_t *out_target) {
    if (validation.coordinateExtents == nil
            || validation.dataStrides == nil
            || out_target == NULL)
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
        if (dimension == validation.axis) coordinate = index;
        if (coordinate > UINT64_MAX / strides[dimension]) return NO;
        uint64_t contribution = coordinate * strides[dimension];
        if (target > UINT64_MAX - contribution) return NO;
        target += contribution;
    }
    if (remaining != 0U) return NO;
    *out_target = target;
    return YES;
}

static BOOL index_validation_value(
        SynaptikMetalIndexValidation *validation,
        const void *contents,
        uint32_t index_type,
        uint64_t ordinal,
        int64_t *out_value) {
    if (validation.indexExtents == nil
            || validation.indexStrides == nil
            || contents == NULL
            || out_value == NULL)
        return NO;
    NSUInteger rank = validation.indexExtents.length / sizeof(uint64_t);
    if (rank != validation.indexStrides.length / sizeof(uint64_t))
        return NO;
    const uint64_t *extents = validation.indexExtents.bytes;
    const uint64_t *strides = validation.indexStrides.bytes;
    uint64_t storage = validation.indexOffset;
    uint64_t remaining = ordinal;
    for (NSUInteger dimension = rank; dimension-- > 0U;) {
        uint64_t extent = extents[dimension];
        if (extent == 0U) return NO;
        uint64_t coordinate = remaining % extent;
        remaining /= extent;
        if (coordinate > UINT64_MAX / strides[dimension]) return NO;
        uint64_t contribution = coordinate * strides[dimension];
        if (storage > UINT64_MAX - contribution) return NO;
        storage += contribution;
    }
    if (remaining != 0U || storage >= validation.indexSpan) return NO;
    *out_value = index_type == SYNAPTIK_METAL_TYPE_INT32
            ? (int64_t)((const int32_t *)contents)[storage]
            : ((const int64_t *)contents)[storage];
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
        int64_t value = 0;
        if (!index_validation_value(
                    validation, contents, index_type, ordinal, &value))
            return NO;
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
    if (width == 0U || validation.indexSpan == 0U
            || validation.indexSpan > UINT64_MAX / width)
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    uint64_t required = validation.indexSpan * width;
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
            validation.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS;
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
        int64_t value = 0;
        if (!index_validation_value(
                    validation, contents, index_type, ordinal, &value))
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        uint64_t bound = nd
                ? bounds[ordinal % validation.tupleDepth]
                : validation.bound;
        if (value < 0 || (uint64_t)value >= bound)
            return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
        if (scatter_elements && !scatter_target_linear(
                    validation, ordinal, (uint64_t)value, &targets[ordinal]))
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
        case SYNAPTIK_METAL_TYPE_FLOAT16:
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
        case SYNAPTIK_METAL_TYPE_FLOAT16:
            return MPSDataTypeFloat16;
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

static BOOL task0065_dropout_metadata(
        uint64_t probability_bits,
        uint64_t *threshold,
        uint32_t *complement_bits) {
    if (threshold == NULL || complement_bits == NULL) return NO;
    double probability = 0.0;
    memcpy(&probability, &probability_bits, sizeof(probability));
    if (!isfinite(probability) || probability < 0.0 || probability >= 1.0) return NO;

    uint64_t magnitude = probability_bits & UINT64_C(0x7fffffffffffffff);
    if (magnitude == 0U) {
        *threshold = 0U;
    } else {
        uint64_t exponent = (magnitude >> 52U) & UINT64_C(0x7ff);
        if (exponent == 0U) {
            *threshold = 1U;
        } else {
            uint64_t significand =
                    UINT64_C(0x0010000000000000)
                    | (magnitude & UINT64_C(0x000fffffffffffff));
            uint64_t denominator_shift = 1022U - exponent;
            if (denominator_shift == 0U) {
                *threshold = significand;
            } else if (denominator_shift >= 63U) {
                *threshold = 1U;
            } else {
                uint64_t denominator = UINT64_C(1) << denominator_shift;
                *threshold = (significand + denominator - 1U) / denominator;
            }
        }
    }
    float complement = (float)(1.0 - probability);
    memcpy(complement_bits, &complement, sizeof(complement));
    return YES;
}


static id<MTLBuffer> custom_metadata(
        id<MTLDevice> device, const void *bytes, NSUInteger length) {
    if (device == nil || bytes == NULL || length == 0U) return nil;
    return [device newBufferWithBytes:bytes
            length:length options:MTLResourceStorageModeShared];
}

static BOOL operation_is_task0066_selected(uint32_t operation) {
    return (operation >= SYNAPTIK_METAL_CUSTOM_RESHAPE
                    && operation <= SYNAPTIK_METAL_CUSTOM_CONTIGUOUS)
            || (operation >= SYNAPTIK_METAL_CUSTOM_GATHER
                    && operation <= SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS)
            || (operation >= SYNAPTIK_METAL_CUSTOM_CAST
                    && operation <= SYNAPTIK_METAL_BOOL_NOT)
            || operation == SYNAPTIK_METAL_BOOL_WHERE
            || operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
            || (operation >= SYNAPTIK_METAL_CUSTOM_GATHER_ND
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_FOLD3D);
}



static BOOL operation_uses_custom_kernel(uint32_t operation) {
    return (operation >= SYNAPTIK_METAL_CUSTOM_RESHAPE
                    && operation <= SYNAPTIK_METAL_CUSTOM_CONTIGUOUS)
            || (operation >= SYNAPTIK_METAL_CUSTOM_GATHER
                    && operation <= SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS)
            || (operation >= SYNAPTIK_METAL_CUSTOM_GT
                    && operation <= SYNAPTIK_METAL_CUSTOM_CUM_PROD)
            || (operation >= SYNAPTIK_METAL_BOOL_IS_FINITE
                    && operation <= SYNAPTIK_METAL_BOOL_NOT)
            || operation == SYNAPTIK_METAL_CUSTOM_CONV2D
            || operation == SYNAPTIK_METAL_CUSTOM_CONV3D
            || operation == SYNAPTIK_METAL_BOOL_WHERE
            || (operation >= SYNAPTIK_METAL_CUSTOM_FLOOR
                    && operation <= SYNAPTIK_METAL_CUSTOM_RELU)
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
                    && operation <= SYNAPTIK_METAL_CUSTOM_INITIAL_STATE)
            || (operation >= SYNAPTIK_METAL_CUSTOM_PROD
                    && operation <= SYNAPTIK_METAL_CUSTOM_ANY)
            || (operation >= SYNAPTIK_METAL_CUSTOM_ARG_MAX
                    && operation <= SYNAPTIK_METAL_CUSTOM_ARG_MIN)
            || operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ADD
            || operation == SYNAPTIK_METAL_MPSGRAPH_L1_NORM;
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
static BOOL variance_uses_task0069_custom_kernel(
        SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes,
        const uint8_t *value_types) {
    if (node.operation != SYNAPTIK_METAL_MPSGRAPH_VARIANCE
            || shapes == nil || value_types == NULL
            || value_types[node.first_input] != SYNAPTIK_METAL_TYPE_FLOAT32
            || value_types[node.output] != SYNAPTIK_METAL_TYPE_FLOAT32)
        return NO;
    BOOL keep_dimensions = NO;
    uint64_t correction = 0U;
    MPSShape *input = shapes[node.first_input];
    MPSShape *output = shapes[node.output];
    NSArray<NSNumber *> *axes = node_statistical_axes(
            node, input, output, &keep_dimensions, &correction);
    return axes != nil
            && axes.count == 1U
            && axes[0].unsignedIntegerValue == 0U
            && correction == 0U
            && input.count == 1U
            && input[0].unsignedLongLongValue == 1U
            && output.count == (keep_dimensions ? 1U : 0U)
            && (!keep_dimensions || output[0].unsignedLongLongValue == 1U);
}
static NSString *low_precision_custom_function(
        SynaptikMetalDecodedNode node, const uint8_t *value_types);



static BOOL node_uses_custom_kernel(
        SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes,
        const uint8_t *value_types) {
    return (node.operation == SYNAPTIK_METAL_MPSGRAPH_VARIANCE
                    ? variance_uses_task0069_custom_kernel(node, shapes, value_types)
                    : operation_uses_custom_kernel(node.operation))
            || matmul_uses_custom_kernel(node, shapes, value_types)
            || low_precision_custom_function(node, value_types) != nil;
}
static BOOL matmul_shapes_match(
        MPSShape *left, MPSShape *right, MPSShape *output) {
    if (left == nil || right == nil || output == nil
            || left.count > SYNAPTIK_MAX_RANK
            || right.count > SYNAPTIK_MAX_RANK
            || output.count > SYNAPTIK_MAX_RANK)
        return NO;
    uint64_t left_dimensions[SYNAPTIK_MAX_RANK] = {0};
    uint64_t right_dimensions[SYNAPTIK_MAX_RANK] = {0};
    uint64_t output_dimensions[SYNAPTIK_MAX_RANK] = {0};
    for (NSUInteger axis = 0U; axis < left.count; axis++)
        left_dimensions[axis] = left[axis].unsignedLongLongValue;
    for (NSUInteger axis = 0U; axis < right.count; axis++)
        right_dimensions[axis] = right[axis].unsignedLongLongValue;
    for (NSUInteger axis = 0U; axis < output.count; axis++)
        output_dimensions[axis] = output[axis].unsignedLongLongValue;
    return synaptik_matmul_dimensions_match(
            (uint32_t)left.count,
            left_dimensions,
            (uint32_t)right.count,
            right_dimensions,
            (uint32_t)output.count,
            output_dimensions);
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
    if (node.operation != SYNAPTIK_METAL_CUSTOM_EXPAND_DIMS
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
                    && operation <= SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS)
            || operation == SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW
            || (operation >= SYNAPTIK_METAL_BOOL_IS_FINITE
                    && operation <= SYNAPTIK_METAL_BOOL_NOT)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_SCALAR_POW)
            || (operation >= SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL
                    && operation <= SYNAPTIK_METAL_MPSGRAPH_LOG1P)
            || operation == SYNAPTIK_METAL_MPSGRAPH_EXP
            || operation == SYNAPTIK_METAL_CUSTOM_SIGMOID
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

static BOOL synaptik_is_low_precision_type(uint8_t type) {
    return type == SYNAPTIK_METAL_TYPE_BFLOAT16
            || type == SYNAPTIK_METAL_TYPE_FLOAT16;
}

static NSString *low_precision_custom_function(
        SynaptikMetalDecodedNode node, const uint8_t *value_types) {
    if (value_types == NULL) return nil;
    uint8_t input_type = node.input_count == 0U
            ? SYNAPTIK_METAL_TYPE_UNAVAILABLE
            : value_types[node.first_input];
    uint8_t output_type = value_types[node.output];
    BOOL input_bfloat = input_type == SYNAPTIK_METAL_TYPE_BFLOAT16;
    BOOL input_float16 = input_type == SYNAPTIK_METAL_TYPE_FLOAT16;
    BOOL output_low = synaptik_is_low_precision_type(output_type);

    if (node.operation == SYNAPTIK_METAL_CUSTOM_CAST) {
        return input_float16 || output_type == SYNAPTIK_METAL_TYPE_FLOAT16
                ? @"lp_cast" : nil;
    }
    if (node.operation == SYNAPTIK_METAL_BOOL_IS_FINITE
            || node.operation == SYNAPTIK_METAL_BOOL_IS_NAN
            || node.operation == SYNAPTIK_METAL_BOOL_IS_INF) {
        return input_float16 ? @"lp_predicate" : nil;
    }
    if (node.operation == SYNAPTIK_METAL_BOOL_WHERE) {
        return value_types[node.second_input] == SYNAPTIK_METAL_TYPE_FLOAT16
                        || value_types[node.auxiliary] == SYNAPTIK_METAL_TYPE_FLOAT16
                        || output_type == SYNAPTIK_METAL_TYPE_FLOAT16
                ? @"lp_where" : nil;
    }
    if (!input_bfloat && !input_float16) {
        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
                || node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                || node.operation == SYNAPTIK_METAL_CUSTOM_CONV3D) {
            return output_low ? (node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
                    ? @"lp_matmul"
                    : node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                            ? @"lp_conv2d" : @"lp_conv3d") : nil;
        }
        return nil;
    }

#define SYNAPTIK_LP_PICK(BFLOAT_NAME, FLOAT16_NAME) \
    (input_bfloat ? @BFLOAT_NAME : @FLOAT16_NAME)
    switch ((SynaptikMetalOperation)node.operation) {
        case SYNAPTIK_METAL_MPSGRAPH_NEG:
            return SYNAPTIK_LP_PICK("lp_neg_bf16", "lp_neg_f16");
        case SYNAPTIK_METAL_MPSGRAPH_ABS:
            return SYNAPTIK_LP_PICK("lp_abs_bf16", "lp_abs_f16");
        case SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL:
            return SYNAPTIK_LP_PICK("lp_reciprocal_bf16", "lp_reciprocal_f16");
        case SYNAPTIK_METAL_MPSGRAPH_EXP:
            return SYNAPTIK_LP_PICK("lp_exp_bf16", "lp_exp_f16");
        case SYNAPTIK_METAL_MPSGRAPH_ADD:
            return SYNAPTIK_LP_PICK("lp_add_bf16", "lp_add_f16");
        case SYNAPTIK_METAL_MPSGRAPH_SUB:
            return SYNAPTIK_LP_PICK("lp_sub_bf16", "lp_sub_f16");
        case SYNAPTIK_METAL_MPSGRAPH_MUL:
            return SYNAPTIK_LP_PICK("lp_mul_bf16", "lp_mul_f16");
        case SYNAPTIK_METAL_MPSGRAPH_DIV:
            return SYNAPTIK_LP_PICK("lp_div_bf16", "lp_div_f16");
        case SYNAPTIK_METAL_CUSTOM_TENSOR_MIN:
            return SYNAPTIK_LP_PICK("lp_min_bf16", "lp_min_f16");
        case SYNAPTIK_METAL_CUSTOM_TENSOR_MAX:
            return SYNAPTIK_LP_PICK("lp_max_bf16", "lp_max_f16");
        case SYNAPTIK_METAL_CUSTOM_GT:
            return SYNAPTIK_LP_PICK("lp_gt_bf16", "lp_gt_f16");
        case SYNAPTIK_METAL_CUSTOM_GE:
            return SYNAPTIK_LP_PICK("lp_ge_bf16", "lp_ge_f16");
        case SYNAPTIK_METAL_CUSTOM_LT:
            return SYNAPTIK_LP_PICK("lp_lt_bf16", "lp_lt_f16");
        case SYNAPTIK_METAL_CUSTOM_LE:
            return SYNAPTIK_LP_PICK("lp_le_bf16", "lp_le_f16");
        case SYNAPTIK_METAL_CUSTOM_EQ:
            return SYNAPTIK_LP_PICK("lp_eq_bf16", "lp_eq_f16");
        case SYNAPTIK_METAL_CUSTOM_NE:
            return SYNAPTIK_LP_PICK("lp_ne_bf16", "lp_ne_f16");
        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD:
            return SYNAPTIK_LP_PICK("lp_scalar_add_bf16", "lp_scalar_add_f16");
        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_SUB:
            return SYNAPTIK_LP_PICK("lp_scalar_sub_bf16", "lp_scalar_sub_f16");
        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL:
            return SYNAPTIK_LP_PICK("lp_scalar_mul_bf16", "lp_scalar_mul_f16");
        case SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV:
            return SYNAPTIK_LP_PICK("lp_scalar_div_bf16", "lp_scalar_div_f16");
        case SYNAPTIK_METAL_CUSTOM_SCALAR_MIN:
            return SYNAPTIK_LP_PICK("lp_scalar_min_bf16", "lp_scalar_min_f16");
        case SYNAPTIK_METAL_CUSTOM_SCALAR_MAX:
            return SYNAPTIK_LP_PICK("lp_scalar_max_bf16", "lp_scalar_max_f16");
        case SYNAPTIK_METAL_CUSTOM_CLAMP:
            return SYNAPTIK_LP_PICK("lp_clamp_bf16", "lp_clamp_f16");
        case SYNAPTIK_METAL_CUSTOM_FLOOR:
            return SYNAPTIK_LP_PICK("lp_floor_bf16", "lp_floor_f16");
        case SYNAPTIK_METAL_CUSTOM_CEIL:
            return SYNAPTIK_LP_PICK("lp_ceil_bf16", "lp_ceil_f16");
        case SYNAPTIK_METAL_CUSTOM_SIGN:
            return SYNAPTIK_LP_PICK("lp_sign_bf16", "lp_sign_f16");
        case SYNAPTIK_METAL_CUSTOM_RELU:
            return SYNAPTIK_LP_PICK("lp_relu_bf16", "lp_relu_f16");
        case SYNAPTIK_METAL_MPSGRAPH_SUM:
            return SYNAPTIK_LP_PICK("lp_sum_bf16", "lp_sum_f16");
        case SYNAPTIK_METAL_MPSGRAPH_MEAN:
            return SYNAPTIK_LP_PICK("lp_mean_bf16", "lp_mean_f16");
        case SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN:
            return SYNAPTIK_LP_PICK(
                    "lp_reduction_min_bf16", "lp_reduction_min_f16");
        case SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX:
            return SYNAPTIK_LP_PICK(
                    "lp_reduction_max_bf16", "lp_reduction_max_f16");
        case SYNAPTIK_METAL_CUSTOM_CUM_SUM:
            return SYNAPTIK_LP_PICK("lp_scan_sum_bf16", "lp_scan_sum_f16");
        case SYNAPTIK_METAL_CUSTOM_CUM_PROD:
            return SYNAPTIK_LP_PICK("lp_scan_prod_bf16", "lp_scan_prod_f16");
        case SYNAPTIK_METAL_MPSGRAPH_MSE:
            return @"lp_mse";
        case SYNAPTIK_METAL_MPSGRAPH_MATMUL:
            return output_low ? @"lp_matmul" : nil;
        case SYNAPTIK_METAL_CUSTOM_CONV2D:
            return output_low ? @"lp_conv2d" : nil;
        case SYNAPTIK_METAL_CUSTOM_CONV3D:
            return output_low ? @"lp_conv3d" : nil;
        case SYNAPTIK_METAL_CUSTOM_MAX_POOL2D:
            return @"lp_max_pool2d";
        case SYNAPTIK_METAL_CUSTOM_MAX_POOL3D:
            return @"lp_max_pool3d";
        case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D:
            return @"lp_average_pool2d";
        case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D:
            return @"lp_average_pool3d";
        case SYNAPTIK_METAL_CUSTOM_SORT:
            return @"lp_sort";
        case SYNAPTIK_METAL_CUSTOM_ARGSORT:
            return @"lp_argsort";
        case SYNAPTIK_METAL_CUSTOM_TOP_K:
            return @"lp_top_k";
        case SYNAPTIK_METAL_CUSTOM_ARG_MAX:
            return @"lp_arg_max";
        case SYNAPTIK_METAL_CUSTOM_ARG_MIN:
            return @"lp_arg_min";
        case SYNAPTIK_METAL_CUSTOM_SCATTER_ADD:
            return @"lp_scatter_add";
        case SYNAPTIK_METAL_MPSGRAPH_L1_NORM:
            return @"lp_l1_norm";
        case SYNAPTIK_METAL_MPSGRAPH_VARIANCE:
            return @"lp_variance";
        case SYNAPTIK_METAL_CUSTOM_DROPOUT:
            return SYNAPTIK_LP_PICK("lp_dropout_bf16", "lp_dropout_f16");
        default:
            return nil;
    }
#undef SYNAPTIK_LP_PICK
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
    if (node.operation == SYNAPTIK_METAL_CUSTOM_DROPOUT) return @"dropout_f32_0065";
    if (node.operation == SYNAPTIK_METAL_CUSTOM_INITIAL_STATE)
        return @"initial_state_0065";
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
        case SYNAPTIK_METAL_BOOL_IS_FINITE:
        case SYNAPTIK_METAL_BOOL_IS_NAN:
        case SYNAPTIK_METAL_BOOL_IS_INF:
        case SYNAPTIK_METAL_BOOL_NOT: return @"exact_predicate_0066";
        case SYNAPTIK_METAL_BOOL_AND:
        case SYNAPTIK_METAL_BOOL_OR: return @"exact_logical_binary_0066";
        case SYNAPTIK_METAL_BOOL_WHERE: return @"exact_where_0066";
        case SYNAPTIK_METAL_CUSTOM_FLOOR: return @"exact_floor";
        case SYNAPTIK_METAL_CUSTOM_CEIL: return @"exact_ceil";
        case SYNAPTIK_METAL_CUSTOM_SIGN: return @"exact_sign";
        case SYNAPTIK_METAL_CUSTOM_RELU: return @"exact_relu";
        case SYNAPTIK_METAL_CUSTOM_CAST: return @"exact_cast_0066";
        case SYNAPTIK_METAL_CUSTOM_GATHER: return @"exact_gather_0066";
        case SYNAPTIK_METAL_CUSTOM_ONE_HOT: return @"exact_one_hot_0066";
        case SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS:
        case SYNAPTIK_METAL_CUSTOM_GATHER_ND: return @"exact_gather_index_0066";
        case SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS:
            return stage == 0U
                    ? @"exact_scatter_copy_0066"
                    : @"exact_scatter_elements_0066";
        case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND:
            return stage == 0U ? @"exact_scatter_copy_0066" : @"exact_scatter_nd_0066";
        case SYNAPTIK_METAL_CUSTOM_SORT: return @"exact_sort_0063";
        case SYNAPTIK_METAL_CUSTOM_ARGSORT: return @"exact_argsort_0063";
        case SYNAPTIK_METAL_CUSTOM_TOP_K: return @"exact_top_k_0063";
        case SYNAPTIK_METAL_CUSTOM_ARG_MAX: return @"exact_arg_max_0063";
        case SYNAPTIK_METAL_CUSTOM_ARG_MIN: return @"exact_arg_min_0063";
        case SYNAPTIK_METAL_CUSTOM_RESHAPE:
        case SYNAPTIK_METAL_CUSTOM_EXPAND:
        case SYNAPTIK_METAL_CUSTOM_PERMUTE:
        case SYNAPTIK_METAL_CUSTOM_EXPAND_DIMS:
        case SYNAPTIK_METAL_CUSTOM_SQUEEZE:
        case SYNAPTIK_METAL_CUSTOM_CONTIGUOUS:
        case SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS:
        case SYNAPTIK_METAL_CUSTOM_SELECT:
        case SYNAPTIK_METAL_CUSTOM_PAD:
        case SYNAPTIK_METAL_CUSTOM_SLICE:
        case SYNAPTIK_METAL_CUSTOM_TILE:
        case SYNAPTIK_METAL_CUSTOM_UNFOLD2D:
        case SYNAPTIK_METAL_CUSTOM_UNFOLD3D: return @"exact_move_unary_0066";
        case SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE: return @"exact_slice_update_0066";
        case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS:
        case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
        case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: return @"exact_fold_0066";
        case SYNAPTIK_METAL_CUSTOM_PROD:
        case SYNAPTIK_METAL_CUSTOM_ALL:
        case SYNAPTIK_METAL_CUSTOM_ANY: return @"exact_reduce_0060";
        case SYNAPTIK_METAL_MPSGRAPH_L1_NORM: return @"l1_norm_f32_0069";
        case SYNAPTIK_METAL_MPSGRAPH_VARIANCE: return @"variance_f32_0069";
        case SYNAPTIK_METAL_CUSTOM_SCATTER_ADD: return @"scatter_add_f32_0069";
        case SYNAPTIK_METAL_CUSTOM_CONCAT:
        case SYNAPTIK_METAL_CUSTOM_STACK: return @"exact_compose_0066";
        default: return nil;
    }
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
            && (value_states[value] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                    || value_states[value] == SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT)) {
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

static BOOL custom_broadcast_physical_layout(
        MPSShape *source,
        MPSShape *output,
        uint32_t value,
        const uint64_t *value_strides,
        const uint64_t *layout_offsets,
        const uint8_t *value_states,
        uint64_t strides[16],
        uint64_t *offset) {
    uint64_t physical[16] = {0};
    if (source == nil || output == nil || source.count > output.count
            || !custom_physical_layout(
                    source, value, value_strides, layout_offsets, value_states,
                    physical, offset))
        return NO;
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
                ? 0U : physical[source_axis];
    }
    return YES;
}
static BOOL configure_index_validation_layout(
        SynaptikMetalIndexValidation *validation,
        NSArray<MPSShape *> *shapes,
        const uint64_t *value_strides,
        const uint64_t *layout_offsets,
        const uint8_t *physical_states) {
    if (validation == nil || validation.indexValue >= shapes.count)
        return NO;
    MPSShape *shape = shapes[validation.indexValue];
    NSUInteger rank = shape.count;
    NSMutableData *extents =
            [NSMutableData dataWithLength:rank * sizeof(uint64_t)];
    NSMutableData *strides =
            [NSMutableData dataWithLength:rank * sizeof(uint64_t)];
    if (extents == nil || strides == nil) return NO;
    uint64_t *extent_cells = extents.mutableBytes;
    uint64_t *stride_cells = strides.mutableBytes;
    uint64_t offset = 0U;
    if (!custom_physical_layout(
                shape,
                validation.indexValue,
                value_strides,
                layout_offsets,
                physical_states,
                stride_cells,
                &offset))
        return NO;
    uint64_t span = offset;
    for (NSUInteger axis = 0U; axis < rank; axis++) {
        uint64_t extent = shape[axis].unsignedLongLongValue;
        extent_cells[axis] = extent;
        if (extent == 0U
                || extent - 1U > UINT64_MAX / stride_cells[axis])
            return NO;
        uint64_t contribution = (extent - 1U) * stride_cells[axis];
        if (span > UINT64_MAX - contribution) return NO;
        span += contribution;
    }
    if (span == UINT64_MAX) return NO;
    validation.indexOffset = offset;
    validation.indexSpan = span + 1U;
    validation.indexExtents = extents;
    validation.indexStrides = strides;
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
        id<MTLLibrary> low_precision_library,
        uint32_t stage) {
    NSString *name = low_precision_custom_function(node, value_types);
    id<MTLLibrary> selected_library = name == nil ? library : low_precision_library;
    if (name == nil) name = custom_function(node, value_types, stage);
    id<MTLFunction> function =
            name == nil ? nil : [selected_library newFunctionWithName:name];
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = function == nil ? nil
            : [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil || error != nil) return nil;
    SynaptikMetalProgramStep *step = [SynaptikMetalProgramStep new];
    if (step == nil) return nil;
    step.custom = YES;
    step.operation = node.operation;
    step.stage = stage;
    step.firstInput = node.input_count == 0U ? node.output : node.first_input;
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

    MPSShape *input = shapes[node.input_count == 0U ? node.output : node.first_input];
    MPSShape *output = shapes[node.output];
    BOOL task0063 = (node.operation >= SYNAPTIK_METAL_CUSTOM_SORT
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_TOP_K)
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_ARG_MAX
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_ARG_MIN);
    BOOL task0059 = node.operation == SYNAPTIK_METAL_CUSTOM_CAST
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_MSE
            || node.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
            || node.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_SELECT
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_TILE)
            || (node.operation >= SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS
                    && node.operation <= SYNAPTIK_METAL_MPSGRAPH_FOLD3D)
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_PROD
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_ANY);
    BOOL task0066 = (node.operation >= SYNAPTIK_METAL_CUSTOM_RESHAPE
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_CONTIGUOUS)
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_GATHER
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS)
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_CAST
                    && node.operation <= SYNAPTIK_METAL_BOOL_NOT)
            || node.operation == SYNAPTIK_METAL_BOOL_WHERE
            || node.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
            || node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ADD
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_GATHER_ND
                    && node.operation <= SYNAPTIK_METAL_MPSGRAPH_FOLD3D);
    BOOL task0064 = node.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
            || node.operation == SYNAPTIK_METAL_CUSTOM_CONV3D
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_MAX_POOL2D
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D);
    BOOL task0065 = node.operation == SYNAPTIK_METAL_CUSTOM_DROPOUT
            || node.operation == SYNAPTIK_METAL_CUSTOM_INITIAL_STATE;
    BOOL task0069_l1 = node.operation == SYNAPTIK_METAL_MPSGRAPH_L1_NORM;
    BOOL task0069_scatter = node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ADD;
    BOOL task0069_variance = node.operation == SYNAPTIK_METAL_MPSGRAPH_VARIANCE;
    if (task0065) {
        SynaptikMetalRngDropoutMeta meta = {0};
        if (node.operation == SYNAPTIK_METAL_CUSTOM_INITIAL_STATE) {
            meta.first = node.attribute_values[0];
            meta.second = node.attribute_values[1];
            meta.count = 1U;
        } else {
            uint64_t element_count = shape_element_count(output);
            if (element_count == 0U || element_count > UINT32_MAX
                    || !task0065_dropout_metadata(
                            node.attribute_values[0],
                            &meta.first,
                            &meta.complementBits))
                return nil;
            meta.count = (uint32_t)element_count;
        }
        step.grid = MTLSizeMake((NSUInteger)meta.count, 1U, 1U);
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (task0063) {
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
    } else if (task0059 || task0064 || task0066
            || task0069_l1 || task0069_scatter || task0069_variance) {
        SynaptikMetalDataMeta meta = {0};
        meta.elementCount = shape_element_count(output);
        if ((node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS
                            || node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND)
                        && stage == 1U)
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
        if (node.operation == SYNAPTIK_METAL_BOOL_WHERE) {
            meta.sourceType = value_types[node.second_input];
            meta.reserved = value_types[node.auxiliary];
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
        if (task0066) {
            for (uint32_t role = 0U; role < node.input_count; role++) {
                uint32_t value = node.inputs[role];
                MPSShape *role_shape = shapes[value];
                meta.roleRanks[role] = (uint32_t)role_shape.count;
                for (NSUInteger axis = 0U; axis < role_shape.count; axis++)
                    meta.roleDims[role][axis] =
                            role_shape[axis].unsignedLongLongValue;
                if (!custom_physical_layout(
                            role_shape,
                            value,
                            value_strides,
                            layout_offsets,
                            value_states,
                            meta.roleStrides[role],
                            &meta.roleOffsets[role]))
                    return nil;
            }
        }
        for (uint32_t word = 0U; word < node.attribute_count; word++)
            meta.attrs[word] = node.attribute_values[word];
        if (node.attribute_kind == SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS
                && node.attribute_count == 1U)
            meta.attrs[0] = node.axis;
        if (node.operation == SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS)
            meta.attrs[2] = node.axis;
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
        if (task0069_l1 || task0069_variance)
            meta.inputExtents[0] = shape_element_count(input);
        if (node.operation == SYNAPTIK_METAL_MPSGRAPH_MSE)
            meta.inputExtents[0] = shape_element_count(input);
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
        if (node.operation == SYNAPTIK_METAL_BOOL_WHERE) {
            MPSShape *when_false = shapes[node.auxiliary];
            meta.inputExtents[0] = when_false.count;
            for (NSUInteger axis = 0U; axis < when_false.count; axis++)
                meta.inputExtents[1U + axis] =
                        when_false[axis].unsignedLongLongValue;
            if (!custom_physical_layout(
                        when_false, node.auxiliary, value_strides, layout_offsets,
                        value_states, meta.outputStrides, &meta.outputOffset))
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
        if (task0069_variance) {
            if ((meta.width != 4U && meta.width != 2U)
                    || meta.sourceType != meta.targetType
                    || (!synaptik_is_low_precision_type(meta.sourceType)
                            && meta.sourceType != SYNAPTIK_METAL_TYPE_FLOAT32)
                    || meta.elementCount != 1U
                    || meta.inputExtents[0] != 1U
                    || meta.inputRank != 1U
                    || meta.attributeKind != 38U
                    || meta.attributeCount != 4U
                    || meta.attrs[0] != 1U
                    || meta.attrs[1] != 0U
                    || meta.attrs[2] > 1U
                    || meta.attrs[3] != 0U
                    || meta.outputRank != (meta.attrs[2] == 1U ? 1U : 0U)
                    || (meta.outputRank == 1U && meta.outputDims[0] != 1U)
                    || meta.inputOffset != 0U
                    || meta.outputOffset != 0U
                    || meta.inputStrides[0] != 1U
                    || (meta.outputRank == 1U && meta.outputStrides[0] != 1U))
                return nil;
            meta.reserved = UINT32_C(0x3f800000);
            meta.gridWidth = 1U;
            meta.gridHeight = 1U;
            step.grid = MTLSizeMake(1U, 1U, 1U);
            step.metadata = custom_metadata(device, &meta, sizeof(meta));
        } else if (task0069_l1) {
            if ((meta.width != 4U && meta.width != 2U)
                    || meta.elementCount != 1U
                    || meta.inputExtents[0] == 0U
                    || meta.inputExtents[0] > UINT32_MAX / meta.width
                    || meta.inputRank != 1U
                    || meta.outputRank > 1U
                    || (meta.outputRank == 1U && meta.outputDims[0] != 1U)
                    || meta.inputOffset != 0U
                    || meta.outputOffset != 0U
                    || meta.inputStrides[0] != 1U
                    || (meta.outputRank == 1U && meta.outputStrides[0] != 1U))
                return nil;
            meta.gridWidth = 1U;
            meta.gridHeight = 1U;
            step.grid = MTLSizeMake(1U, 1U, 1U);
            step.metadata = custom_metadata(device, &meta, sizeof(meta));
        } else if (task0069_scatter) {
            if ((meta.width != 4U && meta.width != 2U)
                    || meta.sourceType != meta.targetType
                    || (!synaptik_is_low_precision_type(meta.sourceType)
                            && meta.sourceType != SYNAPTIK_METAL_TYPE_FLOAT32)
                    || (meta.indexType != SYNAPTIK_METAL_TYPE_INT32
                            && meta.indexType != SYNAPTIK_METAL_TYPE_INT64)
                    || meta.inputCount != 3U
                    || meta.inputRank != 1U
                    || meta.auxiliaryRank != 1U
                    || meta.outputRank != 1U
                    || meta.roleRanks[2] != 1U
                    || meta.elementCount == 0U
                    || meta.elementCount > UINT32_MAX / meta.width
                    || meta.auxiliaryDims[0] == 0U
                    || meta.auxiliaryDims[0] > UINT32_MAX
                            / value_type_width(meta.indexType)
                    || meta.auxiliaryDims[0] > UINT32_MAX / meta.width
                    || meta.roleDims[2][0] != meta.auxiliaryDims[0]
                    || meta.inputDims[0] != meta.outputDims[0]
                    || meta.inputOffset != 0U
                    || meta.outputOffset != 0U
                    || meta.roleOffsets[2] != 0U
                    || meta.inputStrides[0] != 1U
                    || meta.outputStrides[0] != 1U
                    || meta.roleStrides[2][0] != 1U)
                return nil;
            meta.gridWidth = meta.elementCount;
            meta.gridHeight = 1U;
            step.grid = MTLSizeMake((NSUInteger)meta.elementCount, 1U, 1U);
            step.metadata = custom_metadata(device, &meta, sizeof(meta));
        } else if (task0064) {
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
            || (node.operation >= SYNAPTIK_METAL_MPSGRAPH_ADD
                    && node.operation <= SYNAPTIK_METAL_MPSGRAPH_DIV)
            || node.operation == SYNAPTIK_METAL_BOOL_AND
            || node.operation == SYNAPTIK_METAL_BOOL_OR) {
        SynaptikMetalBinaryMeta meta = {0};
        meta.rank = (uint32_t)output.count;
        meta.elementCount = shape_element_count(output);
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        MPSShape *right = shapes[node.second_input];
        if (!custom_broadcast_physical_layout(
                        input, output, node.first_input, value_strides, layout_offsets,
                        value_states, meta.leftStrides, &meta.leftOffset)
                || !custom_broadcast_physical_layout(
                        right, output, node.second_input, value_strides, layout_offsets,
                        value_states, meta.rightStrides, &meta.rightOffset))
            return nil;
        for (NSUInteger axis = 0U; axis < output.count; axis++)
            meta.dims[axis] = output[axis].unsignedLongLongValue;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if (node.operation == SYNAPTIK_METAL_BOOL_WHERE) {
        SynaptikMetalTernaryMeta meta = {0};
        meta.rank = (uint32_t)output.count;
        meta.elementCount = shape_element_count(output);
        for (NSUInteger axis = 0U; axis < output.count; axis++)
            meta.dims[axis] = output[axis].unsignedLongLongValue;
        if (!custom_broadcast_physical_layout(
                        input, output, node.first_input, value_strides, layout_offsets,
                        value_states, meta.conditionStrides, &meta.conditionOffset)
                || !custom_broadcast_physical_layout(
                        shapes[node.second_input], output, node.second_input, value_strides,
                        layout_offsets, value_states, meta.trueStrides, &meta.trueOffset)
                || !custom_broadcast_physical_layout(
                        shapes[node.auxiliary], output, node.auxiliary, value_strides,
                        layout_offsets, value_states, meta.falseStrides, &meta.falseOffset))
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
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_NEG
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_ABS
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_RECIPROCAL
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_EXP
            || (node.operation >= SYNAPTIK_METAL_CUSTOM_FLOOR
                    && node.operation <= SYNAPTIK_METAL_CUSTOM_RELU)
            || node.operation == SYNAPTIK_METAL_CUSTOM_SIGMOID) {
        SynaptikMetalPointMeta meta = {0};
        meta.elementCount = shape_element_count(output);
        MTLSize grid = MTLSizeMake(0U, 0U, 0U);
        if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
            return nil;
        step.grid = grid;
        step.metadata = custom_metadata(device, &meta, sizeof(meta));
    } else if ((node.operation >= SYNAPTIK_METAL_MPSGRAPH_SCALAR_ADD
                    && node.operation <= SYNAPTIK_METAL_MPSGRAPH_SCALAR_DIV)
            || node.operation == SYNAPTIK_METAL_CUSTOM_SCALAR_MIN
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
    } else if (node.operation == SYNAPTIK_METAL_MPSGRAPH_SUM
            || node.operation == SYNAPTIK_METAL_MPSGRAPH_MEAN
            || node.operation == SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN
            || node.operation == SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX) {
        SynaptikMetalReductionMeta meta = {0};
        meta.rank = (uint32_t)input.count;
        BOOL keep_dimensions = NO;
        NSArray<NSNumber *> *axes =
                node_reduction_axes(node, input, output, &keep_dimensions);
        if (axes == nil) return nil;
        for (NSNumber *axis in axes)
            meta.axesMask |= UINT32_C(1) << axis.unsignedIntValue;
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

static BOOL synaptik_pointmeta_reflection_is_exact(id<MTLBufferBinding> buffer) {
    if (buffer == nil
            || buffer.bufferDataType != MTLDataTypeStruct
            || buffer.bufferDataSize != sizeof(SynaptikMetalPointMeta)
            || buffer.bufferAlignment != _Alignof(SynaptikMetalPointMeta)
            || buffer.bufferStructType == nil
            || buffer.bufferPointerType == nil
            || buffer.bufferPointerType.elementType != MTLDataTypeStruct
            || buffer.bufferPointerType.access != MTLBindingAccessReadOnly
            || buffer.bufferPointerType.alignment != _Alignof(SynaptikMetalPointMeta)
            || buffer.bufferPointerType.dataSize != sizeof(SynaptikMetalPointMeta)
            || buffer.bufferPointerType.elementStructType != buffer.bufferStructType)
        return NO;
    NSArray<MTLStructMember *> *members = buffer.bufferStructType.members;
    if (members.count != 5U) return NO;
    NSString *names[5] = {
        @"elementCount", @"gridWidth", @"gridHeight", @"scalar", @"reserved"
    };
    NSUInteger offsets[5] = {
        offsetof(SynaptikMetalPointMeta, elementCount),
        offsetof(SynaptikMetalPointMeta, gridWidth),
        offsetof(SynaptikMetalPointMeta, gridHeight),
        offsetof(SynaptikMetalPointMeta, scalar),
        offsetof(SynaptikMetalPointMeta, reserved),
    };
    MTLDataType types[5] = {
        MTLDataTypeULong,
        MTLDataTypeULong,
        MTLDataTypeULong,
        MTLDataTypeUInt,
        MTLDataTypeUInt,
    };
    for (NSUInteger index = 0U; index < 5U; index++) {
        MTLStructMember *member = members[index];
        if (![member.name isEqualToString:names[index]]
                || member.offset != offsets[index]
                || member.dataType != types[index]
                || [buffer.bufferStructType memberByName:names[index]] != member)
            return NO;
    }
    return YES;
}

static SynaptikMetalProgramStep *make_anchor_epilogue_step(
        uint32_t ordinal,
        SynaptikPointwiseStepRecord record,
        const uint32_t *members,
        const SynaptikPointwiseInstructionRecord *instructions,
        const SynaptikMetalDecodedNode *nodes,
        NSArray<MPSShape *> *shapes,
        const uint8_t *value_types,
        const uint64_t *value_strides,
        const uint64_t *layout_offsets,
        const uint8_t *value_states,
        const uint32_t *local_transpose_sources,
        const uint32_t *local_singleton_height_sources,
        id<MTLDevice> device,
        id<MTLLibrary> library) {
    SynaptikMetalDecodedNode anchor = nodes[members[record.member_start]];
    (void)ordinal;
    SynaptikMetalDecodedNode terminal =
            nodes[members[record.member_start + record.member_count - 1U]];
    SynaptikMetalDecodedNode execution = anchor;
    execution.output = terminal.output;
    execution.outputs[0] = terminal.output;
    SynaptikMetalProgramStep *step = make_custom_step(
            execution,
            shapes,
            value_types,
            value_strides,
            layout_offsets,
            value_states,
            local_transpose_sources,
            local_singleton_height_sources,
            device,
            library,
            nil,
            0U);
    if (step == nil || step.metadata == nil
            || step.metadata.length != sizeof(SynaptikMetalDataMeta))
        return nil;
    SynaptikMetalAnchorEpilogueMeta meta = {0};
    memcpy(&meta.data, step.metadata.contents, sizeof(meta.data));
    uint32_t external = UINT32_MAX;
    for (uint32_t index = 0U; index < record.instruction_count; index++) {
        SynaptikPointwiseInstructionRecord instruction =
                instructions[record.instruction_start + index];
        if (instruction.opcode == 1U) {
            meta.flags |= 1U;
            meta.scalarBits = (uint32_t)instruction.raw0;
        } else if (instruction.opcode == 2U) {
            external = (uint32_t)instruction.raw1;
            if (instruction.raw0 != 0U) meta.flags |= 16U;
        } else if (instruction.opcode == 3U) {
            meta.flags |= 4U;
        } else if (instruction.opcode == 4U) {
            meta.flags |= 8U;
            meta.lowerBits = (uint32_t)instruction.raw0;
            meta.upperBits = (uint32_t)instruction.raw1;
        } else {
            return nil;
        }
    }
    BOOL has_add = external != UINT32_MAX;
    if (has_add) {
        if (external >= shapes.count
                || shapes[external].count > meta.data.outputRank)
            return nil;
        meta.addRank = (uint32_t)shapes[external].count;
        meta.addOffset = layout_offsets[external];
        uint32_t shift = meta.data.outputRank - meta.addRank;
        for (uint32_t axis = shift; axis < meta.data.outputRank; axis++) {
            uint32_t external_axis = axis - shift;
            meta.addStrides[axis] =
                    shapes[external][external_axis].unsignedLongLongValue == 1U
                    ? 0U
                    : value_strides[(size_t)external * SYNAPTIK_MAX_RANK
                            + external_axis];
        }
    }
    NSString *function_name = nil;
    if (record.flags == 1U) {
        uint8_t left = value_types[anchor.first_input];
        uint8_t right = value_types[anchor.second_input];
        if (left == SYNAPTIK_METAL_TYPE_FLOAT32
                && right == SYNAPTIK_METAL_TYPE_FLOAT32)
            function_name = has_add
                    ? @"matmul_f32_f32_f32_add_epilogue_0071"
                    : @"matmul_f32_f32_f32_epilogue_0071";
        else if (left == SYNAPTIK_METAL_TYPE_FLOAT32
                && right == SYNAPTIK_METAL_TYPE_BFLOAT16)
            function_name = has_add
                    ? @"matmul_f32_bf16_f32_add_epilogue_0071"
                    : @"matmul_f32_bf16_f32_epilogue_0071";
        else if (left == SYNAPTIK_METAL_TYPE_BFLOAT16
                && right == SYNAPTIK_METAL_TYPE_FLOAT32)
            function_name = has_add
                    ? @"matmul_bf16_f32_f32_add_epilogue_0071"
                    : @"matmul_bf16_f32_f32_epilogue_0071";
    } else if (record.flags == 2U) {
        function_name = has_add
                ? @"conv2d_add_epilogue_0071"
                : @"conv2d_epilogue_0071";
    }
    id<MTLFunction> function =
            function_name == nil ? nil : [library newFunctionWithName:function_name];
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = function == nil ? nil
            : [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLBuffer> metadata = custom_metadata(device, &meta, sizeof(meta));
    if (pipeline == nil || error != nil || metadata == nil) return nil;
    step.pipeline = pipeline;
    step.threadsPerThreadgroup = MAX(
            (NSUInteger)1U,
            MIN(pipeline.maxTotalThreadsPerThreadgroup, pipeline.threadExecutionWidth));
    step.metadata = metadata;
    step.output = terminal.output;
    step.outputValues = @[@(terminal.output)];
    step.anchorEpilogue = YES;
    step.hasExternalAdd = has_add;
    step.epilogueInput = external;
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
    step.planOrdinal = ordinal;
    step.planKind = SYNAPTIK_ANCHOR_EPILOGUE_STEP;
#endif
    return step;
}

static SynaptikMetalProgramStep *make_generated_pointwise_step(
        uint32_t step_ordinal,
        uint32_t input_value,
        uint32_t output_value,
        NSArray<MPSShape *> *shapes,
        id<MTLDevice> device,
        id<MTLLibrary> library) {
    if (input_value >= shapes.count || output_value >= shapes.count
            || ![shapes[input_value] isEqualToArray:shapes[output_value]])
        return nil;
    NSString *name = [NSString stringWithFormat:@"synaptik_pw_g1_s%u", step_ordinal];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) return nil;
    NSError *error = nil;
    MTLAutoreleasedComputePipelineReflection reflection = nil;
    id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function
            options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
            reflection:&reflection
            error:&error];
    if (pipeline == nil || error != nil || reflection == nil
            || reflection.bindings.count != 3U)
        return nil;
    BOOL seen[3] = {NO, NO, NO};
    for (id<MTLBinding> binding in reflection.bindings) {
        if (binding.type != MTLBindingTypeBuffer || binding.index >= 3U
                || seen[binding.index] || !binding.isUsed || !binding.isArgument)
            return nil;
        seen[binding.index] = YES;
        id<MTLBufferBinding> buffer = (id<MTLBufferBinding>)binding;
        NSString *expected_name =
                binding.index == 0U ? @"input" : binding.index == 1U ? @"output" : @"meta";
        if (![binding.name isEqualToString:expected_name]) return nil;
        if (binding.index == 0U) {
            if (binding.access != MTLBindingAccessReadOnly
                    || buffer.bufferAlignment != _Alignof(uint32_t)
                    || buffer.bufferPointerType == nil
                    || buffer.bufferPointerType.elementType != MTLDataTypeUInt
                    || buffer.bufferPointerType.access != MTLBindingAccessReadOnly
                    || buffer.bufferPointerType.alignment != _Alignof(uint32_t)
                    || buffer.bufferPointerType.dataSize != sizeof(uint32_t))
                return nil;
        } else if (binding.index == 1U) {
            if (binding.access != MTLBindingAccessReadWrite
                    || buffer.bufferAlignment != _Alignof(uint32_t)
                    || buffer.bufferPointerType == nil
                    || buffer.bufferPointerType.elementType != MTLDataTypeUInt
                    || buffer.bufferPointerType.access != MTLBindingAccessReadWrite
                    || buffer.bufferPointerType.alignment != _Alignof(uint32_t)
                    || buffer.bufferPointerType.dataSize != sizeof(uint32_t))
                return nil;
        } else if (binding.access != MTLBindingAccessReadOnly
                || !synaptik_pointmeta_reflection_is_exact(buffer)) {
            return nil;
        }
    }
    if (!seen[0] || !seen[1] || !seen[2]) return nil;
    SynaptikMetalPointMeta meta = {0};
    meta.elementCount = shape_element_count(shapes[output_value]);
    MTLSize grid = MTLSizeMake(0U, 0U, 0U);
    if (!custom_grid(meta.elementCount, &grid, &meta.gridWidth, &meta.gridHeight))
        return nil;
    SynaptikMetalProgramStep *step = [SynaptikMetalProgramStep new];
    if (step == nil) return nil;
    step.custom = YES;
    step.operation = 0U;
    step.stage = 0U;
    step.firstInput = input_value;
    step.secondInput = UINT32_MAX;
    step.auxiliaryInput = 0U;
    step.output = output_value;
    step.outputValues = @[@(output_value)];
    step.pipeline = pipeline;
    step.threadsPerThreadgroup = MAX(
            (NSUInteger)1U,
            MIN(pipeline.maxTotalThreadsPerThreadgroup, pipeline.threadExecutionWidth));
    step.grid = grid;
    step.metadata = custom_metadata(device, &meta, sizeof(meta));
    return step.metadata == nil ? nil : step;
}

static NSArray<NSNumber *> *translate_program_values(
        NSArray<NSNumber *> *values,
        const uint32_t *program_to_slot,
        uint32_t value_count) {
    if (values == nil) return nil;
    NSMutableArray<NSNumber *> *translated =
            [NSMutableArray arrayWithCapacity:values.count];
    if (translated == nil) return nil;
    for (NSNumber *encoded in values) {
        NSUInteger value = encoded.unsignedIntegerValue;
        if (value >= value_count || program_to_slot[value] == UINT32_MAX) return nil;
        [translated addObject:@(program_to_slot[value])];
    }
    return [translated copy];
}

static BOOL translate_program_step(
        SynaptikMetalProgramStep *step,
        const uint32_t *program_to_slot,
        uint32_t value_count) {
    if (step == nil || step.firstInput >= value_count || step.output >= value_count
            || program_to_slot[step.firstInput] == UINT32_MAX
            || program_to_slot[step.output] == UINT32_MAX)
        return NO;
    step.firstInput = program_to_slot[step.firstInput];
    step.output = program_to_slot[step.output];
    if (step.secondInput < value_count)
        step.secondInput = program_to_slot[step.secondInput];
    if (step.auxiliaryInput < value_count)
        step.auxiliaryInput = program_to_slot[step.auxiliaryInput];
    if (step.hasExternalAdd) {
        if (step.epilogueInput >= value_count
                || program_to_slot[step.epilogueInput] == UINT32_MAX)
            return NO;
        step.epilogueInput = program_to_slot[step.epilogueInput];
    }
    if (step.outputValues != nil) {
        step.outputValues = translate_program_values(
                step.outputValues, program_to_slot, value_count);
        if (step.outputValues == nil) return NO;
    }
    if (step.inputValues != nil) {
        step.inputValues = translate_program_values(
                step.inputValues, program_to_slot, value_count);
        if (step.inputValues == nil) return NO;
    }
    if (step.feedValues != nil) {
        step.feedValues = translate_program_values(
                step.feedValues, program_to_slot, value_count);
        if (step.feedValues == nil) return NO;
    }
    return YES;
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
        uint32_t target_count, const uint32_t *target_indices,
        const SynaptikPointwisePlan *fusion, void **out_executable) {
    if (out_executable == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_executable = NULL;
    BOOL has_layouts = value_strides != NULL && layout_offsets != NULL
            && layout_spans != NULL && declared_states != NULL;
    BOOL lacks_layouts = value_strides == NULL && layout_offsets == NULL
            && layout_spans == NULL && declared_states == NULL;
    if (context == NULL
            || (route != SYNAPTIK_METAL_ROUTE_MPSGRAPH
                    && route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
            || value_count == 0U || node_count == 0U || target_count == 0U
            || value_ranks == NULL || value_dimensions == NULL || declared_types == NULL
            || (!has_layouts && !lacks_layouts)
            || nodes == NULL || target_indices == NULL
            || (feed_count == 0U ? feed_indices != NULL : feed_indices == NULL)
            || (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                    ? fusion == NULL
                            || fusion->step_count == 0U
                            || fusion->materialized_count == 0U
                            || fusion->steps == NULL
                            || fusion->members == NULL
                            || fusion->bindings == NULL
                            || fusion->materialized_values == NULL
                            || fusion->program_to_slot == NULL
                            || (fusion->instruction_count != 0U
                                    && fusion->instructions == NULL)
                    : fusion != NULL))
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
        if (route == SYNAPTIK_METAL_ROUTE_MPSGRAPH) {
            for (uint32_t value = 0U; value < value_count; value++)
                if (declared_types[value] == SYNAPTIK_METAL_TYPE_BFLOAT16
                        || declared_types[value] == SYNAPTIK_METAL_TYPE_FLOAT16)
                    return SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION;
        }
        BOOL contains_matmul = NO;
        BOOL contains_custom = NO;
        BOOL contains_low_precision = NO;
        uint32_t fusion_step_cursor = 0U;
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
            BOOL zero_input = node.input_count == 0U;
            if (node.input_count > SYNAPTIK_MAX_NODE_INPUTS
                    || (zero_input
                            ? node.first_input != UINT32_MAX
                            : (node.first_input >= value_count
                                    || states[node.first_input]
                                            == SYNAPTIK_METAL_VALUE_UNAVAILABLE))
                    || node.output >= value_count
                    || node.output_count == 0U
                    || node.output_count > SYNAPTIK_MAX_NODE_OUTPUTS)
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
            BOOL custom_node = node_uses_custom_kernel(node, shapes, value_types)
                    || (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                            && node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
                            && (local_transpose[node.first_input] != 0U
                                    || local_transpose[node.second_input] != 0U));
            contains_low_precision |=
                    low_precision_custom_function(node, value_types) != nil;
            BOOL anchor_step = NO;
            if (fusion != NULL) {
                while (fusion_step_cursor < fusion->step_count
                        && node_index >= fusion->steps[fusion_step_cursor].member_start
                                + fusion->steps[fusion_step_cursor].member_count)
                    fusion_step_cursor++;
                if (fusion_step_cursor >= fusion->step_count)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                uint32_t kind = fusion->steps[fusion_step_cursor].kind;
                anchor_step = kind == SYNAPTIK_ANCHOR_EPILOGUE_STEP;
                if ((kind == 1U && !custom_node)
                        || (kind == 2U && custom_node)
                        || (kind == SYNAPTIK_POINTWISE_GENERATED_STEP
                                && (!custom_node
                                    || node.operation < SYNAPTIK_METAL_CUSTOM_FLOOR
                                    || node.operation > SYNAPTIK_METAL_CUSTOM_RELU)))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (route == SYNAPTIK_METAL_ROUTE_MPSGRAPH
                    && (!operation_has_direct_mpsgraph(node.operation)
                            || low_precision_custom_function(node, value_types) != nil
                            || operation_is_task0066_selected(node.operation)
                            || node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ADD
                            || node.operation == SYNAPTIK_METAL_MPSGRAPH_L1_NORM
                            || variance_uses_task0069_custom_kernel(
                                    node, shapes, value_types)
                            || matmul_uses_custom_kernel(node, shapes, value_types)))
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION;
            if (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                    && (custom_node || anchor_step))
                contains_custom = YES;
            BOOL affine_view = NO;
            switch ((SynaptikMetalOperation)node.operation) {
                case SYNAPTIK_METAL_CUSTOM_INITIAL_STATE: {
                    MPSShape *state = shapes[node.output];
                    if (!zero_input || node.second_input != UINT32_MAX
                            || node.auxiliary != 0U
                            || node.output_count != 1U
                            || node.attribute_kind
                                    != SYNAPTIK_METAL_CUSTOM_ATTR_GRAPH_RNG_STATE
                            || node.attribute_count != 2U
                            || state.count != 1U
                            || state[0].unsignedLongLongValue != 2U
                            || (has_layouts && layout_spans[node.output] > UINT32_MAX))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_DROPOUT: {
                    if (zero_input || node.input_count != 2U
                            || node.output_count != 3U
                            || node.auxiliary != 0U
                            || node.second_input >= value_count
                            || node.attribute_kind
                                    != SYNAPTIK_METAL_CUSTOM_ATTR_DROPOUT
                            || node.attribute_count != 1U)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    MPSShape *input = shapes[node.first_input];
                    MPSShape *state = shapes[node.second_input];
                    MPSShape *output = shapes[node.output];
                    MPSShape *mask = shapes[node.outputs[1]];
                    MPSShape *next_state = shapes[node.outputs[2]];
                    uint64_t threshold = 0U;
                    uint32_t complement_bits = 0U;
                    BOOL distinct = node.first_input != node.second_input
                            && node.first_input != node.outputs[0]
                            && node.first_input != node.outputs[1]
                            && node.first_input != node.outputs[2]
                            && node.second_input != node.outputs[0]
                            && node.second_input != node.outputs[1]
                            && node.second_input != node.outputs[2]
                            && node.outputs[0] != node.outputs[1]
                            && node.outputs[0] != node.outputs[2]
                            && node.outputs[1] != node.outputs[2];
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || states[node.second_input]
                                    != SYNAPTIK_METAL_VALUE_CANONICAL
                            || !distinct
                            || !task0065_dropout_metadata(
                                    node.attribute_values[0],
                                    &threshold,
                                    &complement_bits)
                            || !task0063_shape_is_uint32_bounded(input, YES)
                            || ![input isEqualToArray:output]
                            || ![input isEqualToArray:mask]
                            || state.count != 1U
                            || state[0].unsignedLongLongValue != 2U
                            || ![state isEqualToArray:next_state]
                            || (has_layouts
                                    && (layout_spans[node.first_input] > UINT32_MAX
                                            || layout_spans[node.second_input]
                                                    > UINT32_MAX
                                            || layout_spans[node.output] > UINT32_MAX
                                            || layout_spans[node.outputs[1]]
                                                    > UINT32_MAX
                                            || layout_spans[node.outputs[2]]
                                                    > UINT32_MAX)))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SIGMOID:
                    if ((route != SYNAPTIK_METAL_ROUTE_MPSGRAPH
                                    && route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
                            || states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || shapes[node.first_input].count == 0U
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]]
                            || shape_element_count(shapes[node.first_input])
                                    > UINT32_MAX / sizeof(float))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_EXP:
                    if ((route != SYNAPTIK_METAL_ROUTE_MPSGRAPH
                                    && route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
                            || states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || shapes[node.first_input].count == 0U
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]]
                            || shape_element_count(shapes[node.first_input])
                                    > UINT32_MAX / (synaptik_is_low_precision_type(
                                            declared_types[node.first_input])
                                            ? sizeof(uint16_t) : sizeof(float)))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
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
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || shapes[node.first_input].count == 0U
                            || shapes[node.output].count == 0U
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_RELU:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_BOOL_IS_FINITE:
                case SYNAPTIK_METAL_BOOL_IS_NAN:
                case SYNAPTIK_METAL_BOOL_IS_INF:
                case SYNAPTIK_METAL_BOOL_NOT:
                    if (states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
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
                case SYNAPTIK_METAL_CUSTOM_TENSOR_MAX: {
                    BOOL affine_binary =
                            node.operation == SYNAPTIK_METAL_MPSGRAPH_ADD
                            || node.operation == SYNAPTIK_METAL_MPSGRAPH_SUB
                            || node.operation == SYNAPTIK_METAL_MPSGRAPH_MUL
                            || node.operation == SYNAPTIK_METAL_MPSGRAPH_DIV
                            || (node.operation >= SYNAPTIK_METAL_CUSTOM_GT
                                    && node.operation <= SYNAPTIK_METAL_CUSTOM_NE);
                    BOOL first_valid =
                            states[node.first_input] != SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            && (states[node.first_input] == SYNAPTIK_METAL_VALUE_CANONICAL
                                    || affine_binary
                                    || node.operation == SYNAPTIK_METAL_BOOL_AND
                                    || node.operation == SYNAPTIK_METAL_BOOL_OR);
                    BOOL second_valid =
                            node.second_input < value_count
                            && states[node.second_input] != SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            && (states[node.second_input] == SYNAPTIK_METAL_VALUE_CANONICAL
                                    || affine_binary
                                    || node.operation == SYNAPTIK_METAL_BOOL_AND
                                    || node.operation == SYNAPTIK_METAL_BOOL_OR);
                    if (!first_valid
                            || !second_valid
                            || !node_has_no_attributes(node)
                            || !shape_broadcasts_exactly_to(
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    used[node.second_input] = 1U;
                    break;
                }
                case SYNAPTIK_METAL_BOOL_WHERE:
                    if (states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || node.second_input >= value_count
                            || node.auxiliary >= value_count
                            || states[node.second_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || states[node.auxiliary] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE
                            || node.attribute_count != 0U
                            || node.axis != UINT32_MAX
                            || !node_values_are_zero_from(node, 0U)
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
                case SYNAPTIK_METAL_CUSTOM_CONTIGUOUS:
                    if (node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_RESHAPE:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_target_matches(node, shapes[node.output])
                            || shape_element_count(shapes[node.first_input])
                                    != shape_element_count(shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_EXPAND:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_target_matches(node, shapes[node.output])
                            || !shape_expands_exactly_to(
                                    shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_PERMUTE:
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
                case SYNAPTIK_METAL_CUSTOM_EXPAND_DIMS:
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
                case SYNAPTIK_METAL_CUSTOM_SQUEEZE:
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
                case SYNAPTIK_METAL_MPSGRAPH_L1_NORM: {
                    BOOL keep_dimensions = NO;
                    MPSShape *input = shapes[node.first_input];
                    MPSShape *output = shapes[node.output];
                    NSArray<NSNumber *> *axes = node_reduction_axes(
                            node, input, output, &keep_dimensions);
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                            && (node.axis != SYNAPTIK_METAL_REDUCTION_MULTI_AXIS
                                    || node.attribute_count != 1U
                                    || node.attribute_values[0] != 0U
                                    || axes.count != 1U
                                    || axes[0].unsignedIntegerValue != 0U
                                    || input.count != 1U
                                    || input[0].unsignedLongLongValue == 0U
                                    || input[0].unsignedLongLongValue > UINT32_MAX / 4U
                                    || output.count != (keep_dimensions ? 1U : 0U)
                                    || (keep_dimensions
                                            && output[0].unsignedLongLongValue != 1U)
                                    || (has_layouts
                                            && declared_states[node.output]
                                                    != SYNAPTIK_METAL_VALUE_CANONICAL)))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_PROD:
                case SYNAPTIK_METAL_CUSTOM_ALL:
                case SYNAPTIK_METAL_CUSTOM_ANY:
                case SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP:
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
                case SYNAPTIK_METAL_MPSGRAPH_VARIANCE: {
                    BOOL keep_dimensions = NO;
                    uint64_t correction = 0U;
                    MPSShape *input = shapes[node.first_input];
                    MPSShape *output = shapes[node.output];
                    NSArray<NSNumber *> *axes = node_statistical_axes(
                            node, input, output, &keep_dimensions, &correction);
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || axes == nil
                            || (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                                    && (axes.count != 1U
                                            || axes[0].unsignedIntegerValue != 0U
                                            || correction != 0U
                                            || input.count != 1U
                                            || input[0].unsignedLongLongValue != 1U
                                            || output.count != (keep_dimensions ? 1U : 0U)
                                            || (keep_dimensions
                                                    && output[0].unsignedLongLongValue != 1U)
                                            || (has_layouts
                                                    && (declared_states[node.first_input]
                                                                    != SYNAPTIK_METAL_VALUE_CANONICAL
                                                            || declared_states[node.output]
                                                                    != SYNAPTIK_METAL_VALUE_CANONICAL)))))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
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
                case SYNAPTIK_METAL_CUSTOM_GATHER: {
                    if (states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || node.second_input >= value_count
                            || states[node.second_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || !node_gather_matches(
                                    node,
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    NSUInteger position = route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                            ? node.second_input
                            : feed_position(node.second_input, feed_count, feed_indices);
                    if (position == NSNotFound)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    SynaptikMetalIndexValidation *validation =
                            [SynaptikMetalIndexValidation new];
                    if (validation == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    validation.operation = node.operation;
                    validation.indexValue = node.second_input;
                    validation.indexType = declared_types[node.second_input];
                    validation.nodeIndex = node_index;
                    validation.feedPosition = position;
                    validation.preflight =
                            feed_position(node.second_input, feed_count, feed_indices)
                            != NSNotFound;
                    validation.elementCount =
                            shape_element_count(shapes[node.second_input]);
                    validation.bound = [shapes[node.first_input][node.axis]
                            unsignedLongLongValue];
                    validation.axis = node.axis;
                    [index_validations addObject:validation];
                    used[node.second_input] = 1U;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS: {
                    if (states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || node.second_input >= value_count
                            || node.auxiliary >= value_count
                            || states[node.second_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || states[node.auxiliary] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || !node_scatter_elements_matches(
                                    node,
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.auxiliary],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    NSUInteger position = route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                            ? node.second_input
                            : feed_position(node.second_input, feed_count, feed_indices);
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
                    validation.indexValue = node.second_input;
                    validation.indexType = declared_types[node.second_input];
                    validation.nodeIndex = node_index;
                    validation.feedPosition = position;
                    validation.preflight =
                            feed_position(node.second_input, feed_count, feed_indices)
                            != NSNotFound;
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
                case SYNAPTIK_METAL_CUSTOM_SCATTER_ADD: {
                    if (route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                            || !has_layouts
                            || node.input_count != 3U
                            || node.output_count != 1U
                            || node.attribute_kind != 3U
                            || node.attribute_count != 1U
                            || node.axis != 0U
                            || node.second_input >= value_count
                            || node.auxiliary >= value_count
                            || node.first_input == node.second_input
                            || node.first_input == node.auxiliary
                            || node.first_input == node.output
                            || node.second_input == node.auxiliary
                            || node.second_input == node.output
                            || node.auxiliary == node.output
                            || states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || states[node.second_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || states[node.auxiliary] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || declared_states[node.output]
                                    != SYNAPTIK_METAL_VALUE_CANONICAL) {
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    }
                    MPSShape *data_shape = shapes[node.first_input];
                    MPSShape *index_shape = shapes[node.second_input];
                    MPSShape *update_shape = shapes[node.auxiliary];
                    MPSShape *output_shape = shapes[node.output];
                    uint64_t data_extent = data_shape.count == 1U
                            ? data_shape[0].unsignedLongLongValue : 0U;
                    uint64_t update_extent = update_shape.count == 1U
                            ? update_shape[0].unsignedLongLongValue : 0U;
                    uint64_t index_width =
                            declared_types[node.second_input] == SYNAPTIK_METAL_TYPE_INT32
                            ? sizeof(int32_t)
                            : declared_types[node.second_input] == SYNAPTIK_METAL_TYPE_INT64
                                    ? sizeof(int64_t) : 0U;
                    if (index_shape.count != 1U
                            || output_shape.count != 1U
                            || index_shape[0].unsignedLongLongValue != update_extent
                            || output_shape[0].unsignedLongLongValue != data_extent
                            || data_extent == 0U
                            || update_extent == 0U
                            || data_extent > UINT32_MAX / sizeof(float)
                            || update_extent > UINT32_MAX / sizeof(float)
                            || index_width == 0U
                            || update_extent > UINT32_MAX / index_width
                            || value_strides[
                                    (size_t)node.first_input * SYNAPTIK_MAX_RANK] != 1U
                            || value_strides[
                                    (size_t)node.second_input * SYNAPTIK_MAX_RANK] != 1U
                            || value_strides[
                                    (size_t)node.auxiliary * SYNAPTIK_MAX_RANK] != 1U
                            || value_strides[
                                    (size_t)node.output * SYNAPTIK_MAX_RANK] != 1U
                            || layout_offsets[node.first_input] != 0U
                            || layout_offsets[node.second_input] != 0U
                            || layout_offsets[node.auxiliary] != 0U
                            || layout_offsets[node.output] != 0U
                            || layout_spans[node.first_input] != data_extent
                            || layout_spans[node.second_input] != update_extent
                            || layout_spans[node.auxiliary] != update_extent
                            || layout_spans[node.output] != data_extent) {
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    }
                    NSUInteger position =
                            feed_position(node.second_input, feed_count, feed_indices);
                    if (position == NSNotFound)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    SynaptikMetalIndexValidation *validation =
                            [SynaptikMetalIndexValidation new];
                    if (validation == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    validation.operation = node.operation;
                    validation.indexValue = node.second_input;
                    validation.indexType = declared_types[node.second_input];
                    validation.nodeIndex = node_index;
                    validation.feedPosition = node.second_input;
                    validation.preflight = YES;
                    validation.elementCount = update_extent;
                    validation.bound = data_extent;
                    validation.axis = 0U;
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
                case SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND:
                case SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: {
                    BOOL task0066_operation =
                            operation_is_task0066_selected(node.operation);
                    for (uint32_t input = 0U; input < node.input_count; input++) {
                        uint8_t input_state = states[node.inputs[input]];
                        if ((task0066_operation
                                        && input_state
                                                == SYNAPTIK_METAL_VALUE_UNAVAILABLE)
                                || (!task0066_operation
                                        && input_state
                                                != SYNAPTIK_METAL_VALUE_CANONICAL))
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
                        validation.indexValue = node.second_input;
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
                case SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS:
                    if (states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || !node_unfold_axis_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_CUSTOM_ONE_HOT: {
                    if (states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                            || node.second_input != UINT32_MAX
                            || !node_one_hot_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    NSUInteger position = route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                            ? node.first_input
                            : feed_position(node.first_input, feed_count, feed_indices);
                    if (position == NSNotFound)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    SynaptikMetalIndexValidation *validation =
                            [SynaptikMetalIndexValidation new];
                    if (validation == nil)
                        return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
                    validation.operation = node.operation;
                    validation.indexValue = node.first_input;
                    validation.indexType = declared_types[node.first_input];
                    validation.nodeIndex = node_index;
                    validation.feedPosition = position;
                    validation.preflight =
                            feed_position(node.first_input, feed_count, feed_indices)
                            != NSNotFound;
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
            BOOL low_precision_node =
                    low_precision_custom_function(node, value_types) != nil;
            switch ((SynaptikMetalOperation)node.operation) {
                case SYNAPTIK_METAL_CUSTOM_INITIAL_STATE:
                    type_valid = require_value_type(
                            value_types, node.output, SYNAPTIK_METAL_TYPE_INT64);
                    break;
                case SYNAPTIK_METAL_CUSTOM_DROPOUT: {
                    uint8_t input_type = declared_types[node.first_input];
                    type_valid = (input_type == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || synaptik_is_low_precision_type(input_type))
                            && require_value_type(
                                    value_types,
                                    node.second_input,
                                    SYNAPTIK_METAL_TYPE_INT64)
                            && declared_types[node.outputs[0]] == input_type
                            && require_value_type(
                                    value_types,
                                    node.outputs[1],
                                    SYNAPTIK_METAL_TYPE_BOOL)
                            && require_value_type(
                                    value_types,
                                    node.outputs[2],
                                    SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
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
                case SYNAPTIK_METAL_MPSGRAPH_EXP:
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
                case SYNAPTIK_METAL_CUSTOM_SIGMOID:
                case SYNAPTIK_METAL_MPSGRAPH_SUM:
                case SYNAPTIK_METAL_MPSGRAPH_MEAN:
                case SYNAPTIK_METAL_CUSTOM_SCALAR_MIN:
                case SYNAPTIK_METAL_CUSTOM_SCALAR_MAX:
                case SYNAPTIK_METAL_CUSTOM_CLAMP:
                case SYNAPTIK_METAL_CUSTOM_REDUCTION_MIN:
                case SYNAPTIK_METAL_CUSTOM_REDUCTION_MAX:
                case SYNAPTIK_METAL_CUSTOM_CUM_SUM:
                case SYNAPTIK_METAL_CUSTOM_CUM_PROD:
                case SYNAPTIK_METAL_MPSGRAPH_LOG_SUM_EXP:
                case SYNAPTIK_METAL_MPSGRAPH_VARIANCE:
                case SYNAPTIK_METAL_MPSGRAPH_STANDARD_DEVIATION:
                case SYNAPTIK_METAL_MPSGRAPH_L1_NORM:
                case SYNAPTIK_METAL_MPSGRAPH_L2_NORM:
                    type_valid = low_precision_node
                            ? synaptik_is_low_precision_type(
                                            declared_types[node.first_input])
                                    && declared_types[node.output]
                                            == declared_types[node.first_input]
                            : require_value_type(
                                            value_types, node.first_input,
                                            SYNAPTIK_METAL_TYPE_FLOAT32)
                                    && require_value_type(
                                            value_types, node.output,
                                            SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                case SYNAPTIK_METAL_CUSTOM_RESHAPE:
                case SYNAPTIK_METAL_CUSTOM_EXPAND:
                case SYNAPTIK_METAL_CUSTOM_PERMUTE:
                case SYNAPTIK_METAL_CUSTOM_EXPAND_DIMS:
                case SYNAPTIK_METAL_CUSTOM_SQUEEZE:
                case SYNAPTIK_METAL_CUSTOM_CONTIGUOUS:
                case SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = task0066_is_carrier(type)
                            && type == declared_types[node.output];
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
                    type_valid = low_precision_node
                            ? synaptik_is_low_precision_type(
                                            declared_types[node.first_input])
                                    && declared_types[node.second_input]
                                            == declared_types[node.first_input]
                                    && declared_types[node.output]
                                            == declared_types[node.first_input]
                            : require_value_type(
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
                    BOOL all_output_type = synaptik_is_low_precision_type(output_type);
                    for (uint32_t input = 0U; input < node.input_count; input++) {
                        uint8_t type = declared_types[node.inputs[input]];
                        any_float32 |= type == SYNAPTIK_METAL_TYPE_FLOAT32;
                        inputs_valid &= type == SYNAPTIK_METAL_TYPE_FLOAT32
                                || type == SYNAPTIK_METAL_TYPE_BFLOAT16;
                        all_output_type &= type == output_type;
                    }
                    type_valid = all_output_type
                            || (inputs_valid && any_float32
                                    && output_type == SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_MAX_POOL2D:
                case SYNAPTIK_METAL_CUSTOM_MAX_POOL3D: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = type == declared_types[node.output]
                            && (type == SYNAPTIK_METAL_TYPE_FLOAT64
                                    || type == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || type == SYNAPTIK_METAL_TYPE_BFLOAT16
                                    || type == SYNAPTIK_METAL_TYPE_FLOAT16);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL2D:
                case SYNAPTIK_METAL_CUSTOM_AVERAGE_POOL3D: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = type == declared_types[node.output]
                            && (type == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || synaptik_is_low_precision_type(type));
                    break;
                }
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
                    BOOL homogeneous_low = synaptik_is_low_precision_type(left)
                            && right == left && output == left;
                    type_valid = integral
                            ? output == promoted
                            : homogeneous_low
                                    || (output == SYNAPTIK_METAL_TYPE_FLOAT32
                                            && ((left == SYNAPTIK_METAL_TYPE_FLOAT32
                                                            && right
                                                                    == SYNAPTIK_METAL_TYPE_FLOAT32)
                                                    || (left
                                                                    == SYNAPTIK_METAL_TYPE_BFLOAT16
                                                            && right
                                                                    == SYNAPTIK_METAL_TYPE_FLOAT32)
                                                    || (left
                                                                    == SYNAPTIK_METAL_TYPE_FLOAT32
                                                            && right
                                                                    == SYNAPTIK_METAL_TYPE_BFLOAT16)));
                    break;
                }
                case SYNAPTIK_METAL_BOOL_IS_FINITE:
                case SYNAPTIK_METAL_BOOL_IS_NAN:
                case SYNAPTIK_METAL_BOOL_IS_INF:
                    type_valid = task0066_is_floating(
                                    declared_types[node.first_input])
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
                case SYNAPTIK_METAL_BOOL_WHERE: {
                    uint8_t true_type = declared_types[node.second_input];
                    uint8_t false_type = declared_types[node.auxiliary];
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_BOOL)
                            && declared_types[node.output]
                                    == task0066_promote_floating(
                                            true_type, false_type);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_GT:
                case SYNAPTIK_METAL_CUSTOM_GE:
                case SYNAPTIK_METAL_CUSTOM_LT:
                case SYNAPTIK_METAL_CUSTOM_LE:
                case SYNAPTIK_METAL_CUSTOM_EQ:
                case SYNAPTIK_METAL_CUSTOM_NE:
                    type_valid = (declared_types[node.first_input]
                                            == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || synaptik_is_low_precision_type(
                                            declared_types[node.first_input]))
                            && declared_types[node.second_input]
                                    == declared_types[node.first_input]
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                case SYNAPTIK_METAL_CUSTOM_GATHER: {
                    uint8_t data_type = declared_types[node.first_input];
                    uint8_t index_type = declared_types[node.second_input];
                    type_valid = task0066_is_carrier(data_type)
                            && data_type == declared_types[node.output]
                            && (index_type == SYNAPTIK_METAL_TYPE_INT32
                                    || index_type == SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS: {
                    uint8_t data_type = declared_types[node.first_input];
                    uint8_t index_type = declared_types[node.second_input];
                    type_valid = task0066_is_carrier(data_type)
                            && data_type == declared_types[node.auxiliary]
                            && data_type == declared_types[node.output]
                            && (index_type == SYNAPTIK_METAL_TYPE_INT32
                                    || index_type == SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_ONE_HOT: {
                    uint8_t index_type = declared_types[node.first_input];
                    type_valid = (index_type == SYNAPTIK_METAL_TYPE_INT32
                                    || index_type == SYNAPTIK_METAL_TYPE_INT64)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_BOOL);
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_CAST: {
                    uint8_t source = declared_types[node.first_input];
                    uint8_t target = declared_types[node.output];
                    type_valid = task0066_is_carrier(source)
                            && task0066_is_carrier(target)
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
                                    || type == SYNAPTIK_METAL_TYPE_BFLOAT16
                                    || type == SYNAPTIK_METAL_TYPE_FLOAT16);
                    NSUInteger dimensions = node.operation
                                    == SYNAPTIK_METAL_CUSTOM_UNFOLD2D
                            ? 2U : 3U;
                    uint32_t padded_kind = dimensions == 2U ? 20U : 23U;
                    if (node.attribute_kind == padded_kind
                            && node.attribute_values[dimensions * 4U + 1U] != type)
                        type_valid = NO;
                    break;
                }
                case SYNAPTIK_METAL_CUSTOM_SCATTER_ADD: {
                    uint8_t data_type = declared_types[node.first_input];
                    uint8_t index_type = declared_types[node.second_input];
                    type_valid = (data_type == SYNAPTIK_METAL_TYPE_FLOAT32
                                    || synaptik_is_low_precision_type(data_type))
                            && declared_types[node.auxiliary] == data_type
                            && declared_types[node.output] == data_type
                            && (index_type == SYNAPTIK_METAL_TYPE_INT32
                                    || index_type == SYNAPTIK_METAL_TYPE_INT64);
                    break;
                }
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
                case SYNAPTIK_METAL_MPSGRAPH_FOLD_AXIS: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = task0066_is_carrier(type)
                            && type != SYNAPTIK_METAL_TYPE_BOOL
                            && type == declared_types[node.output];
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_FOLD2D:
                case SYNAPTIK_METAL_MPSGRAPH_FOLD3D: {
                    uint8_t type = declared_types[node.first_input];
                    type_valid = type == declared_types[node.output]
                            && task0066_is_floating(type);
                    break;
                }
                default:
                    break;
            }
            if (!type_valid) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            uint8_t output_state;
            if (node.operation == SYNAPTIK_METAL_CUSTOM_SELECT
                    || node.operation == SYNAPTIK_METAL_CUSTOM_SLICE) {
                output_state = has_layouts
                        ? declared_states[node.output]
                        : SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT;
                if (output_state != SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                        && output_state
                                != SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            } else {
                output_state = affine_view
                        ? SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                        : SYNAPTIK_METAL_VALUE_CANONICAL;
            }
            for (uint32_t output = 0U; output < node.output_count; output++) {
                uint32_t value = node.outputs[output];
                if (has_layouts && declared_states[value] != output_state)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                states[value] = output_state;
                used[value] = 1U;
                produced[value] = 1U;
            }
            if (!zero_input) used[node.first_input] = 1U;
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
        NSMutableData *physical_state_data = [state_data mutableCopy];
        NSMutableData *target_relative_data = [NSMutableData dataWithLength:value_count];
        size_t physical_stride_bytes =
                (size_t)value_count * SYNAPTIK_MAX_RANK * sizeof(uint64_t);
        NSMutableData *physical_stride_data =
                [NSMutableData dataWithLength:physical_stride_bytes];
        NSMutableData *physical_offset_data =
                [NSMutableData dataWithLength:(size_t)value_count * sizeof(uint64_t)];
        NSMutableData *physical_span_data =
                [NSMutableData dataWithLength:(size_t)value_count * sizeof(uint64_t)];
        if (physical_state_data == nil || target_relative_data == nil
                || physical_stride_data == nil || physical_offset_data == nil
                || physical_span_data == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        uint8_t *physical_states = physical_state_data.mutableBytes;
        uint8_t *target_relative = target_relative_data.mutableBytes;
        uint64_t *physical_value_strides = physical_stride_data.mutableBytes;
        uint64_t *physical_layout_offsets = physical_offset_data.mutableBytes;
        uint64_t *physical_layout_spans = physical_span_data.mutableBytes;
        for (uint32_t node_index = 0U; node_index < node_count; node_index++) {
            SynaptikMetalDecodedNode node = nodes[node_index];
            if (node.operation == SYNAPTIK_METAL_CUSTOM_SELECT
                    || node.operation == SYNAPTIK_METAL_CUSTOM_SLICE)
                target_relative[node.output] = 1U;
        }
        for (uint32_t value = 0U; value < value_count; value++) {
            if (physical_states[value] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                    && target_relative[value] != 0U) {
                physical_states[value] = SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT;
            } else if (physical_states[value] == SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                    && local_transpose[value] != 2U
                    && local_singleton_height[value] != 2U) {
                /*
                 * Ordinary affine nodes run as nested MPSGraph steps into their own dense
                 * buffers. Only the authenticated MATMUL/Conv view fusions skip that step and
                 * bind the original source buffer. Custom consumers must index the representation
                 * actually bound at runtime rather than reapplying the declared logical view.
                 */
                physical_states[value] = SYNAPTIK_METAL_VALUE_CANONICAL;
            }
            uint64_t offset = 0U;
            if (!custom_physical_layout(
                        shapes[value], value, value_strides, layout_offsets,
                        physical_states,
                        &physical_value_strides[
                                (size_t)value * SYNAPTIK_MAX_RANK],
                        &offset))
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            physical_layout_offsets[value] = offset;
            physical_layout_spans[value] =
                    physical_states[value] == SYNAPTIK_METAL_VALUE_CANONICAL
                    ? element_counts[value].unsignedLongLongValue
                    : layout_spans[value];
        }
        for (uint32_t node_index = 0U; node_index < node_count; node_index++) {
            if (!task0066_derive_physical_view(
                        nodes[node_index], value_ranks, value_dimensions,
                        physical_value_strides, physical_layout_offsets,
                        physical_layout_spans))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }
        for (SynaptikMetalIndexValidation *validation in index_validations) {
            if (!configure_index_validation_layout(
                        validation,
                        shapes,
                        physical_value_strides,
                        physical_layout_offsets,
                        physical_states))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }
        NSMutableArray<NSNumber *> *bytes = [NSMutableArray arrayWithCapacity:value_count];
        for (uint32_t value = 0; value < value_count; value++) {
            if (used[value] == 0U || value_types[value] == SYNAPTIK_METAL_TYPE_UNAVAILABLE)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            uint64_t width = value_type_width(value_types[value]);
            uint64_t elements =
                    physical_states[value] == SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT
                    ? physical_layout_spans[value]
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
            NSMutableArray<NSNumber *> *stride_values =
                    [NSMutableArray arrayWithCapacity:value_ranks[value]];
            if (stride_values == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            for (uint32_t axis = 0U; axis < value_ranks[value]; axis++)
                [stride_values addObject:@(
                        physical_value_strides[
                                (size_t)value * SYNAPTIK_MAX_RANK + axis])];
            [physical_strides addObject:[stride_values copy]];
            [physical_offsets addObject:@(physical_layout_offsets[value])];
        }
        if (contains_custom) {
            SynaptikMetalContextBox *ctx = (__bridge SynaptikMetalContextBox *)context;
            MTLCompileOptions *options = [MTLCompileOptions new];
            if (options == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            options.mathMode = MTLMathModeSafe;
            options.languageVersion = MTLLanguageVersion3_2;
            NSError *library_error = nil;
            NSString *fixed_source = synaptik_authenticated_fixed_source();
            if (fixed_source == nil)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            NSString *generated_source = SynaptikPointwiseGeneratedSource(
                    fusion->steps,
                    fusion->step_count,
                    fusion->instructions,
                    fusion->instruction_count,
                    fusion->expected_generated_bytes);
            if (generated_source == nil)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            NSString *kernel_source = [fixed_source stringByAppendingString:generated_source];
            if ([kernel_source lengthOfBytesUsingEncoding:NSUTF8StringEncoding]
                            != fusion->expected_total_bytes
                    || !synaptik_assembled_source_is_authentic(
                            fixed_source, generated_source, kernel_source))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            id<MTLLibrary> library = [ctx.device
                    newLibraryWithSource:kernel_source options:options error:&library_error];
            if (library == nil || library_error != nil)
                return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            id<MTLLibrary> low_precision_library = nil;
            if (contains_low_precision) {
                NSString *low_precision_source =
                        synaptik_authenticated_low_precision_source();
                if (low_precision_source == nil)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                NSError *low_precision_error = nil;
                NSString *combined_low_precision_source =
                        [fixed_source stringByAppendingString:low_precision_source];
                low_precision_library = [ctx.device
                        newLibraryWithSource:combined_low_precision_source
                        options:options
                        error:&low_precision_error];
                if (low_precision_library == nil || low_precision_error != nil)
                    return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
            }
            NSMutableArray<SynaptikMetalProgramStep *> *steps =
                    [NSMutableArray arrayWithCapacity:fusion->step_count];
            NSMutableData *generated_mark_data = [NSMutableData
                    dataWithLength:(NSUInteger)node_count * sizeof(uint32_t)];
            if (steps == nil || generated_mark_data == nil)
                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            uint32_t *generated_marks = generated_mark_data.mutableBytes;
            memset(generated_marks, 0xff, (size_t)node_count * sizeof(uint32_t));
            for (uint32_t ordinal = 0U; ordinal < fusion->step_count; ordinal++) {
                SynaptikPointwiseStepRecord record = fusion->steps[ordinal];
                if (record.kind != SYNAPTIK_POINTWISE_GENERATED_STEP
                        && record.kind != SYNAPTIK_ANCHOR_EPILOGUE_STEP)
                    continue;
                if (record.member_start > fusion->member_count
                        || record.member_count > fusion->member_count - record.member_start)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                for (uint32_t member = 0U; member < record.member_count; member++) {
                    uint32_t node = fusion->members[record.member_start + member];
                    if (node >= node_count || generated_marks[node] != UINT32_MAX)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    generated_marks[node] = member == 0U ? ordinal : UINT32_MAX - 1U;
                }
            }
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
            uint32_t plan_ordinal = 0U;
#endif
            for (uint32_t node_index = 0U; node_index < node_count; node_index++) {
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
                while (plan_ordinal < fusion->step_count
                        && node_index >= fusion->steps[plan_ordinal].member_start
                                + fusion->steps[plan_ordinal].member_count)
                    plan_ordinal++;
                if (plan_ordinal >= fusion->step_count)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
#endif
                SynaptikMetalDecodedNode node = nodes[node_index];
                uint32_t generated = generated_marks[node_index];
                if (generated == UINT32_MAX - 1U) continue;
                if (generated != UINT32_MAX) {
                    SynaptikPointwiseStepRecord record = fusion->steps[generated];
                    SynaptikMetalProgramStep *step = nil;
                    if (record.kind == SYNAPTIK_POINTWISE_GENERATED_STEP) {
                        if (record.binding_count != 2U
                                || record.binding_start > fusion->binding_count
                                || record.binding_count
                                        > fusion->binding_count - record.binding_start)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        SynaptikPointwiseBindingRecord input =
                                fusion->bindings[record.binding_start];
                        SynaptikPointwiseBindingRecord output =
                                fusion->bindings[record.binding_start + 1U];
                        step = make_generated_pointwise_step(
                                generated, input.value, output.value,
                                shapes, ctx.device, library);
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
                        if (step != nil) {
                            step.planOrdinal = generated;
                            step.planKind = SYNAPTIK_POINTWISE_GENERATED_STEP;
                        }
#endif
                    } else if (record.kind == SYNAPTIK_ANCHOR_EPILOGUE_STEP) {
                        step = make_anchor_epilogue_step(
                                generated,
                                record,
                                fusion->members,
                                fusion->instructions,
                                nodes,
                                shapes,
                                value_types,
                                physical_value_strides,
                                physical_layout_offsets,
                                physical_states,
                                local_transpose_sources,
                                local_singleton_height_sources,
                                ctx.device,
                                library);
                    }
                    if (step == nil)
                        return SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED;
                    [steps addObject:step];
                    continue;
                }
                if (local_transpose[node.output] == 2U
                        || local_singleton_height[node.output] == 2U) continue;
                BOOL custom_step = node_uses_custom_kernel(node, shapes, value_types)
                        || (node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
                                && (local_transpose[node.first_input] != 0U
                                        || local_transpose[node.second_input] != 0U));
                if (custom_step) {
                    uint32_t final_stage =
                            node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS
                                            || node.operation
                                                    == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND
                                    ? 1U : 0U;
                    for (uint32_t stage = 0U; stage <= final_stage; stage++) {
                        SynaptikMetalProgramStep *step = make_custom_step(
                                node, shapes, value_types, physical_value_strides,
                                physical_layout_offsets, physical_states,
                                local_transpose_sources,
                                local_singleton_height_sources,
                                ctx.device, library, low_precision_library, stage);
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
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
                        step.planOrdinal = plan_ordinal;
                        step.planKind = 1U;
#endif
                        [steps addObject:step];
                    }
                    continue;
                }

                uint32_t input_count = 1U;
                if (node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS)
                    input_count = 3U;
                else if (node.operation == SYNAPTIK_METAL_MPSGRAPH_ADD
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_SUB
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_MUL
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_DIV
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_TENSOR_POW
                        || node.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL
                        || node.operation == SYNAPTIK_METAL_CUSTOM_GATHER
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
                    memcpy(
                            &compact_strides[
                                    (size_t)input_index * SYNAPTIK_MAX_RANK],
                            &physical_value_strides[
                                    (size_t)feed_source * SYNAPTIK_MAX_RANK],
                            SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                    compact_offsets[input_index] =
                            physical_layout_offsets[feed_source];
                    compact_spans[input_index] = physical_layout_spans[feed_source];
                    compact_states[input_index] = physical_states[feed_source];
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
                                        != SYNAPTIK_METAL_CUSTOM_PERMUTE)
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
                    memcpy(
                            &compact_strides[
                                    (size_t)input_count * SYNAPTIK_MAX_RANK],
                            &physical_value_strides[
                                    (size_t)node.output * SYNAPTIK_MAX_RANK],
                            SYNAPTIK_MAX_RANK * sizeof(uint64_t));
                    compact_offsets[input_count] =
                            physical_layout_offsets[node.output];
                    compact_spans[input_count] = physical_layout_spans[node.output];
                    compact_states[input_count] = physical_states[node.output];
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
                    if (node.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS)
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
                        compact_strides,
                        compact_offsets,
                        compact_spans,
                        compact_states,
                        compact_types,
                        compact_node_count,
                        compact_nodes,
                        input_count,
                        compact_feeds,
                        1U,
                        &compact_target,
                        NULL,
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
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
                step.planOrdinal = plan_ordinal;
                step.planKind = 2U;
#endif
                [steps addObject:step];
            }
            NSMutableArray<NSNumber *> *materialized_bytes =
                    [NSMutableArray arrayWithCapacity:fusion->materialized_count];
            NSMutableArray<MPSShape *> *materialized_shapes =
                    [NSMutableArray arrayWithCapacity:fusion->materialized_count];
            NSMutableArray<NSArray<NSNumber *> *> *materialized_strides =
                    [NSMutableArray arrayWithCapacity:fusion->materialized_count];
            NSMutableArray<NSNumber *> *materialized_offsets =
                    [NSMutableArray arrayWithCapacity:fusion->materialized_count];
            NSMutableArray<NSNumber *> *target_bytes =
                    [NSMutableArray arrayWithCapacity:target_count];
            NSMutableArray<NSNumber *> *target_values =
                    [NSMutableArray arrayWithCapacity:target_count];
            if (materialized_bytes == nil || materialized_shapes == nil
                    || materialized_strides == nil || materialized_offsets == nil
                    || target_bytes == nil || target_values == nil)
                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            for (uint32_t slot = 0U; slot < fusion->materialized_count; slot++) {
                uint32_t value = fusion->materialized_values[slot];
                if (value >= value_count || fusion->program_to_slot[value] != slot)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                [materialized_bytes addObject:bytes[value]];
                [materialized_shapes addObject:shapes[value]];
                [materialized_strides addObject:physical_strides[value]];
                [materialized_offsets addObject:physical_offsets[value]];
            }
            for (uint32_t target = 0U; target < target_count; target++) {
                uint32_t value = target_indices[target];
                uint32_t slot = fusion->program_to_slot[value];
                if (slot == UINT32_MAX) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                [target_bytes addObject:bytes[value]];
                [target_values addObject:@(slot)];
            }
            for (SynaptikMetalProgramStep *step in steps)
                if (!translate_program_step(step, fusion->program_to_slot, value_count))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            for (SynaptikMetalIndexValidation *validation in index_validations) {
                NSUInteger value = validation.feedPosition;
                if (value >= value_count || fusion->program_to_slot[value] == UINT32_MAX)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                validation.feedPosition = fusion->program_to_slot[value];
            }
            SynaptikMetalExecutableBox *box = [SynaptikMetalExecutableBox new];
            if (box == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            box.customProgram = YES;
            box.valueCount = fusion->materialized_count;
            box.valueBytes = [materialized_bytes copy];
            box.targetBytes = [target_bytes copy];
            box.targetValueIndices = [target_values copy];
            box.programSteps = [steps copy];
            box.indexValidations = [index_validations copy];
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
            if (fusion->manifest_digest == NULL)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            box.manifestDigest = [NSData
                    dataWithBytes:fusion->manifest_digest
                    length:CC_SHA256_DIGEST_LENGTH];
            if (box.manifestDigest == nil)
                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
#endif
            NSMutableArray<NSNumber *> *canonical_bool_inputs =
                    [NSMutableArray array];
            if (canonical_bool_inputs == nil)
                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            for (uint32_t feed = 0U; feed < feed_count; feed++) {
                uint32_t value = feed_indices[feed];
                if (declared_types[value] == SYNAPTIK_METAL_TYPE_BOOL)
                    [canonical_bool_inputs addObject:@(fusion->program_to_slot[value])];
            }
            box.canonicalBoolInputIndices = [canonical_bool_inputs copy];
            box.inputShapes = [materialized_shapes copy];
            box.inputStrides = [materialized_strides copy];
            box.inputOffsets = [materialized_offsets copy];
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
                case SYNAPTIK_METAL_MPSGRAPH_EXP:
                    output = [graph exponentWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_SIGMOID: {
                    // Reinterpret stored bits before comparison: floating comparison may DAZ
                    // subnormals and would choose the wrong exact Model sign branch.
                    MPSGraphTensor *bits = [graph reinterpretCastTensor:first
                            toType:MPSDataTypeInt32 name:nil];
                    MPSGraphTensor *zero_bits = exact_int32_scalar(graph, 0);
                    MPSGraphTensor *magnitude = [graph bitwiseANDWithPrimaryTensor:bits
                            secondaryTensor:exact_int32_scalar(graph, INT32_MAX) name:nil];
                    MPSGraphTensor *signed_negative = [graph lessThanWithPrimaryTensor:bits
                            secondaryTensor:zero_bits name:nil];
                    MPSGraphTensor *nonzero = [graph greaterThanWithPrimaryTensor:magnitude
                            secondaryTensor:zero_bits name:nil];
                    MPSGraphTensor *negative = [graph logicalANDWithPrimaryTensor:signed_negative
                            secondaryTensor:nonzero name:nil];
                    MPSGraphTensor *one = exact_float32_scalar(graph, UINT32_C(0x3f800000));
                    MPSGraphTensor *negative_input = [graph negativeWithTensor:first name:nil];
                    MPSGraphTensor *exponent_input = [graph selectWithPredicateTensor:negative
                            truePredicateTensor:first
                            falsePredicateTensor:negative_input name:nil];
                    MPSGraphTensor *exponent = [graph exponentWithTensor:exponent_input name:nil];
                    MPSGraphTensor *numerator = [graph selectWithPredicateTensor:negative
                            truePredicateTensor:exponent
                            falsePredicateTensor:one name:nil];
                    MPSGraphTensor *denominator = [graph additionWithPrimaryTensor:one
                            secondaryTensor:exponent name:nil];
                    output = [graph divisionWithPrimaryTensor:numerator
                            secondaryTensor:denominator name:nil];
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
                            if (node.attribute_values[0] == 0U) {
                                // MPSGraph folds scalar +0 to an identity, incorrectly leaving
                                // raw -0 unchanged. Correct only that exact input bit pattern;
                                // an output==0 repair would also alter allowed DAZ/FTZ results.
                                MPSGraphTensor *raw_input = [graph reinterpretCastTensor:first
                                        toType:MPSDataTypeInt32 name:nil];
                                MPSGraphTensor *negative_zero = exact_int32_scalar(
                                        graph, (int32_t)UINT32_C(0x80000000));
                                MPSGraphTensor *is_negative_zero =
                                        [graph equalWithPrimaryTensor:raw_input
                                                secondaryTensor:negative_zero name:nil];
                                output = [graph selectWithPredicateTensor:is_negative_zero
                                        truePredicateTensor:scalar
                                        falsePredicateTensor:output name:nil];
                            }
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
                case SYNAPTIK_METAL_CUSTOM_CONTIGUOUS:
                    output = [graph reshapeTensor:first
                            withShape:shapes[node.output] name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_RESHAPE:
                    output = [graph reshapeTensor:first
                            withShape:node_attribute_array(node) name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_EXPAND:
                    output = [graph broadcastTensor:first
                            toShape:node_attribute_array(node) name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_PERMUTE:
                    output = [graph transposeTensor:first
                            permutation:node_attribute_array(node) name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_EXPAND_DIMS:
                    output = [graph expandDimsOfTensor:first
                            axis:(NSInteger)node.axis name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_SQUEEZE:
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
                case SYNAPTIK_METAL_CUSTOM_UNFOLD_AXIS: {
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
                case SYNAPTIK_METAL_CUSTOM_GATHER:
                    output = [graph gatherWithUpdatesTensor:first
                            indicesTensor:second
                            axis:(NSInteger)node.axis
                            batchDimensions:0U
                            name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS:
                    output = [graph scatterAlongAxis:(NSInteger)node.axis
                            withDataTensor:first
                            updatesTensor:auxiliary
                            indicesTensor:second
                            mode:MPSGraphScatterModeSet
                            name:nil];
                    break;
                case SYNAPTIK_METAL_CUSTOM_ONE_HOT:
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
        descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        descriptor.waitForCompilationCompletion = YES;
        if (descriptor.optimizationLevel != MPSGraphOptimizationLevel1
                || !descriptor.waitForCompilationCompletion)
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

static NSData *synaptik_pointwise_manifest(
        const uint8_t *program,
        uint64_t values_offset,
        uint32_t value_count,
        const uint32_t *value_ranks,
        const uint64_t *value_dimensions,
        uint32_t node_count,
        uint32_t feed_count,
        uint32_t target_count,
        const uint32_t *target_indices,
        const SynaptikPointwisePlan *fusion,
        uint32_t rejected_node,
        uint32_t cap_reason) {
    if (program == NULL || value_ranks == NULL || value_dimensions == NULL
            || target_indices == NULL || fusion == NULL)
        return nil;
    NSMutableString *text = [NSMutableString stringWithCapacity:4096U];
    if (text == nil) return nil;
    [text appendString:@"format 2\nschema 20\ngenerator 2\nroute 3\n"];
    [text appendFormat:@"counts %u %u %u %u %u %u %u %u %u\n",
            value_count, node_count, feed_count, target_count,
            fusion->step_count, fusion->member_count, fusion->binding_count,
            fusion->materialized_count, fusion->instruction_count];
    [text appendFormat:@"stop %u %u\n", rejected_node, cap_reason];
    [text appendFormat:@"source %u 84603 %u\n",
            fusion->expected_generated_bytes, fusion->expected_total_bytes];
    [text appendString:@"caps 32 256 16384 262144 1048576\n"];
    [text appendString:@"execution-caps 64 512\n"];
    [text appendString:@"source-size-table 1\n"];
    [text appendString:@"fixed-corpus-bytes 84603\n"];
    [text appendString:@"opcodes floor=1 ceil=2 sign=3 relu=4\n"];
    [text appendString:@"anchor-opcodes scalar-mul=1 add=2 relu=3 clamp=4\n"];
    [text appendString:@"pointmeta-abi size=32 align=8 elementCount=u64@0 gridWidth=u64@8 gridHeight=u64@16 scalar=u32@24 reserved=u32@28\n"];
    for (uint32_t value = 0U; value < value_count; value++) {
        const uint8_t *descriptor = program + values_offset + (uint64_t)value * 40U;
        uint32_t flags = synaptik_read_le32(descriptor + 16U);
        [text appendFormat:@"value %u %u %u %u %u",
                value,
                synaptik_read_le32(descriptor),
                value_ranks[value],
                (flags & 1U) != 0U ? 1U : 0U,
                (flags & 8U) != 0U ? 1U : 0U];
        for (uint32_t axis = 0U; axis < value_ranks[value]; axis++) {
            [text appendFormat:@" %llu",
                    (unsigned long long)value_dimensions[
                            (size_t)value * SYNAPTIK_MAX_RANK + axis]];
        }
        [text appendString:@"\n"];
    }
    for (uint32_t slot = 0U; slot < fusion->materialized_count; slot++)
        [text appendFormat:@"materialized %u %u\n",
                slot, fusion->materialized_values[slot]];
    for (uint32_t target = 0U; target < target_count; target++) {
        uint32_t value = target_indices[target];
        [text appendFormat:@"target %u %u %u\n",
                target, value, fusion->program_to_slot[value]];
    }
    for (uint32_t ordinal = 0U; ordinal < fusion->step_count; ordinal++) {
        SynaptikPointwiseStepRecord step = fusion->steps[ordinal];
        [text appendFormat:@"step %u %u %u %u %u %u %u %u %u %u\n",
                ordinal, step.kind, step.member_start, step.member_count,
                step.binding_start, step.binding_count,
                step.instruction_start, step.instruction_count,
                step.function_bytes, step.flags];
        if (step.kind == SYNAPTIK_POINTWISE_GENERATED_STEP) {
            if (step.binding_count != 2U
                    || step.binding_start > fusion->binding_count - step.binding_count)
                return nil;
            SynaptikPointwiseBindingRecord output_binding =
                    fusion->bindings[step.binding_start + 1U];
            if (output_binding.access != 2U || output_binding.value >= value_count)
                return nil;
            uint32_t value = output_binding.value;
            uint64_t elements = 1U;
            for (uint32_t axis = 0U; axis < value_ranks[value]; axis++) {
                uint64_t dimension = value_dimensions[
                        (size_t)value * SYNAPTIK_MAX_RANK + axis];
                if (dimension == 0U || elements > UINT64_MAX / dimension)
                    return nil;
                elements *= dimension;
            }
            [text appendFormat:@"pointmeta %u %llu %llu 1 0 0\n",
                    ordinal, (unsigned long long)elements, (unsigned long long)elements];
        }
    }
    for (uint32_t ordinal = 0U; ordinal < fusion->member_count; ordinal++)
        [text appendFormat:@"member %u %u\n", ordinal, fusion->members[ordinal]];
    for (uint32_t ordinal = 0U; ordinal < fusion->binding_count; ordinal++) {
        SynaptikPointwiseBindingRecord binding = fusion->bindings[ordinal];
        [text appendFormat:@"binding %u %u %u %u %u %u\n",
                ordinal, binding.step, binding.argument, binding.access,
                binding.slot, binding.value];
    }
    for (uint32_t ordinal = 0U; ordinal < fusion->instruction_count; ordinal++) {
        SynaptikPointwiseInstructionRecord instruction = fusion->instructions[ordinal];
        [text appendFormat:
                @"instruction %u %u %u %u %u %u %u %u %u %u %u %u %u %016llx %016llx\n",
                ordinal, instruction.step, instruction.relative_node,
                instruction.opcode, instruction.semantic_site,
                instruction.input_count, instruction.input0,
                instruction.input1, instruction.input2, instruction.output,
                instruction.immediate_count, instruction.immediate_type0,
                instruction.immediate_type1,
                (unsigned long long)instruction.raw0,
                (unsigned long long)instruction.raw1];
    }
    [text appendFormat:@"generated-units %u\nend\n", fusion->generated_unit_count];
    return [text dataUsingEncoding:NSASCIIStringEncoding allowLossyConversion:NO];
}

SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_executable_create(
        void *context, const uint8_t *program, uint32_t program_bytes,
        void **out_executable) {
    if (out_executable == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_executable = NULL;
    if (context == NULL || program == NULL || program_bytes < 124U
            || program_bytes > (uint32_t)INT32_MAX)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { @autoreleasepool {
        if (synaptik_read_le32(program) != UINT32_C(0x30324d53)
                || synaptik_read_le32(program + 4U) != 20U
                || synaptik_read_le32(program + 8U) != 124U
                || synaptik_read_le32(program + 12U) != program_bytes)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        uint32_t route = synaptik_read_le32(program + 16U);
        uint32_t generator_schema = synaptik_read_le32(program + 20U);
        uint32_t flags = synaptik_read_le32(program + 24U);
        if ((route != SYNAPTIK_METAL_ROUTE_MPSGRAPH
                    && route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM))
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;

        uint32_t value_count = synaptik_read_le32(program + 28U);
        uint32_t node_count = synaptik_read_le32(program + 32U);
        uint32_t feed_count = synaptik_read_le32(program + 36U);
        uint32_t target_count = synaptik_read_le32(program + 40U);
        uint32_t dimension_count = synaptik_read_le32(program + 44U);
        uint32_t stride_count = synaptik_read_le32(program + 48U);
        uint32_t reference_count = synaptik_read_le32(program + 52U);
        uint32_t attribute_count = synaptik_read_le32(program + 56U);
        uint32_t step_count = synaptik_read_le32(program + 60U);
        uint32_t member_count = synaptik_read_le32(program + 64U);
        uint32_t binding_count = synaptik_read_le32(program + 68U);
        uint32_t materialized_count = synaptik_read_le32(program + 72U);
        uint32_t instruction_count = synaptik_read_le32(program + 76U);
        uint32_t manifest_bytes = synaptik_read_le32(program + 80U);
        uint32_t manifest_digest_bytes = synaptik_read_le32(program + 84U);
        uint32_t generated_unit_count = synaptik_read_le32(program + 88U);
        uint32_t expected_generated_bytes = synaptik_read_le32(program + 92U);
        uint32_t expected_fixed_bytes = synaptik_read_le32(program + 96U);
        uint32_t expected_total_bytes = synaptik_read_le32(program + 100U);
        uint32_t rejected_node = synaptik_read_le32(program + 104U);
        uint32_t cap_reason = synaptik_read_le32(program + 108U);
        uint32_t function_cap = synaptik_read_le32(program + 112U);
        uint32_t generated_cap = synaptik_read_le32(program + 116U);
        uint32_t total_cap = synaptik_read_le32(program + 120U);
        if (value_count == 0U || node_count == 0U || target_count == 0U)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        BOOL extension = route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM;
        if (extension) {
            if (generator_schema != SYNAPTIK_POINTWISE_GENERATOR_SCHEMA || flags != 1U
                    || step_count == 0U || member_count == 0U
                    || member_count != node_count || binding_count == 0U
                    || materialized_count == 0U || materialized_count > value_count
                    || manifest_bytes == 0U || manifest_digest_bytes != CC_SHA256_DIGEST_LENGTH
                    || generated_unit_count > SYNAPTIK_POINTWISE_MAX_UNITS
                    || instruction_count > SYNAPTIK_EXECUTION_MAX_INSTRUCTIONS
                    || expected_fixed_bytes != SYNAPTIK_POINTWISE_FIXED_CORPUS_BYTES
                    || expected_generated_bytes > SYNAPTIK_POINTWISE_MAX_GENERATED_BYTES
                    || expected_total_bytes
                            != expected_fixed_bytes + expected_generated_bytes
                    || expected_total_bytes > SYNAPTIK_POINTWISE_MAX_TOTAL_BYTES
                    || function_cap != SYNAPTIK_POINTWISE_MAX_FUNCTION_BYTES
                    || generated_cap != SYNAPTIK_POINTWISE_MAX_GENERATED_BYTES
                    || total_cap != SYNAPTIK_POINTWISE_MAX_TOTAL_BYTES
                    || cap_reason > 5U
                    || (cap_reason == 0U) != (rejected_node == UINT32_MAX))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        } else if (generator_schema != 0U || flags != 0U || step_count != 0U
                || member_count != 0U || binding_count != 0U || materialized_count != 0U
                || instruction_count != 0U || manifest_bytes != 0U
                || manifest_digest_bytes != 0U || generated_unit_count != 0U
                || expected_generated_bytes != 0U || expected_fixed_bytes != 0U
                || expected_total_bytes != 0U || rejected_node != UINT32_MAX
                || cap_reason != 0U || function_cap != 0U || generated_cap != 0U
                || total_cap != 0U) {
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }

        uint64_t values_offset = 124U;
        uint64_t nodes_offset = values_offset + (uint64_t)value_count * 40U;
        uint64_t dimensions_offset = nodes_offset + (uint64_t)node_count * 32U;
        uint64_t strides_offset =
                dimensions_offset + (uint64_t)dimension_count * 8U;
        uint64_t references_offset =
                strides_offset + (uint64_t)stride_count * 8U;
        uint64_t references_end =
                references_offset + (uint64_t)reference_count * 4U;
        uint64_t attributes_offset = references_end;
        uint64_t steps_offset = attributes_offset + (uint64_t)attribute_count * 8U;
        uint64_t members_offset = steps_offset + (uint64_t)step_count * 40U;
        uint64_t bindings_offset = members_offset + (uint64_t)member_count * 4U;
        uint64_t materialized_offset =
                bindings_offset + (uint64_t)binding_count * 24U;
        uint64_t instructions_offset =
                materialized_offset + (uint64_t)materialized_count * 4U;
        uint64_t manifest_offset =
                instructions_offset + (uint64_t)instruction_count * 64U;
        uint64_t digest_offset = manifest_offset + manifest_bytes;
        uint64_t image_end = digest_offset + manifest_digest_bytes;
        if (image_end != program_bytes)
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
        NSMutableData *rank_zero_anchor_role_data =
                [NSMutableData dataWithLength:(NSUInteger)node_count];
        NSMutableData *feed_data =
                [NSMutableData dataWithLength:(NSUInteger)feed_count * sizeof(uint32_t)];
        NSMutableData *target_data =
                [NSMutableData dataWithLength:(NSUInteger)target_count * sizeof(uint32_t)];
        NSMutableData *topology_data =
                [NSMutableData dataWithLength:(NSUInteger)value_count * 4U];
        if (rank_data == nil || dimension_data == nil || stride_data == nil
                || layout_offset_data == nil || layout_span_data == nil
                || layout_state_data == nil || type_data == nil || node_data == nil
                || rank_zero_anchor_role_data == nil
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
        uint8_t *rank_zero_anchor_roles = rank_zero_anchor_role_data.mutableBytes;
        uint32_t *feed_indices = feed_data.mutableBytes;
        uint32_t *target_indices = target_data.mutableBytes;
        uint8_t *available = topology_data.mutableBytes;
        uint8_t *produced = available + value_count;
        uint8_t *consumed = produced + value_count;
        uint8_t *targeted = consumed + value_count;

        uint32_t dimension_cursor = 0U;
        uint32_t stride_cursor = 0U;
        BOOL contains_declared_low_precision = NO;
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
            if (type < SYNAPTIK_METAL_TYPE_FLOAT32 || type > SYNAPTIK_METAL_TYPE_FLOAT16
                    || rank > SYNAPTIK_MAX_RANK || dimension_offset != dimension_cursor
                    || dimension_cursor > dimension_count
                    || rank > dimension_count - dimension_cursor || (flags & ~15U) != 0U
                    || storage_offset > (uint64_t)INT64_MAX
                    || referenced_span > (uint64_t)INT64_MAX)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            value_ranks[value] = rank;
            declared_types[value] = (uint8_t)type;
            contains_declared_low_precision |=
                    type == SYNAPTIK_METAL_TYPE_BFLOAT16
                    || type == SYNAPTIK_METAL_TYPE_FLOAT16;
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
            for (uint32_t reference = input_offset; reference < output_offset; reference++) {
                uint32_t value = synaptik_read_le32(
                        program + references_offset + (uint64_t)reference * 4U);
                if (value >= value_count || !available[value])
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                consumed[value] = 1U;
            }
            if (attribute_kind == SYNAPTIK_METAL_CUSTOM_ATTR_SCALAR_VALUE
                    || attribute_kind == SYNAPTIK_METAL_CUSTOM_ATTR_CLAMP_RANGE) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint64_t expected_type = declared_types[input_value];
                if (synaptik_attribute_word(words, 0U) != expected_type
                        || (attribute_kind == SYNAPTIK_METAL_CUSTOM_ATTR_CLAMP_RANGE
                                && synaptik_attribute_word(words, 2U)
                                        != expected_type))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            for (uint32_t reference = output_offset;
                    reference < output_offset + output_count; reference++) {
                uint32_t value = synaptik_read_le32(
                        program + references_offset + (uint64_t)reference * 4U);
                if (value >= value_count || available[value])
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                available[value] = 1U;
                produced[value] = 1U;
                if (operation == SYNAPTIK_METAL_CUSTOM_TOP_K
                        || operation == SYNAPTIK_METAL_CUSTOM_DROPOUT)
                    consumed[value] = 1U;
            }
            reference_cursor = output_offset + output_count;
            if (operation == SYNAPTIK_METAL_MPSGRAPH_EXP
                    || operation == SYNAPTIK_METAL_CUSTOM_SIGMOID) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t input_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                uint8_t exp_type = declared_types[input_value];
                if ((route != SYNAPTIK_METAL_ROUTE_MPSGRAPH
                                && route != SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
                        || (exp_type != SYNAPTIK_METAL_TYPE_FLOAT32
                                && !(operation == SYNAPTIK_METAL_MPSGRAPH_EXP
                                        && route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM
                                        && synaptik_is_low_precision_type(exp_type)))
                        || declared_types[output_value] != exp_type
                        || (input_flags & 1U) != 0U || (output_flags & 1U) != 0U
                        || !layout_present[input_value] || !layout_present[output_value]
                        || declared_states[input_value] != SYNAPTIK_METAL_VALUE_CANONICAL
                        || declared_states[output_value] != SYNAPTIK_METAL_VALUE_CANONICAL)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
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
                if ((input_flags & 1U) != (output_flags & 1U))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                if (value_ranks[input_value] == 0U
                        || value_ranks[output_value] == 0U) {
                    if (operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL
                            && route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
                        rank_zero_anchor_roles[node_index] = 1U;
                    else
                        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                }
            }
            if (operation == SYNAPTIK_METAL_CUSTOM_RELU) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                if (value_ranks[input_value] == 0U
                        || value_ranks[output_value] == 0U) {
                    if (route == SYNAPTIK_METAL_ROUTE_CUSTOM_PROGRAM)
                        rank_zero_anchor_roles[node_index] = 2U;
                    else
                        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                }
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
            if (operation == SYNAPTIK_METAL_CUSTOM_INITIAL_STATE) {
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                if ((output_flags & 1U) != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (operation == SYNAPTIK_METAL_CUSTOM_DROPOUT) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t state_value = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(input_offset + 1U) * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t mask_value = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(output_offset + 1U) * 4U);
                uint32_t next_state_value = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(output_offset + 2U) * 4U);
                uint32_t input_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                uint32_t stateful_flags = synaptik_read_le32(
                                program + values_offset
                                        + (uint64_t)state_value * 40U + 16U)
                        | synaptik_read_le32(
                                program + values_offset
                                        + (uint64_t)mask_value * 40U + 16U)
                        | synaptik_read_le32(
                                program + values_offset
                                        + (uint64_t)next_state_value * 40U + 16U);
                if ((input_flags & 1U) != (output_flags & 1U)
                        || (stateful_flags & 1U) != 0U)
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
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint8_t output_type = declared_types[output_value];
                BOOL any_float32 = NO;
                BOOL any_bfloat16 = NO;
                BOOL any_gradient = NO;
                BOOL all_output_type = synaptik_is_low_precision_type(output_type);
                for (uint32_t input = 0U; input < input_count; input++) {
                    uint32_t value = synaptik_read_le32(
                            program + references_offset
                                    + (uint64_t)(input_offset + input) * 4U);
                    uint8_t type = declared_types[value];
                    if (type != SYNAPTIK_METAL_TYPE_FLOAT32
                            && type != SYNAPTIK_METAL_TYPE_BFLOAT16
                            && type != SYNAPTIK_METAL_TYPE_FLOAT16)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    any_float32 |= type == SYNAPTIK_METAL_TYPE_FLOAT32;
                    any_bfloat16 |= type == SYNAPTIK_METAL_TYPE_BFLOAT16;
                    all_output_type &= type == output_type;
                    any_gradient |= (synaptik_read_le32(
                            program + values_offset + (uint64_t)value * 40U + 16U)
                            & 1U) != 0U;
                }
                BOOL output_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U)
                        & 1U) != 0U;
                if (all_output_type
                                ? output_gradient != any_gradient
                                : (!any_float32
                                        || output_type != SYNAPTIK_METAL_TYPE_FLOAT32
                                        || (any_bfloat16
                                                ? any_gradient || output_gradient
                                                : output_gradient != any_gradient)))
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
                BOOL type_valid = input_type == output_type
                        && ((input_type == SYNAPTIK_METAL_TYPE_FLOAT64 && !average)
                                || input_type == SYNAPTIK_METAL_TYPE_FLOAT32
                                || input_type == SYNAPTIK_METAL_TYPE_BFLOAT16
                                || input_type == SYNAPTIK_METAL_TYPE_FLOAT16);
                BOOL input_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U)
                        & 1U) != 0U;
                BOOL output_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U)
                        & 1U) != 0U;
                if (!type_valid || input_gradient != output_gradient)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            BOOL task0069_variance = NO;
            if (operation == SYNAPTIK_METAL_MPSGRAPH_VARIANCE
                    && attribute_kind == 38U
                    && attribute_word_count == 4U
                    && synaptik_attribute_word(words, 0U) == 1U
                    && synaptik_attribute_word(words, 1U) == 0U
                    && synaptik_attribute_word(words, 3U) == 0U) {
                uint32_t variance_input = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t variance_output = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                BOOL keep = synaptik_attribute_word(words, 2U) != 0U;
                task0069_variance =
                        value_ranks[variance_input] == 1U
                        && value_dimensions[
                                (size_t)variance_input * SYNAPTIK_MAX_RANK] == 1U
                        && value_ranks[variance_output] == (keep ? 1U : 0U)
                        && (!keep
                                || value_dimensions[
                                        (size_t)variance_output * SYNAPTIK_MAX_RANK]
                                        == 1U);
            }
            if (operation == SYNAPTIK_METAL_MPSGRAPH_L1_NORM
                    || task0069_variance) {
                uint32_t input_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t input_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)input_value * 40U + 16U);
                uint32_t output_flags = synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U);
                uint8_t input_type = declared_types[input_value];
                if ((input_type != SYNAPTIK_METAL_TYPE_FLOAT32
                                && !synaptik_is_low_precision_type(input_type))
                        || declared_types[output_value] != input_type
                        || (input_flags & 1U) != 0U
                        || (output_flags & 1U) != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ADD) {
                if (input_count != 3U
                        || output_count != 1U) {
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                }
                uint32_t data_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)input_offset * 4U);
                uint32_t index_value = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(input_offset + 1U) * 4U);
                uint32_t update_value = synaptik_read_le32(
                        program + references_offset
                                + (uint64_t)(input_offset + 2U) * 4U);
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                uint32_t gradient_flags =
                        synaptik_read_le32(
                                program + values_offset
                                        + (uint64_t)data_value * 40U + 16U)
                        | synaptik_read_le32(
                                program + values_offset
                                        + (uint64_t)index_value * 40U + 16U)
                        | synaptik_read_le32(
                                program + values_offset
                                        + (uint64_t)update_value * 40U + 16U)
                        | synaptik_read_le32(
                                program + values_offset
                                        + (uint64_t)output_value * 40U + 16U);
                if ((gradient_flags & 1U) != 0U
                        || feed_position(index_value, feed_count, feed_indices)
                                == NSNotFound) {
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                }
            }
            if (operation_is_task0066_selected(operation)) {
                BOOL gradients[SYNAPTIK_MAX_NODE_INPUTS] = {NO};
                BOOL any_input_gradient = NO;
                for (uint32_t input = 0U; input < input_count; input++) {
                    uint32_t value = synaptik_read_le32(
                            program + references_offset
                                    + (uint64_t)(input_offset + input) * 4U);
                    gradients[input] = (synaptik_read_le32(
                            program + values_offset + (uint64_t)value * 40U + 16U)
                            & 1U) != 0U;
                    if (gradients[input]
                            && !task0066_is_floating(declared_types[value]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    any_input_gradient |= gradients[input];
                }
                uint32_t output_value = synaptik_read_le32(
                        program + references_offset + (uint64_t)output_offset * 4U);
                BOOL output_gradient = (synaptik_read_le32(
                        program + values_offset + (uint64_t)output_value * 40U + 16U)
                        & 1U) != 0U;
                if (output_gradient
                        && !task0066_is_floating(declared_types[output_value]))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                BOOL expected = gradients[0];
                if (operation == SYNAPTIK_METAL_CUSTOM_CAST) {
                    uint32_t input_value = synaptik_read_le32(
                            program + references_offset
                                    + (uint64_t)input_offset * 4U);
                    expected = gradients[0]
                            && task0066_is_floating(declared_types[input_value])
                            && task0066_is_floating(declared_types[output_value]);
                } else if ((operation >= SYNAPTIK_METAL_CUSTOM_GT
                                && operation <= SYNAPTIK_METAL_CUSTOM_NE)
                        || (operation >= SYNAPTIK_METAL_BOOL_IS_FINITE
                                && operation <= SYNAPTIK_METAL_BOOL_NOT)) {
                    expected = NO;
                    if (operation == SYNAPTIK_METAL_BOOL_AND
                            || operation == SYNAPTIK_METAL_BOOL_OR
                            || operation == SYNAPTIK_METAL_BOOL_NOT) {
                        if (any_input_gradient)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    }
                } else if (operation == SYNAPTIK_METAL_BOOL_WHERE) {
                    if (gradients[0])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    expected = gradients[1] || gradients[2];
                } else if (operation == SYNAPTIK_METAL_CUSTOM_ONE_HOT) {
                    if (gradients[0])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    expected = NO;
                } else if (operation == SYNAPTIK_METAL_CUSTOM_GATHER
                        || operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
                        || operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND) {
                    if (gradients[1])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    expected = gradients[0];
                } else if (operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS
                        || operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND) {
                    if (gradients[1])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    expected = gradients[0] || gradients[2];
                } else if (operation == SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE) {
                    expected = gradients[0] || gradients[1];
                } else if (operation == SYNAPTIK_METAL_CUSTOM_CONCAT
                        || operation == SYNAPTIK_METAL_CUSTOM_STACK) {
                    expected = any_input_gradient;
                }
                if (output_gradient != expected)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (operation >= SYNAPTIK_METAL_CUSTOM_PROD
                    && operation <= SYNAPTIK_METAL_CUSTOM_ANY) {
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
            node.first_input = input_count == 0U
                    ? UINT32_MAX
                    : synaptik_read_le32(
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
                if (attribute_word_count < 1U
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
                if (attribute_word_count != 2U
                        || !synaptik_scalar_is_valid(words, attribute_word_count, 0U))
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                node.attribute_count = 1U;
                node.attribute_values[0] = synaptik_read_le64(words + 8U);
            } else if (attribute_kind == 8U) {
                if (attribute_word_count != 4U
                        || !synaptik_scalar_is_valid(words, attribute_word_count, 0U)
                        || !synaptik_scalar_is_valid(words, attribute_word_count, 2U)
                        || synaptik_read_le64(words) != synaptik_read_le64(words + 16U))
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
        NSMutableData *step_data = nil;
        NSMutableData *member_data = nil;
        NSMutableData *binding_data = nil;
        NSMutableData *materialized_data = nil;
        NSMutableData *program_to_slot_data = nil;
        NSMutableData *instruction_data = nil;
        SynaptikPointwisePlan fusion_plan = {0};
        const SynaptikPointwisePlan *fusion = NULL;
        if (extension) {
            step_data = [NSMutableData dataWithLength:
                    (NSUInteger)step_count * sizeof(SynaptikPointwiseStepRecord)];
            member_data = [NSMutableData dataWithLength:
                    (NSUInteger)member_count * sizeof(uint32_t)];
            binding_data = [NSMutableData dataWithLength:
                    (NSUInteger)binding_count * sizeof(SynaptikPointwiseBindingRecord)];
            materialized_data = [NSMutableData dataWithLength:
                    (NSUInteger)materialized_count * sizeof(uint32_t)];
            program_to_slot_data = [NSMutableData dataWithLength:
                    (NSUInteger)value_count * sizeof(uint32_t)];
            instruction_data = [NSMutableData dataWithLength:
                    (NSUInteger)instruction_count * sizeof(SynaptikPointwiseInstructionRecord)];
            if (step_data == nil || member_data == nil || binding_data == nil
                    || materialized_data == nil || program_to_slot_data == nil
                    || instruction_data == nil)
                return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
            SynaptikPointwiseStepRecord *steps = step_data.mutableBytes;
            uint32_t *members = member_data.mutableBytes;
            SynaptikPointwiseBindingRecord *bindings = binding_data.mutableBytes;
            uint32_t *materialized_values = materialized_data.mutableBytes;
            uint32_t *program_to_slot = program_to_slot_data.mutableBytes;
            SynaptikPointwiseInstructionRecord *instructions =
                    instruction_data.mutableBytes;
            memset(program_to_slot, 0xff, (size_t)value_count * sizeof(uint32_t));
            uint32_t prior_value = UINT32_MAX;
            for (uint32_t slot = 0U; slot < materialized_count; slot++) {
                uint32_t value = synaptik_read_le32(
                        program + materialized_offset + (uint64_t)slot * 4U);
                if (value >= value_count
                        || (slot != 0U && value <= prior_value)
                        || program_to_slot[value] != UINT32_MAX)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                materialized_values[slot] = value;
                program_to_slot[value] = slot;
                prior_value = value;
            }
            for (uint32_t feed = 0U; feed < feed_count; feed++)
                if (program_to_slot[feed_indices[feed]] == UINT32_MAX)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            for (uint32_t target = 0U; target < target_count; target++)
                if (program_to_slot[target_indices[target]] == UINT32_MAX)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            for (uint32_t member = 0U; member < member_count; member++) {
                members[member] = synaptik_read_le32(
                        program + members_offset + (uint64_t)member * 4U);
            }
            for (uint32_t binding = 0U; binding < binding_count; binding++) {
                const uint8_t *record =
                        program + bindings_offset + (uint64_t)binding * 24U;
                bindings[binding] = (SynaptikPointwiseBindingRecord) {
                    .step = synaptik_read_le32(record),
                    .argument = synaptik_read_le32(record + 4U),
                    .access = synaptik_read_le32(record + 8U),
                    .slot = synaptik_read_le32(record + 12U),
                    .value = synaptik_read_le32(record + 16U),
                    .reserved = synaptik_read_le32(record + 20U),
                };
                if (bindings[binding].reserved != 0U
                        || bindings[binding].step >= step_count
                        || bindings[binding].access < 1U
                        || bindings[binding].access > 2U
                        || bindings[binding].slot >= materialized_count
                        || bindings[binding].value >= value_count
                        || materialized_values[bindings[binding].slot]
                                != bindings[binding].value)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            for (uint32_t instruction = 0U;
                    instruction < instruction_count;
                    instruction++) {
                const uint8_t *record =
                        program + instructions_offset + (uint64_t)instruction * 64U;
                instructions[instruction] = (SynaptikPointwiseInstructionRecord) {
                    .step = synaptik_read_le32(record),
                    .relative_node = synaptik_read_le32(record + 4U),
                    .opcode = synaptik_read_le32(record + 8U),
                    .semantic_site = synaptik_read_le32(record + 12U),
                    .input_count = synaptik_read_le32(record + 16U),
                    .input0 = synaptik_read_le32(record + 20U),
                    .input1 = synaptik_read_le32(record + 24U),
                    .input2 = synaptik_read_le32(record + 28U),
                    .output = synaptik_read_le32(record + 32U),
                    .immediate_count = synaptik_read_le32(record + 36U),
                    .immediate_type0 = synaptik_read_le32(record + 40U),
                    .immediate_type1 = synaptik_read_le32(record + 44U),
                    .raw0 = synaptik_read_le64(record + 48U),
                    .raw1 = synaptik_read_le64(record + 56U),
                };
            }
            uint32_t member_cursor = 0U;
            uint32_t binding_cursor = 0U;
            uint32_t instruction_cursor = 0U;
            uint32_t generated_units = 0U;
            uint32_t anchor_units = 0U;
            for (uint32_t ordinal = 0U; ordinal < step_count; ordinal++) {
                const uint8_t *record =
                        program + steps_offset + (uint64_t)ordinal * 40U;
                steps[ordinal] = (SynaptikPointwiseStepRecord) {
                    .kind = synaptik_read_le32(record),
                    .member_start = synaptik_read_le32(record + 4U),
                    .member_count = synaptik_read_le32(record + 8U),
                    .binding_start = synaptik_read_le32(record + 12U),
                    .binding_count = synaptik_read_le32(record + 16U),
                    .instruction_start = synaptik_read_le32(record + 20U),
                    .instruction_count = synaptik_read_le32(record + 24U),
                    .function_bytes = synaptik_read_le32(record + 28U),
                    .flags = synaptik_read_le32(record + 32U),
                    .reserved = synaptik_read_le32(record + 36U),
                };
                SynaptikPointwiseStepRecord step = steps[ordinal];
                if (step.kind < 1U || step.kind > SYNAPTIK_ANCHOR_EPILOGUE_STEP
                        || (contains_declared_low_precision && step.kind > 2U)
                        || step.member_count == 0U
                        || step.member_start != member_cursor
                        || step.binding_start != binding_cursor
                        || step.instruction_start != instruction_cursor
                        || step.member_count > member_count - member_cursor
                        || step.binding_count > binding_count - binding_cursor
                        || step.instruction_count
                                > instruction_count - instruction_cursor
                        || (step.kind == SYNAPTIK_ANCHOR_EPILOGUE_STEP
                                ? step.flags < 1U || step.flags > 2U
                                : step.flags != 0U)
                        || step.reserved != 0U)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                for (uint32_t relative = 0U; relative < step.member_count; relative++)
                    if (members[member_cursor + relative] != member_cursor + relative)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                uint32_t first_member = members[member_cursor];
                if (first_member >= node_count)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                SynaptikMetalDecodedNode first = nodes[first_member];
                uint32_t expected_bindings = first.input_count + first.output_count;
                if (step.kind == SYNAPTIK_POINTWISE_GENERATED_STEP) {
                    if (step.member_count < 2U || step.member_count > 8U
                            || step.instruction_count != step.member_count
                            || step.binding_count != 2U
                            || first.input_count != 1U || first.output_count != 1U
                            || first.attribute_kind != 0U || first.attribute_count != 0U)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    expected_bindings = 2U;
                    generated_units++;
                    uint32_t computed = SynaptikPointwiseFunctionBytes(
                            ordinal, instructions + step.instruction_start,
                            step.instruction_count);
                    if (step.function_bytes > SYNAPTIK_POINTWISE_MAX_FUNCTION_BYTES
                            || computed == 0U || computed != step.function_bytes)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    if (rejected_node != UINT32_MAX && first_member >= rejected_node)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    for (uint32_t relative = 0U;
                            relative < step.member_count;
                            relative++) {
                        uint32_t member = members[step.member_start + relative];
                        SynaptikMetalDecodedNode node = nodes[member];
                        SynaptikPointwiseInstructionRecord instruction =
                                instructions[step.instruction_start + relative];
                        if (node.operation < SYNAPTIK_METAL_CUSTOM_FLOOR
                                || node.operation > SYNAPTIK_METAL_CUSTOM_RELU
                                || node.input_count != 1U || node.output_count != 1U
                                || node.attribute_kind != 0U || node.attribute_count != 0U
                                || declared_types[node.first_input]
                                        != SYNAPTIK_METAL_TYPE_FLOAT32
                                || declared_types[node.output]
                                        != SYNAPTIK_METAL_TYPE_FLOAT32
                                || declared_states[node.first_input]
                                        != SYNAPTIK_METAL_VALUE_CANONICAL
                                || declared_states[node.output]
                                        != SYNAPTIK_METAL_VALUE_CANONICAL
                                || value_ranks[node.first_input] != value_ranks[node.output]
                                || memcmp(
                                        &value_dimensions[
                                                (size_t)node.first_input * SYNAPTIK_MAX_RANK],
                                        &value_dimensions[
                                                (size_t)node.output * SYNAPTIK_MAX_RANK],
                                        SYNAPTIK_MAX_RANK * sizeof(uint64_t)) != 0
                                || (synaptik_read_le32(
                                        program + values_offset
                                                + (uint64_t)node.first_input * 40U + 16U) & 1U)
                                        != (synaptik_read_le32(
                                                program + values_offset
                                                        + (uint64_t)node.output * 40U + 16U)
                                                & 1U)
                                || instruction.step != ordinal
                                || instruction.relative_node != relative
                                || instruction.opcode != node.operation - 59U
                                || instruction.semantic_site != 1U
                                || instruction.input_count != 1U
                                || instruction.input0 != relative
                                || instruction.input1 != UINT32_MAX
                                || instruction.input2 != UINT32_MAX
                                || instruction.output != relative + 1U
                                || instruction.immediate_count != 0U
                                || instruction.immediate_type0 != 0U
                                || instruction.immediate_type1 != 0U
                                || instruction.raw0 != 0U || instruction.raw1 != 0U)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        if (relative != 0U
                                && nodes[member - 1U].output != node.first_input)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                        if (relative + 1U < step.member_count
                                && program_to_slot[node.output] != UINT32_MAX)
                            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    }
                    SynaptikMetalDecodedNode last =
                            nodes[members[step.member_start + step.member_count - 1U]];
                    if (program_to_slot[first.first_input] == UINT32_MAX
                            || program_to_slot[last.output] == UINT32_MAX)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                } else if (step.kind == SYNAPTIK_ANCHOR_EPILOGUE_STEP) {
                    anchor_units++;
                    if (!synaptik_validate_anchor_step(
                                program,
                                values_offset,
                                node_count,
                                nodes,
                                value_ranks,
                                value_dimensions,
                                declared_states,
                                declared_types,
                                targeted,
                                program_to_slot,
                                ordinal,
                                step,
                                members,
                                bindings,
                                instructions))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    for (uint32_t relative = 1U;
                            relative < step.member_count;
                            relative++) {
                        uint32_t member = members[step.member_start + relative];
                        uint32_t operation = nodes[member].operation;
                        if (rank_zero_anchor_roles[member] == 1U
                                && relative == 1U
                                && operation == SYNAPTIK_METAL_MPSGRAPH_SCALAR_MUL)
                            rank_zero_anchor_roles[member] = 0U;
                        if (rank_zero_anchor_roles[member] == 2U
                                && relative + 1U == step.member_count
                                && operation == SYNAPTIK_METAL_CUSTOM_RELU)
                            rank_zero_anchor_roles[member] = 0U;
                    }
                } else if (step.member_count != 1U || step.instruction_count != 0U
                        || step.function_bytes != 0U
                        || step.binding_count != expected_bindings) {
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                }
                if (step.kind != SYNAPTIK_ANCHOR_EPILOGUE_STEP
                        && step.binding_count != expected_bindings)
                    return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                for (uint32_t argument = 0U; argument < step.binding_count; argument++) {
                    SynaptikPointwiseBindingRecord binding =
                            bindings[step.binding_start + argument];
                    uint32_t expected_value;
                    uint32_t expected_access;
                    if (step.kind == SYNAPTIK_POINTWISE_GENERATED_STEP) {
                        SynaptikMetalDecodedNode last =
                                nodes[members[step.member_start + step.member_count - 1U]];
                        expected_value = argument == 0U ? first.first_input : last.output;
                        expected_access = argument == 0U ? 1U : 2U;
                    } else if (step.kind == SYNAPTIK_ANCHOR_EPILOGUE_STEP) {
                        continue;
                    } else if (argument < first.input_count) {
                        expected_value = first.inputs[argument];
                        expected_access = 1U;
                    } else {
                        expected_value = first.outputs[argument - first.input_count];
                        expected_access = 2U;
                    }
                    if (binding.step != ordinal || binding.argument != argument
                            || binding.access != expected_access
                            || binding.value != expected_value)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                }
                member_cursor += step.member_count;
                binding_cursor += step.binding_count;
                instruction_cursor += step.instruction_count;
            }
            if (member_cursor != member_count || binding_cursor != binding_count
                    || instruction_cursor != instruction_count
                    || generated_units != generated_unit_count
                    || anchor_units > SYNAPTIK_ANCHOR_EPILOGUE_MAX_UNITS)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            uint32_t counted_generated_bytes = SynaptikPointwiseGeneratedBytes(
                    steps, step_count, instructions, instruction_count);
            if (counted_generated_bytes == UINT32_MAX
                    || counted_generated_bytes != expected_generated_bytes
                    || (rejected_node != UINT32_MAX && rejected_node >= node_count))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            fusion_plan = (SynaptikPointwisePlan) {
                .step_count = step_count,
                .steps = steps,
                .member_count = member_count,
                .members = members,
                .binding_count = binding_count,
                .bindings = bindings,
                .materialized_count = materialized_count,
                .materialized_values = materialized_values,
                .program_to_slot = program_to_slot,
                .instruction_count = instruction_count,
                .instructions = instructions,
                .generated_unit_count = generated_unit_count,
                .expected_generated_bytes = expected_generated_bytes,
                .expected_total_bytes = expected_total_bytes,
            };
            if (!synaptik_validate_pointwise_cap_stop(
                        program,
                        values_offset,
                        value_count,
                        node_count,
                        nodes,
                        value_ranks,
                        value_dimensions,
                        declared_states,
                        declared_types,
                        target_count,
                        target_indices,
                        &fusion_plan,
                        contains_declared_low_precision,
                        rejected_node,
                        cap_reason))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
            fusion_plan.manifest_digest = program + digest_offset;
#endif
            fusion = &fusion_plan;
            unsigned char computed_digest[CC_SHA256_DIGEST_LENGTH];
            CC_SHA256(program + manifest_offset, (CC_LONG)manifest_bytes, computed_digest);
            if (memcmp(computed_digest, program + digest_offset, CC_SHA256_DIGEST_LENGTH) != 0)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            NSData *canonical = synaptik_pointwise_manifest(
                    program,
                    values_offset,
                    value_count,
                    value_ranks,
                    value_dimensions,
                    node_count,
                    feed_count,
                    target_count,
                    target_indices,
                    fusion,
                    rejected_node,
                    cap_reason);
            if (canonical == nil || canonical.length != manifest_bytes
                    || memcmp(canonical.bytes, program + manifest_offset, manifest_bytes) != 0)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            if (synaptik_authenticated_fixed_source() == nil)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t node = 0U; node < node_count; node++)
            if (rank_zero_anchor_roles[node] != 0U)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
        if (contains_unsupported)
            return SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION;
        return synaptik_metal_create_decoded(
                context, route, value_count, value_ranks, value_dimensions,
                value_strides, layout_offsets, layout_spans, declared_states,
                declared_types, node_count, nodes, feed_count,
                feed_count == 0U ? NULL : feed_indices,
                target_count, target_indices, fusion, out_executable);
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
        if (step.indexValidation != nil && !step.indexValidation.preflight) {
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
            BOOL scatter = step.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ADD
                    || ((step.operation == SYNAPTIK_METAL_CUSTOM_SCATTER_ELEMENTS
                                    || step.operation
                                            == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ND)
                            && step.stage == 1U);
            BOOL task0063 = (step.operation >= SYNAPTIK_METAL_CUSTOM_SORT
                            && step.operation <= SYNAPTIK_METAL_CUSTOM_TOP_K)
                    || (step.operation >= SYNAPTIK_METAL_CUSTOM_ARG_MAX
                            && step.operation <= SYNAPTIK_METAL_CUSTOM_ARG_MIN);
            BOOL compose = step.operation == SYNAPTIK_METAL_CUSTOM_CONCAT
                    || step.operation == SYNAPTIK_METAL_CUSTOM_STACK;
            BOOL binary = (step.operation >= SYNAPTIK_METAL_CUSTOM_GT
                            && step.operation <= SYNAPTIK_METAL_CUSTOM_TENSOR_MAX)
                    || (step.operation >= SYNAPTIK_METAL_MPSGRAPH_ADD
                            && step.operation <= SYNAPTIK_METAL_MPSGRAPH_DIV)
                    || step.operation == SYNAPTIK_METAL_MPSGRAPH_MSE
                    || step.operation == SYNAPTIK_METAL_BOOL_AND
                    || step.operation == SYNAPTIK_METAL_BOOL_OR
                    || step.operation == SYNAPTIK_METAL_CUSTOM_GATHER
                    || step.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ELEMENTS
                    || step.operation == SYNAPTIK_METAL_CUSTOM_GATHER_ND
                    || step.operation == SYNAPTIK_METAL_MPSGRAPH_SLICE_UPDATE
                    || step.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL;
            BOOL task0064_conv = step.operation == SYNAPTIK_METAL_CUSTOM_CONV2D
                    || step.operation == SYNAPTIK_METAL_CUSTOM_CONV3D;
            BOOL task0065_initial =
                    step.operation == SYNAPTIK_METAL_CUSTOM_INITIAL_STATE;
            BOOL task0065_dropout =
                    step.operation == SYNAPTIK_METAL_CUSTOM_DROPOUT;
            if (step.anchorEpilogue) {
                if (step.secondInput >= input_count
                        || (step.hasExternalAdd
                                && step.epilogueInput >= input_count))
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                SynaptikMetalBufferBox *second =
                        (__bridge SynaptikMetalBufferBox *)
                                input_buffers[step.secondInput];
                [encoder setBuffer:second.buffer offset:0U atIndex:1U];
                if (step.operation == SYNAPTIK_METAL_MPSGRAPH_MATMUL) {
                    if (step.hasExternalAdd) {
                        SynaptikMetalBufferBox *addend =
                                (__bridge SynaptikMetalBufferBox *)
                                        input_buffers[step.epilogueInput];
                        [encoder setBuffer:addend.buffer offset:0U atIndex:2U];
                        [encoder setBuffer:output.buffer offset:0U atIndex:3U];
                        [encoder setBuffer:step.metadata offset:0U atIndex:4U];
                    } else {
                        [encoder setBuffer:output.buffer offset:0U atIndex:2U];
                        [encoder setBuffer:step.metadata offset:0U atIndex:3U];
                    }
                } else if (step.operation == SYNAPTIK_METAL_CUSTOM_CONV2D) {
                    NSUInteger bias_value = step.auxiliaryInput;
                    if (bias_value >= input_count) bias_value = step.firstInput;
                    SynaptikMetalBufferBox *bias =
                            (__bridge SynaptikMetalBufferBox *)
                                    input_buffers[bias_value];
                    [encoder setBuffer:bias.buffer offset:0U atIndex:2U];
                    if (step.hasExternalAdd) {
                        SynaptikMetalBufferBox *addend =
                                (__bridge SynaptikMetalBufferBox *)
                                        input_buffers[step.epilogueInput];
                        [encoder setBuffer:addend.buffer offset:0U atIndex:3U];
                        [encoder setBuffer:output.buffer offset:0U atIndex:4U];
                        [encoder setBuffer:step.metadata offset:0U atIndex:5U];
                    } else {
                        [encoder setBuffer:output.buffer offset:0U atIndex:3U];
                        [encoder setBuffer:step.metadata offset:0U atIndex:4U];
                    }
                } else {
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                }
            } else if (task0065_initial) {
                if (step.outputValues.count != 1U
                        || step.outputValues[0].unsignedIntegerValue != step.output)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                [encoder setBuffer:step.metadata offset:0U atIndex:1U];
            } else if (task0065_dropout) {
                if (step.secondInput >= input_count || step.outputValues.count != 3U
                        || step.outputValues[0].unsignedIntegerValue != step.output)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                NSUInteger mask_value = step.outputValues[1].unsignedIntegerValue;
                NSUInteger next_state_value =
                        step.outputValues[2].unsignedIntegerValue;
                if (mask_value >= input_count || next_state_value >= input_count)
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                SynaptikMetalBufferBox *state =
                        (__bridge SynaptikMetalBufferBox *)
                                input_buffers[step.secondInput];
                SynaptikMetalBufferBox *mask =
                        (__bridge SynaptikMetalBufferBox *)input_buffers[mask_value];
                SynaptikMetalBufferBox *next_state =
                        (__bridge SynaptikMetalBufferBox *)
                                input_buffers[next_state_value];
                [encoder setBuffer:state.buffer offset:0U atIndex:1U];
                [encoder setBuffer:output.buffer offset:0U atIndex:2U];
                [encoder setBuffer:mask.buffer offset:0U atIndex:3U];
                [encoder setBuffer:next_state.buffer offset:0U atIndex:4U];
                [encoder setBuffer:step.metadata offset:0U atIndex:5U];
            } else if (task0064_conv) {
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
#if defined(SYNAPTIK_METAL_TEST_DISPATCH_OBSERVER)
            if (synaptik_metal_test_dispatch_observer != NULL) {
                SynaptikMetalPointMeta point_meta = {0};
                if (step.planKind == SYNAPTIK_POINTWISE_GENERATED_STEP
                        && step.metadata.length == sizeof(point_meta))
                    memcpy(&point_meta, step.metadata.contents, sizeof(point_meta));
                synaptik_metal_test_dispatch_observer(
                        step.planOrdinal,
                        step.planKind,
                        box.manifestDigest.bytes,
                        step.grid.width,
                        step.grid.height,
                        step.grid.depth,
                        step.threadsPerThreadgroup,
                        step.metadata.length,
                        point_meta.elementCount,
                        point_meta.gridWidth,
                        point_meta.gridHeight,
                        point_meta.scalar,
                        point_meta.reserved);
            }
#endif
            [encoder dispatchThreads:step.grid threadsPerThreadgroup:
                    MTLSizeMake(step.threadsPerThreadgroup, 1U, 1U)];
            cursor++;
            if (cursor < box.programSteps.count
                    && box.programSteps[cursor].custom
                    && box.programSteps[cursor].indexValidation == nil)
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
            for (uint32_t previous = 0; previous < output; previous++)
                if (output_buffers[previous] == output_buffers[output])
                    return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
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
