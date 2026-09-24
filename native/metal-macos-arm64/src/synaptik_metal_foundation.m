#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define SYNAPTIK_EXPORT __attribute__((visibility("default")))
#define SYNAPTIK_MAX_RANK 16U

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
    SYNAPTIK_METAL_STATUS_KERNEL_COMPILATION_FAILED = 12
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
    SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS = 18U
} SynaptikMetalMpsGraphOperationV10;

typedef enum : uint32_t {
    SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE = 0U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_TARGET_SHAPE = 1U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_PERMUTATION = 2U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS = 3U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_REDUCTION = 4U,
    SYNAPTIK_METAL_MPSGRAPH_ATTR_DEPTH = 5U
} SynaptikMetalMpsGraphAttributeV10;

typedef enum : uint32_t {
    SYNAPTIK_METAL_REDUCTION_FULL = 1U,
    SYNAPTIK_METAL_REDUCTION_SINGLE_AXIS = 2U,
    SYNAPTIK_METAL_REDUCTION_MULTI_AXIS = 3U,
    SYNAPTIK_METAL_REDUCTION_SUM_TO_SHAPE = 4U
} SynaptikMetalReductionFormV10;

typedef enum : uint8_t {
    SYNAPTIK_METAL_VALUE_UNAVAILABLE = 0U,
    SYNAPTIK_METAL_VALUE_CANONICAL = 1U,
    SYNAPTIK_METAL_VALUE_AFFINE_VIEW = 2U
} SynaptikMetalValueStateV10;

typedef enum : uint8_t {
    SYNAPTIK_METAL_TYPE_UNAVAILABLE = 0U,
    SYNAPTIK_METAL_TYPE_FLOAT32 = 1U,
    SYNAPTIK_METAL_TYPE_INT32 = 2U,
    SYNAPTIK_METAL_TYPE_BOOL = 3U
} SynaptikMetalValueTypeV10;

typedef struct {
    uint32_t operation;
    uint32_t attribute_kind;
    uint32_t first_input;
    uint32_t second_input;
    uint32_t output;
    uint32_t attribute_count;
    uint32_t axis;
    uint32_t auxiliary;
    uint64_t attribute_values[SYNAPTIK_MAX_RANK];
} SynaptikMetalMpsGraphNodeV10;

_Static_assert(sizeof(SynaptikMetalMpsGraphNodeV10) == 160U,
        "MPSGraph v10 node record must be exactly 160 bytes");
_Static_assert(offsetof(SynaptikMetalMpsGraphNodeV10, attribute_values) == 32U,
        "MPSGraph v10 attribute payload must begin at byte 32");


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
@property(nonatomic) NSUInteger feedPosition;
@property(nonatomic) uint64_t elementCount;
@property(nonatomic) uint64_t bound;
@property(nonatomic) uint32_t axis;
@property(nonatomic, strong, nullable) NSMutableData *coordinateExtents;
@property(nonatomic, strong, nullable) NSMutableData *dataStrides;
@property(nonatomic, strong, nullable) NSMutableData *targetScratch;
@end
@implementation SynaptikMetalIndexValidation @end

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

SYNAPTIK_EXPORT uint32_t synaptik_metal_foundation_abi_version(void) { return 4U; }

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
        SynaptikMetalMpsGraphNodeV10 node, uint32_t first) {
    for (uint32_t index = first; index < SYNAPTIK_MAX_RANK; index++)
        if (node.attribute_values[index] != 0U) return NO;
    return YES;
}

static BOOL node_has_no_attributes(SynaptikMetalMpsGraphNodeV10 node) {
    return node.attribute_kind == SYNAPTIK_METAL_MPSGRAPH_ATTR_NONE
            && node.attribute_count == 0U
            && node.axis == UINT32_MAX
            && node.auxiliary == 0U
            && node_values_are_zero_from(node, 0U);
}

static BOOL node_target_matches(
        SynaptikMetalMpsGraphNodeV10 node, MPSShape *output) {
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

static BOOL node_permutation_matches(
        SynaptikMetalMpsGraphNodeV10 node, MPSShape *input, MPSShape *output) {
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

static BOOL node_axis_header_is_valid(SynaptikMetalMpsGraphNodeV10 node) {
    return node.attribute_kind == SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS
            && node.attribute_count == 1U
            && node.axis < SYNAPTIK_MAX_RANK
            && node.auxiliary == 0U
            && node_values_are_zero_from(node, 0U);
}

static BOOL node_expand_dims_matches(
        SynaptikMetalMpsGraphNodeV10 node, MPSShape *input, MPSShape *output) {
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
        SynaptikMetalMpsGraphNodeV10 node, MPSShape *input, MPSShape *output) {
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
        SynaptikMetalMpsGraphNodeV10 node,
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
        SynaptikMetalMpsGraphNodeV10 node,
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
        SynaptikMetalMpsGraphNodeV10 node, MPSShape *input, MPSShape *output) {
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

static BOOL require_value_type(
        uint8_t *types, uint32_t value, SynaptikMetalValueTypeV10 required) {
    if (types[value] != SYNAPTIK_METAL_TYPE_UNAVAILABLE
            && types[value] != required)
        return NO;
    types[value] = required;
    return YES;
}

static uint64_t value_type_width(uint8_t type) {
    switch ((SynaptikMetalValueTypeV10)type) {
        case SYNAPTIK_METAL_TYPE_FLOAT32:
        case SYNAPTIK_METAL_TYPE_INT32:
            return 4U;
        case SYNAPTIK_METAL_TYPE_BOOL:
            return 1U;
        default:
            return 0U;
    }
}

static MPSDataType value_mps_data_type(uint8_t type) {
    switch ((SynaptikMetalValueTypeV10)type) {
        case SYNAPTIK_METAL_TYPE_FLOAT32:
            return MPSDataTypeFloat32;
        case SYNAPTIK_METAL_TYPE_INT32:
            return MPSDataTypeInt32;
        case SYNAPTIK_METAL_TYPE_BOOL:
            return MPSDataTypeBool;
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
        SynaptikMetalMpsGraphNodeV10 node) {
    NSMutableArray<NSNumber *> *result =
            [NSMutableArray arrayWithCapacity:node.attribute_count];
    for (uint32_t index = 0; index < node.attribute_count; index++)
        [result addObject:@((NSUInteger)node.attribute_values[index])];
    return [result copy];
}

static NSArray<NSNumber *> *node_reduction_axes(
        SynaptikMetalMpsGraphNodeV10 node,
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


SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_executable_create(
        void *context, uint32_t node_schema_version, uint32_t value_count,
        const uint32_t *value_ranks, const uint64_t *value_dimensions,
        uint32_t node_count, const SynaptikMetalMpsGraphNodeV10 *nodes,
        uint32_t feed_count, const uint32_t *feed_indices,
        uint32_t target_count, const uint32_t *target_indices, void **out_executable) {
    if (out_executable == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_executable = NULL;
    if (context == NULL || node_schema_version != 10U || value_count == 0U
            || node_count == 0U || feed_count == 0U || target_count == 0U
            || value_ranks == NULL || value_dimensions == NULL || nodes == NULL
            || feed_indices == NULL || target_indices == NULL)
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
        NSMutableArray<SynaptikMetalIndexValidation *> *index_validations =
                [NSMutableArray array];
        if (state_data == nil || type_data == nil || used_data == nil
                || produced_data == nil || targeted_data == nil
                || local_transpose_data == nil || index_validations == nil)
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        uint8_t *states = state_data.mutableBytes;
        uint8_t *value_types = type_data.mutableBytes;
        uint8_t *used = used_data.mutableBytes;
        uint8_t *produced = produced_data.mutableBytes;
        uint8_t *targeted = targeted_data.mutableBytes;
        uint8_t *local_transpose = local_transpose_data.mutableBytes;
        BOOL contains_matmul = NO;
        for (uint32_t feed = 0; feed < feed_count; feed++) {
            uint32_t value = feed_indices[feed];
            if (value >= value_count || value_ranks[value] == 0U
                    || states[value] != SYNAPTIK_METAL_VALUE_UNAVAILABLE)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            states[value] = SYNAPTIK_METAL_VALUE_CANONICAL;
            used[value] = 1U;
        }
        for (uint32_t node_index = 0; node_index < node_count; node_index++) {
            SynaptikMetalMpsGraphNodeV10 node = nodes[node_index];
            if (node.first_input >= value_count
                    || node.output >= value_count
                    || value_ranks[node.first_input] == 0U
                    || states[node.first_input] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                    || states[node.output] != SYNAPTIK_METAL_VALUE_UNAVAILABLE)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            BOOL affine_view = NO;
            switch ((SynaptikMetalMpsGraphOperationV10)node.operation) {
                case SYNAPTIK_METAL_MPSGRAPH_NEG:
                case SYNAPTIK_METAL_MPSGRAPH_ABS:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input != UINT32_MAX
                            || !node_has_no_attributes(node)
                            || ![shapes[node.first_input] isEqualToArray:shapes[node.output]])
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SUB:
                case SYNAPTIK_METAL_MPSGRAPH_MUL:
                case SYNAPTIK_METAL_MPSGRAPH_DIV:
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || node.second_input >= value_count
                            || states[node.second_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || value_ranks[node.second_input] == 0U
                            || !node_has_no_attributes(node)
                            || !shape_broadcasts_exactly_to(
                                    shapes[node.first_input],
                                    shapes[node.second_input],
                                    shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    used[node.second_input] = 1U;
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
                    if (states[node.first_input] == SYNAPTIK_METAL_VALUE_CANONICAL
                            && shapes[node.first_input].count == 2U
                            && shapes[node.output].count == 2U
                            && node.attribute_count == 2U
                            && node.attribute_values[0] == 1U
                            && node.attribute_values[1] == 0U)
                        local_transpose[node.output] = 1U;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND_DIMS:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_expand_dims_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SQUEEZE:
                    affine_view = YES;
                    if (node.second_input != UINT32_MAX
                            || !node_squeeze_matches(
                                    node, shapes[node.first_input], shapes[node.output]))
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_SUM:
                case SYNAPTIK_METAL_MPSGRAPH_MEAN: {
                    BOOL keep_dimensions = NO;
                    NSArray<NSNumber *> *axes = node_reduction_axes(
                            node, shapes[node.first_input], shapes[node.output],
                            &keep_dimensions);
                    if (states[node.first_input] != SYNAPTIK_METAL_VALUE_CANONICAL
                            || axes == nil)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    break;
                }
                case SYNAPTIK_METAL_MPSGRAPH_MATMUL: {
                    if (node.second_input >= value_count
                            || value_ranks[node.second_input] == 0U
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
                    MPSShape *left_shape = shapes[node.first_input];
                    MPSShape *right_shape = shapes[node.second_input];
                    MPSShape *output_shape = shapes[node.output];
                    if (!left_valid || !right_valid
                            || left_shape.count != 2U
                            || right_shape.count != 2U
                            || output_shape.count != 2U
                            || left_shape[1].unsignedLongLongValue
                                    != right_shape[0].unsignedLongLongValue
                            || output_shape[0].unsignedLongLongValue
                                    != left_shape[0].unsignedLongLongValue
                            || output_shape[1].unsignedLongLongValue
                                    != right_shape[1].unsignedLongLongValue)
                        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
                    used[node.second_input] = 1U;
                    contains_matmul = YES;
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
                    validation.operation = node.operation;
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
            switch ((SynaptikMetalMpsGraphOperationV10)node.operation) {
                case SYNAPTIK_METAL_MPSGRAPH_NEG:
                case SYNAPTIK_METAL_MPSGRAPH_ABS:
                case SYNAPTIK_METAL_MPSGRAPH_CONTIGUOUS:
                case SYNAPTIK_METAL_MPSGRAPH_RESHAPE:
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND:
                case SYNAPTIK_METAL_MPSGRAPH_PERMUTE:
                case SYNAPTIK_METAL_MPSGRAPH_EXPAND_DIMS:
                case SYNAPTIK_METAL_MPSGRAPH_SQUEEZE:
                case SYNAPTIK_METAL_MPSGRAPH_SUM:
                case SYNAPTIK_METAL_MPSGRAPH_MEAN:
                    type_valid = require_value_type(
                                    value_types, node.first_input,
                                    SYNAPTIK_METAL_TYPE_FLOAT32)
                            && require_value_type(
                                    value_types, node.output,
                                    SYNAPTIK_METAL_TYPE_FLOAT32);
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ADD:
                case SYNAPTIK_METAL_MPSGRAPH_SUB:
                case SYNAPTIK_METAL_MPSGRAPH_MUL:
                case SYNAPTIK_METAL_MPSGRAPH_DIV:
                case SYNAPTIK_METAL_MPSGRAPH_MATMUL:
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
                default:
                    break;
            }
            if (!type_valid) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            if (value_ranks[node.output] == 0U
                    && node.operation != SYNAPTIK_METAL_MPSGRAPH_SUM
                    && node.operation != SYNAPTIK_METAL_MPSGRAPH_MEAN)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            states[node.output] = affine_view
                    ? SYNAPTIK_METAL_VALUE_AFFINE_VIEW
                    : SYNAPTIK_METAL_VALUE_CANONICAL;
            used[node.first_input] = 1U;
            used[node.output] = 1U;
            produced[node.output] = 1U;
        }
        for (uint32_t target = 0; target < target_count; target++) {
            uint32_t value = target_indices[target];
            if (value >= value_count || produced[value] == 0U
                    || states[value] == SYNAPTIK_METAL_VALUE_UNAVAILABLE
                    || targeted[value] != 0U)
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            targeted[value] = 1U;
        }
        NSMutableArray<NSNumber *> *bytes = [NSMutableArray arrayWithCapacity:value_count];
        for (uint32_t value = 0; value < value_count; value++) {
            if (used[value] == 0U || value_types[value] == SYNAPTIK_METAL_TYPE_UNAVAILABLE
                    || (value_ranks[value] == 0U
                            && (produced[value] == 0U || targeted[value] == 0U)))
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            uint64_t width = value_type_width(value_types[value]);
            uint64_t elements = element_counts[value].unsignedLongLongValue;
            if (width == 0U || elements > UINT64_MAX / width
                    || elements * width > (uint64_t)NSUIntegerMax)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            [bytes addObject:@(elements * width)];
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
            SynaptikMetalMpsGraphNodeV10 node = nodes[node_index];
            MPSGraphTensor *first = (MPSGraphTensor *)table[node.first_input];
            MPSGraphTensor *second = node.second_input == UINT32_MAX
                    ? nil : (MPSGraphTensor *)table[node.second_input];
            MPSGraphTensor *auxiliary =
                    node.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS
                    ? (MPSGraphTensor *)table[node.auxiliary] : nil;
            MPSGraphTensor *output = nil;
            switch ((SynaptikMetalMpsGraphOperationV10)node.operation) {
                case SYNAPTIK_METAL_MPSGRAPH_NEG:
                    output = [graph negativeWithTensor:first name:nil];
                    break;
                case SYNAPTIK_METAL_MPSGRAPH_ABS:
                    output = [graph absoluteWithTensor:first name:nil];
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
        *out_executable = (__bridge_retained void *)box;
        return SYNAPTIK_METAL_STATUS_OK;
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

SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_executable_run(
        void *executable, uint32_t input_count, void *const *input_buffers,
        uint32_t output_count, void *const *output_buffers) {
    if (executable == NULL || input_count == 0U || output_count == 0U
            || input_buffers == NULL || output_buffers == NULL)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    @try { @autoreleasepool {
        SynaptikMetalExecutableBox *box = (__bridge SynaptikMetalExecutableBox *)executable;
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
        for (SynaptikMetalIndexValidation *validation in box.indexValidations) {
            if (validation.feedPosition >= input_boxes.count
                    || validation.elementCount > UINT64_MAX / sizeof(int32_t))
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            SynaptikMetalBufferBox *buffer = input_boxes[validation.feedPosition];
            uint64_t required = validation.elementCount * sizeof(int32_t);
            if (buffer.logicalByteSize < required)
                return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            const int32_t *indices = (const int32_t *)buffer.buffer.contents;
            if (indices == NULL) return SYNAPTIK_METAL_STATUS_COPY_FAILED;
            BOOL scatter =
                    validation.operation == SYNAPTIK_METAL_MPSGRAPH_SCATTER_ELEMENTS;
            uint64_t *targets = NULL;
            if (scatter) {
                if (validation.targetScratch == nil
                        || validation.elementCount
                                > (uint64_t)NSUIntegerMax / sizeof(uint64_t)
                        || validation.targetScratch.length
                                != (NSUInteger)validation.elementCount * sizeof(uint64_t))
                    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                targets = validation.targetScratch.mutableBytes;
                if (targets == NULL) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            }
            for (uint64_t ordinal = 0U; ordinal < validation.elementCount; ordinal++) {
                int32_t value = indices[ordinal];
                if (value < 0 || (uint64_t)value >= validation.bound)
                    return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
            }
            if (scatter) {
                for (uint64_t ordinal = 0U;
                        ordinal < validation.elementCount;
                        ordinal++)
                    if (!scatter_target_linear(
                            validation, ordinal, indices[ordinal], &targets[ordinal]))
                        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
                qsort(
                        targets,
                        (size_t)validation.elementCount,
                        sizeof(uint64_t),
                        compare_u64);
                for (uint64_t ordinal = 1U;
                        ordinal < validation.elementCount;
                        ordinal++)
                    if (targets[ordinal - 1U] == targets[ordinal])
                        return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
            }
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
