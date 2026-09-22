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

@interface SynaptikMetalExecutableBox : NSObject
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) SynaptikMetalContextBox *context;
@property(nonatomic, copy) NSArray<MPSShape *> *feedShapes;
@property(nonatomic, copy) NSArray<MPSShape *> *targetShapes;
@property(nonatomic, copy) NSArray<NSNumber *> *feedBytes;
@property(nonatomic, copy) NSArray<NSNumber *> *targetBytes;
@property(nonatomic, copy) NSArray<NSNumber *> *feedPermutation;
@property(nonatomic, copy) NSArray<NSNumber *> *targetPermutation;
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

SYNAPTIK_EXPORT uint32_t synaptik_metal_foundation_abi_version(void) { return 3U; }

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

SYNAPTIK_EXPORT int32_t synaptik_metal_mpsgraph_neg_executable_create(
        void *context, uint32_t value_count, const uint32_t *value_ranks,
        const uint64_t *value_dimensions, uint32_t node_count,
        const uint32_t *node_inputs, const uint32_t *node_outputs,
        uint32_t feed_count, const uint32_t *feed_indices,
        uint32_t target_count, const uint32_t *target_indices, void **out_executable) {
    if (out_executable == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    *out_executable = NULL;
    if (context == NULL || value_count == 0U || node_count == 0U || feed_count == 0U
            || target_count == 0U || value_ranks == NULL || value_dimensions == NULL
            || node_inputs == NULL || node_outputs == NULL || feed_indices == NULL
            || target_indices == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    if ((uint64_t)value_count > SIZE_MAX / SYNAPTIK_MAX_RANK)
        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
    @try {
        NSMutableArray<MPSShape *> *shapes = [NSMutableArray arrayWithCapacity:value_count];
        NSMutableArray<NSNumber *> *bytes = [NSMutableArray arrayWithCapacity:value_count];
        for (uint32_t value = 0; value < value_count; value++) {
            uint32_t rank = value_ranks[value];
            if (rank == 0U || rank > SYNAPTIK_MAX_RANK)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            uint64_t elements = 1U;
            NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:rank];
            for (uint32_t axis = 0; axis < SYNAPTIK_MAX_RANK; axis++) {
                uint64_t dimension = value_dimensions[(size_t)value * SYNAPTIK_MAX_RANK + axis];
                if (axis < rank) {
                    if (dimension == 0U || dimension > (uint64_t)NSUIntegerMax
                            || elements > UINT64_MAX / dimension)
                        return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
                    elements *= dimension; [shape addObject:@((NSUInteger)dimension)];
                } else if (dimension != 0U) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            if (elements > UINT64_MAX / sizeof(float)
                    || elements * sizeof(float) > (uint64_t)NSUIntegerMax)
                return SYNAPTIK_METAL_STATUS_UNSUPPORTED_SHAPE;
            [shapes addObject:[shape copy]]; [bytes addObject:@(elements * sizeof(float))];
        }
        uint8_t *available = calloc(value_count, 1), *used = calloc(value_count, 1);
        if (available == NULL || used == NULL) { free(available); free(used);
            return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED; }
        for (uint32_t feed = 0; feed < feed_count; feed++) {
            uint32_t value = feed_indices[feed];
            if (value >= value_count || available[value] != 0U) { free(available); free(used);
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT; }
            available[value] = 1U; used[value] = 1U;
        }
        for (uint32_t node = 0; node < node_count; node++) {
            uint32_t input = node_inputs[node], output = node_outputs[node];
            if (input >= value_count || output >= value_count || available[input] == 0U
                    || available[output] != 0U) { free(available); free(used);
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT; }
            if (![shapes[input] isEqualToArray:shapes[output]]) {
                free(available); free(used);
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            }
            available[output] = 2U; used[input] = used[output] = 1U;
        }
        for (uint32_t target = 0; target < target_count; target++) {
            uint32_t value = target_indices[target];
            if (value >= value_count || available[value] != 2U) { free(available); free(used);
                return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT; }
            for (uint32_t prior = 0; prior < target; prior++) if (target_indices[prior] == value) {
                free(available); free(used); return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT; }
        }
        for (uint32_t value = 0; value < value_count; value++) if (used[value] == 0U) {
            free(available); free(used); return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT; }
        free(available); free(used);

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
            MPSGraphTensor *tensor = [graph placeholderWithShape:shapes[value]
                    dataType:MPSDataTypeFloat32 name:nil];
            MPSGraphShapedType *type = [[MPSGraphShapedType alloc]
                    initWithShape:shapes[value] dataType:MPSDataTypeFloat32];
            if (tensor == nil || type == nil) return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
            table[value] = tensor; [feeds addObject:tensor]; types[tensor] = type;
        }
        for (uint32_t node = 0; node < node_count; node++) {
            id input = table[node_inputs[node]];
            MPSGraphTensor *output = input == NSNull.null ? nil
                    : [graph negativeWithTensor:(MPSGraphTensor *)input name:nil];
            if (output == nil) return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
            table[node_outputs[node]] = output;
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
        if (device == nil || descriptor == nil) return SYNAPTIK_METAL_STATUS_GRAPH_COMPILATION_FAILED;
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
        for (uint32_t feed = 0; feed < feed_count; feed++) { uint32_t value = feed_indices[feed];
            [feed_shapes addObject:shapes[value]]; [feed_bytes addObject:bytes[value]]; }
        NSMutableArray *target_shapes = [NSMutableArray arrayWithCapacity:target_count];
        NSMutableArray *target_bytes = [NSMutableArray arrayWithCapacity:target_count];
        for (uint32_t target = 0; target < target_count; target++) { uint32_t value = target_indices[target];
            [target_shapes addObject:shapes[value]]; [target_bytes addObject:bytes[value]]; }
        SynaptikMetalExecutableBox *box = [SynaptikMetalExecutableBox new];
        if (box == nil) return SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED;
        box.executable = compiled; box.context = ctx; box.feedShapes = feed_shapes;
        box.targetShapes = target_shapes; box.feedBytes = feed_bytes; box.targetBytes = target_bytes;
        box.feedPermutation = feed_permutation; box.targetPermutation = target_permutation;
        *out_executable = (__bridge_retained void *)box;
        return SYNAPTIK_METAL_STATUS_OK;
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
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
    @try {
        SynaptikMetalExecutableBox *box = (__bridge SynaptikMetalExecutableBox *)executable;
        if (input_count != box.feedShapes.count || output_count != box.targetShapes.count)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        NSMutableArray *stable_inputs = [NSMutableArray arrayWithCapacity:input_count];
        NSMutableArray *stable_outputs = [NSMutableArray arrayWithCapacity:output_count];
        for (uint32_t input = 0; input < input_count; input++) {
            if (input_buffers[input] == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            SynaptikMetalBufferBox *buffer = (__bridge SynaptikMetalBufferBox *)input_buffers[input];
            if (buffer.buffer.device != box.context.device
                    || buffer.logicalByteSize < box.feedBytes[input].unsignedLongLongValue)
                return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer.buffer
                    shape:box.feedShapes[input] dataType:MPSDataTypeFloat32];
            if (data == nil) return SYNAPTIK_METAL_STATUS_EXECUTION_FAILED;
            [stable_inputs addObject:data];
        }
        for (uint32_t output = 0; output < output_count; output++) {
            if (output_buffers[output] == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            for (uint32_t input = 0; input < input_count; input++)
                if (input_buffers[input] == output_buffers[output])
                    return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            SynaptikMetalBufferBox *buffer = (__bridge SynaptikMetalBufferBox *)output_buffers[output];
            if (buffer.buffer.device != box.context.device
                    || buffer.logicalByteSize < box.targetBytes[output].unsignedLongLongValue)
                return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
            MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer.buffer
                    shape:box.targetShapes[output] dataType:MPSDataTypeFloat32];
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
    } @catch (__unused NSException *exception) { return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR; }
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
