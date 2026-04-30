#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <string.h>
#import <time.h>

static const char *SYNAPTIK_APPLE_MPS_DEFAULT_UNAVAILABLE_REASON =
        "Apple MPSGraph runtime is unavailable on this machine.";

@interface SynaptikAppleMpsContextBox : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) MPSGraphDevice *graphDevice;
@end

@implementation SynaptikAppleMpsContextBox
@end

@interface SynaptikAppleMpsExecutableBox : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<NSNumber *> *externalInputRanks;
@property(nonatomic, strong) NSArray<NSNumber *> *externalInputDTypes;
@property(nonatomic, strong) NSArray<NSNumber *> *externalInputDim0;
@property(nonatomic, strong) NSArray<NSNumber *> *externalInputDim1;
@property(nonatomic, strong) NSArray<NSNumber *> *externalInputDim2;
@property(nonatomic, strong) NSArray<NSNumber *> *externalInputDim3;
@property(nonatomic, strong) NSArray<NSNumber *> *outputRanks;
@property(nonatomic, strong) NSArray<NSNumber *> *outputDTypes;
@property(nonatomic, strong) NSArray<NSNumber *> *outputDim0;
@property(nonatomic, strong) NSArray<NSNumber *> *outputDim1;
@property(nonatomic, strong) NSArray<NSNumber *> *outputDim2;
@property(nonatomic, strong) NSArray<NSNumber *> *outputDim3;
@property(nonatomic, strong) NSArray<NSNumber *> *outputElementCounts;
@end

@implementation SynaptikAppleMpsExecutableBox
@end

@interface SynaptikAppleMpsBufferBox : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) NSUInteger byteLength;
@property(nonatomic) int32_t storageMode;
@property(nonatomic) BOOL ownsBuffer;
@end

@implementation SynaptikAppleMpsBufferBox
@end

static SynaptikAppleMpsContextBox *SynaptikUnboxContext(void *contextPtr) {
    if (contextPtr == NULL) {
        return nil;
    }
    return (__bridge SynaptikAppleMpsContextBox *) contextPtr;
}

static SynaptikAppleMpsExecutableBox *SynaptikUnboxExecutable(void *executablePtr) {
    if (executablePtr == NULL) {
        return nil;
    }
    return (__bridge SynaptikAppleMpsExecutableBox *) executablePtr;
}

static SynaptikAppleMpsBufferBox *SynaptikUnboxBuffer(void *bufferPtr) {
    if (bufferPtr == NULL) {
        return nil;
    }
    return (__bridge SynaptikAppleMpsBufferBox *) bufferPtr;
}

static NSMutableArray<NSNumber *> *SynaptikShapeFromDims(int32_t rank, NSUInteger dim0, NSUInteger dim1, NSUInteger dim2, NSUInteger dim3) {
    if (rank < 1 || rank > 4) {
        return nil;
    }
    NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:(NSUInteger) rank];
    [shape addObject:@(dim0)];
    if (rank >= 2) [shape addObject:@(dim1)];
    if (rank >= 3) [shape addObject:@(dim2)];
    if (rank >= 4) [shape addObject:@(dim3)];
    return shape;
}

static NSUInteger SynaptikElementCountFromDims(int32_t rank, NSUInteger dim0, NSUInteger dim1, NSUInteger dim2, NSUInteger dim3) {
    if (rank < 1 || rank > 4) {
        return 0;
    }
    NSUInteger elementCount = dim0;
    if (rank >= 2) elementCount *= dim1;
    if (rank >= 3) elementCount *= dim2;
    if (rank >= 4) elementCount *= dim3;
    return elementCount;
}

static int64_t SynaptikNowNs(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return 0;
    }
    return ((int64_t) timestamp.tv_sec * 1000000000LL) + (int64_t) timestamp.tv_nsec;
}

static int32_t SynaptikDecodeIntScalar(const float *nodeScalarValues, int32_t index) {
    if (nodeScalarValues == NULL) {
        return 0;
    }
    uint32_t bits = 0;
    memcpy(&bits, &nodeScalarValues[index], sizeof(float));
    return (int32_t) bits;
}

static MPSGraphTensor *SynaptikReductionSumKeepDims(MPSGraph *graph, MPSGraphTensor *tensor, int32_t axis) {
    if (graph == nil || tensor == nil) {
        return nil;
    }
    if (axis == -1) {
        NSArray<NSNumber *> *shape = tensor.shape;
        if (shape == nil || shape.count == 0) {
            return nil;
        }
        MPSGraphTensor *current = tensor;
        for (NSInteger i = shape.count - 1; i >= 0; i--) {
            current = [graph reductionSumWithTensor:current axis:(NSInteger) i name:@"sum_all_keepdims"];
            if (current == nil) {
                return nil;
            }
        }
        return current;
    }
    return [graph reductionSumWithTensor:tensor axis:axis name:@"sum_keepdims"];
}

static MPSGraphTensor *SynaptikTransposeLastTwoAxes(MPSGraph *graph, MPSGraphTensor *tensor, NSString *name) {
    if (graph == nil || tensor == nil || tensor.shape == nil) {
        return nil;
    }
    NSInteger rank = tensor.shape.count;
    if (rank < 2) {
        return nil;
    }
    NSMutableArray<NSNumber *> *permutation = [NSMutableArray arrayWithCapacity:(NSUInteger) rank];
    for (NSInteger i = 0; i < rank - 2; i++) {
        [permutation addObject:@(i)];
    }
    [permutation addObject:@(rank - 1)];
    [permutation addObject:@(rank - 2)];
    return [graph transposeTensor:tensor permutation:permutation name:name];
}

int synaptik_apple_mps_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil ? 1 : 0;
    }
}

const char *synaptik_apple_mps_unavailable_reason(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return SYNAPTIK_APPLE_MPS_DEFAULT_UNAVAILABLE_REASON;
        }
        return "Apple MPSGraph runtime available.";
    }
}

int32_t synaptik_apple_mps_layout_abi_version(void) {
    return 2;
}

int32_t synaptik_apple_mps_validate_layout_abi_v2(
        int32_t binding_count,
        const int32_t *ranks,
        const int32_t *dtypes,
        const int64_t *storage_offsets,
        const int64_t *logical_element_counts,
        const int64_t *logical_byte_lengths,
        const int64_t *physical_byte_spans,
        const int32_t *access_modes,
        const int32_t *layout_classes,
        const void * const *native_handles,
        const int32_t *shape_offsets,
        const int64_t *shape_values,
        const int32_t *stride_offsets,
        const int64_t *stride_values) {
    if (binding_count < 0) {
        return 1;
    }
    if (binding_count == 0) {
        return 0;
    }
    if (ranks == NULL || dtypes == NULL || storage_offsets == NULL
            || logical_element_counts == NULL || logical_byte_lengths == NULL
            || physical_byte_spans == NULL || access_modes == NULL || layout_classes == NULL
            || native_handles == NULL || shape_offsets == NULL || shape_values == NULL
            || stride_offsets == NULL || stride_values == NULL) {
        return 1;
    }
    for (int32_t i = 0; i < binding_count; i++) {
        if (ranks[i] <= 0 || physical_byte_spans[i] < 0 || native_handles[i] == NULL) {
            return 1;
        }
    }
    return 0;
}

void *synaptik_apple_mps_create_context(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return NULL;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            return NULL;
        }
        SynaptikAppleMpsContextBox *box = [SynaptikAppleMpsContextBox new];
        box.device = device;
        box.queue = queue;
        box.graphDevice = [MPSGraphDevice deviceWithMTLDevice:device];
        if (box.graphDevice == nil) {
            return NULL;
        }
        return (void *) CFBridgingRetain(box);
    }
}

void synaptik_apple_mps_destroy_context(void *context) {
    if (context == NULL) {
        return;
    }
    @autoreleasepool {
        CFBridgingRelease(context);
    }
}

void *synaptik_apple_mps_create_buffer(
        void *context,
        int64_t byte_length,
        int32_t storage_mode,
        const void *initial_data,
        int64_t initial_data_bytes
) {
    @autoreleasepool {
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        if (contextBox == nil || contextBox.device == nil || byte_length <= 0) {
            return NULL;
        }
        if (storage_mode != 1) {
            return NULL;
        }
        id<MTLBuffer> buffer = [contextBox.device newBufferWithLength:(NSUInteger) byte_length
                                                              options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return NULL;
        }
        if (initial_data != NULL && initial_data_bytes > 0) {
            if (initial_data_bytes > byte_length || [buffer contents] == NULL) {
                return NULL;
            }
            memcpy([buffer contents], initial_data, (size_t) initial_data_bytes);
        }
        SynaptikAppleMpsBufferBox *box = [SynaptikAppleMpsBufferBox new];
        box.buffer = buffer;
        box.byteLength = (NSUInteger) byte_length;
        box.storageMode = storage_mode;
        box.ownsBuffer = YES;
        return (void *) CFBridgingRetain(box);
    }
}

int32_t synaptik_apple_mps_write_buffer(
        void *buffer,
        const void *src,
        int64_t byte_length
) {
    @autoreleasepool {
        SynaptikAppleMpsBufferBox *box = SynaptikUnboxBuffer(buffer);
        if (box == nil || box.buffer == nil || src == NULL || byte_length < 0) {
            return 1;
        }
        if ((NSUInteger) byte_length > box.byteLength || [box.buffer contents] == NULL) {
            return 2;
        }
        memcpy([box.buffer contents], src, (size_t) byte_length);
        return 0;
    }
}

int32_t synaptik_apple_mps_read_buffer(
        void *buffer,
        void *dst,
        int64_t byte_length
) {
    @autoreleasepool {
        SynaptikAppleMpsBufferBox *box = SynaptikUnboxBuffer(buffer);
        if (box == nil || box.buffer == nil || dst == NULL || byte_length < 0) {
            return 1;
        }
        if ((NSUInteger) byte_length > box.byteLength || [box.buffer contents] == NULL) {
            return 2;
        }
        memcpy(dst, [box.buffer contents], (size_t) byte_length);
        return 0;
    }
}

void synaptik_apple_mps_destroy_buffer(void *buffer) {
    if (buffer == NULL) {
        return;
    }
    @autoreleasepool {
        CFBridgingRelease(buffer);
    }
}

void *synaptik_apple_mps_compile_partition_f32(
        void *context,
        int32_t external_input_count,
        const int32_t *external_input_ranks,
        const int32_t *external_input_dtypes,
        const int32_t *external_input_dim0,
        const int32_t *external_input_dim1,
        const int32_t *external_input_dim2,
        const int32_t *external_input_dim3,
        int32_t post_op_count,
        const int32_t *node_types,
        const int32_t *input0_kinds,
        const int32_t *input0_indices,
        const int32_t *input1_kinds,
        const int32_t *input1_indices,
        const int32_t *input2_kinds,
        const int32_t *input2_indices,
        const int32_t *input3_kinds,
        const int32_t *input3_indices,
        const float *node_scalar_values,
        const int32_t *output_ranks,
        const int32_t *output_dim0,
        const int32_t *output_dim1,
        const int32_t *output_dim2,
        const int32_t *output_dim3,
        int32_t output_node_count,
        const int32_t *output_node_indices
) {
    @autoreleasepool {
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        if (contextBox == nil || contextBox.graphDevice == nil) {
            return NULL;
        }

        MPSGraph *graph = [MPSGraph new];
        NSMutableArray<MPSGraphTensor *> *externalTensors = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger) external_input_count];
        NSMutableArray<NSNumber *> *externalInputRanksBoxed = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        NSMutableArray<NSNumber *> *externalInputDTypesBoxed = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        NSMutableArray<NSNumber *> *externalInputDim0Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        NSMutableArray<NSNumber *> *externalInputDim1Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        NSMutableArray<NSNumber *> *externalInputDim2Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        NSMutableArray<NSNumber *> *externalInputDim3Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        for (int32_t i = 0; i < external_input_count; i++) {
            int32_t rank = external_input_ranks == NULL ? 0 : external_input_ranks[i];
            if (rank < 1 || rank > 4) {
                return NULL;
            }
            int32_t dtypeCode = external_input_dtypes == NULL ? 0 : external_input_dtypes[i];
            MPSDataType dataType = dtypeCode == 2 ? MPSDataTypeBool : MPSDataTypeFloat32;
            NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:(NSUInteger) rank];
            [shape addObject:@(external_input_dim0[i])];
            if (rank >= 2) [shape addObject:@(external_input_dim1[i])];
            if (rank >= 3) [shape addObject:@(external_input_dim2[i])];
            if (rank >= 4) [shape addObject:@(external_input_dim3[i])];
            NSString *name = [NSString stringWithFormat:@"input_%d", i];
            MPSGraphTensor *tensor = [graph placeholderWithShape:shape dataType:dataType name:name];
            MPSGraphShapedType *type = [[MPSGraphShapedType alloc] initWithShape:shape dataType:dataType];
            if (tensor == nil || type == nil) {
                return NULL;
            }
            [externalTensors addObject:tensor];
            feeds[tensor] = type;
            [externalInputRanksBoxed addObject:@(rank)];
            [externalInputDTypesBoxed addObject:@(dtypeCode)];
            [externalInputDim0Boxed addObject:@(external_input_dim0[i])];
            [externalInputDim1Boxed addObject:@(rank >= 2 ? external_input_dim1[i] : 1)];
            [externalInputDim2Boxed addObject:@(rank >= 3 ? external_input_dim2[i] : 1)];
            [externalInputDim3Boxed addObject:@(rank >= 4 ? external_input_dim3[i] : 1)];
        }
        NSMutableArray<MPSGraphTensor *> *nodeOutputs = [NSMutableArray arrayWithCapacity:(NSUInteger) post_op_count];
        for (int32_t i = 0; i < post_op_count; i++) {
            MPSGraphTensor *(^resolveRef)(int32_t, int32_t) = ^MPSGraphTensor *(int32_t kind, int32_t index) {
                switch (kind) {
                    case 1:
                        return (index >= 0 && index < externalTensors.count) ? externalTensors[(NSUInteger) index] : nil;
                    case 2:
                        return (index >= 0 && index < nodeOutputs.count) ? nodeOutputs[(NSUInteger) index] : nil;
                    default:
                        return nil;
                }
            };
            MPSGraphTensor *input0 = resolveRef(input0_kinds == NULL ? 0 : input0_kinds[i], input0_indices == NULL ? -1 : input0_indices[i]);
            MPSGraphTensor *input1 = resolveRef(input1_kinds == NULL ? 0 : input1_kinds[i], input1_indices == NULL ? -1 : input1_indices[i]);
            MPSGraphTensor *input2 = resolveRef(input2_kinds == NULL ? 0 : input2_kinds[i], input2_indices == NULL ? -1 : input2_indices[i]);
            MPSGraphTensor *input3 = resolveRef(input3_kinds == NULL ? 0 : input3_kinds[i], input3_indices == NULL ? -1 : input3_indices[i]);
            if (input0 == nil) {
                return NULL;
            }
            MPSGraphTensor *outTensor = nil;
            switch (node_types[i]) {
                case 1:
                    if (input1 == nil) return NULL;
                    outTensor = [graph matrixMultiplicationWithPrimaryTensor:input0 secondaryTensor:input1 name:@"matmul"];
                    break;
                case 2:
                    if (input1 == nil) return NULL;
                    outTensor = [graph matrixMultiplicationWithPrimaryTensor:input0 secondaryTensor:input1 name:@"linear_matmul"];
                    if (input2 != nil) {
                        outTensor = [graph additionWithPrimaryTensor:outTensor secondaryTensor:input2 name:@"linear_add"];
                    }
                    break;
                case 3:
                    if (input1 == nil) return NULL;
                    outTensor = [graph additionWithPrimaryTensor:input0 secondaryTensor:input1 name:@"add"];
                    break;
                case 4:
                    if (input1 == nil) return NULL;
                    outTensor = [graph subtractionWithPrimaryTensor:input0 secondaryTensor:input1 name:@"sub"];
                    break;
                case 5:
                    if (input1 == nil) return NULL;
                    outTensor = [graph multiplicationWithPrimaryTensor:input0 secondaryTensor:input1 name:@"mul"];
                    break;
                case 6:
                    if (input1 == nil) return NULL;
                    outTensor = [graph divisionWithPrimaryTensor:input0 secondaryTensor:input1 name:@"div"];
                    break;
                case 7:
                    outTensor = [graph reLUWithTensor:input0 name:@"relu"];
                    break;
                case 8:
                    outTensor = [graph tanhWithTensor:input0 name:@"tanh"];
                    break;
                case 9:
                    outTensor = [graph sigmoidWithTensor:input0 name:@"sigmoid"];
                    break;
                case 10:
                    outTensor = [graph absoluteWithTensor:input0 name:@"abs"];
                    break;
                case 11:
                    outTensor = [graph exponentWithTensor:input0 name:@"exp"];
                    break;
                case 12:
                    outTensor = [graph logarithmWithTensor:input0 name:@"log"];
                    break;
                case 13:
                    outTensor = [graph negativeWithTensor:input0 name:@"neg"];
                    break;
                case 14:
                    outTensor = [graph squareRootWithTensor:input0 name:@"sqrt"];
                    break;
                case 15:
                    outTensor = [graph reciprocalWithTensor:input0 name:@"inv"];
                    break;
                case 16: {
                    MPSGraphTensor *scalarTensor = [graph constantWithScalar:(double) node_scalar_values[i] dataType:MPSDataTypeFloat32];
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph maximumWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"clamp_min"];
                    break;
                }
                case 17: {
                    MPSGraphTensor *scalarTensor = [graph constantWithScalar:(double) node_scalar_values[i] dataType:MPSDataTypeFloat32];
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph minimumWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"clamp_max"];
                    break;
                }
                case 23: {
                    MPSGraphTensor *scalarTensor = [graph constantWithScalar:(double) node_scalar_values[i] dataType:MPSDataTypeFloat32];
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph multiplicationWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"mul_scalar"];
                    break;
                }
                case 24:
                    if (input1 == nil || input2 == nil) return NULL;
                    outTensor = [graph selectWithPredicateTensor:input0 truePredicateTensor:input1 falsePredicateTensor:input2 name:@"where"];
                    break;
                case 25: {
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    outTensor = [graph softMaxWithTensor:input0 axis:axis name:@"softmax"];
                    break;
                }
                case 27: {
                    if (input1 == nil) return NULL;
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    MPSGraphTensor *dot = [graph multiplicationWithPrimaryTensor:input0 secondaryTensor:input1 name:@"softmax_grad_dot"];
                    MPSGraphTensor *sum = [graph reductionSumWithTensor:dot axis:axis name:@"softmax_grad_sum"];
                    MPSGraphTensor *diff = [graph subtractionWithPrimaryTensor:input1 secondaryTensor:sum name:@"softmax_grad_diff"];
                    outTensor = [graph multiplicationWithPrimaryTensor:input0 secondaryTensor:diff name:@"softmax_grad_out"];
                    break;
                }
                case 28: {
                    if (input1 == nil) return NULL;
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    MPSGraphTensor *probs = [graph exponentWithTensor:input0 name:@"log_softmax_grad_probs"];
                    MPSGraphTensor *sum = SynaptikReductionSumKeepDims(graph, input1, axis);
                    MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:probs secondaryTensor:sum name:@"log_softmax_grad_scaled"];
                    outTensor = [graph subtractionWithPrimaryTensor:input1 secondaryTensor:scaled name:@"log_softmax_grad_out"];
                    break;
                }
                case 29:
                case 30: {
                    if (input1 == nil || input2 == nil) return NULL;
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    MPSGraphTensor *mask = [graph equalWithPrimaryTensor:input0 secondaryTensor:input1 name:@"reduce_minmax_grad_mask"];
                    if (mask == nil) return NULL;
                    MPSGraphTensor *maskFloat = [graph castTensor:mask toType:MPSDataTypeFloat32 name:@"reduce_minmax_grad_mask_f32"];
                    if (maskFloat == nil) return NULL;
                    MPSGraphTensor *winnerCount = SynaptikReductionSumKeepDims(graph, maskFloat, axis);
                    if (winnerCount == nil) return NULL;
                    MPSGraphTensor *share = [graph divisionWithPrimaryTensor:input2 secondaryTensor:winnerCount name:@"reduce_minmax_grad_share"];
                    if (share == nil) return NULL;
                    outTensor = [graph multiplicationWithPrimaryTensor:maskFloat secondaryTensor:share name:@"reduce_minmax_grad_out"];
                    break;
                }
                case 31:
                case 32: {
                    if (input1 == nil || input2 == nil) return NULL;
                    BOOL forFirstInput = SynaptikDecodeIntScalar(node_scalar_values, i) != 0;
                    MPSGraphTensor *strictFirst = node_types[i] == 31
                            ? [graph lessThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"min_grad_predicate"]
                            : [graph greaterThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"max_grad_predicate"];
                    MPSGraphTensor *strictSecond = node_types[i] == 31
                            ? [graph greaterThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"min_grad_second_predicate"]
                            : [graph lessThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"max_grad_second_predicate"];
                    MPSGraphTensor *equal = [graph equalWithPrimaryTensor:input0 secondaryTensor:input1 name:@"minmax_grad_equal"];
                    if (strictFirst == nil || strictSecond == nil || equal == nil) return NULL;
                    MPSGraphTensor *zero = [graph constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
                    MPSGraphTensor *half = [graph constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
                    if (zero == nil) return NULL;
                    if (half == nil) return NULL;
                    MPSGraphTensor *halfGrad = [graph multiplicationWithPrimaryTensor:input2 secondaryTensor:half name:@"minmax_grad_half"];
                    if (halfGrad == nil) return NULL;
                    MPSGraphTensor *strictPredicate = forFirstInput ? strictFirst : strictSecond;
                    MPSGraphTensor *strictOut = [graph selectWithPredicateTensor:strictPredicate truePredicateTensor:input2 falsePredicateTensor:zero name:@"minmax_grad_strict"];
                    if (strictOut == nil) return NULL;
                    outTensor = [graph selectWithPredicateTensor:equal truePredicateTensor:halfGrad falsePredicateTensor:strictOut name:@"minmax_grad_out"];
                    break;
                }
                case 33:
                case 34:
                case 35: {
                    if (input1 == nil || input2 == nil || input3 == nil) return NULL;
                    float scale = node_scalar_values == NULL ? 1.0f : node_scalar_values[i];
                    NSInteger rank = input0.shape.count;
                    if (rank < 2) return NULL;
                    int32_t axis = (int32_t) rank - 1;

                    MPSGraphTensor *keyT = SynaptikTransposeLastTwoAxes(graph, input1, @"sdpa_backward_key_t");
                    MPSGraphTensor *scores = keyT == nil ? nil : [graph matrixMultiplicationWithPrimaryTensor:input0 secondaryTensor:keyT name:@"sdpa_backward_scores"];
                    if (scores == nil) return NULL;
                    if (scale != 1.0f) {
                        MPSGraphTensor *scaleTensor = [graph constantWithScalar:(double) scale dataType:MPSDataTypeFloat32];
                        if (scaleTensor == nil) return NULL;
                        scores = [graph multiplicationWithPrimaryTensor:scores secondaryTensor:scaleTensor name:@"sdpa_backward_scaled_scores"];
                        if (scores == nil) return NULL;
                    }

                    MPSGraphTensor *weights = [graph softMaxWithTensor:scores axis:axis name:@"sdpa_backward_weights"];
                    if (weights == nil) return NULL;

                    MPSGraphTensor *valueT = SynaptikTransposeLastTwoAxes(graph, input2, @"sdpa_backward_value_t");
                    if (valueT == nil) return NULL;
                    MPSGraphTensor *dWeights = [graph matrixMultiplicationWithPrimaryTensor:input3 secondaryTensor:valueT name:@"sdpa_backward_dweights"];
                    if (dWeights == nil) return NULL;

                    MPSGraphTensor *dot = [graph multiplicationWithPrimaryTensor:weights secondaryTensor:dWeights name:@"sdpa_backward_dot"];
                    MPSGraphTensor *sum = SynaptikReductionSumKeepDims(graph, dot, axis);
                    if (sum == nil) return NULL;
                    MPSGraphTensor *diff = [graph subtractionWithPrimaryTensor:dWeights secondaryTensor:sum name:@"sdpa_backward_diff"];
                    if (diff == nil) return NULL;
                    MPSGraphTensor *dScores = [graph multiplicationWithPrimaryTensor:weights secondaryTensor:diff name:@"sdpa_backward_dscores"];
                    if (dScores == nil) return NULL;
                    if (scale != 1.0f) {
                        MPSGraphTensor *scaleTensor = [graph constantWithScalar:(double) scale dataType:MPSDataTypeFloat32];
                        if (scaleTensor == nil) return NULL;
                        dScores = [graph multiplicationWithPrimaryTensor:dScores secondaryTensor:scaleTensor name:@"sdpa_backward_scaled_dscores"];
                        if (dScores == nil) return NULL;
                    }

                    switch (node_types[i]) {
                        case 33:
                            outTensor = [graph matrixMultiplicationWithPrimaryTensor:dScores secondaryTensor:input1 name:@"sdpa_backward_query"];
                            break;
                        case 34: {
                            MPSGraphTensor *dScoresT = SynaptikTransposeLastTwoAxes(graph, dScores, @"sdpa_backward_dscores_t");
                            outTensor = dScoresT == nil ? nil : [graph matrixMultiplicationWithPrimaryTensor:dScoresT secondaryTensor:input0 name:@"sdpa_backward_key"];
                            break;
                        }
                        case 35: {
                            MPSGraphTensor *weightsT = SynaptikTransposeLastTwoAxes(graph, weights, @"sdpa_backward_weights_t");
                            outTensor = weightsT == nil ? nil : [graph matrixMultiplicationWithPrimaryTensor:weightsT secondaryTensor:input3 name:@"sdpa_backward_value"];
                            break;
                        }
                        default:
                            outTensor = nil;
                            break;
                    }
                    break;
                }
                case 26: {
                    float scale = node_scalar_values == NULL ? 1.0f : node_scalar_values[i];
                    if (input1 == nil || input2 == nil) return NULL;
                    if (input3 != nil) {
                        outTensor = [graph scaledDotProductAttentionWithQueryTensor:input0
                                                                           keyTensor:input1
                                                                         valueTensor:input2
                                                                          maskTensor:input3
                                                                               scale:scale
                                                                                name:@"sdpa"];
                    } else {
                        outTensor = [graph scaledDotProductAttentionWithQueryTensor:input0
                                                                           keyTensor:input1
                                                                         valueTensor:input2
                                                                               scale:scale
                                                                                name:@"sdpa"];
                    }
                    break;
                }
                case 18: {
                    int32_t rank = output_ranks == NULL ? 0 : output_ranks[i];
                    NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:(NSUInteger) rank];
                    [shape addObject:@(output_dim0[i])];
                    if (rank >= 2) [shape addObject:@(output_dim1[i])];
                    if (rank >= 3) [shape addObject:@(output_dim2[i])];
                    if (rank >= 4) [shape addObject:@(output_dim3[i])];
                    outTensor = [graph reshapeTensor:input0 withShape:shape name:@"reshape"];
                    break;
                }
                case 19:
                    outTensor = [graph identityWithTensor:input0 name:@"contiguous"];
                    break;
                case 20: {
                    uint32_t mode = 0;
                    if (node_scalar_values != NULL) {
                        memcpy(&mode, &node_scalar_values[i], sizeof(float));
                    }
                    int32_t rank = (int32_t) (mode & 0xFF);
                    if (rank < 1 || rank > 4) {
                        return NULL;
                    }
                    NSMutableArray<NSNumber *> *permutation = [NSMutableArray arrayWithCapacity:(NSUInteger) rank];
                    for (int j = 0; j < rank; j++) {
                        [permutation addObject:@((mode >> (8 + j * 4)) & 0xF)];
                    }
                    outTensor = [graph transposeTensor:input0 permutation:permutation name:@"permute"];
                    break;
                }
                case 21: {
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    outTensor = [graph expandDimsOfTensor:input0 axis:axis name:@"expand_dims"];
                    break;
                }
                case 22: {
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    outTensor = [graph squeezeTensor:input0 axis:axis name:@"squeeze"];
                    break;
                }
                default:
                    return NULL;
            }
            if (outTensor == nil) {
                return NULL;
            }
            [nodeOutputs addObject:outTensor];
        }
        if (output_node_count < 1 || output_node_indices == NULL) {
            return NULL;
        }
        NSMutableArray<MPSGraphTensor *> *targetTensors = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        NSMutableArray<NSNumber *> *outputRanksBoxed = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        NSMutableArray<NSNumber *> *outputDTypesBoxed = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        NSMutableArray<NSNumber *> *outputDim0Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        NSMutableArray<NSNumber *> *outputDim1Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        NSMutableArray<NSNumber *> *outputDim2Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        NSMutableArray<NSNumber *> *outputDim3Boxed = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        NSMutableArray<NSNumber *> *outputElementCountsBoxed = [NSMutableArray arrayWithCapacity:(NSUInteger) output_node_count];
        for (int32_t i = 0; i < output_node_count; i++) {
            int32_t output_node_index = output_node_indices[i];
            if (output_node_index < 0 || output_node_index >= nodeOutputs.count) {
                return NULL;
            }
            MPSGraphTensor *outputTensor = nodeOutputs[(NSUInteger) output_node_index];
            if (outputTensor == nil) {
                return NULL;
            }
            [targetTensors addObject:outputTensor];
            int32_t rank = output_ranks == NULL ? 0 : output_ranks[output_node_index];
            if (rank < 1 || rank > 4) {
                return NULL;
            }
            NSUInteger dim0 = (NSUInteger) output_dim0[output_node_index];
            NSUInteger dim1 = (NSUInteger) (rank >= 2 ? output_dim1[output_node_index] : 1);
            NSUInteger dim2 = (NSUInteger) (rank >= 3 ? output_dim2[output_node_index] : 1);
            NSUInteger dim3 = (NSUInteger) (rank >= 4 ? output_dim3[output_node_index] : 1);
            NSUInteger elementCount = SynaptikElementCountFromDims(rank, dim0, dim1, dim2, dim3);
            if (elementCount == 0) {
                return NULL;
            }
            [outputRanksBoxed addObject:@(rank)];
            [outputDTypesBoxed addObject:@(1)];
            [outputDim0Boxed addObject:@(dim0)];
            [outputDim1Boxed addObject:@(dim1)];
            [outputDim2Boxed addObject:@(dim2)];
            [outputDim3Boxed addObject:@(dim3)];
            [outputElementCountsBoxed addObject:@(elementCount)];
        }

        MPSGraphExecutable *executable = [graph compileWithDevice:contextBox.graphDevice
                                                            feeds:feeds
                                                    targetTensors:[targetTensors copy]
                                                 targetOperations:nil
                                            compilationDescriptor:nil];
        if (executable == nil) {
            return NULL;
        }

        SynaptikAppleMpsExecutableBox *box = [SynaptikAppleMpsExecutableBox new];
        box.graph = graph;
        box.executable = executable;
        box.externalInputRanks = [externalInputRanksBoxed copy];
        box.externalInputDTypes = [externalInputDTypesBoxed copy];
        box.externalInputDim0 = [externalInputDim0Boxed copy];
        box.externalInputDim1 = [externalInputDim1Boxed copy];
        box.externalInputDim2 = [externalInputDim2Boxed copy];
        box.externalInputDim3 = [externalInputDim3Boxed copy];
        box.outputRanks = [outputRanksBoxed copy];
        box.outputDTypes = [outputDTypesBoxed copy];
        box.outputDim0 = [outputDim0Boxed copy];
        box.outputDim1 = [outputDim1Boxed copy];
        box.outputDim2 = [outputDim2Boxed copy];
        box.outputDim3 = [outputDim3Boxed copy];
        box.outputElementCounts = [outputElementCountsBoxed copy];
        return (void *) CFBridgingRetain(box);
    }
}

int synaptik_apple_mps_execute_partition_f32(
        void *context,
        void *executable,
        const float * const *external_inputs,
        int32_t external_input_count,
        float **outputs,
        int32_t output_count
) {
    @autoreleasepool {
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        SynaptikAppleMpsExecutableBox *executableBox = SynaptikUnboxExecutable(executable);
        if (contextBox == nil || executableBox == nil) {
            return 1;
        }

        if ((NSUInteger) external_input_count != executableBox.externalInputRanks.count) {
            return 2;
        }

        MPSGraphExecutableExecutionDescriptor *executionDescriptor = [MPSGraphExecutableExecutionDescriptor new];
        executionDescriptor.waitUntilCompleted = YES;

        NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        for (int32_t i = 0; i < external_input_count; i++) {
            int32_t rank = executableBox.externalInputRanks[(NSUInteger) i].intValue;
            if (external_inputs[i] == NULL) {
                return 6;
            }
            if (rank < 1 || rank > 4) {
                return 6;
            }
            NSUInteger dim0 = (NSUInteger) executableBox.externalInputDim0[(NSUInteger) i].intValue;
            NSUInteger dim1 = (NSUInteger) executableBox.externalInputDim1[(NSUInteger) i].intValue;
            NSUInteger dim2 = (NSUInteger) executableBox.externalInputDim2[(NSUInteger) i].intValue;
            NSUInteger dim3 = (NSUInteger) executableBox.externalInputDim3[(NSUInteger) i].intValue;
            int32_t dtypeCode = executableBox.externalInputDTypes[(NSUInteger) i].intValue;
            MPSDataType dataType = dtypeCode == 2 ? MPSDataTypeBool : MPSDataTypeFloat32;
            NSUInteger elementCount = dim0;
            if (rank >= 2) elementCount *= dim1;
            if (rank >= 3) elementCount *= dim2;
            if (rank >= 4) elementCount *= dim3;
            NSUInteger bytes = elementCount * (dtypeCode == 2 ? sizeof(uint8_t) : sizeof(float));
            id<MTLBuffer> buffer = [contextBox.device newBufferWithBytes:external_inputs[i]
                                                                  length:bytes
                                                                 options:MTLResourceStorageModeShared];
            NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:(NSUInteger) rank];
            [shape addObject:@(dim0)];
            if (rank >= 2) [shape addObject:@(dim1)];
            if (rank >= 3) [shape addObject:@(dim2)];
            if (rank >= 4) [shape addObject:@(dim3)];
            MPSGraphTensorData *data = buffer == nil ? nil : [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                                                       shape:shape
                                                                                                    dataType:dataType];
            if (data == nil) {
                return 6;
            }
            [inputs addObject:data];
        }
        NSArray<MPSGraphTensorData *> *results =
                [executableBox.executable runWithMTLCommandQueue:contextBox.queue
                                                     inputsArray:[inputs copy]
                                                    resultsArray:nil
                                             executionDescriptor:executionDescriptor];
        if (results.count < 1 || output_count < 1 || outputs == NULL || results.count < (NSUInteger) output_count) {
            return 7;
        }
        for (int32_t i = 0; i < output_count; i++) {
            MPSGraphTensorData *resultData = results[(NSUInteger) i];
            MPSNDArray *resultArray = resultData.mpsndarray;
            if (resultArray == nil || outputs[i] == NULL) {
                return 8;
            }
            [resultArray readBytes:outputs[i] strideBytes:NULL];
        }
        return 0;
    }
}

int32_t synaptik_apple_mps_execute_partition_f32_buffers(
        void *context,
        void *executable,
        const void * const *external_input_buffers,
        int32_t external_input_count,
        void * const *output_buffers,
        int32_t output_count,
        int64_t *native_device_copy_ns
) {
    @autoreleasepool {
        if (native_device_copy_ns != NULL) {
            *native_device_copy_ns = 0;
        }
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        SynaptikAppleMpsExecutableBox *executableBox = SynaptikUnboxExecutable(executable);
        if (contextBox == nil || executableBox == nil) {
            return 1;
        }
        if ((NSUInteger) external_input_count != executableBox.externalInputRanks.count
                || external_input_buffers == NULL) {
            return 2;
        }
        if ((NSUInteger) output_count != executableBox.outputRanks.count
                || output_buffers == NULL
                || output_count < 1) {
            return 3;
        }

        MPSGraphExecutableExecutionDescriptor *executionDescriptor = [MPSGraphExecutableExecutionDescriptor new];
        executionDescriptor.waitUntilCompleted = YES;

        NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray arrayWithCapacity:(NSUInteger) external_input_count];
        for (int32_t i = 0; i < external_input_count; i++) {
            SynaptikAppleMpsBufferBox *box = SynaptikUnboxBuffer((void *) external_input_buffers[i]);
            if (box == nil || box.buffer == nil) {
                return 4;
            }
            int32_t rank = executableBox.externalInputRanks[(NSUInteger) i].intValue;
            NSUInteger dim0 = (NSUInteger) executableBox.externalInputDim0[(NSUInteger) i].intValue;
            NSUInteger dim1 = (NSUInteger) executableBox.externalInputDim1[(NSUInteger) i].intValue;
            NSUInteger dim2 = (NSUInteger) executableBox.externalInputDim2[(NSUInteger) i].intValue;
            NSUInteger dim3 = (NSUInteger) executableBox.externalInputDim3[(NSUInteger) i].intValue;
            int32_t dtypeCode = executableBox.externalInputDTypes[(NSUInteger) i].intValue;
            MPSDataType dataType = dtypeCode == 2 ? MPSDataTypeBool : MPSDataTypeFloat32;
            NSUInteger elementCount = SynaptikElementCountFromDims(rank, dim0, dim1, dim2, dim3);
            NSUInteger bytes = elementCount * (dtypeCode == 2 ? sizeof(uint8_t) : sizeof(float));
            NSMutableArray<NSNumber *> *shape = SynaptikShapeFromDims(rank, dim0, dim1, dim2, dim3);
            if (shape == nil || elementCount == 0 || box.byteLength < bytes) {
                return 5;
            }
            MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:box.buffer
                                                                               shape:shape
                                                                            dataType:dataType];
            if (data == nil) {
                return 6;
            }
            [inputs addObject:data];
        }

        NSMutableArray<MPSGraphTensorData *> *outputs = [NSMutableArray arrayWithCapacity:(NSUInteger) output_count];
        for (int32_t i = 0; i < output_count; i++) {
            SynaptikAppleMpsBufferBox *box = SynaptikUnboxBuffer(output_buffers[i]);
            if (box == nil || box.buffer == nil) {
                return 7;
            }
            int32_t rank = executableBox.outputRanks[(NSUInteger) i].intValue;
            NSUInteger dim0 = (NSUInteger) executableBox.outputDim0[(NSUInteger) i].intValue;
            NSUInteger dim1 = (NSUInteger) executableBox.outputDim1[(NSUInteger) i].intValue;
            NSUInteger dim2 = (NSUInteger) executableBox.outputDim2[(NSUInteger) i].intValue;
            NSUInteger dim3 = (NSUInteger) executableBox.outputDim3[(NSUInteger) i].intValue;
            int32_t dtypeCode = executableBox.outputDTypes[(NSUInteger) i].intValue;
            MPSDataType dataType = dtypeCode == 2 ? MPSDataTypeBool : MPSDataTypeFloat32;
            NSUInteger elementCount = (NSUInteger) executableBox.outputElementCounts[(NSUInteger) i].unsignedLongLongValue;
            NSUInteger bytes = elementCount * (dtypeCode == 2 ? sizeof(uint8_t) : sizeof(float));
            NSMutableArray<NSNumber *> *shape = SynaptikShapeFromDims(rank, dim0, dim1, dim2, dim3);
            if (shape == nil || elementCount == 0 || box.byteLength < bytes) {
                return 8;
            }
            MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:box.buffer
                                                                               shape:shape
                                                                            dataType:dataType];
            if (data == nil) {
                return 9;
            }
            [outputs addObject:data];
        }

        NSArray<MPSGraphTensorData *> *results =
                [executableBox.executable runWithMTLCommandQueue:contextBox.queue
                                                     inputsArray:[inputs copy]
                                                    resultsArray:[outputs copy]
                                             executionDescriptor:executionDescriptor];
        if (results.count < (NSUInteger) output_count) {
            return 10;
        }
        int64_t copyNs = 0;
        for (int32_t i = 0; i < output_count; i++) {
            SynaptikAppleMpsBufferBox *box = SynaptikUnboxBuffer(output_buffers[i]);
            MPSGraphTensorData *resultData = results[(NSUInteger) i];
            MPSNDArray *resultArray = resultData.mpsndarray;
            if (box == nil || box.buffer == nil || resultArray == nil) {
                return 11;
            }
            int32_t dtypeCode = executableBox.outputDTypes[(NSUInteger) i].intValue;
            NSUInteger elementCount = (NSUInteger) executableBox.outputElementCounts[(NSUInteger) i].unsignedLongLongValue;
            NSUInteger bytes = elementCount * (dtypeCode == 2 ? sizeof(uint8_t) : sizeof(float));
            void *contents = box.buffer.contents;
            if (contents == NULL || box.byteLength < bytes) {
                return 12;
            }
            int64_t copyStart = SynaptikNowNs();
            [resultArray readBytes:contents strideBytes:NULL];
            int64_t copyEnd = SynaptikNowNs();
            if (copyEnd > copyStart) {
                copyNs += copyEnd - copyStart;
            }
        }
        if (native_device_copy_ns != NULL) {
            *native_device_copy_ns = copyNs;
        }
        return 0;
    }
}

void synaptik_apple_mps_destroy_executable(void *executable) {
    if (executable == NULL) {
        return;
    }
    @autoreleasepool {
        CFBridgingRelease(executable);
    }
}
