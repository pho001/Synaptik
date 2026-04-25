#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <string.h>

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
@end

@implementation SynaptikAppleMpsExecutableBox
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
        int32_t output_node_index
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
                    int32_t axis = node_scalar_values == NULL ? 0 : (int32_t) node_scalar_values[i];
                    outTensor = [graph softMaxWithTensor:input0 axis:axis name:@"softmax"];
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
                    int32_t axis = node_scalar_values == NULL ? 0 : (int32_t) node_scalar_values[i];
                    outTensor = [graph expandDimsOfTensor:input0 axis:axis name:@"expand_dims"];
                    break;
                }
                case 22: {
                    int32_t axis = node_scalar_values == NULL ? 0 : (int32_t) node_scalar_values[i];
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
        if (output_node_index < 0 || output_node_index >= nodeOutputs.count) {
            return NULL;
        }
        MPSGraphTensor *outputTensor = nodeOutputs[(NSUInteger) output_node_index];

        MPSGraphExecutable *executable = [graph compileWithDevice:contextBox.graphDevice
                                                            feeds:feeds
                                                    targetTensors:@[ outputTensor ]
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
        return (void *) CFBridgingRetain(box);
    }
}

int synaptik_apple_mps_execute_partition_f32(
        void *context,
        void *executable,
        const float * const *external_inputs,
        int32_t external_input_count,
        float *output
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
        if (results.count < 1) {
            return 7;
        }

        MPSGraphTensorData *resultData = results.firstObject;
        MPSNDArray *resultArray = resultData.mpsndarray;
        if (resultArray == nil) {
            return 8;
        }

        [resultArray readBytes:output strideBytes:NULL];
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
