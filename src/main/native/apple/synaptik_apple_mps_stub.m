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
@property(nonatomic, strong) id<MTLComputePipelineState> layoutContiguousPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> customReluF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> optimizerSgdF32Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> optimizerAdamF32Pipeline;
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

static NSUInteger SynaptikByteSizeForDTypeCode(int32_t dtypeCode) {
    switch (dtypeCode) {
        case 1:
            return sizeof(float);
        case 2:
            return sizeof(uint8_t);
        case 3:
            return sizeof(uint16_t);
        case 4:
            return sizeof(int32_t);
        default:
            return 0;
    }
}

static BOOL SynaptikMpsDataTypeForCode(int32_t dtypeCode, MPSDataType *outDataType) {
    if (outDataType == NULL) {
        return NO;
    }
    switch (dtypeCode) {
        case 1:
            *outDataType = MPSDataTypeFloat32;
            return YES;
        case 2:
            *outDataType = MPSDataTypeBool;
            return YES;
        case 3:
            if (@available(macOS 14.0, iOS 16.0, tvOS 16.0, *)) {
                *outDataType = MPSDataTypeBFloat16;
                return YES;
            }
            return NO;
        case 4:
            *outDataType = MPSDataTypeInt32;
            return YES;
        default:
            return NO;
    }
}

static int32_t SynaptikNodeOutputDTypeCode(const int32_t *nodeOutputDTypes, int32_t nodeIndex) {
    return nodeOutputDTypes == NULL ? 1 : nodeOutputDTypes[nodeIndex];
}

static MPSGraphTensor *SynaptikScalarConstant(
        MPSGraph *graph,
        double value,
        int32_t dtypeCode
) {
    MPSDataType dataType = MPSDataTypeInvalid;
    if (!SynaptikMpsDataTypeForCode(dtypeCode, &dataType)) {
        return nil;
    }
    return [graph constantWithScalar:value dataType:dataType];
}

static int64_t SynaptikNowNs(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return 0;
    }
    return ((int64_t) timestamp.tv_sec * 1000000000LL) + (int64_t) timestamp.tv_nsec;
}

static id<MTLComputePipelineState> SynaptikLayoutContiguousPipeline(id<MTLDevice> device) {
    static NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void synaptik_layout_contiguous(\n"
             "    device const uchar *source [[buffer(0)]],\n"
             "    device uchar *destination [[buffer(1)]],\n"
             "    device const long *shape [[buffer(2)]],\n"
             "    device const long *strides [[buffer(3)]],\n"
             "    constant long &logicalElementCount [[buffer(4)]],\n"
             "    constant int &rank [[buffer(5)]],\n"
             "    constant long &storageOffset [[buffer(6)]],\n"
             "    constant int &elementSize [[buffer(7)]],\n"
             "    uint gid [[thread_position_in_grid]]) {\n"
             "    long linear = (long) gid;\n"
             "    if (linear >= logicalElementCount) { return; }\n"
             "    long sourceIndex = storageOffset;\n"
             "    long remaining = linear;\n"
             "    for (int dim = rank - 1; dim >= 0; dim--) {\n"
             "        long coordinate = remaining % shape[dim];\n"
             "        remaining = remaining / shape[dim];\n"
             "        sourceIndex += coordinate * strides[dim];\n"
             "    }\n"
             "    long sourceByte = sourceIndex * elementSize;\n"
             "    long destinationByte = linear * elementSize;\n"
             "    for (int byteIndex = 0; byteIndex < elementSize; byteIndex++) {\n"
             "        destination[destinationByte + byteIndex] = source[sourceByte + byteIndex];\n"
             "    }\n"
             "}\n";
    if (device == nil) {
        return nil;
    }
    NSError *libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&libraryError];
    if (library == nil) {
        return nil;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"synaptik_layout_contiguous"];
    if (function == nil) {
        return nil;
    }
    NSError *pipelineError = nil;
    return [device newComputePipelineStateWithFunction:function error:&pipelineError];
}

static id<MTLComputePipelineState> SynaptikCustomReluF32Pipeline(id<MTLDevice> device) {
    static NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void synaptik_custom_relu_f32(\n"
             "    device const float *source [[buffer(0)]],\n"
             "    device float *destination [[buffer(1)]],\n"
             "    constant long &logicalElementCount [[buffer(2)]],\n"
             "    uint gid [[thread_position_in_grid]]) {\n"
             "    long linear = (long) gid;\n"
             "    if (linear >= logicalElementCount) { return; }\n"
             "    float value = source[linear];\n"
             "    destination[linear] = value > 0.0f ? value : 0.0f;\n"
             "}\n";
    if (device == nil) {
        return nil;
    }
    NSError *libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&libraryError];
    if (library == nil) {
        return nil;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"synaptik_custom_relu_f32"];
    if (function == nil) {
        return nil;
    }
    NSError *pipelineError = nil;
    return [device newComputePipelineStateWithFunction:function error:&pipelineError];
}

static id<MTLComputePipelineState> SynaptikOptimizerPipeline(id<MTLDevice> device, NSString *functionName) {
    static NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void synaptik_optimizer_sgd_f32(\n"
             "    device const float *parameter [[buffer(0)]],\n"
             "    device const float *gradient [[buffer(1)]],\n"
             "    device float *output [[buffer(2)]],\n"
             "    constant float &learningRate [[buffer(3)]],\n"
             "    constant long &logicalElementCount [[buffer(4)]],\n"
             "    uint gid [[thread_position_in_grid]]) {\n"
             "    long linear = (long) gid;\n"
             "    if (linear >= logicalElementCount) { return; }\n"
             "    output[linear] = parameter[linear] - learningRate * gradient[linear];\n"
             "}\n"
             "kernel void synaptik_optimizer_adam_f32(\n"
             "    device const float *parameter [[buffer(0)]],\n"
             "    device const float *gradient [[buffer(1)]],\n"
             "    device float *firstMoment [[buffer(2)]],\n"
             "    device float *secondMoment [[buffer(3)]],\n"
             "    device float *output [[buffer(4)]],\n"
             "    constant float &learningRate [[buffer(5)]],\n"
             "    constant float &beta1 [[buffer(6)]],\n"
             "    constant float &beta2 [[buffer(7)]],\n"
             "    constant float &epsilon [[buffer(8)]],\n"
             "    constant int &step [[buffer(9)]],\n"
             "    constant long &logicalElementCount [[buffer(10)]],\n"
             "    uint gid [[thread_position_in_grid]]) {\n"
             "    long linear = (long) gid;\n"
             "    if (linear >= logicalElementCount) { return; }\n"
             "    float g = gradient[linear];\n"
             "    float m = beta1 * firstMoment[linear] + (1.0f - beta1) * g;\n"
             "    float v = beta2 * secondMoment[linear] + (1.0f - beta2) * g * g;\n"
             "    firstMoment[linear] = m;\n"
             "    secondMoment[linear] = v;\n"
             "    float bias1 = 1.0f - pow(beta1, (float) step);\n"
             "    float bias2 = 1.0f - pow(beta2, (float) step);\n"
             "    float mHat = m / bias1;\n"
             "    float vHat = v / bias2;\n"
             "    output[linear] = parameter[linear] - learningRate * mHat / (sqrt(vHat) + epsilon);\n"
             "}\n";
    if (device == nil || functionName == nil) {
        return nil;
    }
    NSError *libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&libraryError];
    if (library == nil) {
        return nil;
    }
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (function == nil) {
        return nil;
    }
    NSError *pipelineError = nil;
    return [device newComputePipelineStateWithFunction:function error:&pipelineError];
}

static int32_t SynaptikDecodeIntScalar(const float *nodeScalarValues, int32_t index) {
    if (nodeScalarValues == NULL) {
        return 0;
    }
    uint32_t bits = 0;
    memcpy(&bits, &nodeScalarValues[index], sizeof(float));
    return (int32_t) bits;
}

static int32_t SynaptikDecodeReductionAxis(const float *nodeScalarValues, int32_t index) {
    int32_t encoded = SynaptikDecodeIntScalar(nodeScalarValues, index);
    int32_t axis = encoded & 0xFFFF;
    if ((axis & 0x8000) != 0) {
        axis |= 0xFFFF0000;
    }
    return axis;
}

static void SynaptikDecodeConv2DMode(
        const float *nodeScalarValues,
        int32_t index,
        int32_t *strideH,
        int32_t *strideW,
        int32_t *padH,
        int32_t *padW) {
    uint32_t encoded = (uint32_t) SynaptikDecodeIntScalar(nodeScalarValues, index);
    *strideH = (int32_t) (encoded & 0xFF);
    *strideW = (int32_t) ((encoded >> 8) & 0xFF);
    *padH = (int32_t) ((encoded >> 16) & 0xFF);
    *padW = (int32_t) ((encoded >> 24) & 0xFF);
}

static void SynaptikDecodePool2DMode(
        const float *nodeScalarValues,
        int32_t index,
        int32_t *kernelH,
        int32_t *kernelW,
        int32_t *strideH,
        int32_t *strideW,
        int32_t *padH,
        int32_t *padW,
        BOOL *countIncludePad) {
    uint32_t encoded = (uint32_t) SynaptikDecodeIntScalar(nodeScalarValues, index);
    *kernelH = (int32_t) (encoded & 0xF);
    *kernelW = (int32_t) ((encoded >> 4) & 0xF);
    *strideH = (int32_t) ((encoded >> 8) & 0xF);
    *strideW = (int32_t) ((encoded >> 12) & 0xF);
    *padH = (int32_t) ((encoded >> 16) & 0xF);
    *padW = (int32_t) ((encoded >> 20) & 0xF);
    *countIncludePad = ((encoded >> 24) & 0x1) != 0;
}

static void SynaptikDecodeCrossEntropyLossMode(
        const float *nodeScalarValues,
        int32_t index,
        int32_t *axis,
        int32_t *reduction,
        BOOL *hasIgnoreIndex,
        int32_t *ignoreIndex) {
    uint32_t encoded = 0;
    if (nodeScalarValues != NULL) {
        memcpy(&encoded, &nodeScalarValues[index], sizeof(float));
    }
    if (axis != NULL) {
        *axis = (int32_t) (encoded & 0xFFu);
    }
    if (reduction != NULL) {
        *reduction = (int32_t) ((encoded >> 8) & 0x3u);
    }
    if (hasIgnoreIndex != NULL) {
        *hasIgnoreIndex = ((encoded >> 10) & 0x1u) != 0;
    }
    if (ignoreIndex != NULL) {
        *ignoreIndex = (int32_t) ((int16_t) ((encoded >> 16) & 0xFFFFu));
    }
}

static NSMutableArray<NSNumber *> *SynaptikOutputShapeForNode(
        int32_t index,
        const int32_t *outputRanks,
        const int32_t *outputDim0,
        const int32_t *outputDim1,
        const int32_t *outputDim2,
        const int32_t *outputDim3) {
    int32_t rank = outputRanks == NULL ? 0 : outputRanks[index];
    if (rank < 1 || rank > 4) {
        return nil;
    }
    NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:(NSUInteger) rank];
    [shape addObject:@(outputDim0[index])];
    if (rank >= 2) [shape addObject:@(outputDim1[index])];
    if (rank >= 3) [shape addObject:@(outputDim2[index])];
    if (rank >= 4) [shape addObject:@(outputDim3[index])];
    return shape;
}

static MPSGraphTensor *SynaptikReshapeToNodeOutput(
        MPSGraph *graph,
        MPSGraphTensor *tensor,
        int32_t index,
        const int32_t *outputRanks,
        const int32_t *outputDim0,
        const int32_t *outputDim1,
        const int32_t *outputDim2,
        const int32_t *outputDim3,
        NSString *name) {
    NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
            index,
            outputRanks,
            outputDim0,
            outputDim1,
            outputDim2,
            outputDim3
    );
    if (graph == nil || tensor == nil || shape == nil) {
        return nil;
    }
    if (tensor.shape != nil && [tensor.shape isEqualToArray:shape]) {
        return tensor;
    }
    return [graph reshapeTensor:tensor withShape:shape name:name];
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

static MPSGraphTensor *SynaptikReductionSumAllToShapeOne(MPSGraph *graph, MPSGraphTensor *tensor, NSString *name) {
    if (graph == nil || tensor == nil || tensor.shape == nil || tensor.shape.count == 0) {
        return nil;
    }
    MPSGraphTensor *current = tensor;
    for (NSInteger axis = (NSInteger) tensor.shape.count - 1; axis >= 0; axis--) {
        current = [graph reductionSumWithTensor:current axis:axis name:name];
        if (current == nil) {
            return nil;
        }
    }
    return [graph reshapeTensor:current withShape:@[@1] name:[name stringByAppendingString:@"_shape"]];
}

static MPSGraphTensor *SynaptikReductionAllToOutputShape(
        MPSGraph *graph,
        MPSGraphTensor *tensor,
        int32_t nodeType,
        NSMutableArray<NSNumber *> *outputShape,
        NSString *name) {
    if (graph == nil || tensor == nil || tensor.shape == nil || tensor.shape.count == 0 || outputShape == nil) {
        return nil;
    }
    MPSGraphTensor *current = tensor;
    for (NSInteger axis = (NSInteger) tensor.shape.count - 1; axis >= 0; axis--) {
        switch (nodeType) {
            case 36:
                current = [graph reductionSumWithTensor:current axis:axis name:name];
                break;
            case 37:
                current = [graph meanOfTensor:current axes:@[@(axis)] name:name];
                break;
            case 38:
                current = [graph reductionMinimumWithTensor:current axis:axis name:name];
                break;
            case 39:
                current = [graph reductionMaximumWithTensor:current axis:axis name:name];
                break;
            case 50:
                current = [graph reductionAndWithTensor:current axis:axis name:name];
                break;
            case 51:
                current = [graph reductionOrWithTensor:current axis:axis name:name];
                break;
            default:
                return nil;
        }
        if (current == nil) {
            return nil;
        }
    }
    if (current.shape != nil && [current.shape isEqualToArray:outputShape]) {
        return current;
    }
    return [graph reshapeTensor:current withShape:outputShape name:[name stringByAppendingString:@"_shape"]];
}

static MPSGraphTensor *SynaptikGatherReducedAlongAxis(
        MPSGraph *graph,
        MPSGraphTensor *values,
        MPSGraphTensor *indices,
        int32_t axis,
        NSString *name) {
    if (graph == nil || values == nil || indices == nil || values.shape == nil || indices.shape == nil) {
        return nil;
    }
    NSUInteger valueRank = values.shape.count;
    NSUInteger indexRank = indices.shape.count;
    if (valueRank < 1 || valueRank > 4 || axis < 0 || axis >= (int32_t) valueRank) {
        return nil;
    }
    if (indexRank + 1 == valueRank) {
        MPSGraphTensor *expandedIndices = [graph expandDimsOfTensor:indices axis:axis name:[name stringByAppendingString:@"_indices_expand"]];
        if (expandedIndices == nil) {
            return nil;
        }
        MPSGraphTensor *gathered = [graph gatherAlongAxis:(NSInteger) axis
                                        withUpdatesTensor:values
                                            indicesTensor:expandedIndices
                                                     name:name];
        return gathered == nil ? nil : [graph squeezeTensor:gathered axis:axis name:[name stringByAppendingString:@"_squeeze"]];
    }
    if (indexRank == valueRank) {
        return [graph gatherAlongAxis:(NSInteger) axis
                    withUpdatesTensor:values
                        indicesTensor:indices
                                 name:name];
    }
    return nil;
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

int32_t synaptik_apple_mps_dtype_abi_version(void) {
    return 3;
}

int32_t synaptik_apple_mps_validate_dtype_abi_v3(
        int32_t descriptor_count,
        const int32_t *roles,
        const int32_t *dtypes,
        const int32_t *op_types) {
    (void) op_types;
    if (descriptor_count < 0) {
        return 1;
    }
    if (descriptor_count == 0) {
        return 0;
    }
    if (roles == NULL || dtypes == NULL) {
        return 1;
    }
    for (int32_t i = 0; i < descriptor_count; i++) {
        int32_t role = roles[i];
        int32_t dtype = dtypes[i];
        if (dtype < 1 || dtype > 5) {
            return 1;
        }
        switch (role) {
            case 1: // storage descriptor: all public Synaptik dtypes are representable as storage metadata
                break;
            case 2: // external data input
                if (dtype != 1 && dtype != 2 && dtype != 3 && dtype != 4) {
                    return 2;
                }
                break;
            case 3: // predicate external input
                if (dtype != 2) {
                    return 2;
                }
                break;
            case 4: // native compute
            case 5: // native output
                if (dtype != 1 && dtype != 2 && dtype != 3) {
                    return 2;
                }
                break;
            default:
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

static void *SynaptikCompilePartition(
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
        const int32_t *input4_kinds,
        const int32_t *input4_indices,
        const float *node_scalar_values,
        const int32_t *output_ranks,
        const int32_t *output_dim0,
        const int32_t *output_dim1,
        const int32_t *output_dim2,
        const int32_t *output_dim3,
        const int32_t *node_output_dtypes,
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
            MPSDataType dataType = MPSDataTypeInvalid;
            if (!SynaptikMpsDataTypeForCode(dtypeCode, &dataType)) {
                return NULL;
            }
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
            MPSGraphTensor *input4 = resolveRef(input4_kinds == NULL ? 0 : input4_kinds[i], input4_indices == NULL ? -1 : input4_indices[i]);
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
                case 57:
                    if (input1 == nil) return NULL;
                    outTensor = [graph minimumWithPrimaryTensor:input0 secondaryTensor:input1 name:@"min"];
                    break;
                case 58:
                    if (input1 == nil) return NULL;
                    outTensor = [graph maximumWithPrimaryTensor:input0 secondaryTensor:input1 name:@"max"];
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
                    MPSGraphTensor *scalarTensor = SynaptikScalarConstant(
                            graph,
                            (double) node_scalar_values[i],
                            SynaptikNodeOutputDTypeCode(node_output_dtypes, i)
                    );
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph maximumWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"clamp_min"];
                    break;
                }
                case 17: {
                    MPSGraphTensor *scalarTensor = SynaptikScalarConstant(
                            graph,
                            (double) node_scalar_values[i],
                            SynaptikNodeOutputDTypeCode(node_output_dtypes, i)
                    );
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph minimumWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"clamp_max"];
                    break;
                }
                case 23: {
                    MPSGraphTensor *scalarTensor = SynaptikScalarConstant(
                            graph,
                            (double) node_scalar_values[i],
                            SynaptikNodeOutputDTypeCode(node_output_dtypes, i)
                    );
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph multiplicationWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"mul_scalar"];
                    break;
                }
                case 59: {
                    MPSGraphTensor *scalarTensor = SynaptikScalarConstant(
                            graph,
                            (double) node_scalar_values[i],
                            SynaptikNodeOutputDTypeCode(node_output_dtypes, i)
                    );
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph powerWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"pow_scalar"];
                    break;
                }
                case 60: {
                    NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                            i,
                            output_ranks,
                            output_dim0,
                            output_dim1,
                            output_dim2,
                            output_dim3
                    );
                    if (shape == nil) return NULL;
                    outTensor = [graph broadcastTensor:input0 toShape:shape name:@"expand"];
                    break;
                }
                case 61: {
                    uint32_t mode = 0;
                    if (node_scalar_values != NULL) {
                        memcpy(&mode, &node_scalar_values[i], sizeof(float));
                    }
                    NSUInteger axis = (NSUInteger) (mode & 0xFFFF);
                    NSInteger index = (NSInteger) ((mode >> 16) & 0xFFFF);
                    if (input0.shape == nil || axis >= input0.shape.count) {
                        return NULL;
                    }
                    MPSGraphTensor *slice = [graph sliceTensor:input0
                                                     dimension:axis
                                                         start:index
                                                        length:1
                                                          name:@"select_slice"];
                    outTensor = slice == nil ? nil : [graph squeezeTensor:slice axis:(NSInteger) axis name:@"select"];
                    break;
                }
                case 40: {
                    MPSGraphTensor *scalarTensor = SynaptikScalarConstant(
                            graph,
                            (double) node_scalar_values[i],
                            SynaptikNodeOutputDTypeCode(node_output_dtypes, i)
                    );
                    if (scalarTensor == nil) return NULL;
                    outTensor = [graph additionWithPrimaryTensor:input0 secondaryTensor:scalarTensor name:@"add_scalar"];
                    break;
                }
                case 41:
                    if (input1 == nil) return NULL;
                    outTensor = [graph greaterThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"gt"];
                    break;
                case 42:
                    if (input1 == nil) return NULL;
                    outTensor = [graph greaterThanOrEqualToWithPrimaryTensor:input0 secondaryTensor:input1 name:@"ge"];
                    break;
                case 43:
                    if (input1 == nil) return NULL;
                    outTensor = [graph lessThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"lt"];
                    break;
                case 44:
                    if (input1 == nil) return NULL;
                    outTensor = [graph lessThanOrEqualToWithPrimaryTensor:input0 secondaryTensor:input1 name:@"le"];
                    break;
                case 45:
                    if (input1 == nil) return NULL;
                    outTensor = [graph equalWithPrimaryTensor:input0 secondaryTensor:input1 name:@"eq"];
                    break;
                case 46:
                    if (input1 == nil) return NULL;
                    outTensor = [graph notEqualWithPrimaryTensor:input0 secondaryTensor:input1 name:@"ne"];
                    break;
                case 47:
                    if (input1 == nil) return NULL;
                    outTensor = [graph logicalANDWithPrimaryTensor:input0 secondaryTensor:input1 name:@"logical_and"];
                    break;
                case 48:
                    if (input1 == nil) return NULL;
                    outTensor = [graph logicalORWithPrimaryTensor:input0 secondaryTensor:input1 name:@"logical_or"];
                    break;
                case 49:
                    outTensor = [graph notWithTensor:input0 name:@"logical_not"];
                    break;
                case 24:
                    if (input1 == nil || input2 == nil) return NULL;
                    outTensor = [graph selectWithPredicateTensor:input0 truePredicateTensor:input1 falsePredicateTensor:input2 name:@"where"];
                    break;
                case 25: {
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    outTensor = [graph softMaxWithTensor:input0 axis:axis name:@"softmax"];
                    break;
                }
                case 36:
                case 37:
                case 38:
                case 39:
                case 50:
                case 51: {
                    int32_t axis = SynaptikDecodeReductionAxis(node_scalar_values, i);
                    NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                            i,
                            output_ranks,
                            output_dim0,
                            output_dim1,
                            output_dim2,
                            output_dim3
                    );
                    if (shape == nil) return NULL;
                    if (axis == -1) {
                        outTensor = SynaptikReductionAllToOutputShape(graph, input0, node_types[i], shape, @"reduction_all");
                    } else {
                        NSNumber *axisNumber = @(axis);
                        switch (node_types[i]) {
                            case 36:
                                outTensor = [graph reductionSumWithTensor:input0 axis:axis name:@"sum"];
                                break;
                            case 37:
                                outTensor = [graph meanOfTensor:input0 axes:@[axisNumber] name:@"mean"];
                                break;
                            case 38:
                                outTensor = [graph reductionMinimumWithTensor:input0 axis:axis name:@"reduce_min"];
                                break;
                            case 39:
                                outTensor = [graph reductionMaximumWithTensor:input0 axis:axis name:@"reduce_max"];
                                break;
                            case 50:
                                outTensor = [graph reductionAndWithTensor:input0 axis:axis name:@"reduce_all"];
                                break;
                            case 51:
                                outTensor = [graph reductionOrWithTensor:input0 axis:axis name:@"reduce_any"];
                                break;
                            default:
                                outTensor = nil;
                                break;
                        }
                        outTensor = SynaptikReshapeToNodeOutput(
                                graph,
                                outTensor,
                                i,
                                output_ranks,
                                output_dim0,
                                output_dim1,
                                output_dim2,
                                output_dim3,
                                @"reduction_output_shape"
                        );
                    }
                    break;
                }
                case 52: {
                    if (input1 == nil) return NULL;
                    if (@available(macOS 12.3, iOS 15.4, tvOS 15.4, *)) {
                        int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                        NSUInteger valueRank = input0.shape == nil ? 0 : input0.shape.count;
                        NSUInteger indexRank = input1.shape == nil ? 0 : input1.shape.count;
                        if (valueRank < 1 || valueRank > 4 || axis < 0 || axis >= (int32_t) valueRank) {
                            return NULL;
                        }
                        if (indexRank + 1 == valueRank) {
                            MPSGraphTensor *expandedIndices = [graph expandDimsOfTensor:input1 axis:axis name:@"gather_indices_expand"];
                            if (expandedIndices == nil) return NULL;
                            MPSGraphTensor *gathered = [graph gatherAlongAxis:(NSInteger) axis
                                                            withUpdatesTensor:input0
                                                                indicesTensor:expandedIndices
                                                                         name:@"gather"];
                            outTensor = gathered == nil ? nil : [graph squeezeTensor:gathered axis:axis name:@"gather_output_squeeze"];
                        } else if (indexRank == valueRank) {
                            outTensor = [graph gatherAlongAxis:(NSInteger) axis
                                             withUpdatesTensor:input0
                                                 indicesTensor:input1
                                                          name:@"gather_rank1"];
                        } else {
                            return NULL;
                        }
                    } else {
                        return NULL;
                    }
                    break;
                }
                case 53: {
                    if (input1 == nil) return NULL;
                    if (@available(macOS 12.3, iOS 15.4, tvOS 15.4, *)) {
                        int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                        NSUInteger valueRank = input0.shape == nil ? 0 : input0.shape.count;
                        if (valueRank < 1 || valueRank > 4 || axis < 0 || axis >= (int32_t) valueRank) {
                            return NULL;
                        }
                        outTensor = [graph gatherAlongAxis:(NSInteger) axis
                                         withUpdatesTensor:input0
                                             indicesTensor:input1
                                                      name:@"take_along_axis"];
                    } else {
                        return NULL;
                    }
                    break;
                }
                case 63: {
                    if (input1 == nil || input2 == nil) return NULL;
                    if (@available(macOS 12.3, iOS 15.4, tvOS 15.4, *)) {
                        int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                        NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                                i,
                                output_ranks,
                                output_dim0,
                                output_dim1,
                                output_dim2,
                                output_dim3
                        );
                        if (shape == nil) return NULL;
                        NSUInteger outputRank = shape.count;
                        if (outputRank < 1 || outputRank > 4 || axis < 0 || axis >= (int32_t) outputRank) {
                            return NULL;
                        }
                        MPSGraphTensor *scatterIndices = input1;
                        MPSGraphTensor *scatterUpdates = input2;
                        NSUInteger indexRank = input1.shape == nil ? 0 : input1.shape.count;
                        NSUInteger updateRank = input2.shape == nil ? 0 : input2.shape.count;
                        if (indexRank + 1 == outputRank && updateRank + 1 == outputRank) {
                            scatterIndices = [graph expandDimsOfTensor:input1 axis:axis name:@"scatter_add_indices_expand"];
                            scatterUpdates = [graph expandDimsOfTensor:input2 axis:axis name:@"scatter_add_updates_expand"];
                            if (scatterIndices == nil || scatterUpdates == nil) return NULL;
                        } else if (indexRank != outputRank || updateRank != outputRank) {
                            return NULL;
                        }
                        outTensor = [graph scatterAlongAxis:(NSInteger) axis
                                             withDataTensor:input0
                                              updatesTensor:scatterUpdates
                                              indicesTensor:scatterIndices
                                                       mode:MPSGraphScatterModeAdd
                                                       name:@"scatter_add"];
                    } else {
                        return NULL;
                    }
                    break;
                }
                case 64:
                case 65: {
                    if (input1 == nil) return NULL;
                    if (@available(macOS 12.3, iOS 15.4, tvOS 15.4, *)) {
                        int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                        NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                                i,
                                output_ranks,
                                output_dim0,
                                output_dim1,
                                output_dim2,
                                output_dim3
                        );
                        if (shape == nil) return NULL;
                        NSUInteger outputRank = shape.count;
                        if (outputRank < 1 || outputRank > 4 || axis < 0 || axis >= (int32_t) outputRank) {
                            return NULL;
                        }
                        MPSGraphTensor *scatterIndices = input0;
                        MPSGraphTensor *scatterUpdates = input1;
                        NSUInteger indexRank = input0.shape == nil ? 0 : input0.shape.count;
                        NSUInteger updateRank = input1.shape == nil ? 0 : input1.shape.count;
                        if (indexRank + 1 == outputRank && updateRank + 1 == outputRank) {
                            scatterIndices = [graph expandDimsOfTensor:input0 axis:axis name:@"index_grad_indices_expand"];
                            scatterUpdates = [graph expandDimsOfTensor:input1 axis:axis name:@"index_grad_updates_expand"];
                            if (scatterIndices == nil || scatterUpdates == nil) return NULL;
                        } else if (indexRank != outputRank || updateRank != outputRank) {
                            return NULL;
                        }
                        outTensor = [graph scatterAlongAxis:(NSInteger) axis
                                          withUpdatesTensor:scatterUpdates
                                              indicesTensor:scatterIndices
                                                      shape:shape
                                                       mode:MPSGraphScatterModeAdd
                                                       name:(node_types[i] == 64 ? @"gather_grad" : @"take_along_axis_grad")];
                    } else {
                        return NULL;
                    }
                    break;
                }
                case 54: {
                    if (input1 == nil) return NULL;
                    int32_t strideH = 1;
                    int32_t strideW = 1;
                    int32_t padH = 0;
                    int32_t padW = 0;
                    SynaptikDecodeConv2DMode(node_scalar_values, i, &strideH, &strideW, &padH, &padW);
                    if (strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0) {
                        return NULL;
                    }
                    if (input0.shape == nil || input0.shape.count != 4 || input1.shape == nil || input1.shape.count != 4) {
                        return NULL;
                    }
                    MPSGraphConvolution2DOpDescriptor *descriptor =
                            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger) strideW
                                                                              strideInY:(NSUInteger) strideH
                                                                        dilationRateInX:1
                                                                        dilationRateInY:1
                                                                                  groups:1
                                                                             paddingLeft:(NSUInteger) padW
                                                                            paddingRight:(NSUInteger) padW
                                                                             paddingTop:(NSUInteger) padH
                                                                          paddingBottom:(NSUInteger) padH
                                                                            paddingStyle:MPSGraphPaddingStyleExplicit
                                                                              dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                           weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
                    if (descriptor == nil) return NULL;
                    outTensor = [graph convolution2DWithSourceTensor:input0 weightsTensor:input1 descriptor:descriptor name:@"conv2d"];
                    if (outTensor == nil) return NULL;
                    if (input2 != nil) {
                        NSMutableArray<NSNumber *> *biasShape = [NSMutableArray arrayWithObjects:
                                @(1),
                                @(output_dim1 == NULL ? 1 : output_dim1[i]),
                                @(1),
                                @(1),
                                nil];
                        MPSGraphTensor *bias = [graph reshapeTensor:input2 withShape:biasShape name:@"conv2d_bias_reshape"];
                        if (bias == nil) return NULL;
                        outTensor = [graph additionWithPrimaryTensor:outTensor secondaryTensor:bias name:@"conv2d_bias_add"];
                    }
                    break;
                }
                case 66:
                case 67: {
                    if (input1 == nil) return NULL;
                    int32_t strideH = 1;
                    int32_t strideW = 1;
                    int32_t padH = 0;
                    int32_t padW = 0;
                    SynaptikDecodeConv2DMode(node_scalar_values, i, &strideH, &strideW, &padH, &padW);
                    if (strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0) {
                        return NULL;
                    }
                    if (input0.shape == nil || input0.shape.count != 4 || input1.shape == nil || input1.shape.count != 4) {
                        return NULL;
                    }
                    NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                            i,
                            output_ranks,
                            output_dim0,
                            output_dim1,
                            output_dim2,
                            output_dim3
                    );
                    if (shape == nil || shape.count != 4) return NULL;
                    MPSGraphConvolution2DOpDescriptor *descriptor =
                            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger) strideW
                                                                              strideInY:(NSUInteger) strideH
                                                                        dilationRateInX:1
                                                                        dilationRateInY:1
                                                                                  groups:1
                                                                             paddingLeft:(NSUInteger) padW
                                                                            paddingRight:(NSUInteger) padW
                                                                             paddingTop:(NSUInteger) padH
                                                                          paddingBottom:(NSUInteger) padH
                                                                            paddingStyle:MPSGraphPaddingStyleExplicit
                                                                              dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                           weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
                    if (descriptor == nil) return NULL;
                    if (node_types[i] == 66) {
                        outTensor = [graph convolution2DDataGradientWithIncomingGradientTensor:input1
                                                                                 weightsTensor:input0
                                                                                   outputShape:shape
                                                                  forwardConvolutionDescriptor:descriptor
                                                                                          name:@"conv2d_backward_input"];
                    } else {
                        outTensor = [graph convolution2DWeightsGradientWithIncomingGradientTensor:input1
                                                                                     sourceTensor:input0
                                                                                      outputShape:shape
                                                                     forwardConvolutionDescriptor:descriptor
                                                                                             name:@"conv2d_backward_weight"];
                    }
                    break;
                }
                case 55:
                case 56: {
                    int32_t kernelH = 1;
                    int32_t kernelW = 1;
                    int32_t strideH = 1;
                    int32_t strideW = 1;
                    int32_t padH = 0;
                    int32_t padW = 0;
                    BOOL countIncludePad = NO;
                    SynaptikDecodePool2DMode(node_scalar_values, i, &kernelH, &kernelW, &strideH, &strideW, &padH, &padW, &countIncludePad);
                    if (kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0) {
                        return NULL;
                    }
                    if (node_types[i] == 56 && countIncludePad) {
                        return NULL;
                    }
                    if (input0.shape == nil || input0.shape.count != 4) {
                        return NULL;
                    }
                    MPSGraphPooling2DOpDescriptor *descriptor =
                            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger) kernelW
                                                                         kernelHeight:(NSUInteger) kernelH
                                                                            strideInX:(NSUInteger) strideW
                                                                            strideInY:(NSUInteger) strideH
                                                                         paddingStyle:MPSGraphPaddingStyleExplicit
                                                                            dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
                    if (descriptor == nil) return NULL;
                    descriptor.paddingLeft = (NSUInteger) padW;
                    descriptor.paddingRight = (NSUInteger) padW;
                    descriptor.paddingTop = (NSUInteger) padH;
                    descriptor.paddingBottom = (NSUInteger) padH;
                    if (node_types[i] == 56) {
                        descriptor.includeZeroPadToAverage = NO;
                        outTensor = [graph avgPooling2DWithSourceTensor:input0 descriptor:descriptor name:@"avg_pool2d"];
                    } else {
                        outTensor = [graph maxPooling2DWithSourceTensor:input0 descriptor:descriptor name:@"max_pool2d"];
                    }
                    break;
                }
                case 68: {
                    int32_t kernelH = 1;
                    int32_t kernelW = 1;
                    int32_t strideH = 1;
                    int32_t strideW = 1;
                    int32_t padH = 0;
                    int32_t padW = 0;
                    BOOL countIncludePad = NO;
                    SynaptikDecodePool2DMode(node_scalar_values, i, &kernelH, &kernelW, &strideH, &strideW, &padH, &padW, &countIncludePad);
                    if (kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0 || countIncludePad) {
                        return NULL;
                    }
                    if (input0.shape == nil || input0.shape.count != 4) {
                        return NULL;
                    }
                    NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                            i,
                            output_ranks,
                            output_dim0,
                            output_dim1,
                            output_dim2,
                            output_dim3
                    );
                    if (shape == nil || shape.count != 4) return NULL;
                    MPSGraphPooling2DOpDescriptor *descriptor =
                            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger) kernelW
                                                                         kernelHeight:(NSUInteger) kernelH
                                                                            strideInX:(NSUInteger) strideW
                                                                            strideInY:(NSUInteger) strideH
                                                                         paddingStyle:MPSGraphPaddingStyleExplicit
                                                                            dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
                    if (descriptor == nil) return NULL;
                    descriptor.paddingLeft = (NSUInteger) padW;
                    descriptor.paddingRight = (NSUInteger) padW;
                    descriptor.paddingTop = (NSUInteger) padH;
                    descriptor.paddingBottom = (NSUInteger) padH;
                    descriptor.includeZeroPadToAverage = NO;
                    MPSGraphTensor *sourceShape = [graph constantWithScalar:0.0
                                                                      shape:shape
                                                                   dataType:SynaptikNodeOutputDTypeCode(node_output_dtypes, i) == 3 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32];
                    if (sourceShape == nil) return NULL;
                    outTensor = [graph avgPooling2DGradientWithGradientTensor:input0
                                                                 sourceTensor:sourceShape
                                                                   descriptor:descriptor
                                                                         name:@"avg_pool2d_backward_input"];
                    break;
                }
                case 69: {
                    if (input1 == nil) return NULL;
                    int32_t kernelH = 1;
                    int32_t kernelW = 1;
                    int32_t strideH = 1;
                    int32_t strideW = 1;
                    int32_t padH = 0;
                    int32_t padW = 0;
                    BOOL countIncludePad = NO;
                    SynaptikDecodePool2DMode(node_scalar_values, i, &kernelH, &kernelW, &strideH, &strideW, &padH, &padW, &countIncludePad);
                    if (kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0) {
                        return NULL;
                    }
                    if (input0.shape == nil || input0.shape.count != 4 || input1.shape == nil || input1.shape.count != 4) {
                        return NULL;
                    }
                    MPSGraphPooling2DOpDescriptor *descriptor =
                            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger) kernelW
                                                                         kernelHeight:(NSUInteger) kernelH
                                                                            strideInX:(NSUInteger) strideW
                                                                            strideInY:(NSUInteger) strideH
                                                                         paddingStyle:MPSGraphPaddingStyleExplicit
                                                                            dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
                    if (descriptor == nil) return NULL;
                    descriptor.paddingLeft = (NSUInteger) padW;
                    descriptor.paddingRight = (NSUInteger) padW;
                    descriptor.paddingTop = (NSUInteger) padH;
                    descriptor.paddingBottom = (NSUInteger) padH;
                    outTensor = [graph maxPooling2DGradientWithGradientTensor:input0
                                                                 sourceTensor:input1
                                                                   descriptor:descriptor
                                                                         name:@"max_pool2d_backward_input"];
                    break;
                }
                case 70: {
                    if (input1 == nil) return NULL;
                    int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, i);
                    int32_t axis = 0;
                    int32_t reduction = 2;
                    BOOL hasIgnoreIndex = NO;
                    int32_t ignoreIndex = 0;
                    SynaptikDecodeCrossEntropyLossMode(node_scalar_values, i, &axis, &reduction, &hasIgnoreIndex, &ignoreIndex);
                    if (input0.shape == nil || input1.shape == nil || axis < 0 || axis >= (int32_t) input0.shape.count) {
                        return NULL;
                    }
                    MPSGraphTensor *effectiveIndices = input1;
                    MPSGraphTensor *validMask = nil;
                    if (hasIgnoreIndex) {
                        MPSGraphTensor *ignoreTensor = [graph constantWithScalar:(double) ignoreIndex dataType:MPSDataTypeInt32];
                        MPSGraphTensor *zeroIndex = [graph constantWithScalar:0.0 dataType:MPSDataTypeInt32];
                        if (ignoreTensor == nil || zeroIndex == nil) return NULL;
                        validMask = [graph notEqualWithPrimaryTensor:input1 secondaryTensor:ignoreTensor name:@"ce_indices_valid_mask"];
                        if (validMask == nil) return NULL;
                        effectiveIndices = [graph selectWithPredicateTensor:validMask
                                                         truePredicateTensor:input1
                                                        falsePredicateTensor:zeroIndex
                                                                        name:@"ce_indices_safe"];
                        if (effectiveIndices == nil) return NULL;
                    }
                    MPSGraphTensor *probabilities = [graph softMaxWithTensor:input0 axis:axis name:@"ce_indices_softmax"];
                    MPSGraphTensor *logProbabilities = probabilities == nil ? nil : [graph logarithmWithTensor:probabilities name:@"ce_indices_log"];
                    MPSGraphTensor *targetLogProbability = SynaptikGatherReducedAlongAxis(
                            graph,
                            logProbabilities,
                            effectiveIndices,
                            axis,
                            @"ce_indices_gather"
                    );
                    if (targetLogProbability == nil) return NULL;
                    MPSGraphTensor *perSampleLoss = [graph negativeWithTensor:targetLogProbability name:@"ce_indices_per_sample"];
                    if (perSampleLoss == nil) return NULL;
                    if (hasIgnoreIndex) {
                        MPSGraphTensor *zeroLoss = SynaptikScalarConstant(graph, 0.0, outputDType);
                        if (zeroLoss == nil) return NULL;
                        perSampleLoss = [graph selectWithPredicateTensor:validMask
                                                     truePredicateTensor:perSampleLoss
                                                    falsePredicateTensor:zeroLoss
                                                                    name:@"ce_indices_masked_loss"];
                        if (perSampleLoss == nil) return NULL;
                    }
                    if (reduction == 0) {
                        outTensor = SynaptikReshapeToNodeOutput(
                                graph,
                                perSampleLoss,
                                i,
                                output_ranks,
                                output_dim0,
                                output_dim1,
                                output_dim2,
                                output_dim3,
                                @"ce_indices_none_shape"
                        );
                    } else {
                        MPSGraphTensor *totalLoss = SynaptikReductionSumAllToShapeOne(graph, perSampleLoss, @"ce_indices_sum_all");
                        if (totalLoss == nil) return NULL;
                        if (reduction == 1) {
                            outTensor = totalLoss;
                        } else if (reduction == 2) {
                            MPSGraphTensor *denominator = nil;
                            if (hasIgnoreIndex) {
                                MPSDataType valueDataType = MPSDataTypeInvalid;
                                if (!SynaptikMpsDataTypeForCode(outputDType, &valueDataType)) return NULL;
                                MPSGraphTensor *validFloat = [graph castTensor:validMask toType:valueDataType name:@"ce_indices_valid_value"];
                                denominator = SynaptikReductionSumAllToShapeOne(graph, validFloat, @"ce_indices_valid_count");
                                MPSGraphTensor *one = SynaptikScalarConstant(graph, 1.0, outputDType);
                                if (denominator == nil || one == nil) return NULL;
                                denominator = [graph maximumWithPrimaryTensor:denominator secondaryTensor:one name:@"ce_indices_valid_count_clamped"];
                            } else {
                                NSUInteger sampleCount = 1;
                                for (NSNumber *dim in input1.shape) {
                                    sampleCount *= dim.unsignedIntegerValue;
                                }
                                denominator = SynaptikScalarConstant(graph, (double) sampleCount, outputDType);
                            }
                            if (denominator == nil) return NULL;
                            outTensor = [graph divisionWithPrimaryTensor:totalLoss secondaryTensor:denominator name:@"ce_indices_mean"];
                        } else {
                            return NULL;
                        }
                    }
                    break;
                }
                case 71: {
                    if (input1 == nil || input2 == nil) return NULL;
                    int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, i);
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    if (input0.shape == nil || input1.shape == nil || input2.shape == nil || axis < 0 || axis >= (int32_t) input0.shape.count) {
                        return NULL;
                    }
                    NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                            i,
                            output_ranks,
                            output_dim0,
                            output_dim1,
                            output_dim2,
                            output_dim3
                    );
                    if (shape == nil) return NULL;
                    MPSGraphTensor *probabilities = [graph softMaxWithTensor:input0 axis:axis name:@"ce_indices_grad_softmax"];
                    MPSGraphTensor *expandedScale = [graph expandDimsOfTensor:input2 axis:axis name:@"ce_indices_grad_scale_expand"];
                    if (probabilities == nil || expandedScale == nil) return NULL;
                    MPSGraphTensor *scaledProbabilities = [graph multiplicationWithPrimaryTensor:probabilities
                                                                                 secondaryTensor:expandedScale
                                                                                          name:@"ce_indices_grad_scaled_probs"];
                    if (scaledProbabilities == nil) return NULL;
                    MPSGraphTensor *scatterIndices = input1;
                    MPSGraphTensor *scatterUpdates = input2;
                    NSUInteger outputRank = shape.count;
                    NSUInteger indexRank = input1.shape.count;
                    NSUInteger updateRank = input2.shape.count;
                    if (indexRank + 1 == outputRank && updateRank + 1 == outputRank) {
                        scatterIndices = [graph expandDimsOfTensor:input1 axis:axis name:@"ce_indices_grad_indices_expand"];
                        scatterUpdates = [graph expandDimsOfTensor:input2 axis:axis name:@"ce_indices_grad_updates_expand"];
                        if (scatterIndices == nil || scatterUpdates == nil) return NULL;
                    } else if (indexRank != outputRank || updateRank != outputRank) {
                        return NULL;
                    }
                    MPSGraphTensor *zeroBase = [graph constantWithScalar:0.0
                                                                   shape:shape
                                                                dataType:(outputDType == 3 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32)];
                    if (zeroBase == nil) return NULL;
                    MPSGraphTensor *targetScale = [graph scatterAlongAxis:(NSInteger) axis
                                                           withDataTensor:zeroBase
                                                            updatesTensor:scatterUpdates
                                                            indicesTensor:scatterIndices
                                                                     mode:MPSGraphScatterModeAdd
                                                                     name:@"ce_indices_grad_target_scatter"];
                    if (targetScale == nil) return NULL;
                    outTensor = [graph subtractionWithPrimaryTensor:scaledProbabilities
                                                    secondaryTensor:targetScale
                                                               name:@"ce_indices_grad_out"];
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
                    int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, i);
                    MPSDataType valueDataType = MPSDataTypeInvalid;
                    if (!SynaptikMpsDataTypeForCode(outputDType, &valueDataType)) return NULL;
                    int32_t axis = SynaptikDecodeIntScalar(node_scalar_values, i);
                    MPSGraphTensor *mask = [graph equalWithPrimaryTensor:input0 secondaryTensor:input1 name:@"reduce_minmax_grad_mask"];
                    if (mask == nil) return NULL;
                    MPSGraphTensor *maskFloat = [graph castTensor:mask toType:valueDataType name:@"reduce_minmax_grad_mask_value"];
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
                    int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, i);
                    BOOL forFirstInput = SynaptikDecodeIntScalar(node_scalar_values, i) != 0;
                    MPSGraphTensor *strictFirst = node_types[i] == 31
                            ? [graph lessThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"min_grad_predicate"]
                            : [graph greaterThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"max_grad_predicate"];
                    MPSGraphTensor *strictSecond = node_types[i] == 31
                            ? [graph greaterThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"min_grad_second_predicate"]
                            : [graph lessThanWithPrimaryTensor:input0 secondaryTensor:input1 name:@"max_grad_second_predicate"];
                    MPSGraphTensor *equal = [graph equalWithPrimaryTensor:input0 secondaryTensor:input1 name:@"minmax_grad_equal"];
                    if (strictFirst == nil || strictSecond == nil || equal == nil) return NULL;
                    MPSGraphTensor *zero = SynaptikScalarConstant(graph, 0.0, outputDType);
                    MPSGraphTensor *half = SynaptikScalarConstant(graph, 0.5, outputDType);
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
                    int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, i);
                    NSInteger rank = input0.shape.count;
                    if (rank < 2) return NULL;
                    int32_t axis = (int32_t) rank - 1;

                    MPSGraphTensor *keyT = SynaptikTransposeLastTwoAxes(graph, input1, @"sdpa_backward_key_t");
                    MPSGraphTensor *scores = keyT == nil ? nil : [graph matrixMultiplicationWithPrimaryTensor:input0 secondaryTensor:keyT name:@"sdpa_backward_scores"];
                    if (scores == nil) return NULL;
                    if (scale != 1.0f) {
                        MPSGraphTensor *scaleTensor = SynaptikScalarConstant(graph, (double) scale, outputDType);
                        if (scaleTensor == nil) return NULL;
                        scores = [graph multiplicationWithPrimaryTensor:scores secondaryTensor:scaleTensor name:@"sdpa_backward_scaled_scores"];
                        if (scores == nil) return NULL;
                    }
                    if (input4 != nil) {
                        MPSGraphTensor *maskFill = SynaptikScalarConstant(graph, -1.0e9, outputDType);
                        if (maskFill == nil) return NULL;
                        scores = [graph selectWithPredicateTensor:input4 truePredicateTensor:scores falsePredicateTensor:maskFill name:@"sdpa_backward_masked_scores"];
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
                    if (input4 != nil) {
                        MPSGraphTensor *zero = SynaptikScalarConstant(graph, 0.0, outputDType);
                        if (zero == nil) return NULL;
                        dScores = [graph selectWithPredicateTensor:input4 truePredicateTensor:dScores falsePredicateTensor:zero name:@"sdpa_backward_masked_dscores"];
                        if (dScores == nil) return NULL;
                    }
                    if (scale != 1.0f) {
                        MPSGraphTensor *scaleTensor = SynaptikScalarConstant(graph, (double) scale, outputDType);
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
                    int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, i);
                    if (input1 == nil || input2 == nil) return NULL;
                    MPSGraphTensor *keyT = SynaptikTransposeLastTwoAxes(graph, input1, @"sdpa_key_t");
                    MPSGraphTensor *scores = keyT == nil ? nil : [graph matrixMultiplicationWithPrimaryTensor:input0 secondaryTensor:keyT name:@"sdpa_scores"];
                    if (scores == nil) return NULL;
                    if (scale != 1.0f) {
                        MPSGraphTensor *scaleTensor = SynaptikScalarConstant(graph, (double) scale, outputDType);
                        if (scaleTensor == nil) return NULL;
                        scores = [graph multiplicationWithPrimaryTensor:scores secondaryTensor:scaleTensor name:@"sdpa_scaled_scores"];
                        if (scores == nil) return NULL;
                    }
                    if (input3 != nil) {
                        MPSGraphTensor *maskFill = SynaptikScalarConstant(graph, -1.0e9, outputDType);
                        if (maskFill == nil) return NULL;
                        scores = [graph selectWithPredicateTensor:input3 truePredicateTensor:scores falsePredicateTensor:maskFill name:@"sdpa_masked_scores"];
                        if (scores == nil) return NULL;
                    }
                    MPSGraphTensor *weights = [graph softMaxWithTensor:scores axis:-1 name:@"sdpa_weights"];
                    outTensor = weights == nil ? nil : [graph matrixMultiplicationWithPrimaryTensor:weights secondaryTensor:input2 name:@"sdpa_out"];
                    break;
                }
                case 62: {
                    float scale = node_scalar_values == NULL ? 1.0f : node_scalar_values[i];
                    int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, i);
                    if (input1 == nil) return NULL;
                    MPSGraphTensor *keyT = SynaptikTransposeLastTwoAxes(graph, input1, @"sdpa_weights_key_t");
                    MPSGraphTensor *scores = keyT == nil ? nil : [graph matrixMultiplicationWithPrimaryTensor:input0 secondaryTensor:keyT name:@"sdpa_weights_scores"];
                    if (scores == nil) return NULL;
                    if (scale != 1.0f) {
                        MPSGraphTensor *scaleTensor = SynaptikScalarConstant(graph, (double) scale, outputDType);
                        if (scaleTensor == nil) return NULL;
                        scores = [graph multiplicationWithPrimaryTensor:scores secondaryTensor:scaleTensor name:@"sdpa_weights_scaled_scores"];
                        if (scores == nil) return NULL;
                    }
                    if (input2 != nil) {
                        MPSGraphTensor *maskFill = SynaptikScalarConstant(graph, -1.0e9, outputDType);
                        if (maskFill == nil) return NULL;
                        scores = [graph selectWithPredicateTensor:input2 truePredicateTensor:scores falsePredicateTensor:maskFill name:@"sdpa_weights_masked_scores"];
                        if (scores == nil) return NULL;
                    }
                    outTensor = [graph softMaxWithTensor:scores axis:-1 name:@"sdpa_weights_out"];
                    break;
                }
                case 18: {
                    NSMutableArray<NSNumber *> *shape = SynaptikOutputShapeForNode(
                            i,
                            output_ranks,
                            output_dim0,
                            output_dim1,
                            output_dim2,
                            output_dim3
                    );
                    if (shape == nil) return NULL;
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
            int32_t outputDType = SynaptikNodeOutputDTypeCode(node_output_dtypes, output_node_index);
            MPSDataType outputDataType = MPSDataTypeInvalid;
            if (!SynaptikMpsDataTypeForCode(outputDType, &outputDataType)) {
                return NULL;
            }
            outputTensor = [graph castTensor:outputTensor toType:outputDataType name:@"output_dtype"];
            if (outputTensor == nil) {
                return NULL;
            }
            [targetTensors addObject:outputTensor];
            [outputRanksBoxed addObject:@(rank)];
            [outputDTypesBoxed addObject:@(outputDType)];
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
        const int32_t *input4_kinds,
        const int32_t *input4_indices,
        const float *node_scalar_values,
        const int32_t *output_ranks,
        const int32_t *output_dim0,
        const int32_t *output_dim1,
        const int32_t *output_dim2,
        const int32_t *output_dim3,
        int32_t output_node_count,
        const int32_t *output_node_indices
) {
    return SynaptikCompilePartition(
            context,
            external_input_count,
            external_input_ranks,
            external_input_dtypes,
            external_input_dim0,
            external_input_dim1,
            external_input_dim2,
            external_input_dim3,
            post_op_count,
            node_types,
            input0_kinds,
            input0_indices,
            input1_kinds,
            input1_indices,
            input2_kinds,
            input2_indices,
            input3_kinds,
            input3_indices,
            input4_kinds,
            input4_indices,
            node_scalar_values,
            output_ranks,
            output_dim0,
            output_dim1,
            output_dim2,
            output_dim3,
            NULL,
            output_node_count,
            output_node_indices
    );
}

void *synaptik_apple_mps_compile_partition_dtype_v3(
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
        const int32_t *input4_kinds,
        const int32_t *input4_indices,
        const float *node_scalar_values,
        const int32_t *output_ranks,
        const int32_t *output_dim0,
        const int32_t *output_dim1,
        const int32_t *output_dim2,
        const int32_t *output_dim3,
        const int32_t *node_output_dtypes,
        int32_t output_node_count,
        const int32_t *output_node_indices
) {
    return SynaptikCompilePartition(
            context,
            external_input_count,
            external_input_ranks,
            external_input_dtypes,
            external_input_dim0,
            external_input_dim1,
            external_input_dim2,
            external_input_dim3,
            post_op_count,
            node_types,
            input0_kinds,
            input0_indices,
            input1_kinds,
            input1_indices,
            input2_kinds,
            input2_indices,
            input3_kinds,
            input3_indices,
            input4_kinds,
            input4_indices,
            node_scalar_values,
            output_ranks,
            output_dim0,
            output_dim1,
            output_dim2,
            output_dim3,
            node_output_dtypes,
            output_node_count,
            output_node_indices
    );
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
            MPSDataType dataType = MPSDataTypeInvalid;
            if (!SynaptikMpsDataTypeForCode(dtypeCode, &dataType)) {
                return 6;
            }
            NSUInteger elementCount = dim0;
            if (rank >= 2) elementCount *= dim1;
            if (rank >= 3) elementCount *= dim2;
            if (rank >= 4) elementCount *= dim3;
            NSUInteger byteSize = SynaptikByteSizeForDTypeCode(dtypeCode);
            if (byteSize == 0) {
                return 6;
            }
            NSUInteger bytes = elementCount * byteSize;
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

static int32_t SynaptikExecutePartitionBuffers(
        void *context,
        void *executable,
        const void * const *external_input_buffers,
        int32_t external_input_count,
        void * const *output_buffers,
        int32_t output_count,
        int64_t *native_device_copy_ns,
        BOOL copy_mpsgraph_results_to_outputs
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
            MPSDataType dataType = MPSDataTypeInvalid;
            if (!SynaptikMpsDataTypeForCode(dtypeCode, &dataType)) {
                return 5;
            }
            NSUInteger elementCount = SynaptikElementCountFromDims(rank, dim0, dim1, dim2, dim3);
            NSUInteger byteSize = SynaptikByteSizeForDTypeCode(dtypeCode);
            if (byteSize == 0) {
                return 5;
            }
            NSUInteger bytes = elementCount * byteSize;
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
            MPSDataType dataType = MPSDataTypeInvalid;
            if (!SynaptikMpsDataTypeForCode(dtypeCode, &dataType)) {
                return 8;
            }
            NSUInteger elementCount = (NSUInteger) executableBox.outputElementCounts[(NSUInteger) i].unsignedLongLongValue;
            NSUInteger byteSize = SynaptikByteSizeForDTypeCode(dtypeCode);
            if (byteSize == 0) {
                return 8;
            }
            NSUInteger bytes = elementCount * byteSize;
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
        if (copy_mpsgraph_results_to_outputs) {
            for (int32_t i = 0; i < output_count; i++) {
                SynaptikAppleMpsBufferBox *box = SynaptikUnboxBuffer(output_buffers[i]);
                MPSGraphTensorData *resultData = results[(NSUInteger) i];
                MPSNDArray *resultArray = resultData.mpsndarray;
                if (box == nil || box.buffer == nil || resultArray == nil) {
                    return 11;
                }
                int32_t dtypeCode = executableBox.outputDTypes[(NSUInteger) i].intValue;
                NSUInteger elementCount = (NSUInteger) executableBox.outputElementCounts[(NSUInteger) i].unsignedLongLongValue;
                NSUInteger byteSize = SynaptikByteSizeForDTypeCode(dtypeCode);
                if (byteSize == 0) {
                    return 12;
                }
                NSUInteger bytes = elementCount * byteSize;
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
        }
        if (native_device_copy_ns != NULL) {
            *native_device_copy_ns = copyNs;
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
    return SynaptikExecutePartitionBuffers(
            context,
            executable,
            external_input_buffers,
            external_input_count,
            output_buffers,
            output_count,
            native_device_copy_ns,
            YES
    );
}

int32_t synaptik_apple_mps_probe_output_buffer_write_f32_buffers(
        void *context,
        void *executable,
        const void * const *external_input_buffers,
        int32_t external_input_count,
        void * const *output_buffers,
        int32_t output_count
) {
    return SynaptikExecutePartitionBuffers(
            context,
            executable,
            external_input_buffers,
            external_input_count,
            output_buffers,
            output_count,
            NULL,
            NO
    );
}

int synaptik_apple_mps_layout_contiguous_buffer(
        void *context,
        void *source_buffer,
        void *destination_buffer,
        int32_t dtype_code,
        int32_t rank,
        int64_t *shape,
        int64_t *strides,
        int64_t storage_offset,
        int64_t logical_element_count,
        int64_t source_physical_byte_span,
        int64_t destination_byte_length
) {
    if (context == NULL || source_buffer == NULL || destination_buffer == NULL) {
        return 1;
    }
    if (rank <= 0 || rank > 8 || shape == NULL || strides == NULL) {
        return 2;
    }
    if (storage_offset < 0 || logical_element_count < 0 || source_physical_byte_span < 0 || destination_byte_length < 0) {
        return 3;
    }
    NSUInteger elementByteSize = SynaptikByteSizeForDTypeCode(dtype_code);
    if (elementByteSize == 0) {
        return 4;
    }
    if (destination_byte_length < logical_element_count * (int64_t) elementByteSize) {
        return 4;
    }
    @autoreleasepool {
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        SynaptikAppleMpsBufferBox *sourceBox = SynaptikUnboxBuffer(source_buffer);
        SynaptikAppleMpsBufferBox *destinationBox = SynaptikUnboxBuffer(destination_buffer);
        if (contextBox == nil || sourceBox == nil || destinationBox == nil
                || sourceBox.buffer == nil || destinationBox.buffer == nil) {
            return 5;
        }
        if (sourceBox.byteLength < (NSUInteger) source_physical_byte_span
                || destinationBox.byteLength < (NSUInteger) destination_byte_length) {
            return 6;
        }
        for (int32_t dim = 0; dim < rank; dim++) {
            if (shape[dim] <= 0 || strides[dim] < 0) {
                return 8;
            }
        }
        if (contextBox.layoutContiguousPipeline == nil) {
            contextBox.layoutContiguousPipeline = SynaptikLayoutContiguousPipeline(contextBox.device);
        }
        if (contextBox.layoutContiguousPipeline == nil) {
            return 10;
        }
        id<MTLBuffer> shapeBuffer = [contextBox.device newBufferWithBytes:shape
                                                                    length:(NSUInteger) rank * sizeof(int64_t)
                                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> strideBuffer = [contextBox.device newBufferWithBytes:strides
                                                                     length:(NSUInteger) rank * sizeof(int64_t)
                                                                    options:MTLResourceStorageModeShared];
        if (shapeBuffer == nil || strideBuffer == nil) {
            return 11;
        }
        id<MTLCommandBuffer> commandBuffer = [contextBox.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            return 12;
        }
        int32_t rankValue = rank;
        int32_t elementSizeValue = (int32_t) elementByteSize;
        int64_t elementCountValue = logical_element_count;
        int64_t storageOffsetValue = storage_offset;
        [encoder setComputePipelineState:contextBox.layoutContiguousPipeline];
        [encoder setBuffer:sourceBox.buffer offset:0 atIndex:0];
        [encoder setBuffer:destinationBox.buffer offset:0 atIndex:1];
        [encoder setBuffer:shapeBuffer offset:0 atIndex:2];
        [encoder setBuffer:strideBuffer offset:0 atIndex:3];
        [encoder setBytes:&elementCountValue length:sizeof(int64_t) atIndex:4];
        [encoder setBytes:&rankValue length:sizeof(int32_t) atIndex:5];
        [encoder setBytes:&storageOffsetValue length:sizeof(int64_t) atIndex:6];
        [encoder setBytes:&elementSizeValue length:sizeof(int32_t) atIndex:7];
        NSUInteger threads = MIN((NSUInteger) contextBox.layoutContiguousPipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) 256);
        if (threads == 0) {
            return 13;
        }
        MTLSize gridSize = MTLSizeMake((NSUInteger) logical_element_count, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threads, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return 14;
        }
        return 0;
    }
}

int synaptik_apple_mps_custom_relu_f32_buffer(
        void *context,
        void *source_buffer,
        void *destination_buffer,
        int64_t logical_element_count,
        int64_t source_byte_length,
        int64_t destination_byte_length
) {
    if (context == NULL || source_buffer == NULL || destination_buffer == NULL) {
        return 1;
    }
    if (logical_element_count < 0 || source_byte_length < 0 || destination_byte_length < 0) {
        return 2;
    }
    int64_t requiredBytes = logical_element_count * (int64_t) sizeof(float);
    if (source_byte_length < requiredBytes || destination_byte_length < requiredBytes) {
        return 3;
    }
    if (logical_element_count == 0) {
        return 0;
    }
    @autoreleasepool {
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        SynaptikAppleMpsBufferBox *sourceBox = SynaptikUnboxBuffer(source_buffer);
        SynaptikAppleMpsBufferBox *destinationBox = SynaptikUnboxBuffer(destination_buffer);
        if (contextBox == nil || sourceBox == nil || destinationBox == nil
                || sourceBox.buffer == nil || destinationBox.buffer == nil) {
            return 4;
        }
        if (sourceBox.byteLength < (NSUInteger) source_byte_length
                || destinationBox.byteLength < (NSUInteger) destination_byte_length) {
            return 5;
        }
        if (contextBox.customReluF32Pipeline == nil) {
            contextBox.customReluF32Pipeline = SynaptikCustomReluF32Pipeline(contextBox.device);
        }
        if (contextBox.customReluF32Pipeline == nil) {
            return 6;
        }
        id<MTLCommandBuffer> commandBuffer = [contextBox.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            return 7;
        }
        int64_t elementCountValue = logical_element_count;
        [encoder setComputePipelineState:contextBox.customReluF32Pipeline];
        [encoder setBuffer:sourceBox.buffer offset:0 atIndex:0];
        [encoder setBuffer:destinationBox.buffer offset:0 atIndex:1];
        [encoder setBytes:&elementCountValue length:sizeof(int64_t) atIndex:2];
        NSUInteger threads = MIN((NSUInteger) contextBox.customReluF32Pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) 256);
        if (threads == 0) {
            return 8;
        }
        MTLSize gridSize = MTLSizeMake((NSUInteger) logical_element_count, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threads, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return 9;
        }
        return 0;
    }
}

int synaptik_apple_mps_optimizer_sgd_f32_buffer(
        void *context,
        void *parameter_buffer,
        void *gradient_buffer,
        void *output_buffer,
        float learning_rate,
        int64_t logical_element_count,
        int64_t parameter_byte_length,
        int64_t gradient_byte_length,
        int64_t output_byte_length
) {
    if (context == NULL || parameter_buffer == NULL || gradient_buffer == NULL || output_buffer == NULL) {
        return 1;
    }
    if (logical_element_count < 0 || parameter_byte_length < 0 || gradient_byte_length < 0 || output_byte_length < 0) {
        return 2;
    }
    int64_t requiredBytes = logical_element_count * (int64_t) sizeof(float);
    if (parameter_byte_length < requiredBytes || gradient_byte_length < requiredBytes || output_byte_length < requiredBytes) {
        return 3;
    }
    if (logical_element_count == 0) {
        return 0;
    }
    @autoreleasepool {
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        SynaptikAppleMpsBufferBox *parameterBox = SynaptikUnboxBuffer(parameter_buffer);
        SynaptikAppleMpsBufferBox *gradientBox = SynaptikUnboxBuffer(gradient_buffer);
        SynaptikAppleMpsBufferBox *outputBox = SynaptikUnboxBuffer(output_buffer);
        if (contextBox == nil || parameterBox == nil || gradientBox == nil || outputBox == nil
                || parameterBox.buffer == nil || gradientBox.buffer == nil || outputBox.buffer == nil) {
            return 4;
        }
        if (parameterBox.byteLength < (NSUInteger) parameter_byte_length
                || gradientBox.byteLength < (NSUInteger) gradient_byte_length
                || outputBox.byteLength < (NSUInteger) output_byte_length) {
            return 5;
        }
        if (contextBox.optimizerSgdF32Pipeline == nil) {
            contextBox.optimizerSgdF32Pipeline = SynaptikOptimizerPipeline(contextBox.device, @"synaptik_optimizer_sgd_f32");
        }
        if (contextBox.optimizerSgdF32Pipeline == nil) {
            return 6;
        }
        id<MTLCommandBuffer> commandBuffer = [contextBox.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            return 7;
        }
        int64_t elementCountValue = logical_element_count;
        [encoder setComputePipelineState:contextBox.optimizerSgdF32Pipeline];
        [encoder setBuffer:parameterBox.buffer offset:0 atIndex:0];
        [encoder setBuffer:gradientBox.buffer offset:0 atIndex:1];
        [encoder setBuffer:outputBox.buffer offset:0 atIndex:2];
        [encoder setBytes:&learning_rate length:sizeof(float) atIndex:3];
        [encoder setBytes:&elementCountValue length:sizeof(int64_t) atIndex:4];
        NSUInteger threads = MIN((NSUInteger) contextBox.optimizerSgdF32Pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) 256);
        if (threads == 0) {
            return 8;
        }
        [encoder dispatchThreads:MTLSizeMake((NSUInteger) logical_element_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        return commandBuffer.status == MTLCommandBufferStatusCompleted ? 0 : 9;
    }
}

int synaptik_apple_mps_optimizer_adam_f32_buffer(
        void *context,
        void *parameter_buffer,
        void *gradient_buffer,
        void *first_moment_buffer,
        void *second_moment_buffer,
        void *output_buffer,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        int32_t step,
        int64_t logical_element_count,
        int64_t parameter_byte_length,
        int64_t gradient_byte_length,
        int64_t first_moment_byte_length,
        int64_t second_moment_byte_length,
        int64_t output_byte_length
) {
    if (context == NULL || parameter_buffer == NULL || gradient_buffer == NULL
            || first_moment_buffer == NULL || second_moment_buffer == NULL || output_buffer == NULL) {
        return 1;
    }
    if (step <= 0 || logical_element_count < 0 || parameter_byte_length < 0 || gradient_byte_length < 0
            || first_moment_byte_length < 0 || second_moment_byte_length < 0 || output_byte_length < 0) {
        return 2;
    }
    int64_t requiredBytes = logical_element_count * (int64_t) sizeof(float);
    if (parameter_byte_length < requiredBytes || gradient_byte_length < requiredBytes
            || first_moment_byte_length < requiredBytes || second_moment_byte_length < requiredBytes
            || output_byte_length < requiredBytes) {
        return 3;
    }
    if (logical_element_count == 0) {
        return 0;
    }
    @autoreleasepool {
        SynaptikAppleMpsContextBox *contextBox = SynaptikUnboxContext(context);
        SynaptikAppleMpsBufferBox *parameterBox = SynaptikUnboxBuffer(parameter_buffer);
        SynaptikAppleMpsBufferBox *gradientBox = SynaptikUnboxBuffer(gradient_buffer);
        SynaptikAppleMpsBufferBox *firstMomentBox = SynaptikUnboxBuffer(first_moment_buffer);
        SynaptikAppleMpsBufferBox *secondMomentBox = SynaptikUnboxBuffer(second_moment_buffer);
        SynaptikAppleMpsBufferBox *outputBox = SynaptikUnboxBuffer(output_buffer);
        if (contextBox == nil || parameterBox == nil || gradientBox == nil || firstMomentBox == nil
                || secondMomentBox == nil || outputBox == nil || parameterBox.buffer == nil
                || gradientBox.buffer == nil || firstMomentBox.buffer == nil || secondMomentBox.buffer == nil
                || outputBox.buffer == nil) {
            return 4;
        }
        if (parameterBox.byteLength < (NSUInteger) parameter_byte_length
                || gradientBox.byteLength < (NSUInteger) gradient_byte_length
                || firstMomentBox.byteLength < (NSUInteger) first_moment_byte_length
                || secondMomentBox.byteLength < (NSUInteger) second_moment_byte_length
                || outputBox.byteLength < (NSUInteger) output_byte_length) {
            return 5;
        }
        if (contextBox.optimizerAdamF32Pipeline == nil) {
            contextBox.optimizerAdamF32Pipeline = SynaptikOptimizerPipeline(contextBox.device, @"synaptik_optimizer_adam_f32");
        }
        if (contextBox.optimizerAdamF32Pipeline == nil) {
            return 6;
        }
        id<MTLCommandBuffer> commandBuffer = [contextBox.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            return 7;
        }
        int64_t elementCountValue = logical_element_count;
        [encoder setComputePipelineState:contextBox.optimizerAdamF32Pipeline];
        [encoder setBuffer:parameterBox.buffer offset:0 atIndex:0];
        [encoder setBuffer:gradientBox.buffer offset:0 atIndex:1];
        [encoder setBuffer:firstMomentBox.buffer offset:0 atIndex:2];
        [encoder setBuffer:secondMomentBox.buffer offset:0 atIndex:3];
        [encoder setBuffer:outputBox.buffer offset:0 atIndex:4];
        [encoder setBytes:&learning_rate length:sizeof(float) atIndex:5];
        [encoder setBytes:&beta1 length:sizeof(float) atIndex:6];
        [encoder setBytes:&beta2 length:sizeof(float) atIndex:7];
        [encoder setBytes:&epsilon length:sizeof(float) atIndex:8];
        [encoder setBytes:&step length:sizeof(int32_t) atIndex:9];
        [encoder setBytes:&elementCountValue length:sizeof(int64_t) atIndex:10];
        NSUInteger threads = MIN((NSUInteger) contextBox.optimizerAdamF32Pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) 256);
        if (threads == 0) {
            return 8;
        }
        [encoder dispatchThreads:MTLSizeMake((NSUInteger) logical_element_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        return commandBuffer.status == MTLCommandBufferStatusCompleted ? 0 : 9;
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
