#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dlfcn.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

@interface SynaptikProbeWeights : NSObject <MPSCNNConvolutionDataSource>
@property(nonatomic) MPSDataType weightType;
@property(nonatomic) uint16_t weightBits;
@property(nonatomic, strong) MPSCNNConvolutionDescriptor *convolutionDescriptor;
@end

@implementation SynaptikProbeWeights
- (instancetype)initWithType:(MPSDataType)type bits:(uint16_t)bits {
    self = [super init];
    if (self != nil) {
        _weightType = type;
        _weightBits = bits;
        _convolutionDescriptor =
                [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:1
                        kernelHeight:1 inputFeatureChannels:1 outputFeatureChannels:1
                        neuronFilter:nil];
    }
    return self;
}
- (MPSDataType)dataType { return self.weightType; }
- (MPSCNNConvolutionDescriptor *)descriptor { return self.convolutionDescriptor; }
- (void *)weights { return &_weightBits; }
- (float *)biasTerms { return NULL; }
- (BOOL)load { return YES; }
- (void)purge { }
- (NSString *)label { return @"synaptik-low-precision-qualification"; }
- (id)copyWithZone:(NSZone *)zone {
    return [[SynaptikProbeWeights allocWithZone:zone]
            initWithType:self.weightType bits:self.weightBits];
}
@end

static void print_value(const char *key, NSString *value) {
    NSData *bytes = [value dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    if (bytes == nil) exit(2);
    printf("%s\t%.*s\n", key, (int)bytes.length, (const char *)bytes.bytes);
}

static NSString *probe_gpu_family(id<MTLDevice> device) {
    static const struct {
        MTLGPUFamily family;
        __unsafe_unretained NSString *name;
    } families[] = {
        { MTLGPUFamilyApple11, @"APPLE_11" },
        { MTLGPUFamilyApple10, @"APPLE_10" },
        { MTLGPUFamilyApple9, @"APPLE_9" },
        { MTLGPUFamilyApple8, @"APPLE_8" },
        { MTLGPUFamilyApple7, @"APPLE_7" },
        { MTLGPUFamilyApple6, @"APPLE_6" },
        { MTLGPUFamilyApple5, @"APPLE_5" },
        { MTLGPUFamilyApple4, @"APPLE_4" },
        { MTLGPUFamilyApple3, @"APPLE_3" },
        { MTLGPUFamilyApple2, @"APPLE_2" },
        { MTLGPUFamilyApple1, @"APPLE_1" },
    };
    for (NSUInteger index = 0U; index < sizeof(families) / sizeof(families[0]); index++) {
        if ([device supportsFamily:families[index].family]) return families[index].name;
    }
    return nil;
}

static BOOL probe_fp16_convolution(id<MTLDevice> device) {
    SynaptikProbeWeights *weights = [[SynaptikProbeWeights alloc]
            initWithType:MPSDataTypeFloat16 bits:0x3c00U];
    MPSCNNConvolution *convolution = [[MPSCNNConvolution alloc]
            initWithDevice:device weights:weights];
    if (convolution == nil) return NO;
    convolution.accumulatorPrecisionOption =
            MPSNNConvolutionAccumulatorPrecisionOptionFloat;
    BOOL accumulator = convolution.accumulatorPrecisionOption ==
            MPSNNConvolutionAccumulatorPrecisionOptionFloat;
    MPSImageDescriptor *descriptor = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
            width:2 height:2 featureChannels:1];
    MPSImage *source = [[MPSImage alloc] initWithDevice:device imageDescriptor:descriptor];
    MPSImage *destination = [[MPSImage alloc]
            initWithDevice:device imageDescriptor:descriptor];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    if (source == nil || destination == nil || queue == nil || commands == nil) return NO;
    uint16_t sourceWords[4] = {0U, 0U, 0U, 0U};
    [source.texture replaceRegion:MTLRegionMake2D(0U, 0U, 2U, 2U)
            mipmapLevel:0U withBytes:sourceWords bytesPerRow:2U * sizeof(uint16_t)];
    [convolution encodeToCommandBuffer:commands
            sourceImage:source destinationImage:destination];
    [commands commit];
    [commands waitUntilCompleted];
    return accumulator && commands.status == MTLCommandBufferStatusCompleted
            && commands.error == nil;
}

static NSString *probe_bfloat16_construction(id<MTLDevice> device) {
    @try {
        SynaptikProbeWeights *weights = [[SynaptikProbeWeights alloc]
                initWithType:MPSDataTypeBFloat16 bits:0x3f80U];
        MPSCNNConvolution *convolution = [[MPSCNNConvolution alloc]
                initWithDevice:device weights:weights];
        return convolution == nil ? @"REJECTED" : @"CONSTRUCTED_UNDOCUMENTED";
    } @catch (__unused NSException *exception) {
        return @"REJECTED";
    }
}

static BOOL probe_bfloat16_rejection(void) {
    fflush(stdout);
    pid_t child = fork();
    if (child < 0) return NO;
    if (child == 0) {
        (void)freopen("/dev/null", "w", stderr);
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            NSString *result = device == nil ? @"REJECTED"
                    : probe_bfloat16_construction(device);
            _exit([result isEqualToString:@"CONSTRUCTED_UNDOCUMENTED"] ? 0 : 5);
        }
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child) return NO;
    return !WIFEXITED(status) || WEXITSTATUS(status) != 0;
}

static BOOL print_bridge_environment(const char *libraryPath) {
    void *library = dlopen(libraryPath, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) return NO;
    typedef int32_t (*CreateContext)(void **);
    typedef int32_t (*ReadEnvironment)(void *, uint8_t *, uint32_t, uint32_t *);
    typedef int32_t (*ReleaseContext)(void *);
    CreateContext createContext = (CreateContext)dlsym(
            library, "synaptik_metal_context_create");
    ReadEnvironment readEnvironment = (ReadEnvironment)dlsym(
            library, "synaptik_metal_context_certification_environment");
    ReleaseContext releaseContext = (ReleaseContext)dlsym(
            library, "synaptik_metal_context_release");
    if (createContext == NULL || readEnvironment == NULL || releaseContext == NULL) {
        dlclose(library);
        return NO;
    }
    void *context = NULL;
    uint8_t bytes[4096];
    uint32_t length = 0U;
    BOOL ok = createContext(&context) == 0 && context != NULL
            && readEnvironment(context, bytes, sizeof(bytes), &length) == 0
            && length > 0U && length < sizeof(bytes);
    NSString *record = ok ? [[NSString alloc] initWithBytes:bytes length:length
            encoding:NSUTF8StringEncoding] : nil;
    if (context != NULL && releaseContext(context) != 0) ok = NO;
    dlclose(library);
    NSArray<NSString *> *lines = [record componentsSeparatedByString:@"\n"];
    if (!ok || lines.count != 6U
            || ![lines[0] isEqualToString:@"SYNAPTIK_METAL_CERTIFICATION_ENVIRONMENT_V1"]) {
        return NO;
    }
    print_value("gpu-family", lines[1]);
    print_value("os-build", lines[2]);
    print_value("sdk-version", lines[3]);
    print_value("compiler-version", lines[4]);
    print_value("flags-options", lines[5]);
    return YES;
}

static BOOL probe_mpp_pipelines(id<MTLDevice> device, NSString *metallibPath) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:metallibPath] error:&error];
    if (library == nil || error != nil) return NO;
    NSArray<NSString *> *names = @[
        @"mpp_matmul_f16_f32",
        @"mpp_matmul_bf16_f32",
        @"mpp_convolution_f16_f32",
        @"mpp_convolution_bf16_f32"
    ];
    for (NSString *name in names) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (function == nil) return NO;
        error = nil;
        id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil || error != nil || pipeline.threadExecutionWidth == 0U) return NO;
        print_value([[NSString stringWithFormat:@"pipeline-%@", name] UTF8String], @"PASS");
    }
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) return 2;
        BOOL bfloat16Rejected = probe_bfloat16_rejection();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil || !print_bridge_environment(argv[1])) return 3;
        print_value("probe-gpu-family", probe_gpu_family(device));
        BOOL mpsSupported = MPSSupportsMTLDevice(device);
        BOOL fp16Convolution = mpsSupported && probe_fp16_convolution(device);
        print_value("mps-device-support", mpsSupported ? @"PASS" : @"FAIL");
        print_value("mps-fp16-convolution-runtime",
                fp16Convolution ? @"PASS" : @"FAIL");
        print_value("mps-fp16-accumulator-option",
                fp16Convolution ? @"FLOAT" : @"UNAVAILABLE");
        print_value("mps-bfloat16-runtime",
                bfloat16Rejected ? @"REJECTED" : @"ACCEPTED");
        if (!mpsSupported || !fp16Convolution || !bfloat16Rejected
                || !probe_mpp_pipelines(
                        device, [NSString stringWithUTF8String:argv[2]])) return 4;
        return 0;
    }
}
