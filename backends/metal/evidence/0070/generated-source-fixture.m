#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#import "SynaptikPointwiseFusionKernelSource.h"
#import "synaptik_exact_kernels.h"
#import "synaptik_task0059_data_kernels.h"
#import "synaptik_task0060_reduction_kernels.h"
#import "synaptik_task0061_matmul_kernels.h"
#import "synaptik_task0063_ordering_kernels.h"
#import "synaptik_task0064_convolution_pooling_kernels.h"
#import "synaptik_task0069_aggregate_kernels.h"
#import "synaptik_task0065_rng_dropout_kernels.h"
#import "synaptik_task0066_dtype_layout_kernels.h"
#import "synaptik_task0071_anchor_epilogue_kernels.h"
#include <stdint.h>
#include <string.h>

static BOOL fixture_digest(NSString *source, uint8_t digest[CC_SHA256_DIGEST_LENGTH]) {
    NSData *bytes = [source dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    if (bytes == nil || bytes.length > UINT32_MAX) return NO;
    CC_SHA256(bytes.bytes, (CC_LONG)bytes.length, digest);
    return YES;
}

static NSString *fixed_source(void) {
    NSMutableString *source = [NSMutableString string];
    if (source == nil) return nil;
    [source appendString:SynaptikExactKernelSource];
    [source appendString:SynaptikTask0059DataKernelSource];
    [source appendString:SynaptikTask0060ReductionKernelSource];
    [source appendString:SynaptikTask0061MatmulKernelSource];
    [source appendString:SynaptikTask0063OrderingKernelSource];
    [source appendString:SynaptikTask0064ConvolutionPoolingKernelSource];
    [source appendString:SynaptikTask0069AggregateKernelSource];
    [source appendString:SynaptikTask0065RngDropoutKernelSource];
    [source appendString:SynaptikTask0066DtypeLayoutKernelSource];
    [source appendString:SynaptikTask0071AnchorEpilogueKernelSource];
    return [source lengthOfBytesUsingEncoding:NSUTF8StringEncoding]
            == SYNAPTIK_POINTWISE_FIXED_CORPUS_BYTES ? source : nil;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        static const uint32_t opcode_lists[11][8] = {
            {1U},
            {2U},
            {3U},
            {4U},
            {1U, 2U},
            {2U, 3U, 4U},
            {1U, 2U, 3U, 4U},
            {1U, 2U, 3U, 4U, 1U},
            {1U, 2U, 3U, 4U, 1U, 2U},
            {1U, 2U, 3U, 4U, 1U, 2U, 3U},
            {1U, 2U, 3U, 4U, 1U, 2U, 3U, 4U},
        };
        static const uint32_t counts[11] = {1U, 1U, 1U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
        SynaptikPointwiseStepRecord steps[11] = {0};
        SynaptikPointwiseInstructionRecord instructions[39] = {0};
        uint32_t instruction_cursor = 0U;
        for (uint32_t step = 0U; step < 11U; step++) {
            for (uint32_t relative = 0U; relative < counts[step]; relative++) {
                instructions[instruction_cursor + relative] =
                        (SynaptikPointwiseInstructionRecord) {
                            .step = step,
                            .relative_node = relative,
                            .opcode = opcode_lists[step][relative],
                        };
            }
            uint32_t function_bytes = SynaptikPointwiseFunctionBytesWithMode(
                    step, instructions + instruction_cursor, counts[step], YES);
            if (function_bytes == 0U) return 3;
            steps[step] = (SynaptikPointwiseStepRecord) {
                .kind = SYNAPTIK_POINTWISE_GENERATED_STEP,
                .instruction_start = instruction_cursor,
                .instruction_count = counts[step],
                .function_bytes = function_bytes,
            };
            instruction_cursor += counts[step];
        }
        if (instruction_cursor != 39U) return 4;
        uint32_t generated_bytes = SynaptikPointwiseGeneratedBytesWithMode(
                steps, 11U, instructions, instruction_cursor, YES);
        if (generated_bytes == UINT32_MAX) return 5;
        NSString *first = SynaptikPointwiseGeneratedSourceWithMode(
                steps, 11U, instructions, instruction_cursor, generated_bytes, YES);
        NSString *second = SynaptikPointwiseGeneratedSourceWithMode(
                steps, 11U, instructions, instruction_cursor, generated_bytes, YES);
        uint8_t first_digest[CC_SHA256_DIGEST_LENGTH];
        uint8_t second_digest[CC_SHA256_DIGEST_LENGTH];
        if (first == nil || second == nil || ![first isEqualToString:second]
                || [first lengthOfBytesUsingEncoding:NSUTF8StringEncoding] != generated_bytes
                || !fixture_digest(first, first_digest)
                || !fixture_digest(second, second_digest)
                || memcmp(first_digest, second_digest, sizeof(first_digest)) != 0)
            return 6;
        NSString *fixed = fixed_source();
        if (fixed == nil) return 7;
        NSString *assembled = [fixed stringByAppendingString:first];
        NSError *error = nil;
        if (assembled == nil
                || ![assembled writeToFile:[NSString stringWithUTF8String:argv[1]]
                                atomically:YES
                                  encoding:NSUTF8StringEncoding
                                     error:&error]
                || error != nil)
            return 8;
    }
    return 0;
}
