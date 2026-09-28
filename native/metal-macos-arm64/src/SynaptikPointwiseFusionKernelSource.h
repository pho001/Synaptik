#ifndef SYNAPTIK_POINTWISE_FUSION_KERNEL_SOURCE_H
#define SYNAPTIK_POINTWISE_FUSION_KERNEL_SOURCE_H

#import <Foundation/Foundation.h>
#include <stdint.h>

#define SYNAPTIK_POINTWISE_GENERATOR_SCHEMA 1U
#define SYNAPTIK_POINTWISE_GENERATED_STEP 3U
#define SYNAPTIK_POINTWISE_FIXED_CORPUS_BYTES 77411U
#define SYNAPTIK_POINTWISE_MAX_UNITS 32U
#define SYNAPTIK_POINTWISE_MAX_INSTRUCTIONS 256U
#define SYNAPTIK_POINTWISE_MAX_FUNCTION_BYTES 16384U
#define SYNAPTIK_POINTWISE_MAX_GENERATED_BYTES 262144U
#define SYNAPTIK_POINTWISE_MAX_TOTAL_BYTES 1048576U

typedef struct {
    uint32_t kind;
    uint32_t member_start;
    uint32_t member_count;
    uint32_t binding_start;
    uint32_t binding_count;
    uint32_t instruction_start;
    uint32_t instruction_count;
    uint32_t function_bytes;
    uint32_t flags;
    uint32_t reserved;
} SynaptikPointwiseStepRecord;

typedef struct {
    uint32_t step;
    uint32_t relative_node;
    uint32_t opcode;
    uint32_t semantic_site;
    uint32_t input_count;
    uint32_t input0;
    uint32_t input1;
    uint32_t input2;
    uint32_t output;
    uint32_t immediate_count;
    uint32_t immediate_type0;
    uint32_t immediate_type1;
    uint64_t raw0;
    uint64_t raw1;
} SynaptikPointwiseInstructionRecord;

static uint32_t SynaptikPointwiseDecimalDigits(uint32_t value) {
    uint32_t digits = 1U;
    while (value >= 10U) {
        value /= 10U;
        digits++;
    }
    return digits;
}

static uint32_t SynaptikPointwiseHelperBytes(uint32_t opcode) {
    return opcode == 1U ? 10U : opcode >= 2U && opcode <= 4U ? 9U : 0U;
}

static NSString *SynaptikPointwiseHelperName(uint32_t opcode) {
    switch (opcode) {
        case 1U: return @"floor_bits";
        case 2U: return @"ceil_bits";
        case 3U: return @"sign_bits";
        case 4U: return @"relu_bits";
        default: return nil;
    }
}

static uint32_t SynaptikPointwiseFunctionBytes(
        uint32_t step,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count) {
    if (instructions == NULL || instruction_count < 2U || instruction_count > 8U) return 0U;
    uint64_t bytes = 345U + SynaptikPointwiseDecimalDigits(step)
            + SynaptikPointwiseDecimalDigits(instruction_count);
    for (uint32_t index = 0U; index < instruction_count; index++) {
        uint32_t helper = SynaptikPointwiseHelperBytes(instructions[index].opcode);
        if (helper == 0U) return 0U;
        bytes += 16U + SynaptikPointwiseDecimalDigits(index)
                + SynaptikPointwiseDecimalDigits(index + 1U) + helper;
    }
    return bytes <= UINT32_MAX ? (uint32_t)bytes : 0U;
}

static NSString *SynaptikPointwiseGeneratedSource(
        const SynaptikPointwiseStepRecord *steps,
        uint32_t step_count,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count,
        uint32_t expected_generated_bytes) {
    if ((step_count != 0U && steps == NULL)
            || (instruction_count != 0U && instructions == NULL)) return nil;
    uint32_t unit_count = 0U;
    for (uint32_t step = 0U; step < step_count; step++) {
        if (steps[step].kind == SYNAPTIK_POINTWISE_GENERATED_STEP) unit_count++;
    }
    if (unit_count == 0U) return expected_generated_bytes == 0U ? @"" : nil;
    NSMutableString *source = [NSMutableString stringWithString:
            @"\n// synaptik pointwise fusion generator schema 1\n"];
    if (source == nil) return nil;
    for (uint32_t step = 0U; step < step_count; step++) {
        SynaptikPointwiseStepRecord record = steps[step];
        if (record.kind != SYNAPTIK_POINTWISE_GENERATED_STEP) continue;
        if (record.instruction_count < 2U || record.instruction_count > 8U
                || record.instruction_start > instruction_count
                || record.instruction_count > instruction_count - record.instruction_start)
            return nil;
        const SynaptikPointwiseInstructionRecord *unit = instructions + record.instruction_start;
        uint32_t function_bytes = SynaptikPointwiseFunctionBytes(
                step, unit, record.instruction_count);
        if (function_bytes == 0U || function_bytes != record.function_bytes) return nil;
        NSUInteger before = [source lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
        [source appendFormat:
                @"kernel void synaptik_pw_g1_s%u(device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]], constant PointMeta &meta [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {\n",
                step];
        [source appendString:@"  ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight);\n"];
        [source appendString:@"  if (linear >= meta.elementCount) return;\n"];
        [source appendString:@"  uint v0 = input[linear];\n"];
        for (uint32_t index = 0U; index < record.instruction_count; index++) {
            NSString *helper = SynaptikPointwiseHelperName(unit[index].opcode);
            if (helper == nil) return nil;
            [source appendFormat:@"  uint v%u = %@(v%u);\n", index + 1U, helper, index];
        }
        [source appendFormat:@"  output[linear] = v%u;\n", record.instruction_count];
        [source appendString:@"}\n"];
        NSUInteger after = [source lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
        if (after - before != function_bytes) return nil;
    }
    return [source lengthOfBytesUsingEncoding:NSUTF8StringEncoding] == expected_generated_bytes
            ? [source copy] : nil;
}

#endif
