#ifndef SYNAPTIK_POINTWISE_FUSION_KERNEL_SOURCE_H
#define SYNAPTIK_POINTWISE_FUSION_KERNEL_SOURCE_H

#import <Foundation/Foundation.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SYNAPTIK_POINTWISE_GENERATOR_SCHEMA 1U
#define SYNAPTIK_POINTWISE_GENERATED_STEP 3U
#define SYNAPTIK_POINTWISE_FIXED_CORPUS_BYTES 77444U
#define SYNAPTIK_POINTWISE_MAX_UNITS 32U
#define SYNAPTIK_POINTWISE_MAX_INSTRUCTIONS 256U
#define SYNAPTIK_POINTWISE_MAX_FUNCTION_BYTES 16384U
#define SYNAPTIK_POINTWISE_MAX_GENERATED_BYTES 262144U
#define SYNAPTIK_POINTWISE_MAX_TOTAL_BYTES 1048576U
#define SYNAPTIK_POINTWISE_GENERATED_PREAMBLE_BYTES 49U

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

typedef struct {
    uint64_t maximum_units;
    uint64_t maximum_instructions;
    uint64_t maximum_function_bytes;
    uint64_t maximum_generated_bytes;
    uint64_t maximum_total_bytes;
    uint64_t fixed_bytes;
} SynaptikPointwiseCapLimits;

typedef struct {
    uint64_t units;
    uint64_t instructions;
    uint64_t function_bytes;
    uint64_t generated_bytes;
} SynaptikPointwiseCapProjection;

static inline SynaptikPointwiseCapLimits SynaptikPointwiseProductionCapLimits(void) {
    return (SynaptikPointwiseCapLimits) {
        .maximum_units = SYNAPTIK_POINTWISE_MAX_UNITS,
        .maximum_instructions = SYNAPTIK_POINTWISE_MAX_INSTRUCTIONS,
        .maximum_function_bytes = SYNAPTIK_POINTWISE_MAX_FUNCTION_BYTES,
        .maximum_generated_bytes = SYNAPTIK_POINTWISE_MAX_GENERATED_BYTES,
        .maximum_total_bytes = SYNAPTIK_POINTWISE_MAX_TOTAL_BYTES,
        .fixed_bytes = SYNAPTIK_POINTWISE_FIXED_CORPUS_BYTES,
    };
}

static inline uint32_t SynaptikPointwiseCapReason(
        SynaptikPointwiseCapLimits limits,
        SynaptikPointwiseCapProjection projection) {
    if (projection.units > limits.maximum_units) return 1U;
    if (projection.instructions > limits.maximum_instructions) return 2U;
    if (projection.function_bytes > limits.maximum_function_bytes) return 3U;
    if (projection.generated_bytes > limits.maximum_generated_bytes) return 4U;
    if (limits.fixed_bytes > limits.maximum_total_bytes
            || projection.generated_bytes > limits.maximum_total_bytes - limits.fixed_bytes)
        return 5U;
    return 0U;
}

typedef struct {
    uint8_t *bytes;
    uint64_t capacity;
    uint64_t count;
    BOOL failed;
} SynaptikPointwiseSourceSink;

static inline BOOL SynaptikPointwiseSinkWrite(
        SynaptikPointwiseSourceSink *sink,
        const char *bytes,
        size_t length) {
    if (sink == NULL || bytes == NULL || sink->failed
            || sink->count > UINT64_MAX - (uint64_t)length) {
        if (sink != NULL) sink->failed = YES;
        return NO;
    }
    if (sink->bytes != NULL) {
        if (sink->count + (uint64_t)length > sink->capacity) {
            sink->failed = YES;
            return NO;
        }
        memcpy(sink->bytes + sink->count, bytes, length);
    }
    sink->count += (uint64_t)length;
    return YES;
}

static inline BOOL SynaptikPointwiseSinkFormat(
        SynaptikPointwiseSourceSink *sink,
        const char *format,
        ...) {
    char buffer[512];
    va_list arguments;
    va_start(arguments, format);
    int length = vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    if (length < 0 || (size_t)length >= sizeof(buffer)) {
        if (sink != NULL) sink->failed = YES;
        return NO;
    }
    return SynaptikPointwiseSinkWrite(sink, buffer, (size_t)length);
}

static inline const char *SynaptikPointwiseHelperName(uint32_t opcode) {
    switch (opcode) {
        case 1U: return "floor_bits";
        case 2U: return "ceil_bits";
        case 3U: return "sign_bits";
        case 4U: return "relu_bits";
        default: return NULL;
    }
}

static inline BOOL SynaptikPointwiseEmitFunction(
        SynaptikPointwiseSourceSink *sink,
        uint32_t step,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count,
        BOOL allow_audit_singleton) {
    if (sink == NULL || instructions == NULL || instruction_count == 0U
            || instruction_count > 8U || (!allow_audit_singleton && instruction_count < 2U))
        return NO;
    if (!SynaptikPointwiseSinkFormat(
                sink,
                "kernel void synaptik_pw_g1_s%u(device const uint *input [[buffer(0)]], device uint *output [[buffer(1)]], constant PointMeta &meta [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {\n",
                step)
            || !SynaptikPointwiseSinkWrite(
                sink,
                "  ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight);\n",
                sizeof("  ulong linear = linear_id(gid, meta.gridWidth, meta.gridHeight);\n") - 1U)
            || !SynaptikPointwiseSinkWrite(
                sink,
                "  if (linear >= meta.elementCount) return;\n",
                sizeof("  if (linear >= meta.elementCount) return;\n") - 1U)
            || !SynaptikPointwiseSinkWrite(
                sink,
                "  uint v0 = input[linear];\n",
                sizeof("  uint v0 = input[linear];\n") - 1U))
        return NO;
    for (uint32_t index = 0U; index < instruction_count; index++) {
        const char *helper = SynaptikPointwiseHelperName(instructions[index].opcode);
        if (helper == NULL || !SynaptikPointwiseSinkFormat(
                    sink,
                    "  uint v%u = %s(v%u);\n",
                    index + 1U,
                    helper,
                    index))
            return NO;
    }
    return SynaptikPointwiseSinkFormat(
            sink,
            "  output[linear] = v%u;\n}\n",
            instruction_count);
}

static inline uint32_t SynaptikPointwiseFunctionBytesWithMode(
        uint32_t step,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count,
        BOOL allow_audit_singleton) {
    SynaptikPointwiseSourceSink sink = {0};
    if (!SynaptikPointwiseEmitFunction(
                &sink, step, instructions, instruction_count, allow_audit_singleton)
            || sink.failed || sink.count > UINT32_MAX)
        return 0U;
    return (uint32_t)sink.count;
}

static inline uint32_t SynaptikPointwiseFunctionBytes(
        uint32_t step,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count) {
    return SynaptikPointwiseFunctionBytesWithMode(
            step, instructions, instruction_count, NO);
}

static inline BOOL SynaptikPointwiseEmitGeneratedSource(
        SynaptikPointwiseSourceSink *sink,
        const SynaptikPointwiseStepRecord *steps,
        uint32_t step_count,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count,
        BOOL allow_audit_singletons,
        SynaptikPointwiseCapProjection *projection) {
    if (sink == NULL || projection == NULL
            || (step_count != 0U && steps == NULL)
            || (instruction_count != 0U && instructions == NULL))
        return NO;
    uint32_t generated_units = 0U;
    uint32_t generated_instructions = 0U;
    uint32_t largest_function = 0U;
    for (uint32_t step = 0U; step < step_count; step++)
        if (steps[step].kind == SYNAPTIK_POINTWISE_GENERATED_STEP) generated_units++;
    if (generated_units != 0U
            && !SynaptikPointwiseSinkWrite(
                sink,
                "\n// synaptik pointwise fusion generator schema 1\n",
                SYNAPTIK_POINTWISE_GENERATED_PREAMBLE_BYTES))
        return NO;
    for (uint32_t step = 0U; step < step_count; step++) {
        SynaptikPointwiseStepRecord record = steps[step];
        if (record.kind != SYNAPTIK_POINTWISE_GENERATED_STEP) continue;
        if (record.instruction_count == 0U || record.instruction_count > 8U
                || (!allow_audit_singletons && record.instruction_count < 2U)
                || record.instruction_start > instruction_count
                || record.instruction_count > instruction_count - record.instruction_start)
            return NO;
        uint64_t before = sink->count;
        if (!SynaptikPointwiseEmitFunction(
                    sink,
                    step,
                    instructions + record.instruction_start,
                    record.instruction_count,
                    allow_audit_singletons))
            return NO;
        uint64_t function_bytes = sink->count - before;
        if (function_bytes == 0U || function_bytes > UINT32_MAX
                || record.function_bytes != (uint32_t)function_bytes)
            return NO;
        if (function_bytes > largest_function) largest_function = (uint32_t)function_bytes;
        if (generated_instructions > UINT32_MAX - record.instruction_count) return NO;
        generated_instructions += record.instruction_count;
    }
    *projection = (SynaptikPointwiseCapProjection) {
        .units = generated_units,
        .instructions = generated_instructions,
        .function_bytes = largest_function,
        .generated_bytes = sink->count,
    };
    return !sink->failed
            && SynaptikPointwiseCapReason(
                    SynaptikPointwiseProductionCapLimits(), *projection) == 0U;
}

static inline uint32_t SynaptikPointwiseGeneratedBytesWithMode(
        const SynaptikPointwiseStepRecord *steps,
        uint32_t step_count,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count,
        BOOL allow_audit_singletons) {
    SynaptikPointwiseSourceSink sink = {0};
    SynaptikPointwiseCapProjection projection = {0};
    if (!SynaptikPointwiseEmitGeneratedSource(
                &sink,
                steps,
                step_count,
                instructions,
                instruction_count,
                allow_audit_singletons,
                &projection)
            || sink.failed || sink.count > UINT32_MAX)
        return UINT32_MAX;
    return (uint32_t)sink.count;
}

static inline uint32_t SynaptikPointwiseGeneratedBytes(
        const SynaptikPointwiseStepRecord *steps,
        uint32_t step_count,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count) {
    return SynaptikPointwiseGeneratedBytesWithMode(
            steps, step_count, instructions, instruction_count, NO);
}

static inline NSString *SynaptikPointwiseGeneratedSourceWithMode(
        const SynaptikPointwiseStepRecord *steps,
        uint32_t step_count,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count,
        uint32_t expected_generated_bytes,
        BOOL allow_audit_singletons) {
    uint32_t counted = SynaptikPointwiseGeneratedBytesWithMode(
            steps,
            step_count,
            instructions,
            instruction_count,
            allow_audit_singletons);
    if (counted == UINT32_MAX || counted != expected_generated_bytes) return nil;
    if (counted == 0U) return @"";
    NSMutableData *data = [NSMutableData dataWithLength:counted];
    if (data == nil) return nil;
    SynaptikPointwiseSourceSink sink = {
        .bytes = data.mutableBytes,
        .capacity = counted,
    };
    SynaptikPointwiseCapProjection projection = {0};
    if (!SynaptikPointwiseEmitGeneratedSource(
                &sink,
                steps,
                step_count,
                instructions,
                instruction_count,
                allow_audit_singletons,
                &projection)
            || sink.failed || sink.count != counted)
        return nil;
    return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
}

static inline NSString *SynaptikPointwiseGeneratedSource(
        const SynaptikPointwiseStepRecord *steps,
        uint32_t step_count,
        const SynaptikPointwiseInstructionRecord *instructions,
        uint32_t instruction_count,
        uint32_t expected_generated_bytes) {
    return SynaptikPointwiseGeneratedSourceWithMode(
            steps,
            step_count,
            instructions,
            instruction_count,
            expected_generated_bytes,
            NO);
}

#endif
