#import "SynaptikPointwiseFusionKernelSource.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    static const uint32_t ordinals[] = {
        9U, 10U,
        99U, 100U,
        999U, 1000U,
        9999U, 10000U,
        99999U, 100000U,
        999999U, 1000000U,
        9999999U, 10000000U,
        99999999U, 100000000U,
        999999999U, 1000000000U,
        UINT32_MAX,
    };
    static const uint32_t expected_bytes[] = {
        402U, 403U,
        403U, 404U,
        404U, 405U,
        405U, 406U,
        406U, 407U,
        407U, 408U,
        408U, 409U,
        409U, 410U,
        410U, 411U,
        411U,
    };
    _Static_assert(
            sizeof(ordinals) / sizeof(ordinals[0])
                    == sizeof(expected_bytes) / sizeof(expected_bytes[0]),
            "count ledger mismatch");
    SynaptikPointwiseInstructionRecord instructions[2] = {
        {.opcode = 1U},
        {.opcode = 2U},
    };
    for (size_t index = 0U; index < sizeof(ordinals) / sizeof(ordinals[0]); index++) {
        uint32_t counted = SynaptikPointwiseFunctionBytes(
                ordinals[index], instructions, 2U);
        if (counted != expected_bytes[index]) return 1;
        uint8_t bytes[512] = {0};
        SynaptikPointwiseSourceSink sink = {
            .bytes = bytes,
            .capacity = sizeof(bytes) - 1U,
        };
        if (!SynaptikPointwiseEmitFunction(
                    &sink, ordinals[index], instructions, 2U, NO)
                || sink.failed || sink.count != counted)
            return 2;
        char expected_name[64];
        int name_bytes = snprintf(
                expected_name,
                sizeof(expected_name),
                "synaptik_pw_g1_s%u(",
                ordinals[index]);
        if (name_bytes <= 0 || (size_t)name_bytes >= sizeof(expected_name)
                || strstr((const char *)bytes, expected_name) == NULL)
            return 3;
    }
    SynaptikPointwiseStepRecord step = {
        .kind = SYNAPTIK_POINTWISE_GENERATED_STEP,
        .instruction_count = 2U,
        .function_bytes = SynaptikPointwiseFunctionBytes(0U, instructions, 2U),
    };
    uint32_t generated_bytes =
            SynaptikPointwiseGeneratedBytes(&step, 1U, instructions, 2U);
    NSString *generated = SynaptikPointwiseGeneratedSource(
            &step, 1U, instructions, 2U, generated_bytes);
    if (generated == nil
            || [generated rangeOfString:
                    @"// synaptik pointwise fusion generator schema 2\n"].location
                    == NSNotFound
            || [generated rangeOfString:@"generator schema 1"].location != NSNotFound)
        return 4;
    puts("Task 0070 pointwise source-count fixture passed");
    return 0;
}
