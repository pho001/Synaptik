#import "SynaptikPointwiseFusionKernelSource.h"
#include <stdio.h>

static int expect_reason(
        SynaptikPointwiseCapLimits limits,
        SynaptikPointwiseCapProjection projection,
        uint32_t expected) {
    return SynaptikPointwiseCapReason(limits, projection) == expected ? 0 : 1;
}

int main(void) {
    SynaptikPointwiseCapLimits limits = {
        .maximum_units = 10U,
        .maximum_instructions = 20U,
        .maximum_function_bytes = 30U,
        .maximum_generated_bytes = 40U,
        .maximum_total_bytes = 100U,
        .fixed_bytes = 50U,
    };
    SynaptikPointwiseCapProjection projection = {0};

    projection.units = limits.maximum_units;
    if (expect_reason(limits, projection, 0U)) return 1;
    projection.units++;
    if (expect_reason(limits, projection, 1U)) return 2;

    projection = (SynaptikPointwiseCapProjection) {
        .instructions = limits.maximum_instructions,
    };
    if (expect_reason(limits, projection, 0U)) return 3;
    projection.instructions++;
    if (expect_reason(limits, projection, 2U)) return 4;

    projection = (SynaptikPointwiseCapProjection) {
        .function_bytes = limits.maximum_function_bytes,
    };
    if (expect_reason(limits, projection, 0U)) return 5;
    projection.function_bytes++;
    if (expect_reason(limits, projection, 3U)) return 6;

    projection = (SynaptikPointwiseCapProjection) {
        .generated_bytes = limits.maximum_generated_bytes,
    };
    if (expect_reason(limits, projection, 0U)) return 7;
    projection.generated_bytes++;
    if (expect_reason(limits, projection, 4U)) return 8;

    limits.maximum_generated_bytes = 100U;
    projection.generated_bytes = limits.maximum_total_bytes - limits.fixed_bytes;
    if (expect_reason(limits, projection, 0U)) return 9;
    projection.generated_bytes++;
    if (expect_reason(limits, projection, 5U)) return 10;

    limits = (SynaptikPointwiseCapLimits) {
        .maximum_units = 10U,
        .maximum_instructions = 20U,
        .maximum_function_bytes = 30U,
        .maximum_generated_bytes = 40U,
        .maximum_total_bytes = 80U,
        .fixed_bytes = 50U,
    };
    projection = (SynaptikPointwiseCapProjection) {
        .units = 11U,
        .instructions = 21U,
        .function_bytes = 31U,
        .generated_bytes = 41U,
    };
    if (expect_reason(limits, projection, 1U)) return 11;
    projection.units = 10U;
    if (expect_reason(limits, projection, 2U)) return 12;
    projection.instructions = 20U;
    if (expect_reason(limits, projection, 3U)) return 13;
    projection.function_bytes = 30U;
    if (expect_reason(limits, projection, 4U)) return 14;
    projection.generated_bytes = 40U;
    if (expect_reason(limits, projection, 5U)) return 15;

    puts("Task 0070 pointwise cap fixture passed");
    return 0;
}
