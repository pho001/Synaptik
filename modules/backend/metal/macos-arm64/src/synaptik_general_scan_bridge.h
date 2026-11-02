#ifndef SYNAPTIK_GENERAL_SCAN_BRIDGE_H
#define SYNAPTIK_GENERAL_SCAN_BRIDGE_H

#include "synaptik_general_scan_value_validation.h"

/* Included after the ordinary builder, fixed binding codec and Scan plan declarations.
 * Physical body storage is dense; semantic ROW facts remain in immutable validation roles
 * and the separately bound control buffers. No invocation resource is retained here.
 */
static SynaptikMetalProgramStep *synaptik_build_scan_ordinary_step(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes, const uint8_t *types,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        id<MTLDevice> device, id<MTLLibrary> library, id<MTLLibrary> lowLibrary,
        const SynaptikMetalScanStepControl *control, const int32_t *allValueRowAxes,
        const uint64_t *allValueRowInner) {
    if (control == NULL || allValueRowAxes == NULL || allValueRowInner == NULL
            || !synaptik_scan_validate_ordinary(node, shapes, types, strides, offsets, states)) return nil;
    NSData *roles = synaptik_scan_validation_roles(
            node, shapes, types, allValueRowAxes, allValueRowInner);
    if (roles == nil) return nil;

    BOOL hasOutput = NO;
    for (uint32_t role = 0U; role < node.output_count; role++) {
        uint64_t elements;
        if (!synaptik_scan_shape_product(shapes[node.outputs[role]], &elements)) return nil;
        hasOutput |= elements != 0U;
    }

    SynaptikMetalProgramStep *step;
    if (hasOutput) {
        /* The shared builder selects the controlled function before its only pipeline
         * construction. A missing controlled implementation is a hard admission failure.
         * Sibling outputs, not merely output zero, determine whether construction is needed.
         */
        step = make_custom_step(node, shapes, types, strides, offsets, states,
                NULL, NULL, device, library, lowLibrary, 0U, YES);
        if (step == nil || step.pipeline == nil || step.metadata == nil
                || step.grid.width == 0U || step.grid.height == 0U || step.grid.depth == 0U
                || step.threadsPerThreadgroup == 0U) return nil;
    } else {
        /* No positive-grid stand-in: this marker still carries the full pre-write value
         * validation contract, even when nonempty input domains can fail for empty results.
         */
        step = [SynaptikMetalProgramStep new];
        if (step == nil) return nil;
        step.custom = YES;
        step.operation = node.operation;
        step.stage = 0U;
        step.firstInput = node.input_count == 0U ? node.output : node.first_input;
        step.secondInput = node.second_input;
        step.auxiliaryInput = node.auxiliary;
        step.output = node.output;
        step.scratchInput = UINT32_MAX;
        step.grid = MTLSizeMake(0U, 0U, 0U);
    }
    step.scanValidationRoles = roles;
    return step;
}

static int32_t synaptik_encode_scan_step(SynaptikMetalProgramStep *step,
        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> __unsafe_unretained const *buffers,
        uint32_t bufferCount, const NSUInteger *offsets) {
    if (step == nil) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    MTLSize grid = step.grid;
    if (grid.width == 0U && grid.height == 0U && grid.depth == 0U)
        return SYNAPTIK_METAL_STATUS_OK;
    if (step.pipeline == nil || step.metadata == nil || step.threadsPerThreadgroup == 0U
            || grid.width == 0U || grid.height == 0U || grid.depth == 0U)
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    if (encoder == nil || buffers == NULL || bufferCount == 0U || offsets == NULL)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;

    /* The caller already bound control at 28/29. Borrow current bindings on every dispatch:
     * carry promotion changes offsets between iterations, including scratch's workspace base.
     */
    SynaptikMetalStepBindings bindings = {bufferCount, NULL, nil, offsets, buffers};
    if (!synaptik_encode_fixed_step(step, encoder, bindings))
        return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    [encoder dispatchThreads:grid threadsPerThreadgroup:
            MTLSizeMake(step.threadsPerThreadgroup, 1U, 1U)];
    return SYNAPTIK_METAL_STATUS_OK;
}

static int32_t synaptik_validate_scan_step(SynaptikMetalProgramStep *step,
        SynaptikMetalDecodedNode node, id<MTLBuffer> __unsafe_unretained const *buffers,
        uint32_t bufferCount, const NSUInteger *offsets,
        const uint64_t *control, const SynaptikMetalScanStepControl *stepControl) {
    return synaptik_scan_validate_value_domains(
            step, node, buffers, bufferCount, offsets, control, stepControl);
}

#endif
