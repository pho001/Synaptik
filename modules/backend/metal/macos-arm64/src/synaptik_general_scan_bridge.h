#ifndef SYNAPTIK_GENERAL_SCAN_BRIDGE_H
#define SYNAPTIK_GENERAL_SCAN_BRIDGE_H

#include "synaptik_general_scan_value_validation.h"
#include "synaptik_general_scan_coordinates.h"

/* Included after the ordinary builder, fixed binding codec and Scan plan declarations.
 * Physical body storage is dense; semantic ROW facts remain in immutable validation roles
 * and the separately bound control buffers. No invocation resource is retained here.
 */
static BOOL synaptik_append_scan_ordinary_steps(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes, const uint8_t *types,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        id<MTLDevice> device, id<MTLLibrary> library, id<MTLLibrary> lowLibrary,
        const SynaptikMetalScanStepControl *control, const int32_t *allValueRowAxes,
        const uint64_t *allValueRowInner, uint32_t semanticScatterReduction,
        NSMutableArray<SynaptikMetalProgramStep *> *sink) {
    if (control == NULL || allValueRowAxes == NULL || allValueRowInner == NULL || sink == nil
            || shapes == nil || types == NULL || strides == NULL || offsets == NULL || states == NULL)
        return NO;
    NSData *roles = synaptik_scan_validation_roles(
            node, shapes, types, allValueRowAxes, allValueRowInner, semanticScatterReduction);
    if (roles == nil) return NO;

    BOOL hasOutput = NO;
    for (uint32_t role = 0U; role < node.output_count; role++) {
        uint64_t elements;
        if (!synaptik_scan_shape_product(shapes[node.outputs[role]], &elements)) return NO;
        hasOutput |= elements != 0U;
    }

    /* The ordinary scatter owner handles its output-writer domains in one stage.
     * Legacy exact replacement retains base-copy then updates, with one preflight
     * before either stage. The enclosing plan flattens this immutable sequence.
     */
    BOOL exactScatter = (node.operation == 18U || node.operation == 72U)
            && !synaptik_scatter_handles(node, types);
    uint32_t stageCount = exactScatter ? 2U : 1U;
    uint64_t updateElements = 0U;
    if (exactScatter && !synaptik_scan_shape_product(shapes[node.auxiliary], &updateElements)) return NO;
    for (uint32_t stage = 0U; stage < stageCount; stage++) {
        SynaptikMetalProgramStep *step;
        if (hasOutput && (stage == 0U || updateElements != 0U)) {
            /* Select the controlled function before its only pipeline construction.
             * Missing controlled implementations remain hard admission failures.
             */
            step = make_custom_step(node, shapes, types, strides, offsets, states,
                    NULL, NULL, device, library, lowLibrary, stage, YES, control);
            if (step == nil) return NO;
            if (step.compositionSteps == nil && (step.pipeline == nil || step.metadata == nil
                    || step.grid.width == 0U || step.grid.height == 0U || step.grid.depth == 0U
                    || step.threadsPerThreadgroup == 0U)) return NO;
        } else {
            /* Genuine zero work, never a positive-grid stand-in. Stage zero still
             * carries complete validation for nonempty domains of empty results.
             */
            step = [SynaptikMetalProgramStep new];
            if (step == nil) return NO;
            step.custom = YES;
            step.operation = node.operation;
            step.stage = stage;
            step.firstInput = node.input_count == 0U ? node.output : node.first_input;
            step.secondInput = node.second_input;
            step.auxiliaryInput = node.auxiliary;
            step.output = node.output;
            step.scratchInput = UINT32_MAX;
            step.grid = MTLSizeMake(0U, 0U, 0U);
        }
        if (stage == 0U) step.scanValidationRoles = roles;
        [sink addObject:step];
    }
    return YES;
}

static int32_t synaptik_encode_scan_step(SynaptikMetalProgramStep *step,
        id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> __unsafe_unretained const *buffers,
        uint32_t bufferCount, const NSUInteger *offsets) {
    if (step == nil) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    MTLSize grid = step.grid;
    if (step.compositionSteps == nil && grid.width == 0U && grid.height == 0U && grid.depth == 0U)
        return SYNAPTIK_METAL_STATUS_OK;
    if (step.compositionSteps == nil && (step.pipeline == nil || step.metadata == nil || step.threadsPerThreadgroup == 0U
            || grid.width == 0U || grid.height == 0U || grid.depth == 0U))
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    if (encoder == nil || buffers == NULL || bufferCount == 0U || offsets == NULL)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;

    /* The caller already bound control at 28/29. Borrow current bindings on every dispatch:
     * carry promotion changes offsets between iterations, including scratch's workspace base.
     */
    SynaptikMetalStepBindings bindings = {bufferCount, NULL, nil, offsets, buffers};
    if (!synaptik_dispatch_fixed_step(step, encoder, bindings))
        return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
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
