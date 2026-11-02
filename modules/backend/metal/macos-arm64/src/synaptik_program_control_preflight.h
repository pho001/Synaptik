#ifndef SYNAPTIK_PROGRAM_CONTROL_PREFLIGHT_H
#define SYNAPTIK_PROGRAM_CONTROL_PREFLIGHT_H

/* A feed control is checked before the first partition write. A produced control is checked
 * only after its producer's command completes, immediately before its consuming operation.
 * This bridge borrows the authenticated native table; no per-run argument mirror is allocated. */
static int32_t synaptik_preflight_program_control(
        SynaptikMetalProgramStep *step, uint32_t count, void *const *handles) {
    if (step == nil || handles == NULL || !step.validatesControl) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    NSUInteger role = step.operation == 116U ? step.inputValues.count - 1U : 1U;
    if (role >= step.inputValues.count) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    NSUInteger value = step.inputValues[role].unsignedIntegerValue;
    if (value >= count || handles[value] == NULL) return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    SynaptikMetalBufferBox *input = (__bridge SynaptikMetalBufferBox *)handles[value];
    if (step.operation == 116U)
        return synaptik_scan_preflight_lengths(step.scanPlan, input.buffer, NULL);
    if (step.operation == 87U) {
        if (step.metadata.length != sizeof(SynaptikNormalizationLossMeta) || step.metadata.contents == NULL)
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        return synaptik_normalization_preflight_target(step.metadata.contents,
                input.buffer.contents, input.logicalByteSize, 0U, NULL, NULL);
    }
    if (step.operation >= 103U && step.operation <= 105U) {
        if (step.metadata.length != sizeof(SynaptikRecurrentMeta) || step.metadata.contents == NULL)
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        const SynaptikRecurrentMeta *meta = step.metadata.contents;
        uint64_t bytes = (uint64_t)meta->batch * sizeof(int64_t);
        if (bytes > input.logicalByteSize)
            return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
        return SynaptikValidateRecurrentLengths(input.buffer.contents, (size_t)bytes, meta, NULL)
                ? SYNAPTIK_METAL_STATUS_OK : SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
    }
    return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
}

static BOOL synaptik_step_can_batch(SynaptikMetalProgramStep *step) {
    return step.custom && step.scanPlan == nil && step.indexValidation == nil
            && (!step.validatesControl || step.preflightInputs)
            && step.grid.width != 0U && step.grid.height != 0U && step.grid.depth != 0U;
}

#endif
