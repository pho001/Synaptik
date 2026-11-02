#ifndef SYNAPTIK_PROGRAM_BINDINGS_H
#define SYNAPTIK_PROGRAM_BINDINGS_H

/* One fixed binding codec for ordinary prepared steps and the native Scan loop.
 * A direct invocation borrows its already validated native handle table in-place;
 * Scan borrows its prepared MTLBuffer table and explicit per-value byte offsets.
 * Neither path builds a per-step resource mirror or interprets operation mathematics.
 */
typedef struct {
    NSUInteger count;
    void *const *handles;
    __unsafe_unretained NSArray<id<MTLBuffer>> *buffers;
    const NSUInteger *offsets;
    __unsafe_unretained id<MTLBuffer> const *metalBuffers;
} SynaptikMetalStepBindings;

static id<MTLBuffer> synaptik_step_buffer(SynaptikMetalStepBindings bindings, NSUInteger value) {
    if (value >= bindings.count) return nil;
    return bindings.handles != NULL
            ? ((__bridge SynaptikMetalBufferBox *)bindings.handles[value]).buffer
            : bindings.metalBuffers != NULL ? bindings.metalBuffers[value] : bindings.buffers[value];
}

static BOOL synaptik_bind_step_buffer(
        id<MTLComputeCommandEncoder> encoder, SynaptikMetalStepBindings bindings,
        NSUInteger value, NSUInteger extra_offset, NSUInteger slot) {
    if (value >= bindings.count) return NO;
    id<MTLBuffer> buffer = synaptik_step_buffer(bindings, value);
    NSUInteger offset = bindings.offsets == NULL ? 0U : bindings.offsets[value];
    if (buffer == nil || extra_offset > NSUIntegerMax - offset
            || offset + extra_offset > buffer.length) return NO;
    [encoder setBuffer:buffer offset:offset + extra_offset atIndex:slot];
    return YES;
}

static BOOL synaptik_encode_fixed_step(
        SynaptikMetalProgramStep *step, id<MTLComputeCommandEncoder> encoder,
        SynaptikMetalStepBindings bindings) {
    if (step == nil || encoder == nil || step.pipeline == nil || step.metadata == nil) return NO;
    [encoder setComputePipelineState:step.pipeline];
    uint32_t operation = step.operation;
    NSUInteger first = step.firstInput, output = step.output;
#define SYNAPTIK_BIND(VALUE, SLOT) do { if (!synaptik_bind_step_buffer(encoder, bindings, (VALUE), 0U, (SLOT))) return NO; } while (0)
    if (operation >= 86U && operation <= 93U) {
        if (step.inputValues.count == 0U || step.inputValues.count > 5U
                || step.outputValues.count == 0U || step.outputValues.count > 5U) return NO;
        for (NSUInteger role = 0U; role < 5U; role++) {
            NSUInteger input = step.inputValues[MIN(role, step.inputValues.count - 1U)].unsignedIntegerValue;
            NSUInteger result = step.outputValues[MIN(role, step.outputValues.count - 1U)].unsignedIntegerValue;
            SYNAPTIK_BIND(input, role);
            SYNAPTIK_BIND(result, role + 5U);
        }
        [encoder setBuffer:step.metadata offset:0U atIndex:10U];
    } else if (operation == 35U) {
        if (step.inputValues.count < 3U || step.inputValues.count > 4U
                || step.outputValues.count < 1U || step.outputValues.count > 2U) return NO;
        for (NSUInteger role = 0U; role < 4U; role++) {
            NSUInteger input = step.inputValues[role < step.inputValues.count ? role : 0U].unsignedIntegerValue;
            SYNAPTIK_BIND(input, role);
        }
        SYNAPTIK_BIND(output, 4U);
        if (step.outputValues.count == 2U) {
            SYNAPTIK_BIND(step.outputValues[1].unsignedIntegerValue, 5U);
        } else if (step.scratchBytes != 0U) {
            if (!synaptik_bind_step_buffer(encoder, bindings, step.scratchInput,
                        step.scratchOffset, 5U)) return NO;
        } else {
            SYNAPTIK_BIND(output, 5U);
        }
        [encoder setBuffer:step.metadata offset:0U atIndex:6U];
    } else if (operation >= 103U && operation <= 105U) {
        BOOL lstm = operation == 105U;
        NSUInteger reads = lstm ? 6U : 5U;
        if ((step.inputValues.count != reads && step.inputValues.count != reads + 1U)
                || step.outputValues.count != (lstm ? 3U : 2U)) return NO;
        NSUInteger input_roles[7] = {0U, 1U, 2U, lstm ? 3U : 2U,
                lstm ? 4U : 3U, lstm ? 5U : 4U,
                step.inputValues.count == reads + 1U ? reads : (lstm ? 4U : 3U)};
        for (NSUInteger slot = 0U; slot < 7U; slot++)
            SYNAPTIK_BIND(step.inputValues[input_roles[slot]].unsignedIntegerValue, slot);
        SYNAPTIK_BIND(step.outputValues[0].unsignedIntegerValue, 7U);
        SYNAPTIK_BIND(step.outputValues[1].unsignedIntegerValue, 8U);
        SYNAPTIK_BIND(step.outputValues[lstm ? 2U : 1U].unsignedIntegerValue, 9U);
        [encoder setBuffer:step.metadata offset:0U atIndex:10U];
    } else {
        SYNAPTIK_BIND(first, 0U);
        BOOL scatter = operation == 70U
                || ((operation == 18U || operation == 72U) && (step.extended || step.stage == 1U));
        BOOL reduction_mask = step.extended && step.metadata.length == sizeof(SynaptikMetalDataMeta)
                && ((operation == 13U || operation == 14U || operation == 106U
                        || (operation >= 111U && operation <= 115U)))
                && ((const SynaptikMetalDataMeta *)step.metadata.contents)->sourceType == 1U;
        BOOL binary = (operation >= 2U && operation <= 5U)
                || (operation >= 20U && operation <= 27U) || operation == 38U
                || operation == 43U || operation == 44U || operation == 85U
                || operation == 15U || operation == 16U || operation == 69U
                || operation == 71U || operation == 76U;
        if (operation == 102U) {
            [encoder setBuffer:step.metadata offset:0U atIndex:1U];
        } else if (operation == 101U) {
            if (step.outputValues.count != 3U) return NO;
            SYNAPTIK_BIND(step.secondInput, 1U);
            SYNAPTIK_BIND(output, 2U);
            SYNAPTIK_BIND(step.outputValues[1].unsignedIntegerValue, 3U);
            SYNAPTIK_BIND(step.outputValues[2].unsignedIntegerValue, 4U);
            [encoder setBuffer:step.metadata offset:0U atIndex:5U];
        } else if (operation == 36U || operation == 37U) {
            SYNAPTIK_BIND(step.secondInput, 1U);
            SYNAPTIK_BIND(step.auxiliaryInput < bindings.count ? step.auxiliaryInput : first, 2U);
            SYNAPTIK_BIND(output, 3U);
            [encoder setBuffer:step.metadata offset:0U atIndex:4U];
        } else if ((operation >= 94U && operation <= 96U) || operation == 109U || operation == 110U) {
            SYNAPTIK_BIND(output, 1U);
            if (operation == 96U) {
                if (step.outputValues.count != 2U) return NO;
                SYNAPTIK_BIND(step.outputValues[1].unsignedIntegerValue, 2U);
                [encoder setBuffer:step.metadata offset:0U atIndex:3U];
            } else [encoder setBuffer:step.metadata offset:0U atIndex:2U];
        } else if (operation == 77U || operation == 78U) {
            if (step.inputValues.count == 0U || step.inputValues.count > 16U) return NO;
            for (NSUInteger slot = 0U; slot < 16U; slot++)
                SYNAPTIK_BIND(step.inputValues[MIN(slot, step.inputValues.count - 1U)].unsignedIntegerValue, slot);
            SYNAPTIK_BIND(output, 16U);
            [encoder setBuffer:step.metadata offset:0U atIndex:17U];
        } else if (scatter || operation == 51U) {
            SYNAPTIK_BIND(step.secondInput, 1U);
            SYNAPTIK_BIND(step.auxiliaryInput, 2U);
            SYNAPTIK_BIND(output, 3U);
            [encoder setBuffer:step.metadata offset:0U atIndex:4U];
        } else if (reduction_mask || binary) {
            SYNAPTIK_BIND(step.secondInput < bindings.count ? step.secondInput : first, 1U);
            SYNAPTIK_BIND(output, 2U);
            [encoder setBuffer:step.metadata offset:0U atIndex:3U];
        } else {
            SYNAPTIK_BIND(output, 1U);
            [encoder setBuffer:step.metadata offset:0U atIndex:2U];
        }
    }
#undef SYNAPTIK_BIND
    return YES;
}

#endif
