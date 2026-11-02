#ifndef SYNAPTIK_RECURRENT_HOST_H
#define SYNAPTIK_RECURRENT_HOST_H

/* Host-only component: import after SynaptikMetalDecodedNode, value type/state
 * enums, SynaptikRecurrentMeta and custom_physical_layout are defined. This
 * header contributes no MSL source and creates no pipeline or buffer.
 *
 * These checks use actual native decoded node and physical representation
 * arrays, independently of Java provider/preparer answers. The outer image
 * parser still owns descriptor spans, gradient flags/OR, topology, ownership
 * and the actual bound byte extents; this API has no gradient/span arguments.
 */
static BOOL synaptik_recurrent_native_value(
        uint32_t value, NSArray<MPSShape *> *shapes,
        const uint8_t *types, const uint64_t *strides, const uint64_t *offsets,
        const uint8_t *states, uint8_t type, NSUInteger rank,
        const uint64_t *expected) {
    if (value >= shapes.count || types[value] != type
            || states[value] != SYNAPTIK_METAL_VALUE_CANONICAL
            || offsets[value] != 0U) return NO;
    MPSShape *shape = shapes[value];
    if (shape == nil || shape.count != rank || rank > SYNAPTIK_MAX_RANK) return NO;
    uint64_t canonical[SYNAPTIK_MAX_RANK] = {0};
    uint64_t elements = 1U;
    BOOL empty = NO;
    for (NSUInteger reverse = rank; reverse > 0U; --reverse) {
        NSUInteger axis = reverse - 1U;
        if (shape[axis].longLongValue < 0
                || shape[axis].unsignedLongLongValue != expected[axis]
                || expected[axis] > UINT32_MAX) return NO;
        canonical[axis] = elements;
        if (strides[(size_t)value * SYNAPTIK_MAX_RANK + axis] != elements) return NO;
        uint64_t extent = expected[axis];
        empty |= extent == 0U;
        if (extent != 0U && elements > (uint64_t)INT64_MAX / extent) return NO;
        elements *= extent;
    }
    uint64_t width = type == SYNAPTIK_METAL_TYPE_INT64 ? 8U : 4U;
    if (elements > (uint64_t)INT64_MAX / width) return NO;
    /* The shared physical helper's positive-extent domain is intentionally
     * unchanged. Empty recurrence still requires exact canonical stored strides,
     * zero offset and zero elements; there is no fabricated one-element tensor.
     */
    if (!empty) {
        uint64_t physical[SYNAPTIK_MAX_RANK] = {0};
        uint64_t offset = UINT64_MAX;
        if (!custom_physical_layout(shape, value, strides, offsets, states, physical, &offset)
                || offset != 0U) return NO;
        for (NSUInteger axis = 0U; axis < rank; ++axis)
            if (physical[axis] != canonical[axis]) return NO;
    }
    return YES;
}

/* Returns NO without touching meta/grid on malformed geometry. Successful
 * B=0 produces a zero-width grid: the caller must omit dispatch, not substitute
 * a positive batch. All present outputs require independent bound buffers.
 * Before ANY output write, the invocation owner must call
 * SynaptikValidateRecurrentLengths over the complete authenticated INT64 feed.
 */
static BOOL synaptik_recurrent_metadata(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes,
        const uint8_t *types, const uint64_t *strides, const uint64_t *offsets,
        const uint8_t *states, SynaptikRecurrentMeta *meta, MTLSize *grid) {
    if (shapes == nil || types == NULL || strides == NULL || offsets == NULL
            || states == NULL || meta == NULL || grid == NULL) return NO;
    uint32_t gates;
    switch (node.operation) {
        case 103U: gates = 1U; break; /* RNN_TANH */
        case 104U: gates = 3U; break; /* GRU_RESET_AFTER */
        case 105U: gates = 4U; break; /* LSTM */
        default: return NO;
    }
    uint32_t baseReads = gates == 4U ? 6U : 5U;
    if ((node.input_count != baseReads && node.input_count != baseReads + 1U)
            || node.output_count != (gates == 4U ? 3U : 2U)
            || node.attribute_kind != 35U || node.attribute_count != 1U
            || (node.attribute_values[0] != 1U && node.attribute_values[0] != 2U)
            || node.axis != UINT32_MAX
            || node.first_input != node.inputs[0]
            || node.second_input != node.inputs[1]
            || node.auxiliary != node.inputs[2]
            || node.output != node.outputs[0]) return NO;
    for (uint32_t i = 1U; i < SYNAPTIK_MAX_ATTRIBUTE_WORDS; ++i)
        if (node.attribute_values[i] != 0U) return NO;
    for (uint32_t i = 0U; i < node.input_count; ++i)
        if (node.inputs[i] >= shapes.count) return NO;
    for (uint32_t i = 0U; i < node.output_count; ++i) {
        uint32_t output = node.outputs[i];
        if (output >= shapes.count) return NO;
        for (uint32_t j = 0U; j < node.input_count; ++j)
            if (output == node.inputs[j]) return NO;
        for (uint32_t j = 0U; j < i; ++j)
            if (output == node.outputs[j]) return NO;
    }
    MPSShape *x = shapes[node.inputs[0]], *h = shapes[node.inputs[2]];
    if (x == nil || h == nil || x.count != 3U || h.count != 2U) return NO;
    uint64_t time = x[0].unsignedLongLongValue;
    uint64_t batch = x[1].unsignedLongLongValue;
    uint64_t input = x[2].unsignedLongLongValue;
    uint64_t hidden = h[1].unsignedLongLongValue;
    if (time > UINT32_MAX || batch > UINT32_MAX || input == 0U || input > UINT32_MAX
            || hidden == 0U || hidden > UINT32_MAX / gates) return NO;
    uint64_t packed = hidden * gates;
    uint64_t inputShape[3] = {time, batch, input};
    uint64_t lengthShape[1] = {batch};
    uint64_t stateShape[2] = {batch, hidden};
    uint64_t weightShape[2] = {packed, input};
    uint64_t hiddenWeightShape[2] = {packed, hidden};
    uint64_t biasShape[1] = {packed};
    uint64_t outputShape[3] = {time, batch, hidden};
    if (!synaptik_recurrent_native_value(node.inputs[0], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_FLOAT32, 3U, inputShape)
            || !synaptik_recurrent_native_value(node.inputs[1], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_INT64, 1U, lengthShape)
            || !synaptik_recurrent_native_value(node.inputs[2], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_FLOAT32, 2U, stateShape)) return NO;
    uint32_t weightRole = 3U;
    if (gates == 4U) {
        if (!synaptik_recurrent_native_value(node.inputs[3], shapes, types, strides, offsets, states,
                    SYNAPTIK_METAL_TYPE_FLOAT32, 2U, stateShape)) return NO;
        weightRole = 4U;
    }
    if (!synaptik_recurrent_native_value(node.inputs[weightRole], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_FLOAT32, 2U, weightShape)
            || !synaptik_recurrent_native_value(node.inputs[weightRole + 1U], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_FLOAT32, 2U, hiddenWeightShape)) return NO;
    if (node.input_count == baseReads + 1U
            && !synaptik_recurrent_native_value(node.inputs[weightRole + 2U], shapes, types, strides, offsets, states,
                    SYNAPTIK_METAL_TYPE_FLOAT32, 1U, biasShape)) return NO;
    if (!synaptik_recurrent_native_value(node.outputs[0], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_FLOAT32, 3U, outputShape)
            || !synaptik_recurrent_native_value(node.outputs[1], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_FLOAT32, 2U, stateShape)) return NO;
    if (gates == 4U && !synaptik_recurrent_native_value(node.outputs[2], shapes, types, strides, offsets, states,
                SYNAPTIK_METAL_TYPE_FLOAT32, 2U, stateShape)) return NO;
    SynaptikRecurrentMeta result = {
        .kind = node.operation, .time = (uint32_t)time, .batch = (uint32_t)batch,
        .inputSize = (uint32_t)input, .hiddenSize = (uint32_t)hidden,
        .reverse = node.attribute_values[0] == 2U ? 1U : 0U,
        .hasBias = node.input_count == baseReads + 1U ? 1U : 0U, .reserved = 0U
    };
    *meta = result;
    *grid = MTLSizeMake((NSUInteger)batch, 1U, 1U);
    return YES;
}

#endif
