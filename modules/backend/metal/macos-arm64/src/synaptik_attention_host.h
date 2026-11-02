#ifndef SYNAPTIK_ATTENTION_HOST_H
#define SYNAPTIK_ATTENTION_HOST_H

/**
 * Import after SynaptikMetalDecodedNode, SynaptikAttentionMeta,
 * custom_physical_layout and custom_grid. No shader source or ABI export.
 * This independently checks native occurrence/shape/carrier/physical arithmetic;
 * the authenticated decoder retains responsibility for raw descriptor gradient
 * flags and logical layout identity, which are not arguments to this helper.
 * Canonical BOOL bytes must additionally be preflighted before any dispatch write.
 */
static BOOL synaptik_attention_product(uint64_t left, uint64_t right, uint64_t *result) {
    if (right != 0U && left > (uint64_t)INT64_MAX / right) return NO;
    *result = left * right;
    return YES;
}

static BOOL synaptik_attention_layout(
        MPSShape *shape, uint32_t value, const uint64_t *strides,
        const uint64_t *offsets, const uint8_t *states, uint32_t width,
        uint64_t physical[16], uint64_t *offset, BOOL canonical) {
    if (shape == nil || shape.count > 16U || strides == NULL || offsets == NULL || states == NULL)
        return NO;
    uint64_t count = 1U, canonicalStrides[16] = {0};
    BOOL empty = NO;
    for (NSUInteger reverse = 0U; reverse < shape.count; ++reverse) {
        NSUInteger axis = shape.count - 1U - reverse;
        if (shape[axis].longLongValue < 0) return NO;
        uint64_t extent = shape[axis].unsignedLongLongValue;
        canonicalStrides[axis] = count;
        empty |= extent == 0U;
        if (!synaptik_attention_product(count, extent, &count)) return NO;
    }
    uint64_t bytes;
    if (!synaptik_attention_product(count, width, &bytes)) return NO;
    if (states[value] != SYNAPTIK_METAL_VALUE_CANONICAL
            && states[value] != SYNAPTIK_METAL_VALUE_AFFINE_VIEW
            && states[value] != SYNAPTIK_METAL_VALUE_MATERIALIZED_LAYOUT) return NO;
    if (canonical && states[value] != SYNAPTIK_METAL_VALUE_CANONICAL) return NO;
    if (states[value] == SYNAPTIK_METAL_VALUE_CANONICAL) {
        if (offsets[value] != 0U) return NO;
        for (NSUInteger axis = 0U; axis < shape.count; ++axis)
            if (strides[(size_t)value * SYNAPTIK_MAX_RANK + axis] != canonicalStrides[axis]) return NO;
    }
    // The common layout helper's historical positive-extent guard must not turn
    // an exact canonical empty attention value into an unsupported occurrence.
    if (empty) {
        if (states[value] != SYNAPTIK_METAL_VALUE_CANONICAL || offsets[value] != 0U) return NO;
        memcpy(physical, canonicalStrides, sizeof(canonicalStrides));
        *offset = 0U;
        return YES;
    }
    if (!custom_physical_layout(shape, value, strides, offsets, states, physical, offset)) return NO;
    uint64_t span = *offset;
    if (span > (uint64_t)INT64_MAX) return NO;
    for (NSUInteger axis = 0U; axis < shape.count; ++axis) {
        uint64_t delta;
        if (physical[axis] > (uint64_t)INT64_MAX
                || !synaptik_attention_product(shape[axis].unsignedLongLongValue - 1U,
                        physical[axis], &delta)
                || span > (uint64_t)INT64_MAX - delta) return NO;
        span += delta;
    }
    if (span == (uint64_t)INT64_MAX || !synaptik_attention_product(span + 1U, width, &bytes)) return NO;
    uint32_t rank = (uint32_t)shape.count;
    uint64_t dimensions[16] = {0};
    for (uint32_t axis = 0U; axis < rank; ++axis) dimensions[axis] = shape[axis].unsignedLongLongValue;
    uint64_t referencedSpan = span + 1U;
    uint8_t present = 1U;
    // Raw descriptor view authorization is checked by the authenticated decoder;
    // this compact physical seam independently proves the addressed base layout.
    return synaptik_storage_layout_supported(0U, &rank, dimensions, physical,
            offset, &referencedSpan, &present, &present, 0U);
}

static BOOL synaptik_attention_metadata(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes,
        const uint8_t *types, const uint64_t *strides, const uint64_t *offsets,
        const uint8_t *states, SynaptikAttentionMeta *meta, MTLSize *grid,
        NSUInteger *workspaceBytes) {
    if (meta == NULL || grid == NULL || workspaceBytes == NULL || shapes == nil || types == NULL
            || node.operation != 35U || node.attribute_kind != 39U
            || node.input_count < 3U || node.input_count > 4U
            || node.output_count < 1U || node.output_count > 2U
            || node.first_input != node.inputs[0] || node.second_input != node.inputs[1]
            || node.auxiliary != node.inputs[2] || node.output != node.outputs[0]
            || node.axis != UINT32_MAX) return NO;
    SynaptikAttentionMeta m = {0};
    uint64_t present = node.attribute_values[0], causal;
    if (present == 0U && node.attribute_count == 2U) {
        causal = node.attribute_values[1];
    } else if (present == 1U && node.attribute_count == 4U
            && node.attribute_values[1] == SYNAPTIK_METAL_TYPE_FLOAT32
            && node.attribute_values[2] <= UINT32_MAX) {
        uint32_t bits = (uint32_t)node.attribute_values[2];
        memcpy(&m.scale, &bits, sizeof(bits));
        if (!isfinite(m.scale) || m.scale <= 0.0f) return NO;
        causal = node.attribute_values[3];
        m.flags |= 4U;
    } else return NO;
    if (causal > 1U) return NO;
    for (uint32_t word = node.attribute_count; word < SYNAPTIK_MAX_ATTRIBUTE_WORDS; ++word)
        if (node.attribute_values[word] != 0U) return NO;
    if (causal != 0U) m.flags |= 2U;
    if (node.input_count == 4U) m.flags |= 1U;
    for (uint32_t input = 0U; input < node.input_count; ++input) {
        uint32_t index = node.inputs[input];
        if (index >= shapes.count || types[index] != (input == 3U
                ? SYNAPTIK_METAL_TYPE_BOOL : SYNAPTIK_METAL_TYPE_FLOAT32)) return NO;
    }
    for (uint32_t output = 0U; output < node.output_count; ++output) {
        uint32_t index = node.outputs[output];
        if (index >= shapes.count || types[index] != SYNAPTIK_METAL_TYPE_FLOAT32) return NO;
        for (uint32_t input = 0U; input < node.input_count; ++input)
            if (index == node.inputs[input]) return NO;
        if (output != 0U && index == node.outputs[0]) return NO;
    }
    MPSShape *q = shapes[node.inputs[0]], *k = shapes[node.inputs[1]], *v = shapes[node.inputs[2]];
    if (q.count < 2U || k.count < 2U || v.count < 2U
            || q.count > 16U || k.count > 16U || v.count > 16U) return NO;
    m.embedding = q[q.count - 1U].unsignedLongLongValue;
    m.queryLength = q[q.count - 2U].unsignedLongLongValue;
    m.keyLength = k[k.count - 2U].unsignedLongLongValue;
    m.valueEmbedding = v[v.count - 1U].unsignedLongLongValue;
    if (m.embedding == 0U || m.embedding != k[k.count - 1U].unsignedLongLongValue
            || m.keyLength != v[v.count - 2U].unsignedLongLongValue) return NO;
    m.batchRank = (uint32_t)MAX(q.count, MAX(k.count, v.count)) - 2U;
    for (uint32_t axis = 0U; axis < m.batchRank; ++axis) m.batchDims[axis] = 1U;
    for (uint32_t role = 0U; role < 3U; ++role) {
        MPSShape *shape = shapes[node.inputs[role]];
        NSUInteger batchRank = shape.count - 2U;
        for (NSUInteger axis = 0U; axis < batchRank; ++axis) {
            uint32_t target = m.batchRank - (uint32_t)batchRank + (uint32_t)axis;
            uint64_t extent = shape[axis].unsignedLongLongValue;
            if (m.batchDims[target] == 1U) m.batchDims[target] = extent;
            else if (extent != 1U && extent != m.batchDims[target]) return NO;
        }
    }
    uint64_t batchCount = 1U;
    for (uint32_t axis = 0U; axis < m.batchRank; ++axis)
        if (!synaptik_attention_product(batchCount, m.batchDims[axis], &batchCount)) return NO;
    uint64_t workspace, outputCount;
    if (!synaptik_attention_product(batchCount, m.queryLength, &m.rowCount)
            || !synaptik_attention_product(m.rowCount, m.keyLength, &workspace)
            || !synaptik_attention_product(workspace, 4U, &workspace)
            || !synaptik_attention_product(m.rowCount, m.valueEmbedding, &outputCount)
            || !synaptik_attention_product(outputCount, 4U, &outputCount)) return NO;
    for (uint32_t port = 0U; port < node.output_count; ++port) {
        uint32_t index = node.outputs[port];
        MPSShape *shape = shapes[index];
        if (shape.count != m.batchRank + 2U) return NO;
        for (uint32_t axis = 0U; axis < m.batchRank; ++axis)
            if (shape[axis].unsignedLongLongValue != m.batchDims[axis]) return NO;
        if (shape[m.batchRank].unsignedLongLongValue != m.queryLength
                || shape[m.batchRank + 1U].unsignedLongLongValue
                    != (port == 0U ? m.valueEmbedding : m.keyLength)) return NO;
        uint64_t physical[16] = {0}, offset;
        if (!synaptik_attention_layout(shape, index, strides, offsets, states, 4U, physical, &offset, YES))
            return NO;
    }
    uint64_t physical[4][16] = {{0}};
    for (uint32_t role = 0U; role < node.input_count; ++role) {
        uint32_t index = node.inputs[role];
        MPSShape *shape = shapes[index];
        if (!synaptik_attention_layout(shape, index, strides, offsets, states,
                role == 3U ? 1U : 4U, physical[role], &m.roleOffsets[role], NO)) return NO;
        if (role < 3U) {
            NSUInteger batchRank = shape.count - 2U;
            for (NSUInteger axis = 0U; axis < batchRank; ++axis) {
                uint32_t target = m.batchRank - (uint32_t)batchRank + (uint32_t)axis;
                m.roleBatchStrides[role][target] = shape[axis].unsignedLongLongValue == 1U
                        ? 0U : physical[role][axis];
            }
        } else {
            if (shape.count > m.batchRank + 2U) return NO;
            uint32_t padding = m.batchRank + 2U - (uint32_t)shape.count;
            for (NSUInteger axis = 0U; axis < shape.count; ++axis) {
                uint32_t target = padding + (uint32_t)axis;
                uint64_t extent = shape[axis].unsignedLongLongValue;
                uint64_t expected = target < m.batchRank ? m.batchDims[target]
                        : target == m.batchRank ? m.queryLength : m.keyLength;
                if (extent != 1U && extent != expected) return NO;
                uint64_t stride = extent == 1U ? 0U : physical[role][axis];
                if (target < m.batchRank) m.roleBatchStrides[role][target] = stride;
                else if (target == m.batchRank) m.maskQueryStride = stride;
                else m.maskKeyStride = stride;
            }
        }
    }
    m.queryStride = physical[0][q.count - 2U];
    m.queryEmbeddingStride = physical[0][q.count - 1U];
    m.keyStride = physical[1][k.count - 2U];
    m.keyEmbeddingStride = physical[1][k.count - 1U];
    m.valueStride = physical[2][v.count - 2U];
    m.valueEmbeddingStride = physical[2][v.count - 1U];
    if (m.rowCount == 0U) {
        *grid = MTLSizeMake(0U, 0U, 0U);
    } else if (!custom_grid(m.rowCount, grid, &m.gridWidth, &m.gridHeight)) return NO;
    *workspaceBytes = (NSUInteger)workspace;
    *meta = m;
    return YES;
}

#endif
