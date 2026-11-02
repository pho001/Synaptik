#ifndef SYNAPTIK_SCATTER_HOST_H
#define SYNAPTIK_SCATTER_HOST_H

/* Include after the decoded node/DataMeta PODs, custom_physical_layout and custom_grid.
 * Raw program gradient flags are independently checked by the native image decoder:
 * index has no gradient, output = base OR update; integral/F64 numerical rows have none.
 * This constructor has only actual decoded physical facts, never a provider assertion.
 */
static BOOL synaptik_scatter_handles(SynaptikMetalDecodedNode node, const uint8_t *types) {
    if (types == NULL || node.input_count != 3U || node.output_count != 1U
            || (node.operation != 18U && node.operation != 70U && node.operation != 72U))
        return NO;
    uint32_t type = types[node.inputs[0]];
    uint64_t reduction = node.operation == 70U ? 2U
            : node.attribute_count == 2U ? node.attribute_values[1] : 0U;
    return type == 1U || ((type == 2U || type == 6U) && reduction >= 2U)
            || (type == 4U && reduction >= 4U);
}

/* The same bounded affine READ domain as Java: broadcast views are legal;
 * nonbroadcast positive strides use the existing sorted-stride sufficient proof.
 * All span calculations are checked before multiplication and fit Metal's retained
 * unsigned-32-bit byte-resource bound. No logical-to-canonical layout substitution.
 */
static BOOL synaptik_scatter_read_layout(
        MPSShape *shape, uint32_t value, uint32_t width,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        uint64_t result[16], uint64_t *offset, uint64_t *count) {
    if (shape == nil || shape.count > 16U || width == 0U || count == NULL
            || !custom_physical_layout(shape, value, strides, offsets, states, result, offset))
        return NO;
    uint64_t limit = UINT32_MAX / width, elements = 1U, span = 1U;
    BOOL broadcast = NO;
    uint32_t order[16] = {0}, axes = 0U;
    for (NSUInteger axis = 0U; axis < shape.count; ++axis) {
        uint64_t extent = shape[axis].unsignedLongLongValue;
        if (extent == 0U || extent > limit || elements > limit / extent
                || result[axis] > INT64_MAX)
            return NO;
        elements *= extent;
        if (result[axis] == 0U) broadcast = YES;
        if (extent > 1U) {
            uint64_t steps = extent - 1U;
            if (result[axis] > (limit - span) / steps) return NO;
            span += steps * result[axis];
            uint32_t place = axes;
            while (place > 0U && result[order[place - 1U]] > result[axis]) {
                order[place] = order[place - 1U]; --place;
            }
            order[place] = (uint32_t)axis; ++axes;
        }
    }
    if (*offset > limit - span) return NO;
    if (broadcast) {
        if (states == NULL || states[value] != SYNAPTIK_METAL_VALUE_AFFINE_VIEW) return NO;
    } else {
        uint64_t covered = 1U;
        for (uint32_t item = 0U; item < axes; ++item) {
            uint32_t axis = order[item];
            if (result[axis] < covered) return NO;
            covered += (shape[axis].unsignedLongLongValue - 1U) * result[axis];
        }
    }
    *count = elements;
    return YES;
}

static BOOL synaptik_scatter_metadata(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes, const uint8_t *types,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        SynaptikMetalDataMeta *metadata, MTLSize *grid) {
    if (metadata == NULL || grid == NULL || shapes == nil || types == NULL
            || node.input_count != 3U || node.output_count != 1U)
        return NO;
    for (uint32_t role = 0U; role < 3U; ++role)
        if (node.inputs[role] >= shapes.count || node.inputs[role] == node.outputs[0]) return NO;
    if (node.outputs[0] >= shapes.count || node.first_input != node.inputs[0]
            || node.second_input != node.inputs[1] || node.auxiliary != node.inputs[2]
            || node.output != node.outputs[0] || !synaptik_scatter_handles(node, types))
        return NO;
    BOOL add = node.operation == 70U, nd = node.operation == 72U;
    uint32_t attribute = add ? 3U : nd ? 13U : 12U;
    uint32_t words = add ? 1U : 2U;
    if (node.attribute_kind != attribute || node.attribute_count != words) return NO;
    for (uint32_t word = words; word < SYNAPTIK_MAX_ATTRIBUTE_WORDS; ++word)
        if (node.attribute_values[word] != 0U) return NO;
    uint64_t axis = add ? node.axis : node.attribute_values[0];
    uint64_t reduction = add ? 2U : node.attribute_values[1];
    if (axis > 16U || reduction < 1U || reduction > 5U
            || (!nd && node.axis != axis)) return NO;
    uint32_t type = types[node.first_input], index_type = types[node.second_input];
    if ((index_type != 2U && index_type != 6U) || types[node.auxiliary] != type
            || types[node.output] != type || (type == 4U && reduction < 4U)) return NO;
    MPSShape *data = shapes[node.first_input], *indices = shapes[node.second_input];
    MPSShape *updates = shapes[node.auxiliary], *output = shapes[node.output];
    if (data == nil || indices == nil || updates == nil || output == nil
            || data.count < 1U || data.count > 16U || indices.count > 16U
            || updates.count > 16U || output.count > 16U
            || ![data isEqualToArray:output]) return NO;
    if (nd) {
        if (indices.count < 1U || axis >= indices.count || axis > data.count) return NO;
        uint64_t depth = indices[indices.count - 1U].unsignedLongLongValue;
        if (depth < 1U || depth > data.count - axis
                || updates.count != indices.count - 1U + data.count - axis - depth) return NO;
        for (NSUInteger a = 0U; a < axis; ++a)
            if (indices[a].unsignedLongLongValue != data[a].unsignedLongLongValue) return NO;
        for (NSUInteger a = 0U; a + 1U < indices.count; ++a)
            if (updates[a].unsignedLongLongValue != indices[a].unsignedLongLongValue) return NO;
        for (NSUInteger a = axis + depth; a < data.count; ++a)
            if (updates[indices.count - 1U + a - axis - depth].unsignedLongLongValue
                    != data[a].unsignedLongLongValue) return NO;
    } else if (add) {
        if (axis >= data.count || updates.count != data.count - 1U + indices.count) return NO;
        for (NSUInteger a = 0U; a < axis; ++a)
            if (updates[a].unsignedLongLongValue != data[a].unsignedLongLongValue) return NO;
        for (NSUInteger a = 0U; a < indices.count; ++a)
            if (updates[axis + a].unsignedLongLongValue != indices[a].unsignedLongLongValue) return NO;
        for (NSUInteger a = axis + 1U; a < data.count; ++a)
            if (updates[a - 1U + indices.count].unsignedLongLongValue != data[a].unsignedLongLongValue) return NO;
    } else {
        if (axis >= data.count || indices.count != data.count || ![indices isEqualToArray:updates]) return NO;
        for (NSUInteger a = 0U; a < data.count; ++a)
            if (a != axis && updates[a].unsignedLongLongValue != data[a].unsignedLongLongValue) return NO;
    }
    SynaptikMetalDataMeta result = {0};
    result.operation = node.operation;
    result.inputRank = (uint32_t)data.count;
    result.outputRank = (uint32_t)output.count;
    result.auxiliaryRank = (uint32_t)indices.count;
    result.sourceType = type;
    result.targetType = type;
    result.indexType = index_type;
    result.width = type == 1U || type == 2U ? 4U : 8U;
    result.inputCount = 3U;
    result.attributeKind = attribute;
    result.attributeCount = words;
    result.attrs[0] = axis;
    result.attrs[1] = reduction;
    for (uint32_t role = 0U; role < 3U; ++role) {
        uint32_t value = node.inputs[role];
        MPSShape *shape = shapes[value];
        uint64_t ignored;
        uint32_t width = role == 1U ? (index_type == 2U ? 4U : 8U) : result.width;
        if (!synaptik_scatter_read_layout(shape, value, width, strides, offsets, states,
                    result.roleStrides[role], &result.roleOffsets[role], &ignored)) return NO;
        result.roleRanks[role] = (uint32_t)shape.count;
        for (NSUInteger a = 0U; a < shape.count; ++a)
            result.roleDims[role][a] = shape[a].unsignedLongLongValue;
    }
    if (!synaptik_scatter_read_layout(output, node.output, result.width, strides, offsets, states,
                result.outputStrides, &result.outputOffset, &result.elementCount)
            || result.outputOffset != 0U) return NO;
    uint64_t expected = 1U;
    for (NSUInteger reverse = 0U; reverse < output.count; ++reverse) {
        NSUInteger axis_index = output.count - 1U - reverse;
        if (result.outputStrides[axis_index] != expected) return NO;
        result.outputDims[axis_index] = output[axis_index].unsignedLongLongValue;
        expected *= result.outputDims[axis_index];
    }
    if (!custom_grid(result.elementCount, grid, &result.gridWidth, &result.gridHeight)) return NO;
    *metadata = result;
    return YES;
}

#endif
