#ifndef SYNAPTIK_EXACT_HOST_H
#define SYNAPTIK_EXACT_HOST_H

/*
 * Include after DecodedNode/DataMeta, node_has_no_attributes,
 * custom_physical_layout, custom_grid, value_type_width and shape_element_count.
 * Reference indices/types/states must already belong to the authenticated value
 * table. Metadata independently rederives operation grammar, physical mapping,
 * shape and promotion. The raw parser additionally validates gradient flags and
 * scalar dtype words BEFORE decoding: DecodedNode does not retain either fact.
 * No public ABI or new metadata layout. Outputs must be canonical.
 */
static BOOL synaptik_exact_integral(uint8_t type) {
    return type == SYNAPTIK_METAL_TYPE_INT32 || type == SYNAPTIK_METAL_TYPE_INT64;
}

static BOOL synaptik_exact_unary(uint32_t operation) {
    return operation == 1U || operation == 12U || (operation >= 60U && operation <= 63U);
}

static BOOL synaptik_exact_binary(uint32_t operation) {
    return operation == 2U || operation == 3U || operation == 4U
            || (operation >= 20U && operation <= 27U);
}

static BOOL synaptik_exact_scalar(uint32_t operation) {
    return (operation >= 28U && operation <= 30U)
            || (operation >= 46U && operation <= 48U);
}

static BOOL synaptik_exact_handles(SynaptikMetalDecodedNode node, const uint8_t *types) {
    if (types == NULL || node.input_count == 0U || node.first_input == UINT32_MAX) return NO;
    if (!synaptik_exact_unary(node.operation) && !synaptik_exact_binary(node.operation)
            && !synaptik_exact_scalar(node.operation)) return NO;
    uint8_t left = types[node.first_input];
    if (node.operation >= 20U && node.operation <= 25U && node.input_count == 2U
            && node.second_input != UINT32_MAX && task0066_is_floating(left)
            && task0066_is_floating(types[node.second_input])) return YES;
    if (left == SYNAPTIK_METAL_TYPE_FLOAT64 || synaptik_exact_integral(left)) return YES;
    return synaptik_exact_binary(node.operation) && node.input_count == 2U
            && node.second_input != UINT32_MAX
            && (types[node.second_input] == SYNAPTIK_METAL_TYPE_FLOAT64
                    || synaptik_exact_integral(types[node.second_input]));
}

static BOOL synaptik_exact_physical(
        MPSShape *shape, uint32_t value, uint8_t type, BOOL output,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        uint64_t physical[16], uint64_t *offset) {
    if (shape == nil || shape.count > SYNAPTIK_MAX_RANK
            || !custom_physical_layout(shape, value, strides, offsets, states, physical, offset)) return NO;
    if (*offset > INT64_MAX) return NO;
    uint64_t span = *offset, count = 1U, canonical = 1U;
    BOOL zero = NO;
    NSUInteger order[16], order_count = 0U;
    for (NSUInteger reverse = 0U; reverse < shape.count; reverse++) {
        NSUInteger axis = shape.count - 1U - reverse;
        uint64_t dim = shape[axis].unsignedLongLongValue, stride = physical[axis];
        if (dim == 0U || dim > INT64_MAX || stride > INT64_MAX || count > INT64_MAX / dim) return NO;
        count *= dim;
        if (output && stride != canonical) return NO;
        canonical *= dim;
        zero |= stride == 0U;
        if (dim - 1U != 0U && stride > (INT64_MAX - span) / (dim - 1U)) return NO;
        span += (dim - 1U) * stride;
        if (dim > 1U && stride != 0U) {
            NSUInteger insertion = order_count;
            while (insertion > 0U && physical[order[insertion - 1U]] > stride) {
                order[insertion] = order[insertion - 1U]; insertion--;
            }
            order[insertion] = axis; order_count++;
        }
    }
    if (span >= INT64_MAX || value_type_width(type) == 0U
            || span + 1U > INT64_MAX / value_type_width(type)) return NO;
    if (output) return *offset == 0U;
    if (zero && (states == NULL || states[value] != SYNAPTIK_METAL_VALUE_AFFINE_VIEW)) return NO;
    uint64_t covered = 1U;
    for (NSUInteger i = 0U; i < order_count; i++) {
        NSUInteger axis = order[i];
        if (physical[axis] < covered) return NO;
        covered += (shape[axis].unsignedLongLongValue - 1U) * physical[axis];
    }
    return YES;
}

static BOOL synaptik_exact_f64_nan(uint64_t word) {
    return (word & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000)
            && (word & UINT64_C(0x000fffffffffffff)) != 0U;
}

static BOOL synaptik_exact_clamp_ordered(uint64_t lower, uint64_t upper) {
    if (synaptik_exact_f64_nan(lower) || synaptik_exact_f64_nan(upper)) return YES;
    if ((lower & UINT64_C(0x7fffffffffffffff)) == 0U
            && (upper & UINT64_C(0x7fffffffffffffff)) == 0U) return YES;
    uint64_t left_key = (lower >> 63U) != 0U ? ~lower : lower ^ UINT64_C(0x8000000000000000);
    uint64_t right_key = (upper >> 63U) != 0U ? ~upper : upper ^ UINT64_C(0x8000000000000000);
    return left_key <= right_key;
}

static BOOL synaptik_exact_metadata(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes, const uint8_t *types,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        SynaptikMetalDataMeta *meta, MTLSize *grid) {
    if (meta == NULL || grid == NULL || shapes == nil || types == NULL
            || node.first_input >= shapes.count || node.output >= shapes.count
            || !synaptik_exact_handles(node, types) || node.output_count != 1U
            || node.outputs[0] != node.output || node.inputs[0] != node.first_input) return NO;
    BOOL binary = synaptik_exact_binary(node.operation), unary = synaptik_exact_unary(node.operation);
    if (node.input_count != (binary ? 2U : 1U)
            || (!binary && node.second_input != UINT32_MAX)
            || (binary && (node.second_input >= shapes.count || node.inputs[1] != node.second_input))) return NO;
    MPSShape *left = shapes[node.first_input], *output = shapes[node.output];
    uint8_t left_type = types[node.first_input], target = types[node.output];
    memset(meta, 0, sizeof(*meta));
    meta->operation = node.operation;
    meta->inputRank = (uint32_t)left.count; meta->outputRank = (uint32_t)output.count;
    meta->sourceType = left_type; meta->targetType = target; meta->reserved = left_type;
    meta->width = (uint32_t)value_type_width(left_type); meta->inputCount = node.input_count;
    meta->attributeKind = node.attribute_kind; meta->attributeCount = node.attribute_count;
    if (!synaptik_exact_physical(left, node.first_input, left_type, NO, strides, offsets, states,
                meta->inputStrides, &meta->inputOffset)
            || !synaptik_exact_physical(output, node.output, target, YES, strides, offsets, states,
                meta->outputStrides, &meta->outputOffset)) return NO;
    for (NSUInteger axis = 0U; axis < left.count; axis++) meta->inputDims[axis] = left[axis].unsignedLongLongValue;
    for (NSUInteger axis = 0U; axis < output.count; axis++) meta->outputDims[axis] = output[axis].unsignedLongLongValue;
    if (unary) {
        if (!node_has_no_attributes(node) || left_type != SYNAPTIK_METAL_TYPE_FLOAT64
                || target != left_type || ![left isEqualToArray:output]) return NO;
    } else if (binary) {
        MPSShape *right = shapes[node.second_input];
        uint8_t right_type = types[node.second_input];
        BOOL integral = synaptik_exact_integral(left_type) && synaptik_exact_integral(right_type);
        BOOL floating = task0066_is_floating(left_type) && task0066_is_floating(right_type);
        BOOL extreme = node.operation == 26U || node.operation == 27U;
        BOOL comparison = node.operation >= 20U && node.operation <= 25U;
        uint8_t promoted = integral
                ? (left_type == SYNAPTIK_METAL_TYPE_INT64 || right_type == SYNAPTIK_METAL_TYPE_INT64
                        ? SYNAPTIK_METAL_TYPE_INT64 : SYNAPTIK_METAL_TYPE_INT32)
                : SYNAPTIK_METAL_TYPE_FLOAT64;
        if (!node_has_no_attributes(node) || (!integral && !(floating
                    && (comparison || (extreme && (left_type == SYNAPTIK_METAL_TYPE_FLOAT64
                            || right_type == SYNAPTIK_METAL_TYPE_FLOAT64)))))
                || target != (comparison ? SYNAPTIK_METAL_TYPE_BOOL : promoted)
                || output.count != MAX(left.count, right.count)) return NO;
        meta->auxiliaryRank = (uint32_t)right.count; meta->reserved = right_type;
        if (!synaptik_exact_physical(right, node.second_input, right_type, NO, strides, offsets, states,
                    meta->auxiliaryStrides, &meta->auxiliaryOffset)) return NO;
        for (NSUInteger axis = 0U; axis < right.count; axis++) meta->auxiliaryDims[axis] = right[axis].unsignedLongLongValue;
        for (NSUInteger axis = 0U; axis < output.count; axis++) {
            uint64_t a = axis < output.count - left.count ? 1U
                    : left[axis - (output.count - left.count)].unsignedLongLongValue;
            uint64_t b = axis < output.count - right.count ? 1U
                    : right[axis - (output.count - right.count)].unsignedLongLongValue;
            if ((a != b && a != 1U && b != 1U) || output[axis].unsignedLongLongValue != MAX(a, b)) return NO;
        }
    } else {
        BOOL clamp = node.operation == 30U, extreme = node.operation >= 28U && node.operation <= 30U;
        if ((!synaptik_exact_integral(left_type) && !(extreme && left_type == SYNAPTIK_METAL_TYPE_FLOAT64))
                || (clamp && left_type != SYNAPTIK_METAL_TYPE_FLOAT64)
                || target != left_type || ![left isEqualToArray:output]
                || node.attribute_kind != (clamp ? 8U : 7U) || node.attribute_count != (clamp ? 2U : 1U)
                || node.axis != UINT32_MAX || node.auxiliary != 0U) return NO;
        for (uint32_t i = 0U; i < node.attribute_count; i++) {
            if (left_type == SYNAPTIK_METAL_TYPE_INT32 && node.attribute_values[i] > UINT32_MAX) return NO;
            meta->attrs[i] = node.attribute_values[i];
        }
        if (clamp && !synaptik_exact_clamp_ordered(meta->attrs[0], meta->attrs[1])) return NO;
        for (uint32_t i = node.attribute_count; i < SYNAPTIK_MAX_ATTRIBUTE_WORDS; i++)
            if (node.attribute_values[i] != 0U) return NO;
    }
    meta->elementCount = shape_element_count(output);
    return custom_grid(meta->elementCount, grid, &meta->gridWidth, &meta->gridHeight);
}

#endif
