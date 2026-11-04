#ifndef SYNAPTIK_REDUCTION_HOST_H
#define SYNAPTIK_REDUCTION_HOST_H

/* Host-only cold construction. Include after decoded node/DataMeta types, existing
 * node_reduction_axes/node_statistical_axes, custom_grid and custom_physical_layout.
 * The raw envelope's common storage-layout owner has already authenticated references,
 * original view flags (including zero-stride => view), nonbroadcast-axis injectivity,
 * declared spans and BOOL storage. This owner independently checks numerical grammar
 * and actual affine byte bounds before metadata creation; it must not infer an original
 * view flag from a derived physical state or duplicate the shared layout validator.
 */
static BOOL synaptik_reduction_handles(SynaptikMetalDecodedNode node, const uint8_t *types) {
    if (types == NULL || node.input_count == 0U || node.first_input == UINT32_MAX) return NO;
    uint8_t type = types[node.first_input];
    BOOL numeric = type == 1U || type == 5U || type == 7U;
    if (node.operation == 13U)
        return numeric || type == 2U || type == 6U;
    if (node.operation == 31U || node.operation == 32U)
        return numeric || type == 2U || type == 4U || type == 6U;
    return numeric && (node.operation == 14U || node.operation == 106U
            || (node.operation >= 111U && node.operation <= 115U));
}

static BOOL synaptik_reduction_span(MPSShape *shape, const uint64_t *strides,
        uint64_t offset, uint32_t width) {
    if (width == 0U || offset == UINT64_MAX) return NO;
    for (NSNumber *extent in shape)
        if (extent.unsignedLongLongValue == 0U) return YES;
    uint64_t span = offset + 1U;
    for (NSUInteger axis = 0U; axis < shape.count; axis++) {
        uint64_t extent = shape[axis].unsignedLongLongValue;
        if (extent - 1U != 0U
                && strides[axis] > (UINT64_MAX - span) / (extent - 1U)) return NO;
        span += (extent - 1U) * strides[axis];
    }
    return span <= UINT32_MAX / width;
}

/* Empty numeric SUM and authenticated Scan-local reductions use their real geometry,
 * never a B=1 surrogate. Family admission remains separate from this shared axis proof;
 * the common positive-domain owner remains authoritative for nonempty inputs. */
static NSArray<NSNumber *> *synaptik_reduction_empty_axes(
        SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *output) {
    if ((node.operation != 13U && node.operation != 14U
                    && node.operation != 31U && node.operation != 32U
                    && node.operation != 106U && node.operation != 111U
                    && node.operation != 114U && node.operation != 115U)
            || node.attribute_kind != SYNAPTIK_METAL_MPSGRAPH_ATTR_REDUCTION
            || node.second_input != UINT32_MAX || node.attribute_count > 16U
            || node.auxiliary > 1U || !node_values_are_zero_from(node, node.attribute_count)) return nil;
    BOOL empty = NO;
    for (NSNumber *extent in input) empty |= extent.unsignedLongLongValue == 0U;
    if (!empty) return nil;
    NSMutableIndexSet *seen = [NSMutableIndexSet indexSet];
    NSMutableArray<NSNumber *> *axes = [NSMutableArray array];
    if (node.axis == SYNAPTIK_METAL_REDUCTION_SUM_TO_SHAPE) {
        if (node.operation != 13U || node.auxiliary != 0U
                || node.attribute_count != output.count || output.count > input.count) return nil;
        NSUInteger prefix = input.count - output.count;
        for (NSUInteger axis = 0U; axis < input.count; axis++) {
            uint64_t source = input[axis].unsignedLongLongValue;
            if (axis < prefix) [axes addObject:@(axis)];
            else {
                uint64_t target = node.attribute_values[axis - prefix];
                if (target != output[axis - prefix].unsignedLongLongValue
                        || (target != 1U && target != source)) return nil;
                if (target == 1U && target != source) [axes addObject:@(axis)];
            }
        }
        return [axes copy];
    }
    if (node.axis == SYNAPTIK_METAL_REDUCTION_FULL) {
        if (node.attribute_count != 0U || node.auxiliary != 0U || output.count != 0U) return nil;
        for (NSUInteger axis = 0U; axis < input.count; axis++) [axes addObject:@(axis)];
        return [axes copy];
    }
    if ((node.axis != SYNAPTIK_METAL_REDUCTION_SINGLE_AXIS
                    && node.axis != SYNAPTIK_METAL_REDUCTION_MULTI_AXIS)
            || (node.axis == SYNAPTIK_METAL_REDUCTION_SINGLE_AXIS && node.attribute_count != 1U)) return nil;
    for (uint32_t index = 0U; index < node.attribute_count; index++) {
        uint64_t axis = node.attribute_values[index];
        if (axis >= input.count || [seen containsIndex:(NSUInteger)axis]) return nil;
        [seen addIndex:(NSUInteger)axis]; [axes addObject:@(axis)];
    }
    NSMutableArray<NSNumber *> *expected = [NSMutableArray array];
    for (NSUInteger axis = 0U; axis < input.count; axis++) {
        if (![seen containsIndex:axis]) [expected addObject:input[axis]];
        else if (node.auxiliary != 0U) [expected addObject:@1];
    }
    return [expected isEqualToArray:output] ? [axes copy] : nil;
}

static BOOL synaptik_reduction_metadata(SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes, const uint8_t *types, const uint64_t *strides,
        const uint64_t *offsets, const uint8_t *states,
        SynaptikMetalDataMeta *meta, MTLSize *grid) {
    if (meta == NULL || grid == NULL || node.first_input >= shapes.count
            || node.output >= shapes.count || !synaptik_reduction_handles(node, types)
            || node.output_count != 1U || types[node.first_input] != types[node.output]) return NO;
    MPSShape *input = shapes[node.first_input], *output = shapes[node.output];
    if (input.count > 16U || output.count > 16U) return NO;
    BOOL masked = node.attribute_kind == 37U;
    BOOL statistical = node.operation == 112U || node.operation == 113U;
    if (node.input_count != (masked ? 2U : 1U)) return NO;
    if (masked && ((types[node.first_input] != 1U && types[node.first_input] != 5U
                    && types[node.first_input] != 7U)
            || (node.operation != 13U && node.operation != 14U)
            || node.second_input >= shapes.count || types[node.second_input] != 3U
            || node.attribute_count != 1U || node.attribute_values[0] >= input.count)) return NO;
    SynaptikMetalDecodedNode shape_node = node;
    if (masked) {
        shape_node.attribute_kind = SYNAPTIK_METAL_MPSGRAPH_ATTR_REDUCTION;
        shape_node.second_input = UINT32_MAX;
        shape_node.axis = SYNAPTIK_METAL_REDUCTION_SINGLE_AXIS;
        shape_node.auxiliary = 0U;
    }
    BOOL keep = NO; uint64_t correction = 0U;
    BOOL empty_input = NO;
    for (NSNumber *extent in input) empty_input |= extent.unsignedLongLongValue == 0U;
    if (empty_input && types[node.first_input] != 1U
            && types[node.first_input] != 5U && types[node.first_input] != 7U) return NO;
    /* The statistical owner rejects a selected zero extent before division, while
     * permitting an unselected zero axis and valid denominator with empty output. */
    NSArray<NSNumber *> *axes = statistical
            ? node_statistical_axes(shape_node, input, output, &keep, &correction)
            : empty_input
                    ? synaptik_reduction_empty_axes(shape_node, input, output)
                    : node_reduction_axes(shape_node, input, output, &keep);
    if (axes == nil || (statistical && node.attribute_kind != 38U)
            || (!statistical && node.attribute_kind == 38U)) return NO;
    if ((node.operation == 111U || node.operation == 114U || node.operation == 115U)
            && node.axis != SYNAPTIK_METAL_REDUCTION_MULTI_AXIS) return NO;
    memset(meta, 0, sizeof(*meta));
    meta->operation = node.operation; meta->inputCount = node.input_count;
    meta->inputRank = (uint32_t)input.count; meta->outputRank = (uint32_t)output.count;
    meta->sourceType = types[node.first_input]; meta->targetType = types[node.output];
    meta->width = (uint32_t)value_type_width(types[node.first_input]);
    meta->attributeKind = node.attribute_kind; meta->attributeCount = node.attribute_count;
    meta->elementCount = shape_element_count(output);
    for (NSUInteger axis = 0U; axis < input.count; axis++)
        meta->inputDims[axis] = input[axis].unsignedLongLongValue;
    for (NSUInteger axis = 0U; axis < output.count; axis++)
        meta->outputDims[axis] = output[axis].unsignedLongLongValue;
    for (NSNumber *axis in axes) meta->attrs[0] |= UINT64_C(1) << axis.unsignedIntValue;
    meta->attrs[1] = correction;
    if (!custom_physical_layout(input, node.first_input, strides, offsets, states,
                    meta->inputStrides, &meta->inputOffset)
            || !custom_physical_layout(output, node.output, strides, offsets, states,
                    meta->outputStrides, &meta->outputOffset)
            || !synaptik_reduction_span(input, meta->inputStrides, meta->inputOffset, meta->width)
            || !synaptik_reduction_span(output, meta->outputStrides, meta->outputOffset, meta->width)
            || meta->outputOffset != 0U) return NO;
    uint64_t canonical = 1U;
    for (NSUInteger reverse = 0U; reverse < output.count; reverse++) {
        NSUInteger axis = output.count - 1U - reverse;
        if (meta->outputStrides[axis] != canonical) return NO;
        uint64_t extent = output[axis].unsignedLongLongValue;
        if (extent != 0U && canonical > UINT64_MAX / extent) return NO;
        canonical *= extent;
    }
    if (masked) {
        MPSShape *mask = shapes[node.second_input];
        if (mask.count > input.count) return NO;
        meta->auxiliaryRank = (uint32_t)mask.count;
        NSUInteger prefix = input.count - mask.count;
        for (NSUInteger axis = 0U; axis < mask.count; axis++) {
            uint64_t extent = mask[axis].unsignedLongLongValue;
            if (extent != 1U && extent != input[prefix + axis].unsignedLongLongValue) return NO;
            meta->auxiliaryDims[axis] = extent;
        }
        if (!custom_physical_layout(mask, node.second_input, strides, offsets, states,
                        meta->auxiliaryStrides, &meta->auxiliaryOffset)
                || !synaptik_reduction_span(mask, meta->auxiliaryStrides, meta->auxiliaryOffset, 1U)) return NO;
    }
    if (meta->elementCount == 0U) {
        *grid = MTLSizeMake(0U, 0U, 0U);
        return YES; /* The ordinary/Scan dispatcher skips authenticated empty output grids. */
    }
    return custom_grid(meta->elementCount, grid, &meta->gridWidth, &meta->gridHeight);
}

#endif
