#ifndef SYNAPTIK_GENERAL_SCAN_VALIDATION_H
#define SYNAPTIK_GENERAL_SCAN_VALIDATION_H

/* Scan-specific physical admission, not a math inventory/evaluator. Attribute normalization and
 * operation equations remain with the same ordinary owners used by the outer executable. Actual
 * zero extents are retained. No sample batch, positive-shape stand-in or runtime data is used.
 * Include after the ordinary and joined numerical validators, before Scan plan construction.
 */
static BOOL synaptik_scan_floating_type(uint8_t type) { return type == 1U || type == 5U || type == 7U; }
static BOOL synaptik_scan_numeric_type(uint8_t type) { return synaptik_scan_floating_type(type) || type == 2U || type == 6U; }
static BOOL synaptik_scan_index_type(uint8_t type) { return type == 2U || type == 6U; }
static BOOL synaptik_scan_carrier_type(uint8_t type) { return synaptik_scan_numeric_type(type) || type == 3U; }
static BOOL synaptik_scan_shape_product(MPSShape *shape, uint64_t *elements) {
    uint64_t value = 1U;
    for (NSUInteger reverse = shape.count; reverse > 0U; reverse--) {
        uint64_t extent = shape[reverse - 1U].unsignedLongLongValue;
        if (extent > INT64_MAX || (extent != 0U && value > INT64_MAX / extent)) return NO;
        value *= extent;
    }
    *elements = value; return YES;
}

/* Existing normalized REDUCTION geometry, reused by exact BOOL/integral families whose math
 * implementation is already ordinary. SUM/MEAN/statistical/masked owners qualify through the
 * joined reduction validator before this branch. */
static BOOL synaptik_scan_exact_reduction_shape(SynaptikMetalDecodedNode node, MPSShape *input, MPSShape *output) {
    if (node.attribute_kind != 4U || node.axis < 1U || node.axis > 4U || node.auxiliary > 1U) return NO;
    if (node.axis == 4U) {
        if (node.operation != 13U || node.attribute_count != output.count || input.count < output.count) return NO;
        NSUInteger leading = input.count - output.count;
        for (NSUInteger a = 0U; a < output.count; a++) {
            uint64_t target = output[a].unsignedLongLongValue, source = input[a + leading].unsignedLongLongValue;
            if (node.attribute_values[a] != target || (target != 1U && target != source)) return NO;
        }
        return YES;
    }
    BOOL reduced[16] = {0};
    if (node.axis == 1U) {
        if (node.attribute_count != 0U) return NO;
        for (NSUInteger a = 0U; a < input.count; a++) reduced[a] = YES;
    } else {
        if (node.axis == 2U && node.attribute_count != 1U) return NO;
        uint64_t previous = UINT64_MAX;
        for (uint32_t i = 0U; i < node.attribute_count; i++) {
            uint64_t axis = node.attribute_values[i];
            if (axis >= input.count || (i != 0U && axis <= previous)) return NO;
            reduced[axis] = YES; previous = axis;
        }
    }
    NSUInteger cursor = 0U;
    for (NSUInteger a = 0U; a < input.count; a++) {
        if (reduced[a] && node.auxiliary == 0U) continue;
        uint64_t expected = reduced[a] ? 1U : input[a].unsignedLongLongValue;
        if (cursor >= output.count || output[cursor++].unsignedLongLongValue != expected) return NO;
    }
    return cursor == output.count;
}

static BOOL synaptik_scan_validate_ordinary(SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes,
        const uint8_t *types, const uint64_t *strides, const uint64_t *offsets, const uint8_t *states) {
    if (!synaptik_scan_ordinary_kind_is_valid(node.operation)
            || node.input_count == 0U || node.output_count == 0U) return NO;
    for (uint32_t i = 0U; i < node.input_count; i++)
        if (node.inputs[i] >= shapes.count || !synaptik_scan_carrier_type(types[node.inputs[i]])) return NO;
    for (uint32_t o = 0U; o < node.output_count; o++)
        if (node.outputs[o] >= shapes.count || !synaptik_scan_carrier_type(types[node.outputs[o]])) return NO;
    int extended = synaptik_validate_extended_native(node, shapes, types, strides, offsets, states);
    if (extended >= 0) return extended == 1;
    MPSShape *input = shapes[node.first_input], *output = shapes[node.output];
    uint8_t inputType = types[node.first_input], outputType = types[node.output];
    BOOL sameShape = [input isEqualToArray:output];
    switch (node.operation) {
        case 1U: case 12U: case 52U: case 53U: case 54U: case 55U: case 56U: case 57U: case 58U: case 59U:
        case 60U: case 61U: case 62U: case 63U: case 64U: case 65U: case 66U: case 67U: case 68U:
            return synaptik_scan_floating_type(inputType) && outputType == inputType && sameShape && node_has_no_attributes(node);
        case 2U: case 3U: case 4U: case 5U: case 38U:
            return synaptik_scan_floating_type(inputType) && synaptik_scan_floating_type(types[node.second_input])
                    && outputType == task0066_promote_floating(inputType, types[node.second_input]) && node_has_no_attributes(node)
                    && shape_broadcasts_exactly_to(input, shapes[node.second_input], output);
        case 20U: case 21U: case 22U: case 23U: case 24U: case 25U:
            return synaptik_scan_numeric_type(inputType) && types[node.second_input] == inputType
                    && outputType == 3U && node_has_no_attributes(node)
                    && shape_broadcasts_exactly_to(input, shapes[node.second_input], output);
        case 26U: case 27U:
            return ((synaptik_scan_floating_type(inputType) && synaptik_scan_floating_type(types[node.second_input])
                            && outputType == task0066_promote_floating(inputType, types[node.second_input]))
                    || (synaptik_scan_index_type(inputType) && types[node.second_input] == inputType && outputType == inputType))
                    && node_has_no_attributes(node) && shape_broadcasts_exactly_to(input, shapes[node.second_input], output);
        case 28U: case 29U: case 46U: case 47U: case 48U: case 49U: case 50U:
            return synaptik_scan_floating_type(inputType) && outputType == inputType && sameShape && node.attribute_kind == 7U && node.attribute_count == 1U;
        case 30U:
            return synaptik_scan_floating_type(inputType) && outputType == inputType && sameShape && node.attribute_kind == 8U && node.attribute_count == 2U;
        case 40U: case 41U: case 42U:
            return synaptik_scan_floating_type(inputType) && outputType == 3U && sameShape && node_has_no_attributes(node);
        case 43U: case 44U:
            return inputType == 3U && types[node.second_input] == 3U && outputType == 3U && node_has_no_attributes(node)
                    && shape_broadcasts_exactly_to(input, shapes[node.second_input], output);
        case 45U:
            return inputType == 3U && outputType == 3U && sameShape && node_has_no_attributes(node);
        case 51U:
            return inputType == 3U && synaptik_scan_floating_type(types[node.second_input])
                    && synaptik_scan_floating_type(types[node.inputs[2]])
                    && outputType == task0066_promote_floating(types[node.second_input], types[node.inputs[2]])
                    && node.attribute_kind == 0U && node.attribute_count == 0U && node.axis == UINT32_MAX
                    && node_values_are_zero_from(node, 0U) && shape_broadcasts_exactly_to(shapes[node.second_input], shapes[node.inputs[2]], output)
                    && shape_broadcasts_exactly_to(input, output, output);
        case 6U: {
            uint64_t sourceElements, resultElements;
            return inputType == outputType && node_target_matches(node, output)
                    && synaptik_scan_shape_product(input, &sourceElements) && synaptik_scan_shape_product(output, &resultElements)
                    && sourceElements == resultElements;
        }
        case 7U:
            return inputType == outputType && node_target_matches(node, output) && shape_broadcasts_exactly_to(input, output, output);
        case 8U: return inputType == outputType && node_permutation_matches(node, input, output);
        case 9U: return inputType == outputType && node_expand_dims_matches(node, input, output);
        case 10U: return inputType == outputType && node_squeeze_matches(node, input, output);
        case 11U: return inputType == outputType && sameShape && node_has_no_attributes(node);
        case 15U: {
            uint8_t right = types[node.second_input];
            BOOL typed = synaptik_scan_floating_type(inputType) && synaptik_scan_floating_type(right)
                    && outputType == task0066_promote_floating(inputType, right);
            if (synaptik_scan_index_type(inputType) && synaptik_scan_index_type(right))
                typed = outputType == (inputType == 6U || right == 6U ? 6U : 2U);
            return typed && node_has_no_attributes(node)
                    && matmul_shapes_match(input, shapes[node.second_input], output);
        }
        case 16U:
            return synaptik_scan_index_type(types[node.second_input]) && inputType == outputType
                    && node_gather_matches(node, input, shapes[node.second_input], output);
        case 17U: return synaptik_scan_index_type(inputType) && outputType == 3U && node_one_hot_matches(node, input, output);
        case 18U:
            return synaptik_scan_index_type(types[node.second_input]) && inputType == outputType
                    && types[node.inputs[2]] == inputType && node.attribute_kind == 3U
                    && node_scatter_elements_matches(node, input, shapes[node.second_input], shapes[node.inputs[2]], output);
        case 19U: return inputType == outputType && node_unfold_axis_matches(node, input, output);
        case 33U: case 34U:
            return synaptik_scan_numeric_type(inputType) && outputType == inputType && sameShape && node.attribute_kind == 9U
                    && node.axis < input.count && node.attribute_count == 2U
                    && node.attribute_values[0] <= 1U && node.attribute_values[1] <= 1U;
        case 39U:
            return sameShape && node.attribute_kind == 10U && node.attribute_count == 1U
                    && node.attribute_values[0] == outputType && task0059_validate_shape(node, shapes);
        case 69U: case 71U:
            return synaptik_scan_index_type(types[node.second_input]) && inputType == outputType && task0059_validate_shape(node, shapes);
        case 72U:
            // Floating/reduction variants remain with the joined scatter owner. These are
            // ordinary exact replacements, whose shape/index contract is already shared.
            return (inputType == 3U || synaptik_scan_index_type(inputType))
                    && outputType == inputType && types[node.auxiliary] == inputType
                    && synaptik_scan_index_type(types[node.second_input])
                    && node.attribute_kind == 13U && node.attribute_count == 2U
                    && node.attribute_values[1] == 1U && task0059_validate_shape(node, shapes);
        case 73U: case 74U: case 75U: case 76U: case 77U: case 78U: case 79U:
            for (uint32_t i = 0U; i < node.input_count; i++) if (types[node.inputs[i]] != outputType) return NO;
            return task0059_validate_shape(node, shapes);
        case 80U:
            return synaptik_scan_numeric_type(inputType) && outputType == inputType
                    && (synaptik_scan_floating_type(inputType)
                            || node.attribute_values[1] >= node.attribute_values[0])
                    && task0059_validate_shape(node, shapes);
        case 81U: case 82U: case 83U: case 84U:
            return synaptik_scan_floating_type(inputType) && outputType == inputType && task0059_validate_shape(node, shapes);
        case 85U: {
            uint64_t reduction = node.attribute_values[0];
            return synaptik_scan_floating_type(inputType) && types[node.second_input] == inputType && outputType == inputType
                    && input.count != 0U && [input isEqualToArray:shapes[node.second_input]]
                    && node.attribute_kind == 25U && node.attribute_count == 1U
                    && reduction >= 1U && reduction <= 3U && node.axis == UINT32_MAX
                    && node.auxiliary == 0U && node_values_are_zero_from(node, 1U)
                    && (reduction == 1U ? sameShape : output.count == 0U);
        }
        case 36U: case 37U: {
            if (!synaptik_scan_floating_type(inputType)) return NO;
            uint8_t promoted = inputType;
            for (uint32_t i = 1U; i < node.input_count; i++) {
                if (!synaptik_scan_floating_type(types[node.inputs[i]])) return NO;
                promoted = task0066_promote_floating(promoted, types[node.inputs[i]]);
            }
            return outputType == promoted && task0064_validate_scan_node(node, shapes, states);
        }
        case 97U: case 98U: case 99U: case 100U:
            return synaptik_scan_floating_type(inputType) && outputType == inputType && task0064_validate_scan_node(node, shapes, states);
        case 31U: case 32U:
            return synaptik_scan_numeric_type(inputType) && outputType == inputType && synaptik_scan_exact_reduction_shape(node, input, output);
        case 106U:
            return synaptik_scan_index_type(inputType) && outputType == inputType && synaptik_scan_exact_reduction_shape(node, input, output);
        case 107U: case 108U:
            return inputType == 3U && outputType == 3U && synaptik_scan_exact_reduction_shape(node, input, output);
        case 94U: case 95U: case 96U: {
            uint64_t axis = node.attribute_values[0];
            if (axis >= input.count) return NO;
            uint64_t elements = 1U;
            for (NSNumber *dimension in input) {
                uint64_t extent = dimension.unsignedLongLongValue;
                if (extent > UINT32_MAX || (extent != 0U && elements > UINT32_MAX / extent)) return NO;
                elements *= extent;
            }
            if (node.operation != 96U)
                return node.attribute_kind == 31U && node.attribute_count == 2U && sameShape
                        && outputType == (node.operation == 94U ? inputType : 6U);
            if (node.attribute_kind != 32U || node.attribute_count != 4U || node.output_count != 2U
                    || input.count != output.count || outputType != inputType || types[node.outputs[1]] != 6U
                    || ![output isEqualToArray:shapes[node.outputs[1]]]) return NO;
            uint64_t k = node.attribute_values[1];
            if (k == 0U || k > input[axis].unsignedLongLongValue) return NO;
            for (NSUInteger a = 0U; a < input.count; a++)
                if (output[a].unsignedLongLongValue != (a == axis ? k : input[a].unsignedLongLongValue)) return NO;
            return YES;
        }
        case 109U: case 110U: {
            uint64_t axis = node.attribute_values[0], keep = node.attribute_values[1];
            if (!synaptik_scan_numeric_type(inputType) || outputType != 6U || node.attribute_kind != 36U
                    || node.attribute_count != 3U || axis >= input.count || keep > 1U
                    || input[axis].unsignedLongLongValue == 0U) return NO;
            NSUInteger cursor = 0U;
            for (NSUInteger a = 0U; a < input.count; a++) {
                if (a == axis && keep == 0U) continue;
                if (cursor >= output.count || output[cursor++].unsignedLongLongValue
                        != (a == axis ? 1U : input[a].unsignedLongLongValue)) return NO;
            }
            return cursor == output.count;
        }
        default: return NO;
    }
}

#endif
