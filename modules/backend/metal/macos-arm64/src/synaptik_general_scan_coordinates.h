#ifndef SYNAPTIK_GENERAL_SCAN_COORDINATES_H
#define SYNAPTIK_GENERAL_SCAN_COORDINATES_H

/* Cold structural counterpart of Model Scan.ordinarySemantics/outputRow. Include after
 * SynaptikMetalDecodedNode and task0059_window_extent. Ordinary signature, descriptor and
 * normalized-attribute validation MUST precede this proof. Input facts have already been
 * proved in forward topology order; output facts are claims, never premises. All body
 * storage is canonical, so no physical-layout inference belongs here. This owner performs
 * no allocation, numerical evaluation, backend admission or alternative graph construction.
 * -1 denotes SHARED; -2 is an unproved coordinate equation, never a published row fact.
 */
static BOOL synaptik_scan_coordinate_singleton(MPSShape *shape, int axis) {
    return axis >= 0 && axis < (int)shape.count && shape[axis].unsignedLongLongValue == 1U;
}

static int synaptik_scan_coordinate_merge(int left, int right) {
    if (left < -1 || right < -1 || (left >= 0 && right >= 0 && left != right)) return -2;
    return left > right ? left : right;
}

static BOOL synaptik_scan_coordinate_same(int row, MPSShape *shape, int axis) {
    return row >= 0 ? row == axis : synaptik_scan_coordinate_singleton(shape, axis);
}

/* Fixed boundary products follow Model's zero-first checked signed-long arithmetic. */
static BOOL synaptik_scan_coordinate_product(MPSShape *shape, int first, int last, uint64_t *result) {
    for (int axis = first; axis < last; axis++)
        if (shape[axis].unsignedLongLongValue == 0U) { *result = 0U; return YES; }
    uint64_t product = 1U;
    for (int axis = first; axis < last; axis++) {
        uint64_t extent = shape[axis].unsignedLongLongValue;
        if (extent > INT64_MAX || product > (uint64_t)INT64_MAX / extent) return NO;
        product *= extent;
    }
    *result = product;
    return YES;
}

static int synaptik_scan_coordinate_pointwise(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows, MPSShape *output) {
    int result = -1;
    for (uint32_t input = 0U; input < node->input_count; input++) {
        uint32_t value = node->inputs[input];
        if (rows[value] >= 0)
            result = synaptik_scan_coordinate_merge(result,
                    rows[value] + (int)output.count - (int)shapes[value].count);
        if (result < -1) return -2;
    }
    if (result >= 0) for (uint32_t input = 0U; input < node->input_count; input++) {
        uint32_t value = node->inputs[input];
        if (rows[value] >= 0) continue;
        MPSShape *shape = shapes[value];
        int aligned = result - (int)output.count + (int)shape.count;
        if (aligned >= 0 && !synaptik_scan_coordinate_singleton(shape, aligned)) return -2;
    }
    return result;
}

static int synaptik_scan_coordinate_reshape(MPSShape *input, int row, MPSShape *output) {
    if (row < 0 || [input isEqualToArray:output]) return row;
    uint64_t before, after;
    if (!synaptik_scan_coordinate_product(input, 0, row, &before)
            || !synaptik_scan_coordinate_product(input, row + 1, (int)input.count, &after)) return -2;
    for (int axis = 0; axis < (int)output.count; axis++) {
        if (input[row].unsignedLongLongValue != output[axis].unsignedLongLongValue) continue;
        uint64_t candidateBefore, candidateAfter;
        if (!synaptik_scan_coordinate_product(output, 0, axis, &candidateBefore)
                || !synaptik_scan_coordinate_product(output, axis + 1, (int)output.count, &candidateAfter)) return -2;
        if (before == candidateBefore && after == candidateAfter) return axis;
    }
    return -2;
}

/* Canonical numerical/control routes are not identical to shape-domain contributors:
 * attention weights omit V, and each of BatchNorm training's five siblings has its own route.
 */
static BOOL synaptik_scan_coordinate_direct(uint32_t operation, uint32_t slot, uint32_t input) {
    if (operation == 89U) {
        switch (slot) {
            case 0U: return input <= 2U;
            case 1U: return input == 0U || input == 3U;
            case 2U: return input == 0U || input == 4U;
            case 3U: case 4U: return input == 0U;
            default: return NO;
        }
    }
    return operation != 35U || slot != 1U || input != 2U;
}

static int synaptik_scan_coordinate_gather(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows) {
    MPSShape *data = shapes[node->inputs[0]], *indices = shapes[node->inputs[1]];
    int dataRow = rows[node->inputs[0]], indexRow = rows[node->inputs[1]];
    if (node->operation == 71U) {
        int prefix = (int)node->attribute_values[0];
        int depth = (int)indices.lastObject.unsignedLongLongValue;
        if (indexRow == (int)indices.count - 1 || (dataRow >= prefix && dataRow < prefix + depth)) return -2;
        int result = dataRow < 0 ? -1 : dataRow < prefix ? dataRow
                : (int)indices.count - 1 + dataRow - prefix - depth;
        if (indexRow >= 0) result = synaptik_scan_coordinate_merge(result, indexRow);
        if (result >= 0 && result < prefix
                && ((dataRow < 0 && !synaptik_scan_coordinate_singleton(data, result))
                    || (indexRow < 0 && !synaptik_scan_coordinate_singleton(indices, result)))) return -2;
        return result;
    }
    int axis = (int)node->axis;
    if (dataRow == axis) return -2;
    if (node->operation == 69U) {
        int result = synaptik_scan_coordinate_merge(dataRow, indexRow);
        if (result >= 0 && ((dataRow < 0 && result != axis && !synaptik_scan_coordinate_singleton(data, result))
                || (indexRow < 0 && !synaptik_scan_coordinate_singleton(indices, result)))) return -2;
        return result;
    }
    int result = dataRow < 0 ? -1 : dataRow < axis ? dataRow : dataRow + (int)indices.count - 1;
    return indexRow < 0 ? result : synaptik_scan_coordinate_merge(result, axis + indexRow);
}

static int synaptik_scan_coordinate_scatter(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows, const uint8_t *terminals) {
    int base = rows[node->inputs[0]], index = rows[node->inputs[1]], update = rows[node->inputs[2]];
    MPSShape *indices = shapes[node->inputs[1]], *updates = shapes[node->inputs[2]];
    BOOL add = node->operation == 70U || (node->attribute_kind == 12U && node->attribute_values[1] == 2U)
            || (node->attribute_kind == 13U && node->attribute_values[1] == 2U);
    if (base < 0 && (!add || terminals[node->inputs[1]] || terminals[node->inputs[2]])) return -2;
    if (node->operation == 72U) {
        int prefix = (int)node->attribute_values[0], depth = (int)indices.lastObject.unsignedLongLongValue;
        if (index == (int)indices.count - 1) return -2;
        if (base < 0) {
            if (update >= (int)indices.count - 1) return index >= 0 ? -2 : -1;
            int contributor = synaptik_scan_coordinate_merge(index, update);
            if (contributor < 0 || contributor >= (int)indices.count - 1
                    || !synaptik_scan_coordinate_same(index, indices, contributor)
                    || !synaptik_scan_coordinate_same(update, updates, contributor)) return -2;
            return -1;
        }
        if (base >= prefix && base < prefix + depth) return -2;
        int mapped = base < prefix ? base : (int)indices.count - 1 + base - prefix - depth;
        if (!synaptik_scan_coordinate_same(update, updates, mapped)
                || (base < prefix && !synaptik_scan_coordinate_same(index, indices, base))
                || (base >= prefix && index >= 0)) return -2;
        return base;
    }
    int axis = (int)node->axis;
    if (node->operation == 18U) {
        if (base < 0) {
            int contributor = synaptik_scan_coordinate_merge(index, update);
            if (contributor < 0 || !synaptik_scan_coordinate_same(index, indices, contributor)
                    || !synaptik_scan_coordinate_same(update, updates, contributor)) return -2;
        } else if (base == axis || !synaptik_scan_coordinate_same(index, indices, base)
                || !synaptik_scan_coordinate_same(update, updates, base)) return -2;
        return base;
    }
    if (base < 0) {
        if (update >= 0 && (update < axis || update >= axis + (int)indices.count)) return index >= 0 ? -2 : -1;
        int contributor = synaptik_scan_coordinate_merge(index, update < 0 ? -1 : update - axis);
        if (contributor < 0 || contributor >= (int)indices.count
                || !synaptik_scan_coordinate_same(index, indices, contributor)
                || !synaptik_scan_coordinate_same(update, updates, axis + contributor)) return -2;
        return -1;
    }
    if (base == axis || index >= 0) return -2;
    int mapped = base < axis ? base : base + (int)indices.count - 1;
    return synaptik_scan_coordinate_same(update, updates, mapped) ? base : -2;
}

static int synaptik_scan_coordinate_reduction(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows, MPSShape *output) {
    MPSShape *input = shapes[node->inputs[0]];
    int row = node->input_count > 1U ? synaptik_scan_coordinate_pointwise(node, shapes, rows, input)
            : rows[node->inputs[0]];
    if (row < 0) return row;
    if (node->attribute_kind == 4U && node->axis == 4U) {
        int removed = (int)input.count - (int)output.count;
        return row < removed || input[row].unsignedLongLongValue != output[row - removed].unsignedLongLongValue
                ? -1 : row - removed;
    }
    BOOL reduced[SYNAPTIK_MAX_RANK] = {0}, keep = NO;
    if (node->attribute_kind == 4U) {
        keep = node->auxiliary != 0U;
        if (node->axis == 1U) for (NSUInteger axis = 0U; axis < input.count; axis++) reduced[axis] = YES;
        else for (uint32_t i = 0U; i < node->attribute_count; i++) reduced[node->attribute_values[i]] = YES;
    } else if (node->attribute_kind == 38U) {
        uint32_t count = (uint32_t)node->attribute_values[0];
        for (uint32_t i = 0U; i < count; i++) reduced[node->attribute_values[i + 1U]] = YES;
        keep = node->attribute_values[count + 1U] != 0U;
    } else if (node->attribute_kind == 36U || node->attribute_kind == 37U) {
        reduced[node->attribute_values[0]] = YES;
        keep = node->attribute_kind == 36U && node->attribute_values[1] != 0U;
    } else return -2;
    if (reduced[row]) return node->operation == 13U ? -1 : -2;
    int result = row;
    if (!keep) for (int axis = 0; axis < row; axis++) if (reduced[axis]) result--;
    return result;
}

static int synaptik_scan_coordinate_matmul(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows) {
    MPSShape *left = shapes[node->inputs[0]], *right = shapes[node->inputs[1]];
    int l = rows[node->inputs[0]], r = rows[node->inputs[1]];
    int lr = (int)left.count, rr = (int)right.count, lk = lr - 1, rk = rr == 1 ? 0 : rr - 2;
    if (l == lk || r == rk) return l == lk && r == rk ? -1 : -2;
    int prefix = MAX(MAX(0, lr - 2), MAX(0, rr - 2));
    int lm = l < 0 ? -1 : l < lr - 2 ? l + prefix - (lr - 2) : prefix;
    int rm = r < 0 ? -1 : r < rr - 2 ? r + prefix - (rr - 2) : prefix + (lr == 1 ? 0 : 1);
    int result = synaptik_scan_coordinate_merge(lm, rm);
    if (result >= 0 && result < prefix) for (uint32_t input = 0U; input < 2U; input++) {
        if (rows[node->inputs[input]] >= 0) continue;
        MPSShape *shape = shapes[node->inputs[input]];
        int axis = result - prefix + MAX(0, (int)shape.count - 2);
        if (axis >= 0 && !synaptik_scan_coordinate_singleton(shape, axis)) return -2;
    }
    return result;
}

static int synaptik_scan_coordinate_batch_norm(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows, uint32_t slot) {
    int channel = (int)node->attribute_values[0], result = -1;
    for (uint32_t input = 0U; input < node->input_count; input++) {
        if (!synaptik_scan_coordinate_direct(node->operation, slot, input)) continue;
        int row = rows[node->inputs[input]];
        if (row < 0) continue;
        if (input == 0U && node->operation == 89U && row != channel) return -2;
        int mapped = slot == 0U ? (input == 0U ? row : channel) : 0;
        result = synaptik_scan_coordinate_merge(result, mapped);
        if (result < -1) return -2;
    }
    if (result >= 0 && (slot != 0U || result == channel))
        for (uint32_t input = 0U; input < node->input_count; input++) {
            if (!synaptik_scan_coordinate_direct(node->operation, slot, input) || rows[node->inputs[input]] >= 0) continue;
            if (!synaptik_scan_coordinate_singleton(shapes[node->inputs[input]], input == 0U ? channel : 0)) return -2;
        }
    return result;
}

static int synaptik_scan_coordinate_attention(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows, uint32_t slot) {
    int prefix = (int)shapes[node->outputs[0]].count - 2, result = -1;
    for (uint32_t input = 0U; input < 3U; input++) {
        MPSShape *shape = shapes[node->inputs[input]];
        int row = rows[node->inputs[input]], rank = (int)shape.count;
        if (row < 0) continue;
        if (row < rank - 2) result = synaptik_scan_coordinate_merge(result, row + prefix - (rank - 2));
        else if (input == 0U && row == rank - 2) result = synaptik_scan_coordinate_merge(result, prefix);
        else if (input == 2U && row == rank - 1) {
            if (slot == 0U) result = synaptik_scan_coordinate_merge(result, prefix + 1);
        } else return -2;
        if (result < -1) return -2;
    }
    if (node->input_count == 4U) {
        int row = rows[node->inputs[3]];
        if (row >= 0) {
            int mapped = row + prefix + 2 - (int)shapes[node->inputs[3]].count;
            if (mapped == prefix + 1) return -2;
            result = synaptik_scan_coordinate_merge(result, mapped);
            if (result < -1) return -2;
        }
    }
    if (result >= 0 && result < prefix) for (uint32_t input = 0U; input < node->input_count; input++) {
        if (rows[node->inputs[input]] >= 0) continue;
        MPSShape *shape = shapes[node->inputs[input]];
        int axis = input == 3U ? result - prefix - 2 + (int)shape.count : result - prefix + (int)shape.count - 2;
        if (axis >= 0 && !synaptik_scan_coordinate_singleton(shape, axis)) return -2;
    }
    if (result == prefix) {
        MPSShape *query = shapes[node->inputs[0]];
        if (rows[node->inputs[0]] < 0 && !synaptik_scan_coordinate_singleton(query, (int)query.count - 2)) return -2;
        if (node->input_count == 4U && rows[node->inputs[3]] < 0) {
            MPSShape *mask = shapes[node->inputs[3]];
            if (mask.count >= 2U && !synaptik_scan_coordinate_singleton(mask, (int)mask.count - 2)) return -2;
        }
    }
    return result;
}

/* Window equations reuse the ordinary checked grid helper. Extent equality is a witness
 * only after the complete boundary/window contributor equation has been proved. */
typedef struct {
    uint32_t dimensions;
    uint64_t kernel[3], stride[3], padding[3], dilation[3];
    BOOL ceil;
} SynaptikScanCoordinateWindow;

static void synaptik_scan_coordinate_window_geometry(const SynaptikMetalDecodedNode *node,
        SynaptikScanCoordinateWindow *window) {
    BOOL fold = node->operation == 82U || node->operation == 84U;
    uint32_t dimensions = node->operation == 81U || node->operation == 82U ? 2U : 3U;
    uint32_t offset = fold ? 1U + (uint32_t)node->attribute_values[0] : 0U;
    window->dimensions = dimensions;
    for (uint32_t axis = 0U; axis < dimensions; axis++) {
        window->kernel[axis] = node->attribute_values[offset + axis];
        window->stride[axis] = node->attribute_values[offset + dimensions + axis];
        window->padding[axis] = node->attribute_values[offset + dimensions * 2U + axis];
        window->dilation[axis] = node->attribute_values[offset + dimensions * 3U + axis];
    }
    window->ceil = node->attribute_values[offset + dimensions * 4U] != 0U;
}

static BOOL synaptik_scan_coordinate_window_identity(const SynaptikScanCoordinateWindow *window, int axis) {
    return window->kernel[axis] == 1U && window->stride[axis] == 1U && window->padding[axis] == 0U;
}

static BOOL synaptik_scan_coordinate_grid_singleton(MPSShape *shape,
        const SynaptikScanCoordinateWindow *window, int axis) {
    uint64_t grid;
    int spatial = axis - 2;
    return task0059_window_extent(shape[axis].unsignedLongLongValue, window->kernel[spatial],
            window->padding[spatial], window->stride[spatial], window->dilation[spatial], window->ceil, &grid)
            && grid == 1U;
}

static BOOL synaptik_scan_coordinate_full_window(MPSShape *shape,
        const SynaptikScanCoordinateWindow *window, int row) {
    int spatial = row - 2;
    if (!synaptik_scan_coordinate_singleton(shape, 1)
            || shape[row].unsignedLongLongValue != window->kernel[spatial]
            || window->padding[spatial] != 0U || window->dilation[spatial] != 1U
            || !synaptik_scan_coordinate_grid_singleton(shape, window, row)) return NO;
    for (int axis = 2; axis < (int)shape.count; axis++)
        if (axis != row && window->kernel[axis - 2] != 1U) return NO;
    return YES;
}

static int synaptik_scan_coordinate_window(const SynaptikMetalDecodedNode *node,
        MPSShape *input, int row, MPSShape *output) {
    if (row < 0) return -1;
    int axis = (int)node->axis;
    if (node->operation == 19U) {
        if (row != axis || (node->attribute_values[0] == 1U && node->attribute_values[1] == 1U)) return row;
        return synaptik_scan_coordinate_singleton(output, axis)
                && input[row].unsignedLongLongValue == node->attribute_values[0] ? (int)output.count - 1 : -2;
    }
    if (node->operation == 80U) {
        if (row == (int)input.count - 1)
            return synaptik_scan_coordinate_singleton(input, axis)
                    && input[row].unsignedLongLongValue == output[axis].unsignedLongLongValue ? axis : -2;
        if (row == axis && (!synaptik_scan_coordinate_singleton(input, (int)input.count - 1)
                || node->attribute_values[1] != 1U)) return -2;
        return row;
    }
    SynaptikScanCoordinateWindow window = {0};
    synaptik_scan_coordinate_window_geometry(node, &window);
    if (row == 0) return 0;
    uint64_t kernel = 1U;
    for (uint32_t i = 0U; i < window.dimensions; i++) {
        if (window.kernel[i] == 0U || kernel > (uint64_t)INT64_MAX / window.kernel[i]) return -2;
        kernel *= window.kernel[i];
    }
    BOOL fold = node->operation == 82U || node->operation == 84U;
    if (row == 1) {
        if (kernel == 1U) return 1;
        if (!fold) return -2;
        for (int candidate = 2; candidate < (int)output.count; candidate++)
            if (input[row].unsignedLongLongValue == output[candidate].unsignedLongLongValue
                    && synaptik_scan_coordinate_full_window(output, &window, candidate)) return candidate;
        return -2;
    }
    if (!fold) {
        if (synaptik_scan_coordinate_full_window(input, &window, row)) return 1;
        if (!synaptik_scan_coordinate_window_identity(&window, row - 2)) return -2;
        for (int other = 2; other < (int)input.count; other++)
            if (other != row && !synaptik_scan_coordinate_grid_singleton(input, &window, other)) return -2;
        return 2;
    }
    for (int candidate = 2; candidate < (int)output.count; candidate++) {
        if (input[row].unsignedLongLongValue != output[candidate].unsignedLongLongValue
                || !synaptik_scan_coordinate_window_identity(&window, candidate - 2)) continue;
        BOOL othersSingleton = YES;
        for (int other = 2; other < (int)output.count; other++)
            if (other != candidate && !synaptik_scan_coordinate_grid_singleton(output, &window, other)) othersSingleton = NO;
        if (othersSingleton) return candidate;
    }
    return -2;
}

static int synaptik_scan_coordinate_convolution(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows) {
    MPSShape *data = shapes[node->inputs[0]], *weight = shapes[node->inputs[1]];
    int dataRow = rows[node->inputs[0]], weightRow = rows[node->inputs[1]], result = -1;
    uint32_t dimensions = node->operation == 36U ? 2U : 3U;
    if (dataRow == 0) result = 0;
    else if (dataRow == 1) {
        if (!synaptik_scan_coordinate_singleton(weight, 1)
                || weight[0].unsignedLongLongValue != data[1].unsignedLongLongValue
                || data[1].unsignedLongLongValue != node->attribute_values[dimensions * 3U]) return -2;
        result = 1;
    } else if (dataRow >= 2) {
        uint32_t spatial = (uint32_t)dataRow - 2U;
        if (weight[dataRow].unsignedLongLongValue != 1U || node->attribute_values[spatial] != 1U
                || node->attribute_values[dimensions + spatial] != 0U) return -2;
        result = dataRow;
    }
    if (weightRow >= 0) {
        if (weightRow != 0) return -2;
        result = synaptik_scan_coordinate_merge(result, 1);
        if (result < -1) return -2;
    }
    if (node->input_count == 3U) {
        int bias = rows[node->inputs[2]];
        if (bias >= 0) result = synaptik_scan_coordinate_merge(result, 1);
        else if (result == 1 && !synaptik_scan_coordinate_singleton(shapes[node->inputs[2]], 0)) return -2;
    }
    if (result == 1 && weightRow < 0 && !synaptik_scan_coordinate_singleton(weight, 0)) return -2;
    return result;
}

static int synaptik_scan_coordinate_loss(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows) {
    BOOL mse = node->operation == 85U, indexed = node->operation == 87U;
    if (node->attribute_values[mse ? 0U : 1U] != 1U) return -2; // Ordinary NONE reduction.
    int axis = mse ? -1 : (int)node->attribute_values[0];
    int left = rows[node->inputs[0]], right = rows[node->inputs[1]];
    if (axis >= 0 && (left == axis || (!indexed && right == axis))) return -2;
    int mappedLeft = left < 0 ? -1 : left - (axis >= 0 && axis < left ? 1 : 0);
    int mappedRight = indexed ? right : right < 0 ? -1 : right - (axis >= 0 && axis < right ? 1 : 0);
    int result = synaptik_scan_coordinate_merge(mappedLeft, mappedRight);
    if (result < -1) return -2;
    int logitsAxis = result + (axis >= 0 && axis <= result ? 1 : 0);
    if (mappedLeft < 0 && !synaptik_scan_coordinate_singleton(shapes[node->inputs[0]], logitsAxis)) return -2;
    if (mappedRight < 0 && !synaptik_scan_coordinate_singleton(shapes[node->inputs[1]], indexed ? result : logitsAxis)) return -2;
    return result;
}

/* Dispatch is the bounded Model PURE structural algebra, not a numerical capability list. */
static int synaptik_scan_coordinate_output(const SynaptikMetalDecodedNode *node,
        NSArray<MPSShape *> *shapes, const int32_t *rows, const uint8_t *terminals, uint32_t slot) {
    MPSShape *input = shapes[node->inputs[0]], *output = shapes[node->outputs[slot]];
    int row = rows[node->inputs[0]], axis = (int)node->axis;
    const uint64_t *attrs = node->attribute_values;
    switch (node->operation) {
        case 2U: case 3U: case 4U: case 5U: case 7U: case 20U: case 21U: case 22U: case 23U:
        case 24U: case 25U: case 26U: case 27U: case 38U: case 43U: case 44U: case 51U:
            return synaptik_scan_coordinate_pointwise(node, shapes, rows, output);
        case 1U: case 11U: case 12U: case 17U: case 28U: case 29U: case 30U: case 39U:
        case 40U: case 41U: case 42U: case 45U: case 46U: case 47U: case 48U: case 49U: case 50U:
        case 52U: case 53U: case 54U: case 55U: case 56U: case 57U: case 58U: case 59U: case 60U:
        case 61U: case 62U: case 63U: case 64U: case 65U: case 66U: case 67U: case 68U: return row;
        case 6U: return synaptik_scan_coordinate_reshape(input, row, output);
        case 8U:
            if (row < 0) return -1;
            for (uint32_t i = 0U; i < node->attribute_count; i++) if (attrs[i] == (uint64_t)row) return (int)i;
            return -2;
        case 9U: return row < 0 ? -1 : row + (axis <= row ? 1 : 0);
        case 73U: axis = (int)attrs[0]; // SELECT removes precisely the selected axis.
            /* fall through */
        case 10U:
            if (row == axis) return -2;
            return row < 0 ? -1 : row - (axis < row ? 1 : 0);
        case 77U: case 78U: {
            BOOL stack = node->operation == 78U;
            int result = -1;
            for (uint32_t i = 0U; i < node->input_count; i++) {
                int r = rows[node->inputs[i]];
                if (r < 0) continue;
                if (!stack && axis == r && node->input_count != 1U) return -2;
                result = synaptik_scan_coordinate_merge(result, r + (stack && axis <= r ? 1 : 0));
                if (result < -1) return -2;
            }
            if (result >= 0) for (uint32_t i = 0U; i < node->input_count; i++) {
                if (rows[node->inputs[i]] >= 0) continue;
                int mapped = result - (stack && axis < result ? 1 : 0);
                if (!synaptik_scan_coordinate_singleton(shapes[node->inputs[i]], mapped)) return -2;
            }
            return result;
        }
        case 75U: case 76U:
            if (node->input_count == 2U) {
                row = synaptik_scan_coordinate_merge(row, rows[node->inputs[1]]);
                if (row < -1) return -2;
                if (row >= 0 && (!synaptik_scan_coordinate_same(rows[node->inputs[0]], input, row)
                        || !synaptik_scan_coordinate_same(rows[node->inputs[1]], shapes[node->inputs[1]], row))) return -2;
            }
            if (node->attribute_kind == 17U) {
                uint32_t count = (uint32_t)attrs[0];
                for (uint32_t i = 0U; i < count; i++) if (row >= 0 && attrs[1U + count * 2U + i] == (uint64_t)row
                        && (attrs[1U + i] != 0U || attrs[1U + count * 3U + i] != 1U
                            || attrs[1U + count + i] != input[row].unsignedLongLongValue)) return -2;
            } else if (node->attribute_kind == 15U) {
                uint32_t rank = (uint32_t)attrs[0];
                if (row >= 0 && (attrs[1U + row] != input[row].unsignedLongLongValue
                        || attrs[2U + rank + row] != 0U)) return -2;
            } else return -2;
            return row;
        case 74U:
            return row >= 0 && (attrs[1U + row] != 0U || attrs[1U + input.count + row] != 0U) ? -2 : row;
        case 79U: return row >= 0 && attrs[1U + row] != 1U ? -2 : row;
        case 16U: case 69U: case 71U: return synaptik_scan_coordinate_gather(node, shapes, rows);
        case 18U: case 70U: case 72U: return synaptik_scan_coordinate_scatter(node, shapes, rows, terminals);
        case 13U: case 14U: case 31U: case 32U: case 106U: case 107U: case 108U:
        case 109U: case 110U: case 111U: case 112U: case 113U: case 114U: case 115U:
            return synaptik_scan_coordinate_reduction(node, shapes, rows, output);
        case 33U: case 34U: case 92U: case 93U: return row == axis ? -2 : row;
        case 94U: case 95U: case 96U: return row >= 0 && row == (int)attrs[0] ? -2 : row;
        case 90U: case 91U:
            if (row >= (int)input.count - (int)attrs[0]) return -2;
            return synaptik_scan_coordinate_pointwise(node, shapes, rows, output);
        case 88U: case 89U: return synaptik_scan_coordinate_batch_norm(node, shapes, rows, slot);
        case 15U: return synaptik_scan_coordinate_matmul(node, shapes, rows);
        case 35U: return synaptik_scan_coordinate_attention(node, shapes, rows, slot);
        case 36U: case 37U: return synaptik_scan_coordinate_convolution(node, shapes, rows);
        case 97U: case 98U: case 99U: case 100U: {
            if (row < 2) return row;
            uint32_t dimensions = node->operation <= 98U ? 2U : 3U, spatial = (uint32_t)row - 2U;
            return attrs[spatial] == 1U && attrs[dimensions + spatial] == 1U
                    && attrs[dimensions * 2U + spatial] == 0U ? row : -2;
        }
        case 19U: case 80U: case 81U: case 82U: case 83U: case 84U:
            return synaptik_scan_coordinate_window(node, input, row, output);
        case 85U: case 86U: case 87U: return synaptik_scan_coordinate_loss(node, shapes, rows);
        default: return -2;
    }
}

/* Independently recomputes EVERY canonical sibling before controlled pipelines are built.
 * rows/terminals are borrowed value-ID-indexed arrays bounded by shapes.count; the node's
 * ordinary equations/attributes and all operand facts must already be validated. batch is
 * the actual declared B, including zero/one, never a positive stand-in. No equal extent
 * invents ROW provenance: equality is used only in Model's complete coordinate witnesses.
 * Returns NO for an unproved equation, terminal/ROW conflict or forged output claim.
 */
static BOOL synaptik_scan_prove_coordinates(SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes,
        const int32_t *rows, const uint8_t *terminals, uint64_t batch) {
    if (shapes == nil || rows == NULL || terminals == NULL || batch > INT64_MAX
            || node.input_count == 0U
            || (node.operation != 77U && node.operation != 78U
                    && node.input_count > SYNAPTIK_MAX_NODE_INPUTS)
            || node.output_count == 0U || node.output_count > SYNAPTIK_MAX_NODE_OUTPUTS
            || node.operation == 0U || node.operation > 115U
            || (node.operation >= 101U && node.operation <= 105U)) return NO;
    BOOL anyRow = NO;
    for (uint32_t i = 0U; i < node.input_count; i++) {
        uint32_t value = node.inputs[i];
        if (value >= shapes.count || shapes[value] == nil || shapes[value].count > SYNAPTIK_MAX_RANK
                || rows[value] < -1 || rows[value] >= (int32_t)shapes[value].count || terminals[value] > 1U) return NO;
        if (rows[value] >= 0) anyRow = YES;
    }
    for (uint32_t slot = 0U; slot < node.output_count; slot++) {
        uint32_t value = node.outputs[slot];
        if (value >= shapes.count || shapes[value] == nil || shapes[value].count > SYNAPTIK_MAX_RANK) return NO;
        int row = anyRow ? synaptik_scan_coordinate_output(&node, shapes, rows, terminals, slot) : -1;
        if (row < -1 || row >= (int)shapes[value].count) return NO;
        BOOL terminal = NO, eliminated = NO;
        for (uint32_t input = 0U; input < node.input_count; input++) {
            if (!synaptik_scan_coordinate_direct(node.operation, slot, input)) continue;
            uint32_t source = node.inputs[input];
            terminal |= terminals[source] != 0U;
            eliminated |= rows[source] >= 0 && row < 0;
        }
        if (row >= 0 && (terminal || shapes[value][row].unsignedLongLongValue != batch)) return NO;
        if (rows[value] != row || terminals[value] != (uint8_t)(terminal || eliminated)) return NO;
    }
    return YES;
}

#endif
