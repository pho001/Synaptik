#ifndef SYNAPTIK_NORMALIZATION_HOST_H
#define SYNAPTIK_NORMALIZATION_HOST_H

/* Import after SynaptikMetalDecodedNode, SynaptikNormalizationLossMeta and
 * custom_physical_layout and the shared broadcast-aware synaptik_storage_layout_supported.
 * No source strings, pipeline creation or ABI exports.
 * The caller independently validates topology, declared gradient role bits and
 * concrete buffer lengths; this helper validates native actual types, complete
 * siblings, scalar words, shapes, affine address arithmetic and canonical writes.
 * Runtime target prevalidation (ignore BEFORE bounds) remains a pre-write step.
 */
static BOOL synaptik_nl_product(MPSShape *shape, NSUInteger first, NSUInteger last, uint64_t *result) {
    uint64_t product = 1U;
    for (NSUInteger a = first; a < last; a++) {
        if (shape[a].longLongValue < 0) return NO;
        uint64_t d = shape[a].unsignedLongLongValue;
        if (d != 0U && product > UINT64_MAX / d) return NO;
        product *= d;
    }
    *result = product;
    return YES;
}

static BOOL synaptik_nl_scalar(const uint64_t *words, uint32_t start, BOOL positive, uint32_t *bits) {
    if (words[start] != 1U || words[start+1U] > UINT32_MAX) return NO;
    uint32_t raw = (uint32_t)words[start+1U];
    float value;
    memcpy(&value, &raw, sizeof(value));
    if (!isfinite(value) || (positive && !(value > 0.0f))) return NO;
    *bits = raw;
    return YES;
}

/* Adapts one resolved role to the existing shared storage proof, without a
 * second nonoverlap algorithm or a blanket exemption for zero-stride axes. */
static BOOL synaptik_nl_storage_supported(MPSShape *shape,
        const SynaptikNormalizationLossView *view, uint64_t span, uint8_t state) {
    uint32_t rank = (uint32_t)shape.count;
    uint64_t dimensions[16] = {0};
    for (uint32_t axis = 0U; axis < rank; axis++)
        dimensions[axis] = shape[axis].unsignedLongLongValue;
    uint8_t present = 1U, isView = state == 2U || state == 3U;
    return synaptik_storage_layout_supported(0U, &rank, dimensions, view->strides,
            &view->offset, &span, &present, &isView, 0U);
}

static BOOL synaptik_nl_layout(MPSShape *shape, uint32_t value, uint8_t type,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        BOOL output, SynaptikNormalizationLossView *view) {
    if (shape == nil || shape.count > 16U || (type != 1U && type != 2U && type != 6U)) return NO;
    uint64_t elements;
    if (!synaptik_nl_product(shape, 0U, shape.count, &elements)) return NO;
    uint64_t width = type == 6U ? 8U : 4U;
    if (elements > NSUIntegerMax / width) return NO;
    if (elements == 0U) {
        // No cell is addressed. custom_physical_layout's nonempty-only premise
        // is not used to reject an otherwise valid empty normalization result.
        view->offset = 0U;
        uint64_t stride = 1U;
        for (NSUInteger a = shape.count; a-- > 0U;) {
            view->strides[a] = stride;
            uint64_t extent = shape[a].unsignedLongLongValue;
            if (extent != 0U && stride > UINT64_MAX / extent) return NO;
            stride *= extent;
        }
        return synaptik_nl_storage_supported(shape, view, 0U, states[value]);
    }
    if (!custom_physical_layout(shape, value, strides, offsets, states, view->strides, &view->offset)) return NO;
    uint64_t last = view->offset, canonical = 1U;
    if (last > NSUIntegerMax / width) return NO;
    for (NSUInteger a = shape.count; a-- > 0U;) {
        uint64_t extent = shape[a].unsignedLongLongValue;
        uint64_t stride = view->strides[a];
        if (output && stride != canonical) return NO;
        if (extent > 1U && stride > (UINT64_MAX-last)/(extent-1U)) return NO;
        last += (extent-1U)*stride;
        if (canonical > UINT64_MAX/extent) return NO;
        canonical *= extent;
    }
    return last < NSUIntegerMax / width && (!output || view->offset == 0U)
            && synaptik_nl_storage_supported(shape, view, last+1U, states[value]);
}

static BOOL synaptik_normalization_metadata(SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes, const uint8_t *types,
        const uint64_t *strides, const uint64_t *offsets, const uint8_t *states,
        SynaptikNormalizationLossMeta *meta, MTLSize *grid) {
    if (shapes == nil || types == NULL || meta == NULL || grid == NULL
            || strides == NULL || offsets == NULL || states == NULL
            || node.operation < 86U || node.operation > 93U || node.input_count < 1U
            || node.input_count > 5U || node.output_count < 1U || node.output_count > 5U) return NO;
    BOOL batch = node.operation == 88U || node.operation == 89U;
    BOOL training = node.operation == 89U;
    BOOL layer = node.operation == 90U;
    BOOL rms = node.operation == 91U;
    BOOL soft = node.operation == 92U || node.operation == 93U;
    BOOL index = node.operation == 87U;
    BOOL loss = node.operation == 86U || index;
    if (node.first_input != node.inputs[0] || node.output != node.outputs[0]
            || node.second_input != (node.input_count >= 2U ? node.inputs[1] : UINT32_MAX)
            || node.auxiliary != (node.input_count >= 3U ? node.inputs[2] : 0U)) return NO;
    if ((batch && (node.input_count != 5U || node.output_count != (training ? 5U : 1U)))
            || (soft && (node.input_count != 1U || node.output_count != 1U))
            || (layer && ((node.input_count != 1U && node.input_count != 3U) || node.output_count != 1U))
            || (rms && ((node.input_count != 1U && node.input_count != 2U) || node.output_count != 1U))
            || (loss && (node.input_count != 2U || node.output_count != 1U))) return NO;
    for (uint32_t i = 0U; i < node.input_count; i++) {
        if (node.inputs[i] >= shapes.count) return NO;
        if (states[node.inputs[i]] < 1U || states[node.inputs[i]] > 3U) return NO;
        uint8_t type = types[node.inputs[i]];
        if (index && i == 1U ? type != 2U && type != 6U : type != 1U) return NO;
    }
    for (uint32_t i = 0U; i < node.output_count; i++) {
        if (node.outputs[i] >= shapes.count || types[node.outputs[i]] != 1U) return NO;
        if (states[node.outputs[i]] != 1U) return NO;
        for (uint32_t j = 0U; j < node.input_count; j++) if (node.outputs[i] == node.inputs[j]) return NO;
        for (uint32_t j = 0U; j < i; j++) if (node.outputs[i] == node.outputs[j]) return NO;
    }
    MPSShape *input = shapes[node.inputs[0]];
    if (input.count < 1U || input.count > 16U) return NO;
    SynaptikNormalizationLossMeta m = {0};
    m.mode = node.operation;
    m.rank = (uint32_t)input.count;
    for (NSUInteger a = 0U; a < input.count; a++) m.dims[a] = input[a].unsignedLongLongValue;
    if (!synaptik_nl_product(input, 0U, input.count, &m.elementCount)) return NO;
    uint32_t expectedAttributes = 0U;
    if (soft) {
        if (node.attribute_kind != 3U || node.attribute_count != 1U || node.axis >= input.count) return NO;
        m.axis = node.axis;
    } else if (layer || rms) {
        if (node.attribute_kind != 28U || node.attribute_count < 4U) return NO;
        uint64_t suffixRank = node.attribute_values[0];
        if (suffixRank < 1U || suffixRank > input.count || node.attribute_count != suffixRank+3U) return NO;
        m.axis = (uint32_t)(input.count-suffixRank);
        for (uint32_t a = 0U; a < suffixRank; a++)
            if (node.attribute_values[a+1U] != m.dims[m.axis+a]) return NO;
        if (!synaptik_nl_scalar(node.attribute_values, (uint32_t)suffixRank+1U, YES, &m.epsilonBits)) return NO;
        MPSShape *suffix = [input subarrayWithRange:NSMakeRange(m.axis, (NSUInteger)suffixRank)];
        for (uint32_t role = 1U; role < node.input_count; role++)
            if (![shapes[node.inputs[role]] isEqualToArray:suffix]) return NO;
        m.flags = node.input_count > 1U ? (layer ? 3U : 1U) : 0U;
        expectedAttributes = node.attribute_count;
    } else {
        if (node.attribute_count < 1U || node.attribute_values[0] >= input.count) return NO;
        m.axis = (uint32_t)node.attribute_values[0];
        if (node.axis != UINT32_MAX) return NO;
        if (batch) {
            if (input.count < 2U || node.attribute_kind != (training ? 30U : 29U)
                    || node.attribute_count != (training ? 5U : 3U)) return NO;
            if (!synaptik_nl_scalar(node.attribute_values, training ? 3U : 1U, YES, &m.epsilonBits)) return NO;
            if (training) {
                if (!synaptik_nl_scalar(node.attribute_values, 1U, NO, &m.momentumBits)) return NO;
                float momentum;
                memcpy(&momentum, &m.momentumBits, sizeof(momentum));
                if (momentum < 0.0f || momentum > 1.0f) return NO;
            }
            MPSShape *channel = @[@(m.dims[m.axis])];
            for (uint32_t role = 1U; role < 5U; role++)
                if (![shapes[node.inputs[role]] isEqualToArray:channel]) return NO;
            if (training) for (uint32_t role = 1U; role < 5U; role++)
                if (![shapes[node.outputs[role]] isEqualToArray:channel]) return NO;
        } else {
            if (node.attribute_kind != (index ? 27U : 26U) || node.attribute_count < 2U
                    || node.attribute_values[1] < 1U || node.attribute_values[1] > 3U) return NO;
            m.reduction = (uint32_t)node.attribute_values[1]-1U;
            if (index) {
                if (node.attribute_count < 3U || node.attribute_values[2] > 1U
                        || node.attribute_count != (node.attribute_values[2] ? 5U : 3U)) return NO;
                m.indexType = types[node.inputs[1]];
                if (node.attribute_values[2]) {
                    if (node.attribute_values[3] != m.indexType
                            || (m.indexType == 2U && node.attribute_values[4] > UINT32_MAX)) return NO;
                    m.flags = 4U;
                    uint64_t raw = node.attribute_values[4];
                    if (m.indexType == 2U) m.ignoreIndex = (int32_t)(uint32_t)raw;
                    else memcpy(&m.ignoreIndex, &raw, sizeof(raw));
                }
            } else if (node.attribute_count != 2U
                    || ![shapes[node.inputs[1]] isEqualToArray:input]) return NO;
        }
        expectedAttributes = node.attribute_count;
    }
    // No alternate unused words or hidden scalar/axis header fields are accepted.
    if (!soft && node.axis != UINT32_MAX) return NO;
    for (uint32_t i = expectedAttributes; i < SYNAPTIK_MAX_ATTRIBUTE_WORDS; i++)
        if (node.attribute_values[i] != 0U) return NO;
    if (layer || rms) {
        if (!synaptik_nl_product(input, 0U, m.axis, &m.rowCount)
                || !synaptik_nl_product(input, m.axis, input.count, &m.featureCount)) return NO;
    } else {
        m.featureCount = m.dims[m.axis];
        m.rowCount = 1U;
        for (uint32_t a = 0U; a < m.rank; a++) if (a != m.axis) {
            if (m.dims[a] != 0U && m.rowCount > UINT64_MAX/m.dims[a]) return NO;
            m.rowCount *= m.dims[a];
        }
    }
    if (training && m.featureCount != 0U && m.rowCount < 2U) return NO;
    NSMutableArray<NSNumber *> *samples = [input mutableCopy];
    [samples removeObjectAtIndex:m.axis];
    if (loss) {
        if (index && ![shapes[node.inputs[1]] isEqualToArray:samples]) return NO;
        if (m.featureCount == 0U && m.rowCount != 0U && !(index && (m.flags & 4U))) return NO;
        MPSShape *expected = m.reduction == 0U ? samples : @[];
        if (![shapes[node.outputs[0]] isEqualToArray:expected]) return NO;
    } else if (![shapes[node.outputs[0]] isEqualToArray:input]) return NO;
    for (uint32_t role = 0U; role < node.input_count; role++) {
        uint32_t value = node.inputs[role];
        if (!synaptik_nl_layout(shapes[value], value, types[value], strides, offsets, states, NO, &m.views[role])) return NO;
    }
    for (uint32_t role = 0U; role < node.output_count; role++) {
        uint32_t value = node.outputs[role];
        SynaptikNormalizationLossView output = {0};
        if (!synaptik_nl_layout(shapes[value], value, types[value], strides, offsets, states, YES, &output)) return NO;
    }
    uint64_t workitems = batch ? m.featureCount : (loss && m.reduction != 0U ? 1U : m.rowCount);
    if (!loss && m.elementCount == 0U) workitems = 0U;
    if (workitems > UINT32_MAX) return NO;
    *meta = m;
    *grid = MTLSizeMake((NSUInteger)workitems, 1U, 1U);
    return YES;
}

/**
 * Validates every eligible index target before any categorical-loss output write.
 * targetBytes addresses the start of the actual target buffer; targetByteLength
 * bounds that complete buffer and targetBindingOffset is its byte binding offset.
 * Metadata view offsets/strides remain element-addressed. A null pointer is valid
 * when no row is eligible; it is rejected before the first eligible target load.
 * The caller synchronizes a produced target before invoking this CPU read-only
 * preflight; feed targets can be checked before the partition's first mutation.
 * GeneralScan's authenticated host eligibility helper must be defined first.
 * Dense targets remain a Model caller obligation, not a new rejection policy.
 */
static int32_t synaptik_normalization_preflight_target(
        const SynaptikNormalizationLossMeta *m,
        const void *targetBytes,
        NSUInteger targetByteLength,
        NSUInteger targetBindingOffset,
        const uint64_t *scanControl,
        const SynaptikMetalScanStepControl *scanStep) {
    if (m == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    if (m->mode != 87U) return SYNAPTIK_METAL_STATUS_OK;
    if (m->rank == 0U || m->rank > 16U || m->axis >= m->rank
            || (m->indexType != 2U && m->indexType != 6U)
            || m->reduction > 2U || (m->flags & ~4U) != 0U
            || (scanControl != NULL && scanStep == NULL))
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    uint64_t rows = 1U;
    for (uint32_t axis = 0U; axis < m->rank; axis++) {
        if (axis == m->axis) continue;
        uint64_t extent = m->dims[axis];
        if (extent != 0U && rows > UINT64_MAX / extent)
            return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        rows *= extent;
    }
    if (rows != m->rowCount || m->featureCount != m->dims[m->axis])
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    if (scanControl != NULL && m->reduction != 0U && scanStep->rowAxis >= 0)
        return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
    uint64_t width = m->indexType == 2U ? 4U : 8U;
    const uint8_t *boundTargetBytes = NULL;
    NSUInteger available = 0U;
    for (uint64_t row = 0U; row < rows; row++) {
        // CE NONE output coordinates are precisely the target's nonclass shape.
        // Reduced CE is SHARED, so its eligibility is the same for every row.
        if (scanControl != NULL && !synaptik_scan_control_output(
                scanControl, scanStep, m->reduction == 0U ? row : 0U)) continue;
        if (boundTargetBytes == NULL) {
            if (targetBytes == NULL) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            if (targetBindingOffset > targetByteLength)
                return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
            boundTargetBytes = (const uint8_t *)targetBytes + targetBindingOffset;
            available = targetByteLength-targetBindingOffset;
        }
        uint64_t remaining = row, address = m->views[1].offset;
        for (uint32_t axis = m->rank; axis-- > 0U;) {
            if (axis == m->axis) continue;
            uint64_t extent = m->dims[axis];
            if (extent == 0U) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
            uint64_t coordinate = remaining % extent;
            remaining /= extent;
            uint32_t targetAxis = axis > m->axis ? axis-1U : axis;
            uint64_t stride = m->views[1].strides[targetAxis];
            if (coordinate != 0U && stride > (UINT64_MAX-address)/coordinate)
                return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
            address += coordinate*stride;
        }
        if (address > NSUIntegerMax/width)
            return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
        NSUInteger byteAddress = (NSUInteger)(address*width);
        if (byteAddress > available || width > available-byteAddress)
            return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
        const uint8_t *bytes = boundTargetBytes + byteAddress;
        int64_t selected;
        if (m->indexType == 2U) {
            int32_t narrow;
            memcpy(&narrow, bytes, sizeof(narrow));
            selected = narrow;
        } else memcpy(&selected, bytes, sizeof(selected));
        // Compare in the original exact signed carrier domain, before bounds.
        if ((m->flags & 4U) != 0U && selected == m->ignoreIndex) continue;
        if (selected < 0 || (uint64_t)selected >= m->featureCount)
            return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
    }
    return SYNAPTIK_METAL_STATUS_OK;
}


#endif
