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
        }
        if (extent > 1U && result[axis] != 0U) {
            uint32_t place = axes;
            while (place > 0U && result[order[place - 1U]] > result[axis]) {
                order[place] = order[place - 1U]; --place;
            }
            order[place] = (uint32_t)axis; ++axes;
        }
    }
    if (*offset > limit - span) return NO;
    if (broadcast && (states == NULL || states[value] != SYNAPTIK_METAL_VALUE_AFFINE_VIEW))
        return NO;
    uint64_t covered = 1U;
    for (uint32_t item = 0U; item < axes; ++item) {
        uint32_t axis = order[item];
        if (result[axis] < covered) return NO;
        covered += (shape[axis].unsignedLongLongValue - 1U) * result[axis];
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

/*
 * Constructs logical index bounds/uniqueness facts only. The existing physical
 * layout pass subsequently installs indexSpan/indexOffset/indexExtents/indexStrides.
 * Its validation scan remains stable node/row-major order and completes before any
 * scatter output write. Only replacement allocates target scratch: arithmetic
 * duplicate destinations are valid and must never enter the uniqueness check.
 */
static SynaptikMetalIndexValidation *synaptik_scatter_index_validation(
        SynaptikMetalDecodedNode node, NSArray<MPSShape *> *shapes,
        const uint8_t *types, uint32_t node_index, BOOL preflight) {
    SynaptikMetalDataMeta logical = {0};
    MTLSize ignored_grid;
    /* Canonical construction checks the complete logical grammar and products;
     * actual affine span/resource checks remain in the real metadata constructor. */
    if (!synaptik_scatter_metadata(
                node, shapes, types, NULL, NULL, NULL, &logical, &ignored_grid))
        return nil;
    SynaptikMetalIndexValidation *validation = [SynaptikMetalIndexValidation new];
    if (validation == nil) return nil;
    validation.operation = node.operation;
    validation.indexValue = node.inputs[1];
    validation.indexType = logical.indexType;
    validation.nodeIndex = node_index;
    validation.feedPosition = node.inputs[1];
    validation.preflight = preflight;
    uint64_t index_count = 1U;
    for (uint32_t axis = 0U; axis < logical.roleRanks[1]; ++axis)
        index_count *= logical.roleDims[1][axis];
    validation.elementCount = index_count;
    BOOL replacement = logical.attrs[1] == 1U;
    uint32_t axis = (uint32_t)logical.attrs[0];
    validation.axis = axis;
    if (node.operation == 72U) {
        uint32_t depth = (uint32_t)logical.roleDims[1][logical.roleRanks[1] - 1U];
        validation.tupleDepth = depth;
        validation.coordinateExtents = [NSMutableData
                dataWithBytes:&logical.roleDims[0][axis]
                length:(NSUInteger)depth * sizeof(uint64_t)];
        if (validation.coordinateExtents == nil) return nil;
        if (replacement) {
            validation.targetCount = index_count / depth;
            validation.prefixExtents = [NSMutableData
                    dataWithBytes:logical.roleDims[1]
                    length:(NSUInteger)(logical.roleRanks[1] - 1U) * sizeof(uint64_t)];
            if (validation.prefixExtents == nil) return nil;
        }
    } else {
        validation.bound = logical.roleDims[0][axis];
        if (replacement) {
            validation.targetCount = index_count;
            validation.coordinateExtents = [NSMutableData
                    dataWithBytes:logical.roleDims[1]
                    length:(NSUInteger)logical.roleRanks[1] * sizeof(uint64_t)];
            if (validation.coordinateExtents == nil) return nil;
        }
    }
    if (replacement) {
        if (validation.targetCount > (uint64_t)NSUIntegerMax / sizeof(uint64_t))
            return nil;
        validation.dataStrides = [NSMutableData
                dataWithBytes:logical.roleStrides[0]
                length:(NSUInteger)logical.roleRanks[0] * sizeof(uint64_t)];
        validation.targetScratch = [NSMutableData
                dataWithLength:(NSUInteger)validation.targetCount * sizeof(uint64_t)];
        if (validation.dataStrides == nil || validation.targetScratch == nil) return nil;
    }
    return validation;
}

/* Logical-only projection helpers for controlled preflight. They deliberately do
 * not use storage strides to infer ROW membership: Scan's authenticated per-role
 * logical row axis/inner facts are the sole activity authority. */
static void synaptik_scatter_decode(
        uint64_t ordinal, const uint64_t *dims, uint32_t rank, uint64_t *coords) {
    for (uint32_t reverse = 0U; reverse < rank; ++reverse) {
        uint32_t axis = rank - 1U - reverse;
        coords[axis] = ordinal % dims[axis];
        ordinal /= dims[axis];
    }
}

static uint64_t synaptik_scatter_linear(
        const uint64_t *coords, const uint64_t *dims, uint32_t rank) {
    uint64_t ordinal = 0U;
    for (uint32_t axis = 0U; axis < rank; ++axis)
        ordinal = ordinal * dims[axis] + coords[axis];
    return ordinal;
}

static int32_t synaptik_scatter_scan_read_index(
        const SynaptikMetalDataMeta *meta, id<MTLBuffer> indices, NSUInteger offset,
        const uint64_t *coords, uint64_t bound, uint64_t *selected) {
    uint64_t storage = meta->roleOffsets[1];
    for (uint32_t axis = 0U; axis < meta->roleRanks[1]; ++axis) {
        uint64_t stride = meta->roleStrides[1][axis];
        if (coords[axis] != 0U && stride > (UINT64_MAX - storage) / coords[axis])
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        storage += coords[axis] * stride;
    }
    uint64_t width = meta->indexType == 2U ? 4U : 8U;
    if (indices == nil || indices.contents == NULL)
        return SYNAPTIK_METAL_STATUS_COPY_FAILED;
    if ((uint64_t)offset > indices.length || storage > UINT64_MAX / width)
        return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    uint64_t available = (uint64_t)indices.length - offset;
    if (available < width || storage > (available - width) / width)
        return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    const uint8_t *source = (const uint8_t *)indices.contents + offset + storage * width;
    int64_t value;
    if (meta->indexType == 2U) {
        int32_t small;
        memcpy(&small, source, sizeof(small));
        value = small;
    } else {
        memcpy(&value, source, sizeof(value));
    }
    if (value < 0 || (uint64_t)value >= bound)
        return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
    *selected = (uint64_t)value;
    return SYNAPTIK_METAL_STATUS_OK;
}

/* Projects exactly one original update scalar. Eligibility precedes every index
 * load, in the same role2-then-role1 order as scatter_reduce_scan. No update value
 * is inspected, including when poison storage belongs to an inactive ROW. */
static int32_t synaptik_scatter_scan_target(
        const SynaptikMetalDataMeta *meta, id<MTLBuffer> indices, NSUInteger offset,
        const uint64_t *control, const SynaptikMetalScanStepControl *step,
        uint64_t update_ordinal, BOOL *eligible, uint64_t *target) {
    *eligible = NO;
    if (!synaptik_scan_control_operand(control, step, 2U, update_ordinal))
        return SYNAPTIK_METAL_STATUS_OK;
    uint64_t update[16] = {0}, idx[16] = {0}, out[16] = {0};
    synaptik_scatter_decode(update_ordinal, meta->roleDims[2], meta->roleRanks[2], update);
    uint32_t axis = (uint32_t)meta->attrs[0], q = meta->roleRanks[1];
    if (meta->operation == 18U) {
        for (uint32_t a = 0U; a < meta->outputRank; ++a) out[a] = idx[a] = update[a];
        uint64_t index_ordinal = synaptik_scatter_linear(idx, meta->roleDims[1], q);
        if (!synaptik_scan_control_operand(control, step, 1U, index_ordinal))
            return SYNAPTIK_METAL_STATUS_OK;
        int32_t status = synaptik_scatter_scan_read_index(
                meta, indices, offset, idx, meta->outputDims[axis], &out[axis]);
        if (status != SYNAPTIK_METAL_STATUS_OK) return status;
    } else if (meta->operation == 70U) {
        for (uint32_t a = 0U; a < axis; ++a) out[a] = update[a];
        for (uint32_t a = 0U; a < q; ++a) idx[a] = update[axis + a];
        for (uint32_t a = axis + 1U; a < meta->outputRank; ++a) out[a] = update[a - 1U + q];
        uint64_t index_ordinal = synaptik_scatter_linear(idx, meta->roleDims[1], q);
        if (!synaptik_scan_control_operand(control, step, 1U, index_ordinal))
            return SYNAPTIK_METAL_STATUS_OK;
        int32_t status = synaptik_scatter_scan_read_index(
                meta, indices, offset, idx, meta->outputDims[axis], &out[axis]);
        if (status != SYNAPTIK_METAL_STATUS_OK) return status;
    } else {
        uint32_t depth = (uint32_t)meta->roleDims[1][q - 1U];
        for (uint32_t a = 0U; a < axis; ++a) out[a] = update[a];
        for (uint32_t a = 0U; a + 1U < q; ++a) idx[a] = update[a];
        for (uint32_t a = axis + depth; a < meta->outputRank; ++a)
            out[a] = update[q - 1U + a - axis - depth];
        for (uint32_t component = 0U; component < depth; ++component) {
            idx[q - 1U] = component;
            uint64_t index_ordinal = synaptik_scatter_linear(idx, meta->roleDims[1], q);
            if (!synaptik_scan_control_operand(control, step, 1U, index_ordinal))
                return SYNAPTIK_METAL_STATUS_OK;
            int32_t status = synaptik_scatter_scan_read_index(meta, indices, offset, idx,
                    meta->outputDims[axis + component], &out[axis + component]);
            if (status != SYNAPTIK_METAL_STATUS_OK) return status;
        }
    }
    *target = synaptik_scatter_linear(out, meta->outputDims, meta->outputRank);
    *eligible = synaptik_scan_control_output(control, step, *target);
    return SYNAPTIK_METAL_STATUS_OK;
}

/*
 * Allocation-free pre-write validation for controlled scatter. Arithmetic checks
 * active membership/bounds only. Replacement first checks the complete active
 * bounds domain, then checks active target uniqueness, preserving failure ordering
 * without hot scratch allocation or any writes. The uncommon NONE check is
 * pairwise; ADD terminals retain one linear preflight over their update domain.
 */
static int32_t synaptik_scatter_scan_preflight_raw(
        const SynaptikMetalDataMeta *meta,
        __unsafe_unretained id<MTLBuffer> const *role_inputs, NSUInteger role_count,
        const NSUInteger *offsets, const uint64_t *control,
        const SynaptikMetalScanStepControl *step) {
    if (meta == NULL || role_inputs == NULL || role_count != 3U || offsets == NULL || control == NULL
            || step == NULL || meta->outputRank == 0U || meta->outputRank > 16U
            || (meta->operation != 18U && meta->operation != 70U && meta->operation != 72U)
            || (meta->indexType != 2U && meta->indexType != 6U))
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    if (!synaptik_scan_control_any(control, step->scope))
        return SYNAPTIK_METAL_STATUS_OK;
    uint64_t update_count = 1U;
    for (uint32_t role = 0U; role < 3U; ++role)
        if (meta->roleRanks[role] > 16U) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    /* Empty updates have no index demand, even when the retained base is empty.
     * A nonempty active update into an empty indexed extent must instead reach
     * the ordinary bound-zero check, not be rejected as malformed geometry. */
    for (uint32_t axis = 0U; axis < meta->roleRanks[2]; ++axis)
        if (meta->roleDims[2][axis] == 0U) return SYNAPTIK_METAL_STATUS_OK;
    for (uint32_t axis = 0U; axis < meta->roleRanks[2]; ++axis) {
        uint64_t extent = meta->roleDims[2][axis];
        if (update_count > UINT64_MAX / extent) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        update_count *= extent;
    }
    /* Nonempty updates imply nonempty logical indices under the checked Model
     * shape equations. Keep this guard before coordinate projection/division. */
    for (uint32_t axis = 0U; axis < meta->roleRanks[1]; ++axis)
        if (meta->roleDims[1][axis] == 0U) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    __unsafe_unretained id<MTLBuffer> indices = role_inputs[1];
    for (uint64_t ordinal = 0U; ordinal < update_count; ++ordinal) {
        BOOL eligible;
        uint64_t target;
        int32_t status = synaptik_scatter_scan_target(
                meta, indices, offsets[1], control, step, ordinal, &eligible, &target);
        if (status != SYNAPTIK_METAL_STATUS_OK) return status;
    }
    if (meta->operation == 70U || meta->attrs[1] != 1U)
        return SYNAPTIK_METAL_STATUS_OK;
    for (uint64_t ordinal = 0U; ordinal < update_count; ++ordinal) {
        BOOL eligible;
        uint64_t target;
        int32_t status = synaptik_scatter_scan_target(
                meta, indices, offsets[1], control, step, ordinal, &eligible, &target);
        if (status != SYNAPTIK_METAL_STATUS_OK) return status;
        if (!eligible) continue;
        for (uint64_t previous = 0U; previous < ordinal; ++previous) {
            BOOL earlier_eligible;
            uint64_t earlier_target;
            status = synaptik_scatter_scan_target(
                    meta, indices, offsets[1], control, step, previous, &earlier_eligible, &earlier_target);
            if (status != SYNAPTIK_METAL_STATUS_OK) return status;
            if (earlier_eligible && target == earlier_target)
                return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
        }
    }
    return SYNAPTIK_METAL_STATUS_OK;
}

/* Ordinary callers may already own an NSArray table. The Scan time loop calls
 * the borrowed raw core directly, without constructing a Foundation collection. */
static int32_t synaptik_scatter_scan_preflight(
        const SynaptikMetalDataMeta *meta, NSArray<id<MTLBuffer>> *role_inputs,
        const NSUInteger *offsets, const uint64_t *control,
        const SynaptikMetalScanStepControl *step) {
    if (role_inputs.count != 3U) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    __unsafe_unretained id<MTLBuffer> roles[3] = {
        role_inputs[0], role_inputs[1], role_inputs[2]
    };
    return synaptik_scatter_scan_preflight_raw(meta, roles, 3U, offsets, control, step);
}

#endif
