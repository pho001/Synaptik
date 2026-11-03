#ifndef SYNAPTIK_GENERAL_SCAN_VALUE_VALIDATION_H
#define SYNAPTIK_GENERAL_SCAN_VALUE_VALIDATION_H

/* Private host preflight, included after the ordinary scatter/normalization owners.
 * The leading records are in exact decoded input order, including nonvalidated numeric
 * roles and roles beyond ScanStepControl's four inline input maps. ROW facts describe
 * the original semantic value, never an inferred equal-sized physical dimension.
 * Bodies bind canonical dense buffers; byte binding offsets remain invocation-owned.
 */
typedef struct {
    uint32_t value, type, width, rank;
    int32_t rowAxis;
    uint32_t reserved;
    uint64_t elements, bytes, rowInner;
    uint64_t dimensions[16], strides[16];
} SynaptikMetalScanValidationRole;

static BOOL synaptik_scan_validation_scatter(uint32_t operation) {
    return operation == 18U || operation == 70U || operation == 72U;
}

/* Every scatter retains one cold canonical DataMeta tail for host validation only.
 * Its reduction is the original semantic wire value, before ordinary decoding can
 * normalize exact replacement to AXIS/count 1. Kernel metadata remains untouched.
 * Genuine empty-output markers consume the same tail and ordinary raw preflight.
 */
static NSData *synaptik_scan_validation_roles(SynaptikMetalDecodedNode node,
        NSArray<MPSShape *> *shapes, const uint8_t *types,
        const int32_t *allValueRowAxes, const uint64_t *allValueRowInner,
        uint32_t semanticScatterReduction) {
    if (shapes == nil || types == NULL || allValueRowAxes == NULL || allValueRowInner == NULL
            || node.output_count == 0U || node.output_count > SYNAPTIK_MAX_NODE_OUTPUTS)
        return nil;
    BOOL scatter = synaptik_scan_validation_scatter(node.operation);
    if (scatter && semanticScatterReduction == 0U) return nil;
    for (uint32_t output = 0U; output < node.output_count; output++) {
        if (node.outputs[output] >= shapes.count || shapes[node.outputs[output]].count > 16U) return nil;
        if (shapes[node.outputs[output]] == nil) return nil;
        for (NSNumber *dimension in shapes[node.outputs[output]]) {
            if (dimension.longLongValue < 0) return nil;
        }
    }
    NSUInteger roleBytes = (NSUInteger)node.input_count * sizeof(SynaptikMetalScanValidationRole);
    NSMutableData *data = [NSMutableData dataWithLength:roleBytes
            + (scatter ? sizeof(SynaptikMetalDataMeta) : 0U)];
    if (data == nil) return nil;
    SynaptikMetalScanValidationRole *roles = data.mutableBytes;
    for (uint32_t role = 0U; role < node.input_count; role++) {
        uint32_t value = node.inputs[role];
        if (value >= shapes.count) return nil;
        MPSShape *shape = shapes[value];
        if (shape == nil || shape.count > 16U) return nil;
        SynaptikMetalScanValidationRole *fact = roles + role;
        fact->value = value;
        fact->type = types[value];
        fact->width = (uint32_t)value_type_width(types[value]);
        fact->rank = (uint32_t)shape.count;
        fact->rowAxis = allValueRowAxes[value];
        fact->rowInner = allValueRowInner[value];
        if (fact->width == 0U || fact->rowAxis < -1 || fact->rowAxis >= (int32_t)fact->rank
                || (fact->rowAxis < 0 && fact->rowInner != 0U)) return nil;
        uint64_t count = 1U, inner = fact->rowAxis < 0 ? 0U : 1U;
        for (uint32_t reverse = fact->rank; reverse > 0U; reverse--) {
            uint32_t axis = reverse - 1U;
            if (shape[axis].longLongValue < 0) return nil;
            uint64_t extent = shape[axis].unsignedLongLongValue;
            fact->dimensions[axis] = extent;
            fact->strides[axis] = count;
            if (extent != 0U && count > UINT64_MAX / extent) return nil;
            count *= extent;
            if ((int32_t)axis > fact->rowAxis && fact->rowAxis >= 0) {
                if (extent != 0U && inner > UINT64_MAX / extent) return nil;
                inner *= extent;
            }
        }
        if (inner != fact->rowInner || count > NSUIntegerMax / fact->width) return nil;
        fact->elements = count;
        fact->bytes = count * fact->width;
    }
    if (scatter) {
        if (node.input_count != 3U || node.output_count != 1U
                || (roles[1].type != 2U && roles[1].type != 6U)
                || node.attribute_count > SYNAPTIK_MAX_ATTRIBUTE_WORDS) return nil;
        SynaptikMetalDataMeta *meta = (SynaptikMetalDataMeta *)((uint8_t *)data.mutableBytes + roleBytes);
        meta->operation = node.operation;
        meta->inputCount = 3U;
        meta->inputRank = roles[0].rank;
        meta->auxiliaryRank = roles[1].rank;
        meta->outputRank = (uint32_t)shapes[node.outputs[0]].count;
        meta->sourceType = roles[0].type;
        meta->targetType = types[node.outputs[0]];
        meta->indexType = roles[1].type;
        meta->width = roles[0].width;
        meta->attributeKind = node.attribute_kind;
        meta->attributeCount = node.attribute_count;
        for (uint32_t word = 0U; word < node.attribute_count; word++)
            meta->attrs[word] = node.attribute_values[word];
        if (node.operation == 18U || node.operation == 70U) meta->attrs[0] = node.axis;
        meta->attrs[1] = semanticScatterReduction;
        for (uint32_t role = 0U; role < 3U; role++) {
            meta->roleTypes[role] = roles[role].type;
            meta->roleRanks[role] = roles[role].rank;
            for (uint32_t axis = 0U; axis < roles[role].rank; axis++) {
                meta->roleDims[role][axis] = roles[role].dimensions[axis];
                meta->roleStrides[role][axis] = roles[role].strides[axis];
            }
        }
        uint64_t stride = 1U;
        for (uint32_t reverse = meta->outputRank; reverse > 0U; reverse--) {
            uint32_t axis = reverse - 1U;
            uint64_t extent = shapes[node.outputs[0]][axis].unsignedLongLongValue;
            meta->outputDims[axis] = extent;
            meta->outputStrides[axis] = stride;
            if (extent != 0U && stride > UINT64_MAX / extent) return nil;
            stride *= extent;
        }
        meta->elementCount = stride;
    }
    return [data copy];
}

static BOOL synaptik_scan_validation_eligible(const SynaptikMetalScanValidationRole *role,
        uint32_t input, uint64_t ordinal, const uint64_t *control,
        const SynaptikMetalScanStepControl *step) {
    uint32_t scope = step->scope;
    if (step->leftScope != UINT32_MAX && input < 2U)
        scope = input == 0U ? step->leftScope : step->rightScope;
    if (role->rowAxis < 0) return synaptik_scan_control_any(control, scope);
    if (control[1] == 0U || role->rowInner == 0U) return NO;
    return synaptik_scan_control_row(control, scope, (ordinal / role->rowInner) % control[1]);
}

/* Resolve only after logical eligibility. Absent values may have no host-accessible
 * backing at all; neither their object nor their contents is inspected by this path.
 */
static int32_t synaptik_scan_validation_bytes(const SynaptikMetalScanValidationRole *role,
        id<MTLBuffer> __unsafe_unretained const *buffers,
        uint32_t bufferCount, const NSUInteger *offsets, const uint8_t **bytes) {
    if (role->value >= bufferCount || role->elements == 0U)
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    __unsafe_unretained id<MTLBuffer> buffer = buffers[role->value];
    if (buffer == nil) return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    NSUInteger offset = offsets[role->value], length = buffer.length;
    if (offset > length || role->bytes > length - offset)
        return SYNAPTIK_METAL_STATUS_INCOMPATIBLE_RESOURCE;
    const uint8_t *contents = buffer.contents;
    if (contents == NULL) return SYNAPTIK_METAL_STATUS_COPY_FAILED;
    *bytes = contents + offset;
    return SYNAPTIK_METAL_STATUS_OK;
}

static int32_t synaptik_scan_validate_value_domains(SynaptikMetalProgramStep *step,
        SynaptikMetalDecodedNode node, id<MTLBuffer> __unsafe_unretained const *buffers,
        uint32_t bufferCount, const NSUInteger *offsets,
        const uint64_t *control, const SynaptikMetalScanStepControl *stepControl) {
    if (step == nil || buffers == NULL || offsets == NULL || control == NULL || stepControl == NULL)
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    __unsafe_unretained NSData *facts = step.scanValidationRoles;
    NSUInteger roleBytes = (NSUInteger)node.input_count * sizeof(SynaptikMetalScanValidationRole);
    NSUInteger expectedBytes = roleBytes
            + (synaptik_scan_validation_scatter(node.operation) ? sizeof(SynaptikMetalDataMeta) : 0U);
    if (facts == nil || facts.length != expectedBytes)
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    const SynaptikMetalScanValidationRole *roles = facts.bytes;
    for (uint32_t input = 0U; input < node.input_count; input++) {
        const SynaptikMetalScanValidationRole *role = roles + input;
        if (role->value != node.inputs[input]) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        if (role->type != SYNAPTIK_METAL_TYPE_BOOL) continue;
        const uint8_t *contents = NULL;
        for (uint64_t ordinal = 0U; ordinal < role->elements; ordinal++) {
            if (!synaptik_scan_validation_eligible(role, input, ordinal, control, stepControl)) continue;
            if (contents == NULL) {
                int32_t status = synaptik_scan_validation_bytes(role, buffers, bufferCount, offsets, &contents);
                if (status != SYNAPTIK_METAL_STATUS_OK) return status;
            }
            if (contents[ordinal] > 1U) return SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT;
        }
    }
    if (synaptik_scan_validation_scatter(node.operation)) {
        if (node.input_count != 3U) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        if (roles[2].elements == 0U || !synaptik_scan_control_any(control, stepControl->scope))
            return SYNAPTIK_METAL_STATUS_OK;
        const SynaptikMetalDataMeta *meta =
                (const SynaptikMetalDataMeta *)((const uint8_t *)roles + roleBytes);
        __unsafe_unretained id<MTLBuffer> inputs[3];
        NSUInteger inputOffsets[3];
        for (uint32_t input = 0U; input < 3U; input++) {
            if (roles[input].value >= bufferCount) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
            inputs[input] = buffers[roles[input].value];
            inputOffsets[input] = offsets[roles[input].value];
        }
        return synaptik_scatter_scan_preflight_raw(meta, inputs, 3U, inputOffsets, control, stepControl);
    }
    if (node.operation == 87U) {
        if (node.input_count != 2U) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        const SynaptikMetalScanValidationRole *target = roles + 1U;
        if (target->elements == 0U || !synaptik_scan_control_any(control, stepControl->scope))
            return SYNAPTIK_METAL_STATUS_OK;
        if (step.metadata == nil || step.metadata.length < sizeof(SynaptikNormalizationLossMeta))
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        const SynaptikNormalizationLossMeta *meta = step.metadata.contents;
        if (meta == NULL) return SYNAPTIK_METAL_STATUS_COPY_FAILED;
        if (meta->reduction != 0U) {
            if (target->rowAxis >= 0 || stepControl->rowAxis >= 0)
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        } else if (target->rowAxis >= 0) {
            if (target->rowAxis != stepControl->rowAxis || target->rowInner != stepControl->rowInner)
                return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        } else if (stepControl->rowAxis >= 0
                && ((uint32_t)stepControl->rowAxis >= target->rank
                    || target->dimensions[stepControl->rowAxis] != 1U))
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        if (target->value >= bufferCount) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        __unsafe_unretained id<MTLBuffer> buffer = buffers[target->value];
        return synaptik_normalization_preflight_target(meta, buffer.contents, buffer.length,
                offsets[target->value], control, stepControl);
    }
    uint32_t indexRole;
    if (node.operation == 17U) indexRole = 0U;
    else if (node.operation == 16U || node.operation == 69U || node.operation == 71U) indexRole = 1U;
    else return SYNAPTIK_METAL_STATUS_OK;
    if (indexRole >= node.input_count || (roles[indexRole].type != 2U && roles[indexRole].type != 6U))
        return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    const SynaptikMetalScanValidationRole *indices = roles + indexRole;
    const SynaptikMetalScanValidationRole *data = roles;
    uint64_t bound = 0U, depth = 0U, batch = 0U;
    if (node.operation == 17U) bound = node.attribute_values[0];
    else if (node.operation == 71U) {
        if (indices->rank == 0U) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        depth = indices->dimensions[indices->rank - 1U];
        batch = node.attribute_values[0];
        if (depth == 0U || batch > data->rank || depth > data->rank - batch)
            return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
    } else {
        if (node.axis >= data->rank) return SYNAPTIK_METAL_STATUS_INTERNAL_ERROR;
        bound = data->dimensions[node.axis];
    }
    const uint8_t *contents = NULL;
    for (uint64_t ordinal = 0U; ordinal < indices->elements; ordinal++) {
        if (!synaptik_scan_validation_eligible(indices, indexRole, ordinal, control, stepControl)) continue;
        if (contents == NULL) {
            int32_t status = synaptik_scan_validation_bytes(indices, buffers, bufferCount, offsets, &contents);
            if (status != SYNAPTIK_METAL_STATUS_OK) return status;
        }
        const uint8_t *bytes = contents + (NSUInteger)(ordinal * indices->width);
        int64_t selected;
        if (indices->type == SYNAPTIK_METAL_TYPE_INT32) {
            int32_t narrow;
            memcpy(&narrow, bytes, sizeof(narrow));
            selected = narrow;
        } else memcpy(&selected, bytes, sizeof(selected));
        uint64_t componentBound = depth == 0U ? bound : data->dimensions[batch + ordinal % depth];
        if (selected < 0 || (uint64_t)selected >= componentBound)
            return SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS;
    }
    return SYNAPTIK_METAL_STATUS_OK;
}

#endif
