#ifndef SYNAPTIK_ORDINARY_ATTRIBUTES_H
#define SYNAPTIK_ORDINARY_ATTRIBUTES_H

/* One normalized ordinary attribute decoder, shared by outer programs and Scan body units.
 * Raw scalar carriers and complete signatures are checked before normalized POD fields erase them.
 * Include after the raw signature/attribute validators and SynaptikMetalDecodedNode declaration.
 */
static BOOL synaptik_decode_attributes_in_context(
        SynaptikMetalDecodedNode *result, const uint8_t *words,
        uint32_t attribute_word_count, const uint8_t *types, BOOL scan_body) {
    if (result == NULL || types == NULL) return NO;
    SynaptikMetalDecodedNode node = *result;
    node.scan_body = scan_body;
    uint32_t operation = node.operation, attribute_kind = node.attribute_kind;
    if (!synaptik_node_signature_is_valid(operation, attribute_kind,
                node.input_count, node.output_count)
            || !synaptik_attribute_is_valid_in_context(operation, attribute_kind, words, attribute_word_count, scan_body))
        return NO;
    if ((attribute_kind == 7U || attribute_kind == 8U)
            && (synaptik_read_le64(words) != types[node.first_input]
                    || (attribute_kind == 8U
                            && synaptik_read_le64(words + 16U) != types[node.first_input]))) return NO;
    node.axis = UINT32_MAX;
    if (attribute_kind == 0U) {
        if (attribute_word_count != 0U)
            return NO;
    } else if (attribute_kind == 1U || attribute_kind == 2U) {
        if (attribute_word_count < 1U
                || synaptik_read_le64(words) != attribute_word_count - 1U
                || attribute_word_count - 1U > SYNAPTIK_MAX_RANK)
            return NO;
        node.attribute_count = attribute_word_count - 1U;
        for (uint32_t word = 0U; word < node.attribute_count; word++)
            node.attribute_values[word] = synaptik_read_le64(words + (uint64_t)(word + 1U) * 8U);
    } else if (attribute_kind == 3U) {
        if (attribute_word_count != 1U)
            return NO;
        uint64_t axis = synaptik_read_le64(words);
        if (axis > UINT32_MAX) return NO;
        node.attribute_count = 1U;
        node.axis = (uint32_t)axis;
    } else if (attribute_kind == 4U) {
        if (attribute_word_count < 3U)
            return NO;
        uint64_t form = synaptik_read_le64(words);
        uint64_t keep = synaptik_read_le64(words + 8U);
        uint64_t count = synaptik_read_le64(words + 16U);
        if (form < 1U || form > 4U || keep > 1U || count > SYNAPTIK_MAX_RANK
                || count != attribute_word_count - 3U
                || (form == 1U && (count != 0U || keep != 0U))
                || (form == 2U && count != 1U)
                || (form == 4U && (operation != 13U || keep != 0U)))
            return NO;
        node.attribute_count = (uint32_t)count;
        node.axis = (uint32_t)form;
        node.auxiliary = (uint32_t)keep;
        for (uint32_t word = 0U; word < node.attribute_count; word++)
            node.attribute_values[word] = synaptik_read_le64(words + (uint64_t)(word + 3U) * 8U);
    } else if (attribute_kind == 5U) {
        if (attribute_word_count != 1U)
            return NO;
        node.attribute_count = 1U;
        node.attribute_values[0] = synaptik_read_le64(words);
    } else if (attribute_kind == 6U) {
        if (attribute_word_count != 3U)
            return NO;
        uint64_t axis = synaptik_read_le64(words);
        if (axis > UINT32_MAX) return NO;
        node.attribute_count = 3U;
        node.axis = (uint32_t)axis;
        node.attribute_values[0] = synaptik_read_le64(words + 8U);
        node.attribute_values[1] = synaptik_read_le64(words + 16U);
    } else if (attribute_kind == 7U) {
        if (attribute_word_count != 2U
                || !synaptik_scalar_is_valid(words, attribute_word_count, 0U))
            return NO;
        node.attribute_count = 1U;
        node.attribute_values[0] = synaptik_read_le64(words + 8U);
    } else if (attribute_kind == 8U) {
        if (attribute_word_count != 4U
                || !synaptik_scalar_is_valid(words, attribute_word_count, 0U)
                || !synaptik_scalar_is_valid(words, attribute_word_count, 2U)
                || synaptik_read_le64(words) != synaptik_read_le64(words + 16U))
            return NO;
        node.attribute_count = 2U;
        node.attribute_values[0] = synaptik_read_le64(words + 8U);
        node.attribute_values[1] = synaptik_read_le64(words + 24U);
    } else if (attribute_kind == 9U) {
        if (attribute_word_count != 3U
                || synaptik_read_le64(words) > UINT32_MAX
                || synaptik_read_le64(words + 8U) > 1U
                || synaptik_read_le64(words + 16U) > 1U)
            return NO;
        node.attribute_count = 2U;
        node.axis = (uint32_t)synaptik_read_le64(words);
        node.attribute_values[0] = synaptik_read_le64(words + 8U);
        node.attribute_values[1] = synaptik_read_le64(words + 16U);
    } else if (attribute_kind == 12U) {
        uint64_t axis = synaptik_read_le64(words);
        uint64_t reduction = synaptik_read_le64(words + 8U);
        if (attribute_word_count != 2U || axis > UINT32_MAX
                || reduction < 1U || reduction > 5U) return NO;
        node.axis = (uint32_t)axis;
        if (reduction == 1U && types[node.first_input] != SYNAPTIK_METAL_TYPE_FLOAT32
                && types[node.first_input] != SYNAPTIK_METAL_TYPE_BFLOAT16
                && types[node.first_input] != SYNAPTIK_METAL_TYPE_FLOAT16) {
            node.attribute_kind = SYNAPTIK_METAL_MPSGRAPH_ATTR_AXIS;
            node.attribute_count = 1U;
        } else {
            node.attribute_count = 2U;
            node.attribute_values[0] = axis;
            node.attribute_values[1] = reduction;
        }
    } else if (attribute_kind >= 10U && attribute_kind <= 41U) {
        node.attribute_count = attribute_word_count;
        for (uint32_t word = 0U; word < attribute_word_count; word++) {
            node.attribute_values[word] =
                    synaptik_read_le64(words + (uint64_t)word * 8U);
        }
    } else {
        return NO;
    }
    *result = node;
    return YES;
}

static BOOL synaptik_decode_ordinary_attributes(SynaptikMetalDecodedNode *node,
        const uint8_t *words, uint32_t count, const uint8_t *types) {
    return synaptik_decode_attributes_in_context(node, words, count, types, NO);
}

static BOOL synaptik_decode_scan_ordinary_attributes(SynaptikMetalDecodedNode *node,
        const uint8_t *words, uint32_t count, const uint8_t *types, uint32_t *semanticScatterReduction) {
    if (semanticScatterReduction != NULL) *semanticScatterReduction = 0U;
    if (node == NULL || !synaptik_scan_ordinary_kind_is_valid(node->operation)
            || !synaptik_decode_attributes_in_context(node, words, count, types, YES)) return NO;
    if (semanticScatterReduction != NULL) {
        if (node->operation == 18U)
            *semanticScatterReduction = node->attribute_kind == 3U ? 1U : (uint32_t)node->attribute_values[1];
        else if (node->operation == 70U) *semanticScatterReduction = 2U;
        else if (node->operation == 72U) *semanticScatterReduction = (uint32_t)node->attribute_values[1];
    }
    return YES;
}

#endif
