#ifndef SYNAPTIK_EXTENDED_GRADIENTS_H
#define SYNAPTIK_EXTENDED_GRADIENTS_H

/* Raw metadata validation, before normalized native nodes erase original per-port flags.
 * This checks only owned extensions; existing ordinary domains retain their other guards.
 * Geometry, scalar carriers, body activity and actual gradients are independently owned.
 */
static BOOL synaptik_extended_gradients_valid(
        uint32_t operation, uint32_t input_count, const uint8_t *types,
        const BOOL *inputs, uint32_t output_count, const BOOL *outputs) {
    if (types == NULL || inputs == NULL || outputs == NULL || output_count == 0U) return NO;
    if (operation == 35U) {
        if (input_count < 3U || input_count > 4U || output_count > 2U
                || (input_count == 4U && inputs[3])) return NO;
        BOOL scores = inputs[0] || inputs[1];
        return outputs[0] == (scores || inputs[2])
                && (output_count == 1U || outputs[1] == scores);
    }
    if (operation == 89U) {
        return input_count == 5U && output_count == 5U
                && outputs[0] == (inputs[0] || inputs[1] || inputs[2])
                && outputs[1] == (inputs[0] || inputs[3])
                && outputs[2] == (inputs[0] || inputs[4])
                && outputs[3] == inputs[0] && outputs[4] == inputs[0];
    }
    if (operation >= 103U && operation <= 105U) {
        if (input_count < 5U || inputs[1]) return NO;
        BOOL gradient = NO;
        for (uint32_t role = 0U; role < input_count; role++)
            if (role != 1U) gradient |= inputs[role];
        for (uint32_t port = 0U; port < output_count; port++)
            if (outputs[port] != gradient) return NO;
        return YES;
    }
    if (operation == 18U || operation == 70U || operation == 72U) {
        if (input_count != 3U || output_count != 1U || inputs[1]) return NO;
        if ((types[0] == 2U || types[0] == 6U || types[0] == 3U)
                && (inputs[0] || inputs[2] || outputs[0])) return NO;
        return outputs[0] == (inputs[0] || inputs[2]);
    }
    if (operation >= 20U && operation <= 25U) {
        if (input_count != 2U || output_count != 1U || outputs[0]) return NO;
        for (uint32_t i = 0U; i < input_count; i++)
            if ((types[i] == 2U || types[i] == 6U || types[i] == 3U) && inputs[i]) return NO;
        return YES;
    }
    if (operation >= 86U && operation <= 93U) {
        if (input_count == 0U || output_count != 1U) return NO;
        if (operation == 87U && (input_count != 2U || inputs[1])) return NO;
        BOOL gradient = NO;
        for (uint32_t role = 0U; role < input_count; role++) gradient |= inputs[role];
        return outputs[0] == gradient;
    }
    /* Index-valued arg extrema do not inherit the data role's gradient bit. */
    BOOL reduction = operation == 13U || operation == 14U || operation == 31U
            || operation == 32U || operation == 106U || (operation >= 111U && operation <= 115U);
    if (reduction && (types[0] == 1U || types[0] == 5U || types[0] == 7U)) {
        return (input_count == 1U || (input_count == 2U && !inputs[1]))
                && output_count == 1U && outputs[0] == inputs[0];
    }
    BOOL elementary = operation == 38U || operation == 50U || operation == 53U
            || operation == 54U || (operation >= 56U && operation <= 59U)
            || (operation >= 65U && operation <= 68U)
            || ((types[0] == 1U || types[0] == 5U || types[0] == 7U) && (operation == 1U || operation == 12U
                    || (operation >= 28U && operation <= 30U)
                    || (operation >= 46U && operation <= 49U)
                    || operation == 52U || (operation >= 60U && operation <= 64U)));
    BOOL exact = input_count != 0U && (types[0] == 4U || types[0] == 2U || types[0] == 6U)
            && (operation == 1U || (operation >= 2U && operation <= 4U)
                    || operation == 12U || (operation >= 26U && operation <= 30U)
                    || (operation >= 46U && operation <= 48U)
                    || (operation >= 60U && operation <= 63U));
    if (elementary || exact) {
        if (input_count == 0U || output_count != 1U) return NO;
        BOOL gradient = NO;
        for (uint32_t role = 0U; role < input_count; role++) {
            if ((types[role] == 2U || types[role] == 6U) && inputs[role]) return NO;
            gradient |= inputs[role];
        }
        return outputs[0] == gradient;
    }
    return YES;
}

/* Same per-port proof for raw outer ordinary instructions and decoded physical body units.
 * Ordinary Model results inherit the OR of floating input roles; three canonical multi-result
 * neural operations have narrower explicitly named role relations above. Integer/BOOL storage
 * never acquires gradient metadata merely because another result has a floating carrier. */
static BOOL synaptik_scan_validate_gradient_flags(
        SynaptikMetalDecodedNode node, const uint8_t *types, const uint8_t *flags) {
    if (types == NULL || flags == NULL
            || node.output_count == 0U || node.output_count > SYNAPTIK_MAX_NODE_OUTPUTS) return NO;
    if (node.operation == 77U || node.operation == 78U) {
        if (node.input_count == 0U || node.output_count != 1U) return NO;
        uint8_t output_type = types[node.outputs[0]];
        BOOL floating = output_type == 1U || output_type == 4U || output_type == 5U || output_type == 7U;
        BOOL gradient = NO;
        for (uint32_t role = 0U; role < node.input_count; role++) {
            uint32_t value = node.inputs[role];
            if (types[value] != output_type || (!floating && flags[value] != 0U)) return NO;
            gradient |= flags[value] != 0U;
        }
        return (flags[node.outputs[0]] != 0U) == gradient;
    }
    if (node.input_count > SYNAPTIK_MAX_NODE_INPUTS) return NO;
    uint8_t inputs_types[SYNAPTIK_MAX_NODE_INPUTS] = {0};
    BOOL inputs[SYNAPTIK_MAX_NODE_INPUTS] = {NO}, outputs[SYNAPTIK_MAX_NODE_OUTPUTS] = {NO};
    BOOL any = NO;
    for (uint32_t role = 0U; role < node.input_count; role++) {
        uint32_t value = node.inputs[role]; uint8_t type = types[value];
        inputs_types[role] = type; inputs[role] = flags[value] != 0U;
        if (inputs[role] && type != 1U && type != 4U && type != 5U && type != 7U) return NO;
        any |= inputs[role];
    }
    for (uint32_t port = 0U; port < node.output_count; port++) outputs[port] = flags[node.outputs[port]] != 0U;
    if (!synaptik_extended_gradients_valid(node.operation, node.input_count, inputs_types,
                inputs, node.output_count, outputs)) return NO;
    if (node.operation == 35U || node.operation == 89U
            || (node.operation >= 103U && node.operation <= 105U)) return YES;
    for (uint32_t port = 0U; port < node.output_count; port++) {
        uint8_t type = types[node.outputs[port]];
        BOOL floating = type == 1U || type == 4U || type == 5U || type == 7U;
        if (outputs[port] != (floating && any)) return NO;
    }
    return YES;
}

#endif
