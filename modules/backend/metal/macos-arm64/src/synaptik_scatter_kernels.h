#ifndef SYNAPTIK_SCATTER_KERNELS_H
#define SYNAPTIK_SCATTER_KERNELS_H

/*
 * Assemble after DataMeta/data coordinate helpers, the shared numerical codec,
 * and the exact/ordering raw comparators (exact_extreme, ordering_nan/key).
 *
 * scatter_reduce: data(0), indices(1), updates(2), output(3), DataMeta(4).
 * Dispatch one data_linear_id per output scalar, elementCount = output count.
 * role 0/1/2 describe the actual base/index/update affine storage. outDims and
 * outputRank describe the canonical output. sourceType/width and indexType are
 * the homogeneous value carrier and INT32/INT64 index carrier, respectively.
 * attrs[0] is normalized axis (18/70), or batchDimensions (72). attrs[1] uses
 * the existing reduction wire: NONE=1, ADD=2, MUL=3, MAX=4, MIN=5; operation 70
 * has intrinsic ADD. All ranks, products and storage spans are host-validated.
 *
 * The existing complete logical index prevalidation MUST finish before this
 * dispatch, including for produced indices. NONE additionally requires global
 * target uniqueness; ADD/MUL/MIN/MAX deliberately permit duplicates. Output and
 * all inputs must be distinct. There is no copy pass, scratch buffer or atomic.
 * F32/BF16/F16 use F32 ADD/MUL working sites and narrow once at the output.
 * INT32/INT64 are modular; F64 retains raw extrema only.
 * Unaddressed words are never passed through arithmetic/comparison.
 * The source also requires ScanStepControl and scan_control_operand/output from
 * SynaptikGeneralScanKernelSource. scatter_reduce_scan appends control buffer28
 * and the shared ScanStepControl buffer29. It guards output before the base load, then
 * role2's original logical update ordinal before role1's original logical index
 * ordinal and before either operand load. Binder-supplied inputRowAxes/Inner,
 * never coincident extents, define ROW membership. Prevalidation must use the
 * identical candidate eligibility; inactive candidates do not validate indices.
 */
static NSString *const SynaptikScatterKernelSource =
@"\n"
"inline ulong scatter_combine(ulong base, ulong update, uint reduction, uint type, thread bool &working) {\n"
"    if (reduction == 1u) return update;\n"
"    if (reduction >= 4u) {\n"
"        bool maximum = reduction == 4u;\n"
"        if (type == 1u) return ulong(exact_extreme(uint(base), uint(update), maximum));\n"
"        if (type == 5u || type == 7u) return lp_extreme(base, update, type, maximum);\n"
"        if (type == 4u) {\n"
"            if (ordering_nan(base, type)) return base;\n"
"            if (ordering_nan(update, type)) return update;\n"
"            ulong sign = 0x8000000000000000ul;\n"
"            if (((base | update) & ~sign) == 0ul) return (maximum ? base & update : base | update) & sign;\n"
"            bool less = ordering_key(base, type) < ordering_key(update, type);\n"
"            return maximum ? (less ? update : base) : (less ? base : update);\n"
"        }\n"
"        bool less = type == 2u ? int(uint(base)) < int(uint(update)) : long(base) < long(update);\n"
"        return maximum ? (less ? update : base) : (less ? base : update);\n"
"    }\n"
"    if (type == 5u || type == 7u) {\n"
"        float left = working ? as_type<float>(uint(base)) : lp_widen(base, type);\n"
"        float right = lp_widen(update, type);\n"
"        working = true;\n"
"        return ulong(as_type<uint>(reduction == 2u ? left + right : left * right));\n"
"    }\n"
"    if (type == 1u) {\n"
"        float left = as_type<float>(uint(base));\n"
"        float right = as_type<float>(uint(update));\n"
"        float result = reduction == 2u ? left + right : left * right;\n"
"        return ulong(as_type<uint>(result));\n"
"    }\n"
"    if (type == 2u) return ulong(reduction == 2u ? uint(base) + uint(update) : uint(base) * uint(update));\n"
"    return reduction == 2u ? base + update : base * update;\n"
"}\n"
"inline ulong scatter_role_offset(thread const ulong *coords, uint role, constant DataMeta &m) {\n"
"    return coords_storage(coords, m.roleStrides[role], m.roleRanks[role], m.roleOffsets[role]);\n"
"}\n"
"inline bool scatter_operand_active(device const ulong *control, constant ScanStepControl *step,\n"
"        thread const ulong *coords, uint role, constant DataMeta &m) {\n"
"    if (step == nullptr) return true;\n"
"    return scan_control_operand(control, *step, role, coords_linear(coords, m.roleDims[role], m.roleRanks[role]));\n"
"}\n"
/* Only contributors on this output's selected-axis line can address it. */
"inline ulong scatter_elements_value(device const uchar *indices, device const uchar *updates,\n"
"        thread const ulong *out, ulong accumulator, thread bool &working, constant DataMeta &m,\n"
"        device const ulong *control, constant ScanStepControl *step) {\n"
"    uint axis = uint(m.attrs[0]);\n"
"    ulong coords[16];\n"
"    for (uint a = 0u; a < m.outputRank; ++a) coords[a] = out[a];\n"
"    for (ulong item = 0ul; item < m.roleDims[2][axis]; ++item) {\n"
"        coords[axis] = item;\n"
"        if (!scatter_operand_active(control, step, coords, 2u, m)) continue;\n"
"        if (!scatter_operand_active(control, step, coords, 1u, m)) continue;\n"
"        long selected = read_index(indices, scatter_role_offset(coords, 1u, m), m.indexType);\n"
"        if (ulong(selected) != out[axis]) continue;\n"
"        ulong word = load_word(updates, scatter_role_offset(coords, 2u, m), m.width);\n"
"        accumulator = scatter_combine(accumulator, word, uint(m.attrs[1]), m.sourceType, working);\n"
"    }\n"
"    return accumulator;\n"
"}\n"
/* Gather's entire index Shape replaces one data axis, including a scalar Shape. */
"inline ulong scatter_add_value(device const uchar *indices, device const uchar *updates,\n"
"        thread const ulong *out, ulong accumulator, thread bool &working, constant DataMeta &m,\n"
"        device const ulong *control, constant ScanStepControl *step) {\n"
"    uint axis = uint(m.attrs[0]), indexRank = m.roleRanks[1];\n"
"    ulong count = 1ul, idx[16], update[16];\n"
"    for (uint a = 0u; a < indexRank; ++a) count *= m.roleDims[1][a];\n"
"    for (uint a = 0u; a < axis; ++a) update[a] = out[a];\n"
"    for (uint a = axis + 1u; a < m.outputRank; ++a) update[a - 1u + indexRank] = out[a];\n"
"    for (ulong ordinal = 0ul; ordinal < count; ++ordinal) {\n"
"        decode_coords(ordinal, m.roleDims[1], indexRank, idx);\n"
"        for (uint a = 0u; a < indexRank; ++a) update[axis + a] = idx[a];\n"
"        if (!scatter_operand_active(control, step, update, 2u, m)) continue;\n"
"        if (!scatter_operand_active(control, step, idx, 1u, m)) continue;\n"
"        long selected = read_index(indices, scatter_role_offset(idx, 1u, m), m.indexType);\n"
"        if (ulong(selected) != out[axis]) continue;\n"
"        ulong word = load_word(updates, scatter_role_offset(update, 2u, m), m.width);\n"
"        accumulator = scatter_combine(accumulator, word, 2u, m.sourceType, working);\n"
"    }\n"
"    return accumulator;\n"
"}\n"
/* Batch and unindexed suffix are fixed by the output; scan only tuple prefixes. */
"inline ulong scatter_nd_value(device const uchar *indices, device const uchar *updates,\n"
"        thread const ulong *out, ulong accumulator, thread bool &working, constant DataMeta &m,\n"
"        device const ulong *control, constant ScanStepControl *step) {\n"
"    uint batch = uint(m.attrs[0]), q = m.roleRanks[1];\n"
"    uint depth = uint(m.roleDims[1][q - 1u]);\n"
"    ulong count = 1ul, idx[16], update[16];\n"
"    for (uint a = 0u; a < batch; ++a) { idx[a] = out[a]; update[a] = out[a]; }\n"
"    for (uint a = batch; a + 1u < q; ++a) count *= m.roleDims[1][a];\n"
"    for (uint a = batch + depth; a < m.outputRank; ++a) update[q - 1u + a - batch - depth] = out[a];\n"
"    for (ulong ordinal = 0ul; ordinal < count; ++ordinal) {\n"
"        ulong remaining = ordinal;\n"
"        for (int a = int(q) - 2; a >= int(batch); --a) {\n"
"            idx[a] = remaining % m.roleDims[1][a]; remaining /= m.roleDims[1][a];\n"
"            update[a] = idx[a];\n"
"        }\n"
"        if (!scatter_operand_active(control, step, update, 2u, m)) continue;\n"
"        bool matches = true;\n"
"        for (uint j = 0u; j < depth; ++j) {\n"
"            idx[q - 1u] = j;\n"
"            if (!scatter_operand_active(control, step, idx, 1u, m)) { matches = false; break; }\n"
"            long selected = read_index(indices, scatter_role_offset(idx, 1u, m), m.indexType);\n"
"            if (ulong(selected) != out[batch + j]) { matches = false; break; }\n"
"        }\n"
"        if (!matches) continue;\n"
"        ulong word = load_word(updates, scatter_role_offset(update, 2u, m), m.width);\n"
"        accumulator = scatter_combine(accumulator, word, uint(m.attrs[1]), m.sourceType, working);\n"
"    }\n"
"    return accumulator;\n"
"}\n"
"inline void scatter_reduce_body(device const uchar *data, device const uchar *indices,\n"
"        device const uchar *updates, device uchar *output, constant DataMeta &m,\n"
"        ulong linear, device const ulong *control, constant ScanStepControl *step) {\n"
"    ulong out[16]; decode_coords(linear, m.outDims, m.outputRank, out);\n"
"    if (control != nullptr && !scan_control_operand(control, *step, 0u, linear)) return;\n"
"    ulong accumulator = load_word(data, scatter_role_offset(out, 0u, m), m.width);\n"
"    bool working = false;\n"
"    if (m.operation == 18u) accumulator = scatter_elements_value(indices, updates, out, accumulator, working, m, control, step);\n"
"    else if (m.operation == 70u) accumulator = scatter_add_value(indices, updates, out, accumulator, working, m, control, step);\n"
"    else accumulator = scatter_nd_value(indices, updates, out, accumulator, working, m, control, step);\n"
"    if (working) accumulator = lp_narrow(as_type<float>(uint(accumulator)), m.targetType);\n"
"    store_word(output, linear, m.width, accumulator);\n"
"}\n"
"kernel void scatter_reduce(device const uchar *data [[buffer(0)]],\n"
"        device const uchar *indices [[buffer(1)]], device const uchar *updates [[buffer(2)]],\n"
"        device uchar *output [[buffer(3)]], constant DataMeta &m [[buffer(4)]],\n"
"        uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong linear = data_linear_id(gid, m);\n"
"    if (linear >= m.elementCount) return;\n"
"    scatter_reduce_body(data, indices, updates, output, m, linear, nullptr, nullptr);\n"
"}\n"
"kernel void scatter_reduce_scan(device const uchar *data [[buffer(0)]],\n"
"        device const uchar *indices [[buffer(1)]], device const uchar *updates [[buffer(2)]],\n"
"        device uchar *output [[buffer(3)]], constant DataMeta &m [[buffer(4)]],\n"
"        device const ulong *scanControl [[buffer(28)]],\n"
"        constant ScanStepControl &scanStep [[buffer(29)]], uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong linear = data_linear_id(gid, m);\n"
"    if (linear >= m.elementCount || !scan_control_output(scanControl, scanStep, linear)) return;\n"
"    scatter_reduce_body(data, indices, updates, output, m, linear, scanControl, &scanStep);\n"
"}\n";

#endif
