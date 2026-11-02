#ifndef SYNAPTIK_AUDIT_REDUCTION_KERNELS_H
#define SYNAPTIK_AUDIT_REDUCTION_KERNELS_H

/*
 * Ordinary reductions. Assemble after DataMeta/coordinate/raw-word helpers,
 * SynaptikFloat32MathKernelSource, SynaptikExactTypedKernelSource and
 * SynaptikGeneralScanKernelSource (the Scan-owned ScanStepControl ABI).
 * This component adds no host POD or workspace.
 *
 * audit_reduce_f32 bindings: 0=data, 1=BOOL mask (data again when unmasked), 2=output,
 * 3=SynaptikMetalDataMeta. One grid workitem owns one output; custom_grid supplies the
 * ordinary elementCount/gridWidth/gridHeight geometry. No atomics or partial buffers.
 *
 * Native preparation independently proves FLOAT32 input/output, ranks <=16, canonical
 * output, checked affine referenced input/mask spans (input zero strides are legal),
 * exact result shape, mask right-aligned broadcast, and logical BOOL contents. attrs[0]
 * is the normalized reduction-axis bitmask; attrs[1] is statistical correction. Other
 * input/output/auxiliary shape, stride and offset fields retain their ordinary meaning.
 * inputCount==2 means masked SUM/MEAN only; auxiliary fields describe that BOOL input.
 * The controlled variant additionally binds scanControl=28 and ScanStepControl=29;
 * only SUM/MEAN use that variant. Its source-row activity test precedes mask/data
 * loads and its divisor is the real selected count. Absent scope is a no-write
 * unit; all-valid empty domains instead publish their ordinary empty identities.
 * VAR/STD require contributorCount>correction before any dispatch. The defensive NaN
 * below is not a substitute for that pre-write validation. No low arithmetic is granted.
 *
 * audit_reduce_exact binds data=0, output=1, DataMeta=2 with the same coordinate
 * metadata. It admits integral modular SUM and stored-value MIN/MAX, including F64
 * extrema without FP64 arithmetic. Exact extrema use the shared typed comparator;
 * F64 SUM/MEAN/PROD and new low domains are deliberately absent.
 *
 * Each selected contributor is visited in input logical row-major order. Named MUL,
 * ADD, SUB, DIV and SQRT sites remain separate under the shared no-fast-math compiler
 * policy. LSE uses a guarded max-shifted realization; constant statistics and point
 * norms preserve their Model identities without spurious overflow or square underflow.
 * SUM/MEAN classify only selected stored NaNs and each infinity sign, independently
 * of finite intermediate overflow: NaN/opposite input infinities dominate, otherwise
 * the sole stored infinity sign wins. MEAN still counts every selected contributor.
 */
static NSString *const SynaptikAuditReductionKernelSource =
@"\n"
"inline ulong audit_reduction_domain(constant DataMeta &m) {\n"
"    ulong count = 1ul;\n"
"    for (uint axis = 0u; axis < m.inputRank; ++axis)\n"
"        if ((m.attrs[0] & (1ul << axis)) != 0ul) count *= m.inDims[axis];\n"
"    return count;\n"
"}\n"
"inline void audit_reduction_point(ulong output, ulong contributor, constant DataMeta &m, thread ulong *coords) {\n"
"    for (int axis = int(m.inputRank) - 1; axis >= 0; --axis) {\n"
"        ulong extent = m.inDims[axis];\n"
"        if ((m.attrs[0] & (1ul << uint(axis))) != 0ul) {\n"
"            coords[axis] = contributor % extent; contributor /= extent;\n"
"        } else { coords[axis] = output % extent; output /= extent; }\n"
"    }\n"
"}\n"
"inline bool audit_reduction_selected(device const uchar *mask, thread const ulong *coords, constant DataMeta &m) {\n"
"    if (m.inputCount != 2u) return true;\n"
"    ulong offset = m.auxiliaryOffset; uint prefix = m.inputRank - m.auxRank;\n"
"    for (uint axis = 0u; axis < m.auxRank; ++axis)\n"
"        offset += (m.auxDims[axis] == 1ul ? 0ul : coords[prefix + axis]) * m.auxiliaryStrides[axis];\n"
"    return mask[offset] != uchar(0);\n"
"}\n"
"inline uint audit_reduction_load(device const uchar *input, thread const ulong *coords, constant DataMeta &m) {\n"
"    return uint(load_word(input, coords_storage(coords, m.inputStrides, m.inputRank, m.inputOffset), 4u));\n"
"}\n"
"inline bool audit_reduction_nan(uint word) { return (word & 0x7fffffffu) > 0x7f800000u; }\n"
"inline bool audit_reduction_inf(uint word) { return (word & 0x7fffffffu) == 0x7f800000u; }\n"
"inline float audit_reduction_add(float a, float b) { return a + b; }\n"
"inline float audit_reduction_mul(float a, float b) { return a * b; }\n"
"inline float audit_reduction_sub(float a, float b) { return a - b; }\n"
"inline float audit_reduction_div(float a, float b) { return a / b; }\n"
"inline uint audit_reduce_f32_impl(device const uchar *input, device const uchar *mask,\n"
"        ulong linear, constant DataMeta &m, device const ulong *scanControl,\n"
"        constant ScanStepControl *scanStep) {\n"
"    ulong domain = audit_reduction_domain(m), selected = 0ul;\n"
"    uint op = m.operation, firstWord = 0u; ulong coords[16];\n"
"    float accumulator = op == 106u ? 1.0f : 0.0f;\n"
"    float maximum = as_type<float>(0xff800000u);\n"
"    bool anyNan = false, anyInf = false, constantValue = true;\n"
"    bool positiveInf = false, negativeInf = false;\n"
"    for (ulong item = 0ul; item < domain; ++item) {\n"
"        audit_reduction_point(linear, item, m, coords);\n"
"        if (scanStep != nullptr && !scan_control_operand(scanControl, *scanStep, 0u,\n"
"                coords_linear(coords, m.inDims, m.inputRank))) continue;\n"
"        if (!audit_reduction_selected(mask, coords, m)) continue;\n"
"        uint word = audit_reduction_load(input, coords, m); float x = as_type<float>(word);\n"
"        if (selected == 0ul) firstWord = word;\n"
"        else if (word != firstWord && ((word | firstWord) & 0x7fffffffu) != 0u) constantValue = false;\n"
"        anyNan = anyNan || audit_reduction_nan(word); anyInf = anyInf || audit_reduction_inf(word);\n"
"        positiveInf = positiveInf || word == 0x7f800000u;\n"
"        negativeInf = negativeInf || word == 0xff800000u;\n"
"        if (op == 111u) maximum = as_type<float>(uint(exact_numeric_extreme(\n"
"                ulong(as_type<uint>(maximum)), 1u, ulong(word), 1u, 1u, true)));\n"
"        else {\n"
"            float term = op == 114u ? as_type<float>(word & 0x7fffffffu)\n"
"                    : op == 115u ? audit_reduction_mul(x, x) : x;\n"
"            accumulator = selected == 0ul ? term : op == 106u\n"
"                    ? audit_reduction_mul(accumulator, term) : audit_reduction_add(accumulator, term);\n"
"        }\n"
"        ++selected;\n"
"    }\n"
"    uint result;\n"
"    if (op == 13u || op == 14u) {\n"
"        if (op == 13u && selected == 1ul) result = firstWord;\n"
"        else if (anyNan || (positiveInf && negativeInf)) result = 0x7fc00000u;\n"
"        else if (positiveInf) result = 0x7f800000u;\n"
"        else if (negativeInf) result = 0xff800000u;\n"
"        else if (op == 14u) result = selected == 0ul ? 0x7fc00000u\n"
"                : as_type<uint>(audit_reduction_div(accumulator, float(selected)));\n"
"        else result = as_type<uint>(accumulator);\n"
"    }\n"
"    else if (op == 106u || op == 114u) result = as_type<uint>(accumulator);\n"
"    else if (op == 115u) {\n"
"        result = anyNan ? 0x7fc00000u : anyInf ? 0x7f800000u\n"
"                : selected == 1ul ? firstWord & 0x7fffffffu\n"
"                : as_type<uint>(synaptik_f32_sqrt(accumulator));\n"
"    } else if (op == 112u || op == 113u) {\n"
"        result = 0x7fc00000u;\n"
"        if (!anyNan && !anyInf && selected > m.attrs[1]) {\n"
"            if (constantValue) result = 0u;\n"
"            else {\n"
"                float mean = audit_reduction_div(accumulator, float(selected)), squares = 0.0f;\n"
"                for (ulong item = 0ul; item < domain; ++item) {\n"
"                    audit_reduction_point(linear, item, m, coords);\n"
"                    float x = as_type<float>(audit_reduction_load(input, coords, m));\n"
"                    float difference = audit_reduction_sub(x, mean);\n"
"                    float square = audit_reduction_mul(difference, difference);\n"
"                    squares = item == 0ul ? square : audit_reduction_add(squares, square);\n"
"                }\n"
"                float variance = audit_reduction_div(squares, float(selected - m.attrs[1]));\n"
"                result = as_type<uint>(op == 113u ? synaptik_f32_sqrt(variance) : variance);\n"
"                if ((result & 0x7fffffffu) == 0u) result = 0u;\n"
"            }\n"
"        }\n"
"    } else {\n"
"        result = 0xff800000u;\n"
"        if (anyNan) result = 0x7fc00000u;\n"
"        else if (maximum == as_type<float>(0x7f800000u)) result = 0x7f800000u;\n"
"        else if (maximum != as_type<float>(0xff800000u)) {\n"
"            if (selected == 1ul) result = firstWord;\n"
"            else {\n"
"                float sum = 0.0f;\n"
"                for (ulong item = 0ul; item < domain; ++item) {\n"
"                    audit_reduction_point(linear, item, m, coords);\n"
"                    float x = as_type<float>(audit_reduction_load(input, coords, m));\n"
"                    float term = synaptik_f32_exp(audit_reduction_sub(x, maximum));\n"
"                    sum = item == 0ul ? term : audit_reduction_add(sum, term);\n"
"                }\n"
"                result = as_type<uint>(audit_reduction_add(maximum, synaptik_f32_log(sum)));\n"
"            }\n"
"        }\n"
"    }\n"
"    return result;\n"
"}\n"
"inline uint audit_reduce_f32_word(device const uchar *input, device const uchar *mask,\n"
"        ulong linear, constant DataMeta &m) {\n"
"    return audit_reduce_f32_impl(input, mask, linear, m, nullptr, nullptr);\n"
"}\n"
"kernel void audit_reduce_f32(device const uchar *input [[buffer(0)]],\n"
"        device const uchar *mask [[buffer(1)]], device uchar *output [[buffer(2)]],\n"
"        constant DataMeta &m [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return;\n"
"    uint result = audit_reduce_f32_word(input, mask, linear, m);\n"
"    ulong destination = logical_storage(linear, m.outDims, m.outputStrides, m.outputRank, m.outputOffset);\n"
"    store_word(output, destination, 4u, ulong(result));\n"
"}\n"
"kernel void audit_reduce_f32_controlled(device const uchar *input [[buffer(0)]],\n"
"        device const uchar *mask [[buffer(1)]], device uchar *output [[buffer(2)]],\n"
"        constant DataMeta &m [[buffer(3)]], device const ulong *scanControl [[buffer(28)]],\n"
"        constant ScanStepControl &scanStep [[buffer(29)]], uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return;\n"
"    if (!scan_control_output(scanControl, scanStep, linear)) return;\n"
"    uint result = audit_reduce_f32_impl(input, mask, linear, m, scanControl, &scanStep);\n"
"    ulong destination = logical_storage(linear, m.outDims, m.outputStrides, m.outputRank, m.outputOffset);\n"
"    store_word(output, destination, 4u, ulong(result));\n"
"}\n"
"inline ulong audit_reduction_extreme_identity(uint type, bool maximum) {\n"
"    if (type == 2u) return maximum ? 0x80000000ul : 0x7ffffffful;\n"
"    if (type == 6u) return maximum ? 0x8000000000000000ul : 0x7ffffffffffffffful;\n"
"    if (type == 4u) return maximum ? 0xfff0000000000000ul : 0x7ff0000000000000ul;\n"
"    return maximum ? 0xff800000ul : 0x7f800000ul;\n"
"}\n"
"kernel void audit_reduce_exact(device const uchar *input [[buffer(0)]],\n"
"        device uchar *output [[buffer(1)]], constant DataMeta &m [[buffer(2)]],\n"
"        uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong linear = data_linear_id(gid, m); if (linear >= m.elementCount) return;\n"
"    ulong domain = audit_reduction_domain(m), accumulator = m.operation == 13u ? 0ul\n"
"            : audit_reduction_extreme_identity(m.sourceType, m.operation == 32u);\n"
"    ulong coords[16];\n"
"    for (ulong item = 0ul; item < domain; ++item) {\n"
"        audit_reduction_point(linear, item, m, coords);\n"
"        ulong source = coords_storage(coords, m.inputStrides, m.inputRank, m.inputOffset);\n"
"        ulong word = load_word(input, source, m.width);\n"
"        if (m.operation == 13u) accumulator = m.sourceType == 2u\n"
"                ? ulong(uint(accumulator) + uint(word)) : accumulator + word;\n"
"        else accumulator = item == 0ul ? word : exact_numeric_extreme(\n"
"                accumulator, m.sourceType, word, m.sourceType, m.targetType, m.operation == 32u);\n"
"    }\n"
"    ulong destination = logical_storage(linear, m.outDims, m.outputStrides, m.outputRank, m.outputOffset);\n"
"    store_word(output, destination, m.width, accumulator);\n"
"}\n";

#endif
