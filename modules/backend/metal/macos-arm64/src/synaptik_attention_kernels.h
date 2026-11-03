#ifndef SYNAPTIK_ATTENTION_KERNELS_H
#define SYNAPTIK_ATTENTION_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/**
 * Prepared typed SDPA geometry. Strides and offsets are in elements, not bytes.
 * Q/K/V/mask batch strides are right-aligned to batchDims, with zero strides for
 * broadcast dimensions. Tail strides are separately normalized (including a
 * broadcast mask's query/key axes). Outputs are canonical, offset-zero arrays.
 *
 * Validate before dispatch: rank <= 14, positive embedding, matching Q/K embedding
 * and K/V length, exact broadcast output/mask geometry, checked spans/products,
 * finite positive original scale (F32 for F16), canonical BOOL mask bytes, disjoint
 * outputs/scratch and inputs, flags restricted to 1|2|4|8, reserved == 0.
 * rowCount = product(batchDims[0:batchRank]) * queryLength.
 *
 * Bindings: 0 Q, 1 K, 2 V, 3 BOOL mask (dummy when absent), 4 output,
 * 5 F32 score/probability workspace, 6 this metadata, 7 optional weights sibling.
 * Buffer 5 requires checked rowCount * keyLength * sizeof(float) bytes. F32
 * siblings may alias buffers 5/7; low siblings never do: working scores and
 * probabilities must remain F32 until each final declared output is published.
 * Dispatch one workitem per row using the ordinary checked 3-D grid geometry;
 * zero rows require no dispatch. Zero keys require no workspace bytes and still
 * publish positive-zero output rows (bind the ordinary non-null empty buffer).
 *
 * Assemble source after DataMeta/load_word/store_word, matmul_fma_0061,
 * synaptik_f32_exp/sqrt, and the ScanStepControl/per-output activity helpers.
 * The Scan binder authenticates each sibling's row fact and identical ordinary
 * node scope: active nonempty Y implies eligible canonical weights for that
 * score row, permitting safe intermediate reuse. Empty Y does not imply empty
 * weights. All canonical writes retain their own slot activity guard.
 * All numeric roles retain explicit carriers and follow original Model promotion.
 */
typedef struct {
    uint64_t batchDims[16];
    uint64_t roleBatchStrides[4][16];
    uint64_t roleOffsets[4];
    uint64_t queryStride;
    uint64_t queryEmbeddingStride;
    uint64_t keyStride;
    uint64_t keyEmbeddingStride;
    uint64_t valueStride;
    uint64_t valueEmbeddingStride;
    uint64_t maskQueryStride;
    uint64_t maskKeyStride;
    uint64_t rowCount;
    uint64_t queryLength;
    uint64_t keyLength;
    uint64_t embedding;
    uint64_t valueEmbedding;
    uint64_t gridWidth;
    uint64_t gridHeight;
    uint32_t scaleBits;
    uint32_t flags;
    uint32_t batchRank;
    uint32_t scaleType;
    uint32_t roleTypes[4];
    uint32_t outputType, reserved;
} SynaptikAttentionMeta;
_Static_assert(sizeof(SynaptikAttentionMeta) == 832, "attention metadata size");
_Static_assert(offsetof(SynaptikAttentionMeta, scaleBits) == 792, "attention scale offset");
_Static_assert(offsetof(SynaptikAttentionMeta, flags) == 796, "attention flags offset");

static NSString *const SynaptikAttentionKernelSource =
@"\n"
"struct AttentionMeta {\n"
"    ulong batchDims[16];\n"
"    ulong roleBatchStrides[4][16];\n"
"    ulong roleOffsets[4];\n"
"    ulong queryStride, queryEmbeddingStride;\n"
"    ulong keyStride, keyEmbeddingStride;\n"
"    ulong valueStride, valueEmbeddingStride;\n"
"    ulong maskQueryStride, maskKeyStride;\n"
"    ulong rowCount, queryLength, keyLength, embedding, valueEmbedding;\n"
"    ulong gridWidth, gridHeight;\n"
"    uint scaleBits, flags, batchRank, scaleType;\n"
"    uint roleTypes[4]; uint outputType, reserved;\n"
"};\n"
"inline float attention_load_working(device const uchar *p, ulong offset) {\n"
"    return as_type<float>(uint(load_word(p, offset, 4u)));\n"
"}\n"
"inline void attention_store_working(device uchar *p, ulong offset, float value) {\n"
"    store_word(p, offset, 4u, ulong(as_type<uint>(value)));\n"
"}\n"
"inline bool attention_eligible(device const uchar *mask, ulong maskBase, ulong query, ulong key, constant AttentionMeta &m) {\n"
"    if ((m.flags & 2u) != 0u && key > query) return false;\n"
"    if ((m.flags & 1u) == 0u) return true;\n"
"    return load_word(mask, maskBase + query * m.maskQueryStride + key * m.maskKeyStride, 1u) != 0ul;\n"
"}\n"
"inline bool attention_scan_row_active(device const ulong *control, constant ScanStepControl &step,\n"
"    uint slot, ulong row, constant AttentionMeta &m) {\n"
"    if (slot >= step.outputCount || step.outputElements[slot] == 0ul) return false;\n"
"    int axis = step.outputRowAxes[slot];\n"
"    uint scope = step.outputScopes[slot];\n"
"    if (axis < 0 || axis == int(m.batchRank + 1u)) return scan_control_any(control, scope);\n"
"    // Score-row coordinates exclude the final feature/key axis. Unlike Y's\n"
"    // flattened ordinal, this mapping survives an empty value embedding.\n"
"    ulong inner = 1ul;\n"
"    if (axis < int(m.batchRank)) {\n"
"        inner = m.queryLength;\n"
"        for (uint a = uint(axis) + 1u; a < m.batchRank; ++a) inner *= m.batchDims[a];\n"
"    }\n"
"    return scan_control_row(control, scope, (row / inner) % control[1]);\n"
"}\n"
"inline void attention_store_weight(device uchar *weights, ulong linear, float value,\n"
"    device const ulong *control, constant ScanStepControl *step) {\n"
"    if (step != nullptr && step->outputCount > 1u\n"
"            && !scan_control_output_at(control, *step, 1u, linear)) return;\n"
"    attention_store_working(weights, linear, value);\n"
"}\n"
"inline void attention_publish_weight(device uchar *working, device uchar *published, ulong linear, float value,\n"
"    constant AttentionMeta &m, device const ulong *control, constant ScanStepControl *step) {\n"
"    attention_store_weight(working,linear,value,control,step);\n"
"    if ((m.flags & 8u) == 0u || working == published || (step != nullptr && !scan_control_output_at(control,*step,1u,linear))) return;\n"
"    synaptik_store_numeric(published,linear,m.outputType,value);\n"
"}\n"
"inline void attention_typed_row_body(device const uchar *query, device const uchar *key,\n"
"    device const uchar *value, device const uchar *mask, device uchar *output,\n"
"    device uchar *weights, device uchar *publishedWeights, constant AttentionMeta &m, ulong row,\n"
"    device const ulong *scanControl, constant ScanStepControl *scanStep) {\n"
"    if (row >= m.rowCount) return;\n"
"    if (scanStep != nullptr && (!scan_control_any_output(scanControl, *scanStep)\n"
"            || (!attention_scan_row_active(scanControl, *scanStep, 0u, row, m)\n"
"                && !attention_scan_row_active(scanControl, *scanStep, 1u, row, m)))) return;\n"
"    ulong queryIndex = row % m.queryLength;\n"
"    ulong batch = row / m.queryLength;\n"
"    ulong bases[4];\n"
"    for (uint role = 0u; role < 4u; ++role) bases[role] = m.roleOffsets[role];\n"
"    for (int axis = int(m.batchRank) - 1; axis >= 0; --axis) {\n"
"        ulong coordinate = batch % m.batchDims[axis];\n"
"        batch /= m.batchDims[axis];\n"
"        for (uint role = 0u; role < 4u; ++role) bases[role] += coordinate * m.roleBatchStrides[role][axis];\n"
"    }\n"
"    ulong weightBase = row * m.keyLength;\n"
"    ulong queryBase = bases[0] + queryIndex * m.queryStride;\n"
"    float scale = (m.flags & 4u) != 0u ? lp_widen(m.scaleBits,m.scaleType) : 1.0f / synaptik_f32_sqrt(float(m.embedding));\n"
"    ulong eligibleCount = 0ul, positiveCount = 0ul;\n"
"    bool hasNaN = false, allNegativeInfinity = true;\n"
"    for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"        // Exclusion precedes every Q/K load and gives exact positive-zero weights.\n"
"        if (!attention_eligible(mask, bases[3], queryIndex, j, m)) {\n"
"            attention_publish_weight(weights,publishedWeights,weightBase+j,0.0f,m,scanControl,scanStep);\n"
"            continue;\n"
"        }\n"
"        float score = 0.0f;\n"
"        ulong keyBase = bases[1] + j * m.keyStride;\n"
"        for (ulong e = 0ul; e < m.embedding; ++e) {\n"
"            float q = synaptik_load_numeric(query, queryBase + e * m.queryEmbeddingStride,m.roleTypes[0]);\n"
"            float k = synaptik_load_numeric(key, keyBase + e * m.keyEmbeddingStride,m.roleTypes[1]);\n"
"            score = matmul_fma_0061(q, k, score);\n"
"        }\n"
"        score = score * scale;\n"
"        attention_store_weight(weights, weightBase + j, score, scanControl, scanStep);\n"
"        ++eligibleCount;\n"
"        hasNaN = hasNaN || isnan(score);\n"
"        if (score == INFINITY) ++positiveCount;\n"
"        if (score != -INFINITY) allNegativeInfinity = false;\n"
"    }\n"
"    // Attention's guards are deliberately not ordinary SOFTMAX semantics.\n"
"    uint mode = eligibleCount == 0ul || allNegativeInfinity ? 0u : hasNaN ? 1u : positiveCount != 0ul ? 2u : 3u;\n"
"    if (mode == 3u) {\n"
"        // Literal named EXP/sum/div sites: do not introduce a maximum shift.\n"
"        float sum = 0.0f;\n"
"        for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"            if (!attention_eligible(mask, bases[3], queryIndex, j, m)) continue;\n"
"            float exponential = synaptik_f32_exp(attention_load_working(weights, weightBase + j));\n"
"            attention_store_weight(weights, weightBase + j, exponential, scanControl, scanStep);\n"
"            sum = sum + exponential;\n"
"        }\n"
"        for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"            if (!attention_eligible(mask, bases[3], queryIndex, j, m)) continue;\n"
"            float probability = attention_load_working(weights, weightBase + j) / sum;\n"
"            attention_publish_weight(weights,publishedWeights,weightBase+j,probability,m,scanControl,scanStep);\n"
"        }\n"
"    } else {\n"
"        float tiedWeight = mode == 2u ? 1.0f / float(positiveCount) : 0.0f;\n"
"        for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"            if (!attention_eligible(mask, bases[3], queryIndex, j, m)) continue;\n"
"            float probability = mode == 1u ? NAN : mode == 2u && attention_load_working(weights, weightBase + j) == INFINITY ? tiedWeight : 0.0f;\n"
"            attention_publish_weight(weights,publishedWeights,weightBase+j,probability,m,scanControl,scanStep);\n"
"        }\n"
"    }\n"
"    for (ulong d = 0ul; d < m.valueEmbedding; ++d) {\n"
"        if (scanStep != nullptr && !scan_control_output_at(scanControl, *scanStep, 0u, row * m.valueEmbedding + d)) continue;\n"
"        float result = mode == 1u ? NAN : 0.0f;\n"
"        bool hasValueTerm = false;\n"
"        if (mode == 2u || mode == 3u) {\n"
"            for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"                if (!attention_eligible(mask, bases[3], queryIndex, j, m)) continue;\n"
"                float probability = attention_load_working(weights, weightBase + j);\n"
"                float v = synaptik_load_numeric(value, bases[2] + j * m.valueStride + d * m.valueEmbeddingStride,m.roleTypes[2]);\n"
"                result = hasValueTerm ? matmul_fma_0061(probability, v, result) : probability * v;\n"
"                hasValueTerm = true;\n"
"            }\n"
"        }\n"
"        synaptik_store_numeric(output,row*m.valueEmbedding+d,m.outputType,result);\n"
"    }\n"
"}\n"
"kernel void attention_typed_rows(device const uchar *query [[buffer(0)]],\n"
"    device const uchar *key [[buffer(1)]], device const uchar *value [[buffer(2)]],\n"
"    device const uchar *mask [[buffer(3)]], device uchar *output [[buffer(4)]],\n"
"    device uchar *weights [[buffer(5)]], constant AttentionMeta &m [[buffer(6)]],\n"
"    device uchar *publishedWeights [[buffer(7)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong row = ulong(gid.x) + m.gridWidth * (ulong(gid.y) + m.gridHeight * ulong(gid.z));\n"
"    attention_typed_row_body(query, key, value, mask, output, weights, publishedWeights, m, row, nullptr, nullptr);\n"
"}\n"
"kernel void attention_typed_rows_scan(device const uchar *query [[buffer(0)]],\n"
"    device const uchar *key [[buffer(1)]], device const uchar *value [[buffer(2)]],\n"
"    device const uchar *mask [[buffer(3)]], device uchar *output [[buffer(4)]],\n"
"    device uchar *weights [[buffer(5)]], constant AttentionMeta &m [[buffer(6)]],\n"
"    device uchar *publishedWeights [[buffer(7)]],\n"
"    device const ulong *scanControl [[buffer(28)]], constant ScanStepControl &scanStep [[buffer(29)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong row = ulong(gid.x) + m.gridWidth * (ulong(gid.y) + m.gridHeight * ulong(gid.z));\n"
"    attention_typed_row_body(query, key, value, mask, output, weights, publishedWeights, m, row, scanControl, &scanStep);\n"
"}\n";

#endif
