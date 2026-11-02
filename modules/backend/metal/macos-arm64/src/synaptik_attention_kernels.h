#ifndef SYNAPTIK_ATTENTION_KERNELS_H
#define SYNAPTIK_ATTENTION_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/**
 * Prepared FLOAT32 SDPA geometry. Strides and offsets are in elements, not bytes.
 * Q/K/V/mask batch strides are right-aligned to batchDims, with zero strides for
 * broadcast dimensions. Tail strides are separately normalized (including a
 * broadcast mask's query/key axes). Outputs are canonical, offset-zero arrays.
 *
 * Validate before dispatch: rank <= 14, positive embedding, matching Q/K embedding
 * and K/V length, exact broadcast output/mask geometry, checked spans/products,
 * finite positive explicit FLOAT32 scale, canonical BOOL mask bytes, disjoint
 * outputs/scratch and inputs, flags restricted to 1|2|4, reserved == 0.
 * rowCount = product(batchDims[0:batchRank]) * queryLength.
 *
 * Bindings: 0 Q, 1 K, 2 V, 3 BOOL mask (dummy when absent), 4 output,
 * 5 canonical weights sibling OR explicitly declared scratch, 6 this metadata.
 * Buffer 5 requires checked rowCount * keyLength * sizeof(float) bytes. Reuse
 * output slot one when present; otherwise its prepared resource is invocation
 * owned through completion of this dispatch. No hidden/native buffer pool.
 * Dispatch one workitem per row using the ordinary checked 3-D grid geometry;
 * zero rows require no dispatch. Zero keys require no workspace bytes and still
 * publish positive-zero output rows (bind the ordinary non-null empty buffer).
 *
 * Assemble source after DataMeta/load_word/store_word, matmul_fma_0061, and
 * synaptik_f32_exp/sqrt. This component grants no low or FLOAT64 arithmetic.
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
    float scale;
    uint32_t flags;
    uint32_t batchRank;
    uint32_t reserved;
} SynaptikAttentionMeta;
_Static_assert(sizeof(SynaptikAttentionMeta) == 808, "attention metadata size");
_Static_assert(offsetof(SynaptikAttentionMeta, scale) == 792, "attention scale offset");
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
"    float scale;\n"
"    uint flags, batchRank, reserved;\n"
"};\n"
"inline float attention_load_f32(device const uchar *p, ulong offset) {\n"
"    return as_type<float>(uint(load_word(p, offset, 4u)));\n"
"}\n"
"inline void attention_store_f32(device uchar *p, ulong offset, float value) {\n"
"    store_word(p, offset, 4u, ulong(as_type<uint>(value)));\n"
"}\n"
"inline bool attention_eligible(device const uchar *mask, ulong maskBase, ulong query, ulong key, constant AttentionMeta &m) {\n"
"    if ((m.flags & 2u) != 0u && key > query) return false;\n"
"    if ((m.flags & 1u) == 0u) return true;\n"
"    return load_word(mask, maskBase + query * m.maskQueryStride + key * m.maskKeyStride, 1u) != 0ul;\n"
"}\n"
"inline void attention_f32_row_body(device const uchar *query, device const uchar *key,\n"
"    device const uchar *value, device const uchar *mask, device uchar *output,\n"
"    device uchar *weights, constant AttentionMeta &m, ulong row,\n"
"    device const ulong *scanControl, constant ScanStepControl *scanStep) {\n"
"    if (row >= m.rowCount) return;\n"
"    if (scanStep != nullptr) {\n"
"        bool featureRow = scanStep->rowAxis == int(m.batchRank + 1u);\n"
"        if (featureRow ? !scan_control_any(scanControl, scanStep->scope)\n"
"                       : !scan_control_output(scanControl, *scanStep, row * m.valueEmbedding)) return;\n"
"    }\n"
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
"    float scale = (m.flags & 4u) != 0u ? m.scale : 1.0f / synaptik_f32_sqrt(float(m.embedding));\n"
"    ulong eligibleCount = 0ul, positiveCount = 0ul;\n"
"    bool hasNaN = false, allNegativeInfinity = true;\n"
"    for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"        // Exclusion precedes every Q/K load and gives exact positive-zero weights.\n"
"        if (!attention_eligible(mask, bases[3], queryIndex, j, m)) {\n"
"            attention_store_f32(weights, weightBase + j, 0.0f);\n"
"            continue;\n"
"        }\n"
"        float score = 0.0f;\n"
"        ulong keyBase = bases[1] + j * m.keyStride;\n"
"        for (ulong e = 0ul; e < m.embedding; ++e) {\n"
"            float q = attention_load_f32(query, queryBase + e * m.queryEmbeddingStride);\n"
"            float k = attention_load_f32(key, keyBase + e * m.keyEmbeddingStride);\n"
"            score = matmul_fma_0061(q, k, score);\n"
"        }\n"
"        score = score * scale;\n"
"        attention_store_f32(weights, weightBase + j, score);\n"
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
"            float exponential = synaptik_f32_exp(attention_load_f32(weights, weightBase + j));\n"
"            attention_store_f32(weights, weightBase + j, exponential);\n"
"            sum = sum + exponential;\n"
"        }\n"
"        for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"            if (!attention_eligible(mask, bases[3], queryIndex, j, m)) continue;\n"
"            float probability = attention_load_f32(weights, weightBase + j) / sum;\n"
"            attention_store_f32(weights, weightBase + j, probability);\n"
"        }\n"
"    } else {\n"
"        float tiedWeight = mode == 2u ? 1.0f / float(positiveCount) : 0.0f;\n"
"        for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"            if (!attention_eligible(mask, bases[3], queryIndex, j, m)) continue;\n"
"            float probability = mode == 1u ? NAN : mode == 2u && attention_load_f32(weights, weightBase + j) == INFINITY ? tiedWeight : 0.0f;\n"
"            attention_store_f32(weights, weightBase + j, probability);\n"
"        }\n"
"    }\n"
"    for (ulong d = 0ul; d < m.valueEmbedding; ++d) {\n"
"        if (scanStep != nullptr && !scan_control_output(scanControl, *scanStep, row * m.valueEmbedding + d)) continue;\n"
"        float result = mode == 1u ? NAN : 0.0f;\n"
"        if (mode == 2u || mode == 3u) {\n"
"            for (ulong j = 0ul; j < m.keyLength; ++j) {\n"
"                if (!attention_eligible(mask, bases[3], queryIndex, j, m)) continue;\n"
"                float probability = attention_load_f32(weights, weightBase + j);\n"
"                float v = attention_load_f32(value, bases[2] + j * m.valueStride + d * m.valueEmbeddingStride);\n"
"                result = matmul_fma_0061(probability, v, result);\n"
"            }\n"
"        }\n"
"        attention_store_f32(output, row * m.valueEmbedding + d, result);\n"
"    }\n"
"}\n"
"kernel void attention_f32_rows(device const uchar *query [[buffer(0)]],\n"
"    device const uchar *key [[buffer(1)]], device const uchar *value [[buffer(2)]],\n"
"    device const uchar *mask [[buffer(3)]], device uchar *output [[buffer(4)]],\n"
"    device uchar *weights [[buffer(5)]], constant AttentionMeta &m [[buffer(6)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong row = ulong(gid.x) + m.gridWidth * (ulong(gid.y) + m.gridHeight * ulong(gid.z));\n"
"    attention_f32_row_body(query, key, value, mask, output, weights, m, row, nullptr, nullptr);\n"
"}\n"
"kernel void attention_f32_rows_scan(device const uchar *query [[buffer(0)]],\n"
"    device const uchar *key [[buffer(1)]], device const uchar *value [[buffer(2)]],\n"
"    device const uchar *mask [[buffer(3)]], device uchar *output [[buffer(4)]],\n"
"    device uchar *weights [[buffer(5)]], constant AttentionMeta &m [[buffer(6)]],\n"
"    device const ulong *scanControl [[buffer(28)]], constant ScanStepControl &scanStep [[buffer(29)]],\n"
"    uint3 gid [[thread_position_in_grid]]) {\n"
"    ulong row = ulong(gid.x) + m.gridWidth * (ulong(gid.y) + m.gridHeight * ulong(gid.z));\n"
"    attention_f32_row_body(query, key, value, mask, output, weights, m, row, scanControl, &scanStep);\n"
"}\n";

#endif
