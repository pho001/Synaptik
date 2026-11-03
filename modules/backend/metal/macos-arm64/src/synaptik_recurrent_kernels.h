#ifndef SYNAPTIK_RECURRENT_KERNELS_H
#define SYNAPTIK_RECURRENT_KERNELS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* Canonical FLOAT32/BFLOAT16/FLOAT16 fixed recurrence; lengths are signed INT64.
 * The native owner checks all shape products/buffer extents and nonaliasing of
 * writes before execution. kind is the existing ordinary wire id 103/104/105;
 * reverse and hasBias are Boolean, valueType is 1/5/7, I/H are positive.
 */
typedef struct {
    uint32_t kind;
    uint32_t time;
    uint32_t batch;
    uint32_t inputSize;
    uint32_t hiddenSize;
    uint32_t reverse;
    uint32_t hasBias;
    uint32_t valueType;
} SynaptikRecurrentMeta;
_Static_assert(sizeof(SynaptikRecurrentMeta) == 32, "recurrent metadata size");
_Static_assert(offsetof(SynaptikRecurrentMeta, valueType) == 28, "recurrent metadata packing");

/* Invocation preflight, not a per-row guard: call for every recurrence before
 * encoding ANY target writes. bytes must be the authenticated CPU-readable
 * contents of the length feed, after any producing step has completed. An
 * internally produced vector requires the same pre-write staging/validation
 * boundary; it cannot be read speculatively before its producer executes.
 * firstInvalid is UINT32_MAX for malformed extent, otherwise the first bad row;
 * it is unchanged on success. No state or output is touched, even for T=0.
 */
static inline bool SynaptikValidateRecurrentLengths(
        const void *bytes, size_t byteLength, const SynaptikRecurrentMeta *m,
        uint32_t *firstInvalid) {
    if (m == NULL || (uint64_t)m->batch * sizeof(int64_t) != byteLength
            || (byteLength != 0 && bytes == NULL)) {
        if (firstInvalid != NULL) *firstInvalid = UINT32_MAX;
        return false;
    }
    for (uint32_t row = 0; row < m->batch; ++row) {
        int64_t count;
        memcpy(&count, (const unsigned char *)bytes + (size_t)row * sizeof(count), sizeof(count));
        if (count < 0 || (uint64_t)count > m->time) {
            if (firstInvalid != NULL) *firstInvalid = row;
            return false;
        }
    }
    return true;
}

/* Assemble after SynaptikNumericCodecSource and SynaptikFloat32MathKernelSource.
 * Shared working dependencies are synaptik_f32_sigmoid(float) and
 * synaptik_f32_tanh(float). Disable fast math and contraction. Projections and
 * gates stay FLOAT32; hidden/cell publications narrow once per step. LSTM hidden
 * consumes the narrowed cell. Initial state movement preserves raw payloads.
 *
 * synaptik_recurrent_numeric dispatch: exactly B workitems (one row each), any
 * ordinary one-dimensional threadgroup width; B=0 needs no dispatch. Bindings:
 *  0 X[T,B,I], 1 lengths[B], 2 H0[B,H], 3 C0[B,H],
 *  4 W[G*H,I], 5 U[G*H,H], 6 bias[G*H],
 *  7 Y[T,B,H], 8 finalH[B,H], 9 finalC[B,H], 10 metadata.
 * Non-LSTM C0/finalC may bind H0/finalH; absent bias may bind W. Those dummy
 * bindings are never accessed. All present outputs must be independent of
 * every read and one another. No scratch, atomics or cell-history allocation.
 * Previous hidden is read only from H0 or the completed previous Y row, never
 * finalH: feature-by-feature writes therefore cannot change another gate's
 * contraction. FinalC is safe in place: a cell feature depends on only its own
 * previous cell, while every gate reads immutable previous hidden.
 */
static NSString *const SynaptikRecurrentKernelSource =
@"\n"
"struct SynaptikRecurrentMeta {\n"
"    uint kind, time, batch, inputSize, hiddenSize, reverse, hasBias, valueType;\n"
"};\n"
"inline float2 synaptik_recurrent_projection(\n"
"        device const uchar *x, device const uchar *previous,\n"
"        device const uchar *w, device const uchar *u, device const uchar *bias,\n"
"        ulong inputBase, ulong hiddenBase, ulong gateRow,\n"
"        constant SynaptikRecurrentMeta &m) {\n"
"    // Nonempty recurrent contractions retain the first actual product's zero sign.\n"
"    ulong weightBase = gateRow * ulong(m.inputSize);\n"
"    float inputSum = synaptik_load_numeric(x, inputBase, m.valueType)\n"
"            * synaptik_load_numeric(w, weightBase, m.valueType);\n"
"    for (uint k = 1u; k < m.inputSize; ++k) {\n"
"        float product = synaptik_load_numeric(x, inputBase + k, m.valueType)\n"
"                * synaptik_load_numeric(w, weightBase + k, m.valueType);\n"
"        inputSum = inputSum + product;\n"
"    }\n"
"    ulong recurrentBase = gateRow * ulong(m.hiddenSize);\n"
"    float hiddenSum = synaptik_load_numeric(previous, hiddenBase, m.valueType)\n"
"            * synaptik_load_numeric(u, recurrentBase, m.valueType);\n"
"    for (uint k = 1u; k < m.hiddenSize; ++k) {\n"
"        float product = synaptik_load_numeric(previous, hiddenBase + k, m.valueType)\n"
"                * synaptik_load_numeric(u, recurrentBase + k, m.valueType);\n"
"        hiddenSum = hiddenSum + product;\n"
"    }\n"
"    if (m.hasBias != 0u) inputSum = inputSum + synaptik_load_numeric(bias, gateRow, m.valueType);\n"
"    return float2(inputSum, hiddenSum);\n"
"}\n"
"kernel void synaptik_recurrent_numeric(\n"
"        device const uchar *x [[buffer(0)]],\n"
"        device const long *lengths [[buffer(1)]],\n"
"        device const uchar *h0 [[buffer(2)]],\n"
"        device const uchar *c0 [[buffer(3)]],\n"
"        device const uchar *w [[buffer(4)]],\n"
"        device const uchar *u [[buffer(5)]],\n"
"        device const uchar *bias [[buffer(6)]],\n"
"        device uchar *y [[buffer(7)]],\n"
"        device uchar *finalH [[buffer(8)]],\n"
"        device uchar *finalC [[buffer(9)]],\n"
"        constant SynaptikRecurrentMeta &m [[buffer(10)]],\n"
"        uint row [[thread_position_in_grid]]) {\n"
"    if (row >= m.batch) return;\n"
"    // Global preflight has already validated every signed length before writes.\n"
"    uint count = uint(lengths[row]);\n"
"    ulong stateBase = ulong(row) * m.hiddenSize;\n"
"    uint width = lp_width(m.valueType);\n"
"    for (uint t = count; t < m.time; ++t) {\n"
"        ulong base = (ulong(t) * m.batch + row) * m.hiddenSize;\n"
"        for (uint j = 0; j < m.hiddenSize; ++j) store_word(y, base + j, width, 0ul);\n"
"    }\n"
"    if (count == 0u) {\n"
"        for (uint j = 0; j < m.hiddenSize; ++j) {\n"
"            ulong state = stateBase + j;\n"
"            copy_word(h0, state, finalH, state, width);\n"
"            if (m.kind == 105u) copy_word(c0, state, finalC, state, width);\n"
"        }\n"
"        return;\n"
"    }\n"
"    if (m.kind == 105u) {\n"
"        for (uint j = 0; j < m.hiddenSize; ++j)\n"
"            copy_word(c0, stateBase + j, finalC, stateBase + j, width);\n"
"    }\n"
"    for (uint visit = 0; visit < count; ++visit) {\n"
"        uint t = m.reverse != 0u ? count - 1u - visit : visit;\n"
"        ulong inputBase = (ulong(t) * m.batch + row) * m.inputSize;\n"
"        ulong outputBase = (ulong(t) * m.batch + row) * m.hiddenSize;\n"
"        device const uchar *previous = visit == 0u ? h0 : y;\n"
"        uint previousTime = m.reverse != 0u ? t + 1u : t - 1u;\n"
"        ulong previousBase = visit == 0u ? stateBase :\n"
"                (ulong(previousTime) * m.batch + row) * m.hiddenSize;\n"
"        for (uint j = 0; j < m.hiddenSize; ++j) {\n"
"            float2 first = synaptik_recurrent_projection(x, previous, w, u, bias,\n"
"                    inputBase, previousBase, j, m);\n"
"            float next;\n"
"            if (m.kind == 103u) {\n"
"                next = synaptik_f32_tanh(first.x + first.y);\n"
"            } else {\n"
"                float2 second = synaptik_recurrent_projection(x, previous, w, u, bias,\n"
"                        inputBase, previousBase, ulong(m.hiddenSize) + j, m);\n"
"                float2 third = synaptik_recurrent_projection(x, previous, w, u, bias,\n"
"                        inputBase, previousBase, 2ul * m.hiddenSize + j, m);\n"
"                float gate0 = synaptik_f32_sigmoid(first.x + first.y);\n"
"                float gate1 = synaptik_f32_sigmoid(second.x + second.y);\n"
"                if (m.kind == 104u) {\n"
"                    float resetHidden = gate0 * third.y;\n"
"                    float candidate = synaptik_f32_tanh(third.x + resetHidden);\n"
"                    float difference = synaptik_load_numeric(previous, previousBase + j, m.valueType) - candidate;\n"
"                    float update = gate1 * difference;\n"
"                    next = candidate + update;\n"
"                } else {\n"
"                    float2 fourth = synaptik_recurrent_projection(x, previous, w, u, bias,\n"
"                            inputBase, previousBase, 3ul * m.hiddenSize + j, m);\n"
"                    float candidate = synaptik_f32_tanh(third.x + third.y);\n"
"                    float outputGate = synaptik_f32_sigmoid(fourth.x + fourth.y);\n"
"                    float retained = gate1 * synaptik_load_numeric(finalC, stateBase + j, m.valueType);\n"
"                    float admitted = gate0 * candidate;\n"
"                    float cell = retained + admitted;\n"
"                    ulong storedCellBits = lp_narrow(cell, m.valueType);\n"
"                    store_word(finalC, stateBase + j, width, storedCellBits);\n"
"                    float storedCell = lp_widen(storedCellBits, m.valueType);\n"
"                    next = outputGate * synaptik_f32_tanh(storedCell);\n"
"                }\n"
"            }\n"
"            ulong storedHidden = lp_narrow(next, m.valueType);\n"
"            store_word(y, outputBase + j, width, storedHidden);\n"
"            if (visit == count - 1u) store_word(finalH, stateBase + j, width, storedHidden);\n"
"        }\n"
"    }\n"
"}\n";

#endif
