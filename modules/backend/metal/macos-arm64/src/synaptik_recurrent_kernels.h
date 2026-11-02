#ifndef SYNAPTIK_RECURRENT_KERNELS_H
#define SYNAPTIK_RECURRENT_KERNELS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* Canonical FLOAT32 fixed recurrence only; lengths are canonical signed INT64.
 * The native owner checks all shape products/buffer extents and nonaliasing of
 * writes before execution. kind is the existing ordinary wire id 103/104/105;
 * reverse and hasBias are Boolean, reserved is zero, I/H are positive.
 */
typedef struct {
    uint32_t kind;
    uint32_t time;
    uint32_t batch;
    uint32_t inputSize;
    uint32_t hiddenSize;
    uint32_t reverse;
    uint32_t hasBias;
    uint32_t reserved;
} SynaptikRecurrentMeta;
_Static_assert(sizeof(SynaptikRecurrentMeta) == 32, "recurrent metadata size");
_Static_assert(offsetof(SynaptikRecurrentMeta, reserved) == 28, "recurrent metadata packing");

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

/* Assemble after SynaptikFloat32MathKernelSource. Shared dependencies are
 * synaptik_f32_sigmoid(float) and synaptik_f32_tanh(float). Disable fast math
 * and contraction, as for the other FLOAT32 named-working-site components.
 *
 * synaptik_recurrent_f32 dispatch: exactly B workitems (one row each), any
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
"    uint kind, time, batch, inputSize, hiddenSize, reverse, hasBias, reserved;\n"
"};\n"
"inline float2 synaptik_recurrent_projection(\n"
"        device const float *x, device const float *previous,\n"
"        device const float *w, device const float *u, device const float *bias,\n"
"        ulong inputBase, ulong hiddenBase, ulong gateRow,\n"
"        constant SynaptikRecurrentMeta &m) {\n"
"    float inputSum = 0.0f, hiddenSum = 0.0f;\n"
"    ulong weightBase = gateRow * ulong(m.inputSize);\n"
"    for (uint k = 0; k < m.inputSize; ++k) {\n"
"        float product = x[inputBase + k] * w[weightBase + k];\n"
"        inputSum = inputSum + product;\n"
"    }\n"
"    ulong recurrentBase = gateRow * ulong(m.hiddenSize);\n"
"    for (uint k = 0; k < m.hiddenSize; ++k) {\n"
"        float product = previous[hiddenBase + k] * u[recurrentBase + k];\n"
"        hiddenSum = hiddenSum + product;\n"
"    }\n"
"    if (m.hasBias != 0u) inputSum = inputSum + bias[gateRow];\n"
"    return float2(inputSum, hiddenSum);\n"
"}\n"
"kernel void synaptik_recurrent_f32(\n"
"        device const float *x [[buffer(0)]],\n"
"        device const long *lengths [[buffer(1)]],\n"
"        device const float *h0 [[buffer(2)]],\n"
"        device const float *c0 [[buffer(3)]],\n"
"        device const float *w [[buffer(4)]],\n"
"        device const float *u [[buffer(5)]],\n"
"        device const float *bias [[buffer(6)]],\n"
"        device float *y [[buffer(7)]],\n"
"        device float *finalH [[buffer(8)]],\n"
"        device float *finalC [[buffer(9)]],\n"
"        constant SynaptikRecurrentMeta &m [[buffer(10)]],\n"
"        uint row [[thread_position_in_grid]]) {\n"
"    if (row >= m.batch) return;\n"
"    // Global preflight has already validated every signed length before writes.\n"
"    uint count = uint(lengths[row]);\n"
"    ulong stateBase = ulong(row) * m.hiddenSize;\n"
"    for (uint t = count; t < m.time; ++t) {\n"
"        ulong base = (ulong(t) * m.batch + row) * m.hiddenSize;\n"
"        for (uint j = 0; j < m.hiddenSize; ++j) y[base + j] = 0.0f;\n"
"    }\n"
"    if (count == 0u) {\n"
"        for (uint j = 0; j < m.hiddenSize; ++j) {\n"
"            ulong state = stateBase + j;\n"
"            reinterpret_cast<device uint *>(finalH)[state] =\n"
"                    reinterpret_cast<device const uint *>(h0)[state];\n"
"            if (m.kind == 105u) reinterpret_cast<device uint *>(finalC)[state] =\n"
"                    reinterpret_cast<device const uint *>(c0)[state];\n"
"        }\n"
"        return;\n"
"    }\n"
"    if (m.kind == 105u) {\n"
"        for (uint j = 0; j < m.hiddenSize; ++j)\n"
"            reinterpret_cast<device uint *>(finalC)[stateBase + j] =\n"
"                    reinterpret_cast<device const uint *>(c0)[stateBase + j];\n"
"    }\n"
"    for (uint visit = 0; visit < count; ++visit) {\n"
"        uint t = m.reverse != 0u ? count - 1u - visit : visit;\n"
"        ulong inputBase = (ulong(t) * m.batch + row) * m.inputSize;\n"
"        ulong outputBase = (ulong(t) * m.batch + row) * m.hiddenSize;\n"
"        device const float *previous = visit == 0u ? h0 : y;\n"
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
"                    float difference = previous[previousBase + j] - candidate;\n"
"                    float update = gate1 * difference;\n"
"                    next = candidate + update;\n"
"                } else {\n"
"                    float2 fourth = synaptik_recurrent_projection(x, previous, w, u, bias,\n"
"                            inputBase, previousBase, 3ul * m.hiddenSize + j, m);\n"
"                    float candidate = synaptik_f32_tanh(third.x + third.y);\n"
"                    float outputGate = synaptik_f32_sigmoid(fourth.x + fourth.y);\n"
"                    float retained = gate1 * finalC[stateBase + j];\n"
"                    float admitted = gate0 * candidate;\n"
"                    float cell = retained + admitted;\n"
"                    finalC[stateBase + j] = cell;\n"
"                    next = outputGate * synaptik_f32_tanh(cell);\n"
"                }\n"
"            }\n"
"            y[outputBase + j] = next;\n"
"            if (visit == count - 1u) finalH[stateBase + j] = next;\n"
"        }\n"
"    }\n"
"}\n";

#endif
