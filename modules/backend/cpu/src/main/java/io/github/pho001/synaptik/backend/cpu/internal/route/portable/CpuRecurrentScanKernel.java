package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLowering.Geometry;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Immutable direct Java FLOAT32 scan kernel. Each invocation supplies its own exact segments.
 * All packed projections from prior hidden state finish before any hidden coordinate is replaced.
 * Invalid/padded coordinates never reach projections. No step allocates storage or inspects a graph.
 */
public final class CpuRecurrentScanKernel {
    private final int batch, input, hidden, packed, step;
    private final boolean reverse;

    /**
     * Retains only checked static loop geometry; no storage or runtime lengths are retained.
     * @param geometry non-null descriptor-validated occurrence geometry
     */
    public CpuRecurrentScanKernel(Geometry geometry) {
        batch = geometry.batch(); input = geometry.inputSize();
        hidden = geometry.hiddenSize(); packed = Math.multiplyExact(geometry.gates(), hidden);
        reverse = geometry.reverse(); step = reverse ? -1 : 1;
    }

    /**
     * Executes the tanh family after complete invocation preflight.
     * @param x exact canonical input, borrowed read-only
     * @param lengths validated INT64 batch lengths
     * @param h0 initial hidden values, copied raw
     * @param w packed input weights
     * @param u packed hidden weights
     * @param bias optional input-side bias, or null
     * @param output writable dense output, disjoint from all inputs and scratch
     * @param h writable carried/final hidden, disjoint from other segments
     * @param scratch exact writable two-projection workspace; null only for an empty domain
     */
    public void rnn(MemorySegment x, MemorySegment lengths, MemorySegment h0,
            MemorySegment w, MemorySegment u, MemorySegment bias, MemorySegment output,
            MemorySegment h, MemorySegment scratch) {
        initialize(output, h0, h);
        for (int b = 0; b < batch; b++) {
            int length = (int) lengths.getAtIndex(JAVA_LONG, b);
            for (int n = 0, t = reverse ? length - 1 : 0; n < length; n++, t += step) {
                project(x, w, u, bias, h, scratch, t, b);
                for (int j = 0; j < hidden; j++) {
                    float next = (float) StrictMath.tanh(get(scratch, j) + get(scratch, packed + j));
                    put(h, (long) b * hidden + j, next);
                    put(output, ((long) t * batch + b) * hidden + j, next);
                }
            }
        }
    }

    /**
     * Executes reset-after GRU with reset/update/candidate packing and n + z*(h-n).
     * @param x exact canonical input, borrowed read-only
     * @param lengths validated INT64 batch lengths
     * @param h0 initial hidden values, copied raw
     * @param w packed input weights
     * @param u packed hidden weights
     * @param bias optional input-side bias, or null
     * @param output writable dense output, disjoint from all inputs and scratch
     * @param h writable carried/final hidden, disjoint from other segments
     * @param scratch exact writable two-projection workspace; null only for an empty domain
     */
    public void gru(MemorySegment x, MemorySegment lengths, MemorySegment h0,
            MemorySegment w, MemorySegment u, MemorySegment bias, MemorySegment output,
            MemorySegment h, MemorySegment scratch) {
        initialize(output, h0, h);
        for (int b = 0; b < batch; b++) {
            int length = (int) lengths.getAtIndex(JAVA_LONG, b);
            for (int n = 0, t = reverse ? length - 1 : 0; n < length; n++, t += step) {
                project(x, w, u, bias, h, scratch, t, b);
                for (int j = 0; j < hidden; j++) {
                    float r = sigmoid(get(scratch, j) + get(scratch, packed + j));
                    float z = sigmoid(get(scratch, hidden + j) + get(scratch, packed + hidden + j));
                    float candidate = (float) StrictMath.tanh(get(scratch, 2 * hidden + j)
                            + r * get(scratch, packed + 2 * hidden + j));
                    long state = (long) b * hidden + j;
                    float next = candidate + z * (get(h, state) - candidate);
                    put(h, state, next);
                    put(output, ((long) t * batch + b) * hidden + j, next);
                }
            }
        }
    }

    /**
     * Executes LSTM with explicit carried cell and input/forget/candidate/output gate packing.
     * @param x exact canonical input, borrowed read-only
     * @param lengths validated INT64 batch lengths
     * @param h0 initial hidden values, copied raw
     * @param c0 initial cell values, copied raw
     * @param w packed input weights
     * @param u packed hidden weights
     * @param bias optional input-side bias, or null
     * @param output writable dense output, disjoint from all inputs and scratch
     * @param h writable carried/final hidden, disjoint from other segments
     * @param c writable carried/final cell, disjoint from other segments
     * @param scratch exact writable two-projection workspace; null only for an empty domain
     */
    public void lstm(MemorySegment x, MemorySegment lengths, MemorySegment h0, MemorySegment c0,
            MemorySegment w, MemorySegment u, MemorySegment bias, MemorySegment output,
            MemorySegment h, MemorySegment c, MemorySegment scratch) {
        initialize(output, h0, h);
        MemorySegment.copy(c0, 0, c, 0, (long) batch * hidden * Float.BYTES);
        for (int b = 0; b < batch; b++) {
            int length = (int) lengths.getAtIndex(JAVA_LONG, b);
            for (int n = 0, t = reverse ? length - 1 : 0; n < length; n++, t += step) {
                project(x, w, u, bias, h, scratch, t, b);
                for (int j = 0; j < hidden; j++) {
                    float i = sigmoid(get(scratch, j) + get(scratch, packed + j));
                    float f = sigmoid(get(scratch, hidden + j) + get(scratch, packed + hidden + j));
                    float g = (float) StrictMath.tanh(get(scratch, 2 * hidden + j)
                            + get(scratch, packed + 2 * hidden + j));
                    float o = sigmoid(get(scratch, 3 * hidden + j)
                            + get(scratch, packed + 3 * hidden + j));
                    long state = (long) b * hidden + j;
                    float nextCell = f * get(c, state) + i * g;
                    float next = o * (float) StrictMath.tanh(nextCell);
                    put(c, state, nextCell); put(h, state, next);
                    put(output, ((long) t * batch + b) * hidden + j, next);
                }
            }
        }
    }

    /**
     * Initializes only preflighted logical output spans, preserving raw carried state bits.
     * @param output exact logical dense output, writable and disjoint
     * @param h0 exact initial hidden source, readable
     * @param h exact carried hidden destination, writable and disjoint
     */
    private void initialize(MemorySegment output, MemorySegment h0, MemorySegment h) {
        output.fill((byte) 0);
        MemorySegment.copy(h0, 0, h, 0, (long) batch * hidden * Float.BYTES);
    }

    /**
     * Completes both packed projections before any hidden value can be overwritten.
     * @param x readable exact time-major input
     * @param w readable input-side packed weights
     * @param u readable recurrent packed weights
     * @param bias readable input-side packed bias, or null
     * @param h readable prior row state in the final hidden buffer
     * @param scratch writable exact two-projection workspace
     * @param t active original time coordinate
     * @param b active original batch coordinate
     */
    private void project(MemorySegment x, MemorySegment w, MemorySegment u, MemorySegment bias,
            MemorySegment h, MemorySegment scratch, int t, int b) {
        long xb = ((long) t * batch + b) * input, hb = (long) b * hidden;
        for (int row = 0; row < packed; row++) {
            float px = 0, ph = 0;
            for (int k = 0; k < input; k++) px += get(x, xb + k) * get(w, (long) row * input + k);
            for (int k = 0; k < hidden; k++) ph += get(h, hb + k) * get(u, (long) row * hidden + k);
            put(scratch, row, px); put(scratch, packed + row, ph);
        }
        if (bias != null) for (int row = 0; row < packed; row++)
            put(scratch, row, get(scratch, row) + get(bias, row));
    }

    /**
     * Evaluates Model's exact sign-guarded FLOAT32 sigmoid branch without positive EXP overflow.
     * @param x gate value, including zero, NaN or infinity
     * @return rounded FLOAT32 branch result with the required special-value class
     */
    private static float sigmoid(float x) {
        if (x >= 0) return 1f / (1f + (float) StrictMath.exp(-x));
        float exp = (float) StrictMath.exp(x);
        return exp / (1f + exp);
    }
    /**
     * Reads a preflighted canonical FLOAT32 element.
     * @param segment exact readable logical span
     * @param element in-bounds logical element index
     * @return represented FLOAT32 value without a carrier conversion
     */
    private static float get(MemorySegment segment, long element) { return segment.getAtIndex(JAVA_FLOAT, element); }
    /**
     * Stores one state or gate value at its canonical coordinate.
     * @param segment exact writable logical span
     * @param element in-bounds logical element index
     * @param value represented FLOAT32 value to publish unchanged
     */
    private static void put(MemorySegment segment, long element, float value) { segment.setAtIndex(JAVA_FLOAT, element, value); }
}
