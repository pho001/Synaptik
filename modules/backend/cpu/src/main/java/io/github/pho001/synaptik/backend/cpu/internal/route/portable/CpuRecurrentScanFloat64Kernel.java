package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLowering.Facts;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Immutable direct Java FLOAT64 scan kernel. Each invocation supplies its own exact segments.
 * All packed projections from prior hidden state finish before any hidden coordinate is replaced.
 * Invalid/padded coordinates never reach projections. No step allocates storage or inspects a graph.
 * Input roles use the prepared nonnegative offsets/strides relative to each supplied segment;
 * output and carried state are dense. Callers preflight every representation and the complete
 * logical length vector before calling a family method; these methods perform result writes.
 */
public final class CpuRecurrentScanFloat64Kernel {
    private final int batch, input, hidden, packed, step;
    private final boolean reverse;
    private final long xo, xt, xb, xi, lo, ls, ho, hb, hj, co, cb, cj;
    private final long wo, wr, wk, uo, ur, uk, bo, bs;

    /**
     * Retains only checked static loop geometry; no storage or runtime lengths are retained.
     * @param facts non-null descriptor-validated occurrence with exact role addressing
     * @throws NullPointerException if facts is null
     */
    public CpuRecurrentScanFloat64Kernel(Facts facts) {
        var geometry = facts.geometry();
        var roles = facts.rolePositions();
        var bindings = facts.accessBindings();
        var x = bindings.get(roles.get(0)); var l = bindings.get(roles.get(1));
        var h = bindings.get(roles.get(2));
        var c = geometry.gates() == 4 ? bindings.get(roles.get(3)) : h;
        int weight = geometry.gates() == 4 ? 4 : 3;
        var w = bindings.get(roles.get(weight)); var u = bindings.get(roles.get(weight + 1));
        var b = geometry.bias() ? bindings.get(roles.get(weight + 2)) : l;
        xo = x.baseElementOffset(); xt = x.effectiveStrides().get(0);
        xb = x.effectiveStrides().get(1); xi = x.effectiveStrides().get(2);
        lo = l.baseElementOffset(); ls = l.effectiveStrides().getFirst();
        ho = h.baseElementOffset(); hb = h.effectiveStrides().get(0); hj = h.effectiveStrides().get(1);
        co = c.baseElementOffset(); cb = c.effectiveStrides().get(0); cj = c.effectiveStrides().get(1);
        wo = w.baseElementOffset(); wr = w.effectiveStrides().get(0); wk = w.effectiveStrides().get(1);
        uo = u.baseElementOffset(); ur = u.effectiveStrides().get(0); uk = u.effectiveStrides().get(1);
        bo = b.baseElementOffset(); bs = b.effectiveStrides().getFirst();
        batch = geometry.batch(); input = geometry.inputSize();
        hidden = geometry.hiddenSize(); packed = Math.multiplyExact(geometry.gates(), hidden);
        reverse = geometry.reverse(); step = reverse ? -1 : 1;
    }

    /**
     * Executes the tanh family after complete invocation preflight.
     * @param x resolved strided input, borrowed read-only
     * @param lengths borrowed INT64 span with every logical length validated in [0,T]; read at
     *     the prepared offset and batch stride
     * @param h0 borrowed initial hidden source at its prepared read layout, gathered as raw bits
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
            int length = (int) lengths.getAtIndex(JAVA_LONG, lo + b * ls);
            for (int n = 0, t = reverse ? length - 1 : 0; n < length; n++, t += step) {
                project(x, w, u, bias, h, scratch, t, b);
                for (int j = 0; j < hidden; j++) {
                    double next = StrictMath.tanh(get(scratch, j) + get(scratch, packed + j));
                    put(h, (long) b * hidden + j, next);
                    put(output, ((long) t * batch + b) * hidden + j, next);
                }
            }
        }
    }

    /**
     * Executes reset-after GRU with reset/update/candidate packing and n + z*(h-n).
     * @param x resolved strided input, borrowed read-only
     * @param lengths borrowed INT64 span with every logical length validated in [0,T]; read at
     *     the prepared offset and batch stride
     * @param h0 borrowed initial hidden source at its prepared read layout, gathered as raw bits
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
            int length = (int) lengths.getAtIndex(JAVA_LONG, lo + b * ls);
            for (int n = 0, t = reverse ? length - 1 : 0; n < length; n++, t += step) {
                project(x, w, u, bias, h, scratch, t, b);
                for (int j = 0; j < hidden; j++) {
                    double r = sigmoid(get(scratch, j) + get(scratch, packed + j));
                    double z = sigmoid(get(scratch, hidden + j) + get(scratch, packed + hidden + j));
                    double candidate = StrictMath.tanh(get(scratch, 2 * hidden + j)
                            + r * get(scratch, packed + 2 * hidden + j));
                    long state = (long) b * hidden + j;
                    double next = candidate + z * (get(h, state) - candidate);
                    put(h, state, next);
                    put(output, ((long) t * batch + b) * hidden + j, next);
                }
            }
        }
    }

    /**
     * Executes LSTM with explicit carried cell and input/forget/candidate/output gate packing.
     * @param x resolved strided input, borrowed read-only
     * @param lengths borrowed INT64 span with every logical length validated in [0,T]; read at
     *     the prepared offset and batch stride
     * @param h0 borrowed initial hidden source at its prepared read layout, gathered as raw bits
     * @param c0 borrowed initial cell source at its prepared read layout, gathered as raw bits
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
        copyState(c0, c, co, cb, cj);
        for (int b = 0; b < batch; b++) {
            int length = (int) lengths.getAtIndex(JAVA_LONG, lo + b * ls);
            for (int n = 0, t = reverse ? length - 1 : 0; n < length; n++, t += step) {
                project(x, w, u, bias, h, scratch, t, b);
                for (int j = 0; j < hidden; j++) {
                    double i = sigmoid(get(scratch, j) + get(scratch, packed + j));
                    double f = sigmoid(get(scratch, hidden + j) + get(scratch, packed + hidden + j));
                    double g = StrictMath.tanh(get(scratch, 2 * hidden + j)
                            + get(scratch, packed + 2 * hidden + j));
                    double o = sigmoid(get(scratch, 3 * hidden + j)
                            + get(scratch, packed + 3 * hidden + j));
                    long state = (long) b * hidden + j;
                    double nextCell = f * get(c, state) + i * g;
                    double next = o * StrictMath.tanh(nextCell);
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
        copyState(h0, h, ho, hb, hj);
    }


    /**
     * Gathers raw initial bits; empty states never dereference an unused origin or stride.
     * @param source borrowed resolved initial state
     * @param target dense disjoint carried state
     * @param offset source origin in elements
     * @param rowStride source batch stride in elements
     * @param featureStride source hidden stride in elements
     */
    private void copyState(MemorySegment source, MemorySegment target,
            long offset, long rowStride, long featureStride) {
        if (batch == 0) return;
        if (rowStride == hidden && featureStride == 1) {
            MemorySegment.copy(source, offset * Double.BYTES, target, 0, (long) batch * hidden * Double.BYTES);
        } else for (int b = 0; b < batch; b++) {
            long address = offset + b * rowStride;
            for (int j = 0; j < hidden; j++, address += featureStride)
                target.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG, (long) b * hidden + j,
                        source.getAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG, address));
        }
    }

    /**
     * Completes both packed projections before any hidden value can be overwritten.
     * @param x readable referenced input span, addressed with prepared time/batch/feature strides
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
        long xbase = xo + t * xt + b * xb, hbase = (long) b * hidden;
        for (int row = 0; row < packed; row++) {
            double px = 0, ph = 0;
            long xa = xbase, wa = wo + row * wr, ua = uo + row * ur;
            for (int k = 0; k < input; k++, xa += xi, wa += wk) px += get(x, xa) * get(w, wa);
            for (int k = 0; k < hidden; k++, ua += uk) ph += get(h, hbase + k) * get(u, ua);
            put(scratch, row, px); put(scratch, packed + row, ph);
        }
        if (bias != null) for (int row = 0; row < packed; row++)
            put(scratch, row, get(scratch, row) + get(bias, bo + row * bs));
    }

    /**
     * Evaluates Model's exact sign-guarded FLOAT64 sigmoid branch without positive EXP overflow.
     * @param x gate value, including zero, NaN or infinity
     * @return rounded FLOAT64 branch result with the required special-value class
     */
    private static double sigmoid(double x) {
        if (x >= 0) return 1d / (1d + StrictMath.exp(-x));
        double exp = StrictMath.exp(x);
        return exp / (1d + exp);
    }
    /**
     * Reads a preflighted physical FLOAT64 element.
     * @param segment exact readable referenced span
     * @param element in-bounds physical element index
     * @return represented FLOAT64 value without a carrier conversion
     */
    private static double get(MemorySegment segment, long element) { return segment.getAtIndex(JAVA_DOUBLE, element); }
    /**
     * Stores one state or gate value at its canonical coordinate.
     * @param segment exact writable logical span
     * @param element in-bounds physical element index
     * @param value represented FLOAT64 value to publish unchanged
     */
    private static void put(MemorySegment segment, long element, double value) { segment.setAtIndex(JAVA_DOUBLE, element, value); }
}
