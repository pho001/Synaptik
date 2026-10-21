package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLowering.Facts;
import java.lang.foreign.MemorySegment;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Immutable direct Java BFLOAT16 scan kernel with FLOAT32 work and accumulation.
 * Hidden and cell publications narrow once to storage; subsequent reads decode that exact state.
 * Gates and projections never narrow to low storage. LSTM hidden consumes the stored cell.
 * Each invocation supplies its own exact segments.
 * All packed projections from prior hidden state finish before any hidden coordinate is replaced.
 * Invalid/padded coordinates never reach projections. No step allocates storage or inspects a graph.
 * Input roles use the prepared nonnegative offsets/strides relative to each supplied segment;
 * output and carried state are dense. Callers preflight every representation and the complete
 * logical length vector before calling a family method; these methods perform result writes.
 */
public final class CpuRecurrentScanBFloat16Kernel {
    private final int batch, input, hidden, packed, step;
    private final boolean reverse;
    private final long xo, xt, xb, xi, lo, ls, ho, hb, hj, co, cb, cj;
    private final long wo, wr, wk, uo, ur, uk, bo, bs;

    /**
     * Retains only checked static loop geometry; no storage or runtime lengths are retained.
     * @param facts non-null descriptor-validated occurrence with exact role addressing
     * @throws NullPointerException if facts is null
     */
    public CpuRecurrentScanBFloat16Kernel(Facts facts) {
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
                    float next = (float) StrictMath.tanh(work(scratch, j) + work(scratch, packed + j));
                    short stored = BFloat16Bits.fromFloat(next);
                    h.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, (long) b * hidden + j, stored);
                    output.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT,
                            ((long) t * batch + b) * hidden + j, stored);
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
                    float r = sigmoid(work(scratch, j) + work(scratch, packed + j));
                    float z = sigmoid(work(scratch, hidden + j) + work(scratch, packed + hidden + j));
                    float candidate = (float) StrictMath.tanh(work(scratch, 2 * hidden + j)
                            + r * work(scratch, packed + 2 * hidden + j));
                    long state = (long) b * hidden + j;
                    float next = candidate + z * (get(h, state) - candidate);
                    short stored = BFloat16Bits.fromFloat(next);
                    h.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, state, stored);
                    output.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT,
                            ((long) t * batch + b) * hidden + j, stored);
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
                    float i = sigmoid(work(scratch, j) + work(scratch, packed + j));
                    float f = sigmoid(work(scratch, hidden + j) + work(scratch, packed + hidden + j));
                    float g = (float) StrictMath.tanh(work(scratch, 2 * hidden + j)
                            + work(scratch, packed + 2 * hidden + j));
                    float o = sigmoid(work(scratch, 3 * hidden + j)
                            + work(scratch, packed + 3 * hidden + j));
                    long state = (long) b * hidden + j;
                    float nextCell = f * get(c, state) + i * g;
                    short storedCell = BFloat16Bits.fromFloat(nextCell);
                    c.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, state, storedCell);
                    float next = o * (float) StrictMath.tanh(BFloat16Bits.toFloat(storedCell));
                    short storedHidden = BFloat16Bits.fromFloat(next);
                    h.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, state, storedHidden);
                    output.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT,
                            ((long) t * batch + b) * hidden + j, storedHidden);
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
            MemorySegment.copy(source, offset * Short.BYTES, target, 0, (long) batch * hidden * Short.BYTES);
        } else for (int b = 0; b < batch; b++) {
            long address = offset + b * rowStride;
            for (int j = 0; j < hidden; j++, address += featureStride)
                target.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, (long) b * hidden + j,
                        source.getAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, address));
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
            float px = 0, ph = 0;
            long xa = xbase, wa = wo + row * wr, ua = uo + row * ur;
            for (int k = 0; k < input; k++, xa += xi, wa += wk) px += get(x, xa) * get(w, wa);
            for (int k = 0; k < hidden; k++, ua += uk) ph += get(h, hbase + k) * get(u, ua);
            putWork(scratch, row, px); putWork(scratch, packed + row, ph);
        }
        if (bias != null) for (int row = 0; row < packed; row++)
            putWork(scratch, row, work(scratch, row) + get(bias, bo + row * bs));
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
     * Decodes one preflighted BFLOAT16 storage element exactly.
     * @param segment exact readable referenced span
     * @param element in-bounds physical element index
     * @return exactly decoded FLOAT32 working value
     */
    private static float get(MemorySegment segment, long element) { return BFloat16Bits.toFloat(segment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, element)); }
    /**
     * Stores one FLOAT32 projection value without low narrowing.
     * @param segment exact writable logical span
     * @param element in-bounds physical element index
     * @param value represented FLOAT32 value to publish unchanged
     */
    private static void putWork(MemorySegment segment, long element, float value) { segment.setAtIndex(JAVA_FLOAT, element, value); }
    /**
     * Reads one FLOAT32 projection without low narrowing.
     * @param segment readable exact projection scratch
     * @param element in-bounds packed projection index
     * @return represented FLOAT32 accumulator
     */
    private static float work(MemorySegment segment, long element) { return segment.getAtIndex(JAVA_FLOAT, element); }
}
