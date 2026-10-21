package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLowering.Facts;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.*;

/**
 * Test-only specialized clean Java scan oracle. No production kernel or execution entry point is
 * called. Cold binding selects twelve primitive family/type loops over the same exact carriers.
 * Two packed work rows complete prior-hidden contractions before replacement. Explicit stored
 * cell/hidden barriers and rounded elementary sites make this a sequential route oracle, not a
 * cross-backend bit-equality policy. All allocation and selection happens outside measured work.
 */
public final class CpuRecurrentScanCleanJavaOracle {
    private CpuRecurrentScanCleanJavaOracle() { }

    /**
     * Selects one independent primitive loop without inspecting runtime lengths.
     * @param facts checked shape and resolved role layout metadata
     * @param boundaries unique reads followed by independent canonical output segments
     * @param scratch independent two-row working storage, null for an empty domain
     * @return direct fixed-family/type action; caller validates lengths before invocation
     */
    public static Runnable bind(Facts facts, MemorySegment[] boundaries, MemorySegment scratch) {
        var a = new Arguments(facts, boundaries, scratch);
        return switch (facts.geometry().dataType()) {
            case FLOAT32 -> switch (facts.geometry().kind()) {
                case RNN_TANH -> () -> rnn32(a);
                case GRU_RESET_AFTER -> () -> gru32(a);
                case LSTM -> () -> lstm32(a);
            };
            case FLOAT64 -> switch (facts.geometry().kind()) {
                case RNN_TANH -> () -> rnn64(a);
                case GRU_RESET_AFTER -> () -> gru64(a);
                case LSTM -> () -> lstm64(a);
            };
            case BFLOAT16 -> switch (facts.geometry().kind()) {
                case RNN_TANH -> () -> rnnBF(a);
                case GRU_RESET_AFTER -> () -> gruBF(a);
                case LSTM -> () -> lstmBF(a);
            };
            case FLOAT16 -> switch (facts.geometry().kind()) {
                case RNN_TANH -> () -> rnnHF(a);
                case GRU_RESET_AFTER -> () -> gruHF(a);
                case LSTM -> () -> lstmHF(a);
            };
            default -> throw new AssertionError("floating oracle only");
        };
    }

    /** Physical read geometry only; deliberately contains no arithmetic or dtype dispatch. */
    private record Read(MemorySegment segment, long offset, long s0, long s1, long s2) { }

    /** Cold snapshot of primitives and direct segments, shared by no invocation. */
    private static final class Arguments {
        final int time, batch, input, hidden, packed, step;
        final boolean reverse;
        final Read x, lengths, h0, c0, w, u, bias;
        final MemorySegment y, h, c, work;
        Arguments(Facts facts, MemorySegment[] boundary, MemorySegment scratch) {
            var g = facts.geometry();
            time = g.time(); batch = g.batch(); input = g.inputSize(); hidden = g.hiddenSize();
            packed = g.gates() * hidden; reverse = g.reverse(); step = reverse ? -1 : 1;
            x = read(facts, boundary, 0); lengths = read(facts, boundary, 1);
            h0 = read(facts, boundary, 2); c0 = g.gates() == 4 ? read(facts, boundary, 3) : null;
            int weight = g.gates() == 4 ? 4 : 3;
            w = read(facts, boundary, weight); u = read(facts, boundary, weight + 1);
            bias = g.bias() ? read(facts, boundary, weight + 2) : null;
            y = boundary[facts.readCount()]; h = boundary[facts.readCount() + 1];
            c = g.gates() == 4 ? boundary[facts.readCount() + 2] : null; work = scratch;
        }
    }

    private static Read read(Facts facts, MemorySegment[] boundary, int role) {
        int position = facts.rolePositions().get(role);
        Binding b = facts.accessBindings().get(position);
        return new Read(boundary[position], b.baseElementOffset(), b.effectiveStrides().get(0),
                b.effectiveStrides().size() > 1 ? b.effectiveStrides().get(1) : 0,
                b.effectiveStrides().size() > 2 ? b.effectiveStrides().get(2) : 0);
    }

    private static void initialize32(Arguments a) {
        a.y.fill((byte) 0);
        copy32(a.h0, a.h, a.batch, a.hidden);
        if (a.c != null) copy32(a.c0, a.c, a.batch, a.hidden);
    }

    private static void copy32(Read from, MemorySegment to, int rows, int width) {
        if (rows == 0) return;
        if (from.s0 == width && from.s1 == 1) {
            MemorySegment.copy(from.segment, from.offset * JAVA_INT.byteSize(), to, 0,
                    (long) rows * width * JAVA_INT.byteSize());
        } else for (int row = 0; row < rows; row++) {
            long address = from.offset + row * from.s0;
            for (int j = 0; j < width; j++, address += from.s1)
                to.setAtIndex(JAVA_INT, (long) row * width + j, from.segment.getAtIndex(JAVA_INT, address));
        }
    }

    private static void project32(Arguments a, int time, int row) {
        long inputBase = a.x.offset + time * a.x.s0 + row * a.x.s1;
        for (int gateRow = 0; gateRow < a.packed; gateRow++) {
            float inputSum = 0, hiddenSum = 0;
            long x = inputBase, w = a.w.offset + gateRow * a.w.s0, u = a.u.offset + gateRow * a.u.s0;
            for (int k = 0; k < a.input; k++, x += a.x.s2, w += a.w.s1)
                inputSum += a.x.segment.getAtIndex(JAVA_FLOAT, x) * a.w.segment.getAtIndex(JAVA_FLOAT, w);
            for (int k = 0; k < a.hidden; k++, u += a.u.s1)
                hiddenSum += a.h.getAtIndex(JAVA_FLOAT, (long) row * a.hidden + k) * a.u.segment.getAtIndex(JAVA_FLOAT, u);
            a.work.setAtIndex(JAVA_FLOAT, gateRow, inputSum);
            a.work.setAtIndex(JAVA_FLOAT, a.packed + gateRow, hiddenSum);
        }
        if (a.bias != null) for (int gateRow = 0; gateRow < a.packed; gateRow++)
            a.work.setAtIndex(JAVA_FLOAT, gateRow,
                    a.work.getAtIndex(JAVA_FLOAT, gateRow) + a.bias.segment.getAtIndex(JAVA_FLOAT, a.bias.offset + gateRow * a.bias.s0));
    }

    private static float sigmoid32(float value) {
        float e = (float) StrictMath.exp(value >= 0 ? -value : value);
        float denominator = 1f + e;
        return value >= 0 ? 1f / denominator : e / denominator;
    }

    private static void rnn32(Arguments a) {
        initialize32(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                project32(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float next = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    a.h.setAtIndex(JAVA_FLOAT, state, next);
                    a.y.setAtIndex(JAVA_FLOAT, ((long) time * a.batch + row) * a.hidden + j, next);
                }
            }
        }
    }

    private static void gru32(Arguments a) {
        initialize32(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                project32(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float reset = sigmoid32(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    float update = sigmoid32(a.work.getAtIndex(JAVA_FLOAT, a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + a.hidden + j));
                    float candidate = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, 2 * a.hidden + j) + reset * a.work.getAtIndex(JAVA_FLOAT, a.packed + 2 * a.hidden + j));
                    float next = candidate + update * (a.h.getAtIndex(JAVA_FLOAT, state) - candidate);
                    a.h.setAtIndex(JAVA_FLOAT, state, next);
                    a.y.setAtIndex(JAVA_FLOAT, ((long) time * a.batch + row) * a.hidden + j, next);
                }
            }
        }
    }

    private static void lstm32(Arguments a) {
        initialize32(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                project32(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float input = sigmoid32(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    float forget = sigmoid32(a.work.getAtIndex(JAVA_FLOAT, a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + a.hidden + j));
                    float candidate = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, 2 * a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + 2 * a.hidden + j));
                    float output = sigmoid32(a.work.getAtIndex(JAVA_FLOAT, 3 * a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + 3 * a.hidden + j));
                    float cell = forget * a.c.getAtIndex(JAVA_FLOAT, state) + input * candidate;
                    a.c.setAtIndex(JAVA_FLOAT, state, cell);
                    float consumedCell = cell;
                    float next = output * (float) StrictMath.tanh(consumedCell);
                    a.h.setAtIndex(JAVA_FLOAT, state, next);
                    a.y.setAtIndex(JAVA_FLOAT, ((long) time * a.batch + row) * a.hidden + j, next);
                }
            }
        }
    }

    private static void initialize64(Arguments a) {
        a.y.fill((byte) 0);
        copy64(a.h0, a.h, a.batch, a.hidden);
        if (a.c != null) copy64(a.c0, a.c, a.batch, a.hidden);
    }

    private static void copy64(Read from, MemorySegment to, int rows, int width) {
        if (rows == 0) return;
        if (from.s0 == width && from.s1 == 1) {
            MemorySegment.copy(from.segment, from.offset * JAVA_LONG.byteSize(), to, 0,
                    (long) rows * width * JAVA_LONG.byteSize());
        } else for (int row = 0; row < rows; row++) {
            long address = from.offset + row * from.s0;
            for (int j = 0; j < width; j++, address += from.s1)
                to.setAtIndex(JAVA_LONG, (long) row * width + j, from.segment.getAtIndex(JAVA_LONG, address));
        }
    }

    private static void project64(Arguments a, int time, int row) {
        long inputBase = a.x.offset + time * a.x.s0 + row * a.x.s1;
        for (int gateRow = 0; gateRow < a.packed; gateRow++) {
            double inputSum = 0, hiddenSum = 0;
            long x = inputBase, w = a.w.offset + gateRow * a.w.s0, u = a.u.offset + gateRow * a.u.s0;
            for (int k = 0; k < a.input; k++, x += a.x.s2, w += a.w.s1)
                inputSum += a.x.segment.getAtIndex(JAVA_DOUBLE, x) * a.w.segment.getAtIndex(JAVA_DOUBLE, w);
            for (int k = 0; k < a.hidden; k++, u += a.u.s1)
                hiddenSum += a.h.getAtIndex(JAVA_DOUBLE, (long) row * a.hidden + k) * a.u.segment.getAtIndex(JAVA_DOUBLE, u);
            a.work.setAtIndex(JAVA_DOUBLE, gateRow, inputSum);
            a.work.setAtIndex(JAVA_DOUBLE, a.packed + gateRow, hiddenSum);
        }
        if (a.bias != null) for (int gateRow = 0; gateRow < a.packed; gateRow++)
            a.work.setAtIndex(JAVA_DOUBLE, gateRow,
                    a.work.getAtIndex(JAVA_DOUBLE, gateRow) + a.bias.segment.getAtIndex(JAVA_DOUBLE, a.bias.offset + gateRow * a.bias.s0));
    }

    private static double sigmoid64(double value) {
        double e = StrictMath.exp(value >= 0 ? -value : value);
        double denominator = 1d + e;
        return value >= 0 ? 1d / denominator : e / denominator;
    }

    private static void rnn64(Arguments a) {
        initialize64(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                project64(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    double next = StrictMath.tanh(a.work.getAtIndex(JAVA_DOUBLE, j) + a.work.getAtIndex(JAVA_DOUBLE, a.packed + j));
                    a.h.setAtIndex(JAVA_DOUBLE, state, next);
                    a.y.setAtIndex(JAVA_DOUBLE, ((long) time * a.batch + row) * a.hidden + j, next);
                }
            }
        }
    }

    private static void gru64(Arguments a) {
        initialize64(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                project64(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    double reset = sigmoid64(a.work.getAtIndex(JAVA_DOUBLE, j) + a.work.getAtIndex(JAVA_DOUBLE, a.packed + j));
                    double update = sigmoid64(a.work.getAtIndex(JAVA_DOUBLE, a.hidden + j) + a.work.getAtIndex(JAVA_DOUBLE, a.packed + a.hidden + j));
                    double candidate = StrictMath.tanh(a.work.getAtIndex(JAVA_DOUBLE, 2 * a.hidden + j) + reset * a.work.getAtIndex(JAVA_DOUBLE, a.packed + 2 * a.hidden + j));
                    double next = candidate + update * (a.h.getAtIndex(JAVA_DOUBLE, state) - candidate);
                    a.h.setAtIndex(JAVA_DOUBLE, state, next);
                    a.y.setAtIndex(JAVA_DOUBLE, ((long) time * a.batch + row) * a.hidden + j, next);
                }
            }
        }
    }

    private static void lstm64(Arguments a) {
        initialize64(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                project64(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    double input = sigmoid64(a.work.getAtIndex(JAVA_DOUBLE, j) + a.work.getAtIndex(JAVA_DOUBLE, a.packed + j));
                    double forget = sigmoid64(a.work.getAtIndex(JAVA_DOUBLE, a.hidden + j) + a.work.getAtIndex(JAVA_DOUBLE, a.packed + a.hidden + j));
                    double candidate = StrictMath.tanh(a.work.getAtIndex(JAVA_DOUBLE, 2 * a.hidden + j) + a.work.getAtIndex(JAVA_DOUBLE, a.packed + 2 * a.hidden + j));
                    double output = sigmoid64(a.work.getAtIndex(JAVA_DOUBLE, 3 * a.hidden + j) + a.work.getAtIndex(JAVA_DOUBLE, a.packed + 3 * a.hidden + j));
                    double cell = forget * a.c.getAtIndex(JAVA_DOUBLE, state) + input * candidate;
                    a.c.setAtIndex(JAVA_DOUBLE, state, cell);
                    double consumedCell = cell;
                    double next = output * StrictMath.tanh(consumedCell);
                    a.h.setAtIndex(JAVA_DOUBLE, state, next);
                    a.y.setAtIndex(JAVA_DOUBLE, ((long) time * a.batch + row) * a.hidden + j, next);
                }
            }
        }
    }

    private static void initializeBF(Arguments a) {
        a.y.fill((byte) 0);
        copyBF(a.h0, a.h, a.batch, a.hidden);
        if (a.c != null) copyBF(a.c0, a.c, a.batch, a.hidden);
    }

    private static void copyBF(Read from, MemorySegment to, int rows, int width) {
        if (rows == 0) return;
        if (from.s0 == width && from.s1 == 1) {
            MemorySegment.copy(from.segment, from.offset * JAVA_SHORT.byteSize(), to, 0,
                    (long) rows * width * JAVA_SHORT.byteSize());
        } else for (int row = 0; row < rows; row++) {
            long address = from.offset + row * from.s0;
            for (int j = 0; j < width; j++, address += from.s1)
                to.setAtIndex(JAVA_SHORT, (long) row * width + j, from.segment.getAtIndex(JAVA_SHORT, address));
        }
    }

    private static void projectBF(Arguments a, int time, int row) {
        long inputBase = a.x.offset + time * a.x.s0 + row * a.x.s1;
        for (int gateRow = 0; gateRow < a.packed; gateRow++) {
            float inputSum = 0, hiddenSum = 0;
            long x = inputBase, w = a.w.offset + gateRow * a.w.s0, u = a.u.offset + gateRow * a.u.s0;
            for (int k = 0; k < a.input; k++, x += a.x.s2, w += a.w.s1)
                inputSum += BFloat16Bits.toFloat(a.x.segment.getAtIndex(JAVA_SHORT, x)) * BFloat16Bits.toFloat(a.w.segment.getAtIndex(JAVA_SHORT, w));
            for (int k = 0; k < a.hidden; k++, u += a.u.s1)
                hiddenSum += BFloat16Bits.toFloat(a.h.getAtIndex(JAVA_SHORT, (long) row * a.hidden + k)) * BFloat16Bits.toFloat(a.u.segment.getAtIndex(JAVA_SHORT, u));
            a.work.setAtIndex(JAVA_FLOAT, gateRow, inputSum);
            a.work.setAtIndex(JAVA_FLOAT, a.packed + gateRow, hiddenSum);
        }
        if (a.bias != null) for (int gateRow = 0; gateRow < a.packed; gateRow++)
            a.work.setAtIndex(JAVA_FLOAT, gateRow,
                    a.work.getAtIndex(JAVA_FLOAT, gateRow) + BFloat16Bits.toFloat(a.bias.segment.getAtIndex(JAVA_SHORT, a.bias.offset + gateRow * a.bias.s0)));
    }

    private static float sigmoidBF(float value) {
        float e = (float) StrictMath.exp(value >= 0 ? -value : value);
        float denominator = 1f + e;
        return value >= 0 ? 1f / denominator : e / denominator;
    }

    private static void rnnBF(Arguments a) {
        initializeBF(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                projectBF(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float next = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    short stored = BFloat16Bits.fromFloat(next);
                    a.h.setAtIndex(JAVA_SHORT, state, stored);
                    a.y.setAtIndex(JAVA_SHORT, ((long) time * a.batch + row) * a.hidden + j, stored);
                }
            }
        }
    }

    private static void gruBF(Arguments a) {
        initializeBF(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                projectBF(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float reset = sigmoidBF(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    float update = sigmoidBF(a.work.getAtIndex(JAVA_FLOAT, a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + a.hidden + j));
                    float candidate = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, 2 * a.hidden + j) + reset * a.work.getAtIndex(JAVA_FLOAT, a.packed + 2 * a.hidden + j));
                    float next = candidate + update * (BFloat16Bits.toFloat(a.h.getAtIndex(JAVA_SHORT, state)) - candidate);
                    short stored = BFloat16Bits.fromFloat(next);
                    a.h.setAtIndex(JAVA_SHORT, state, stored);
                    a.y.setAtIndex(JAVA_SHORT, ((long) time * a.batch + row) * a.hidden + j, stored);
                }
            }
        }
    }

    private static void lstmBF(Arguments a) {
        initializeBF(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                projectBF(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float input = sigmoidBF(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    float forget = sigmoidBF(a.work.getAtIndex(JAVA_FLOAT, a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + a.hidden + j));
                    float candidate = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, 2 * a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + 2 * a.hidden + j));
                    float output = sigmoidBF(a.work.getAtIndex(JAVA_FLOAT, 3 * a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + 3 * a.hidden + j));
                    float cell = forget * BFloat16Bits.toFloat(a.c.getAtIndex(JAVA_SHORT, state)) + input * candidate;
                    short storedCell = BFloat16Bits.fromFloat(cell);
                    a.c.setAtIndex(JAVA_SHORT, state, storedCell);
                    float consumedCell = BFloat16Bits.toFloat(storedCell);
                    float next = output * (float) StrictMath.tanh(consumedCell);
                    short stored = BFloat16Bits.fromFloat(next);
                    a.h.setAtIndex(JAVA_SHORT, state, stored);
                    a.y.setAtIndex(JAVA_SHORT, ((long) time * a.batch + row) * a.hidden + j, stored);
                }
            }
        }
    }

    private static void initializeHF(Arguments a) {
        a.y.fill((byte) 0);
        copyHF(a.h0, a.h, a.batch, a.hidden);
        if (a.c != null) copyHF(a.c0, a.c, a.batch, a.hidden);
    }

    private static void copyHF(Read from, MemorySegment to, int rows, int width) {
        if (rows == 0) return;
        if (from.s0 == width && from.s1 == 1) {
            MemorySegment.copy(from.segment, from.offset * JAVA_SHORT.byteSize(), to, 0,
                    (long) rows * width * JAVA_SHORT.byteSize());
        } else for (int row = 0; row < rows; row++) {
            long address = from.offset + row * from.s0;
            for (int j = 0; j < width; j++, address += from.s1)
                to.setAtIndex(JAVA_SHORT, (long) row * width + j, from.segment.getAtIndex(JAVA_SHORT, address));
        }
    }

    private static void projectHF(Arguments a, int time, int row) {
        long inputBase = a.x.offset + time * a.x.s0 + row * a.x.s1;
        for (int gateRow = 0; gateRow < a.packed; gateRow++) {
            float inputSum = 0, hiddenSum = 0;
            long x = inputBase, w = a.w.offset + gateRow * a.w.s0, u = a.u.offset + gateRow * a.u.s0;
            for (int k = 0; k < a.input; k++, x += a.x.s2, w += a.w.s1)
                inputSum += Float16Bits.toFloat(a.x.segment.getAtIndex(JAVA_SHORT, x)) * Float16Bits.toFloat(a.w.segment.getAtIndex(JAVA_SHORT, w));
            for (int k = 0; k < a.hidden; k++, u += a.u.s1)
                hiddenSum += Float16Bits.toFloat(a.h.getAtIndex(JAVA_SHORT, (long) row * a.hidden + k)) * Float16Bits.toFloat(a.u.segment.getAtIndex(JAVA_SHORT, u));
            a.work.setAtIndex(JAVA_FLOAT, gateRow, inputSum);
            a.work.setAtIndex(JAVA_FLOAT, a.packed + gateRow, hiddenSum);
        }
        if (a.bias != null) for (int gateRow = 0; gateRow < a.packed; gateRow++)
            a.work.setAtIndex(JAVA_FLOAT, gateRow,
                    a.work.getAtIndex(JAVA_FLOAT, gateRow) + Float16Bits.toFloat(a.bias.segment.getAtIndex(JAVA_SHORT, a.bias.offset + gateRow * a.bias.s0)));
    }

    private static float sigmoidHF(float value) {
        float e = (float) StrictMath.exp(value >= 0 ? -value : value);
        float denominator = 1f + e;
        return value >= 0 ? 1f / denominator : e / denominator;
    }

    private static void rnnHF(Arguments a) {
        initializeHF(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                projectHF(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float next = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    short stored = Float16Bits.fromFloat(next);
                    a.h.setAtIndex(JAVA_SHORT, state, stored);
                    a.y.setAtIndex(JAVA_SHORT, ((long) time * a.batch + row) * a.hidden + j, stored);
                }
            }
        }
    }

    private static void gruHF(Arguments a) {
        initializeHF(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                projectHF(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float reset = sigmoidHF(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    float update = sigmoidHF(a.work.getAtIndex(JAVA_FLOAT, a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + a.hidden + j));
                    float candidate = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, 2 * a.hidden + j) + reset * a.work.getAtIndex(JAVA_FLOAT, a.packed + 2 * a.hidden + j));
                    float next = candidate + update * (Float16Bits.toFloat(a.h.getAtIndex(JAVA_SHORT, state)) - candidate);
                    short stored = Float16Bits.fromFloat(next);
                    a.h.setAtIndex(JAVA_SHORT, state, stored);
                    a.y.setAtIndex(JAVA_SHORT, ((long) time * a.batch + row) * a.hidden + j, stored);
                }
            }
        }
    }

    private static void lstmHF(Arguments a) {
        initializeHF(a);
        for (int row = 0; row < a.batch; row++) {
            int count = (int) a.lengths.segment.getAtIndex(JAVA_LONG, a.lengths.offset + row * a.lengths.s0);
            for (int visit = 0, time = a.reverse ? count - 1 : 0; visit < count; visit++, time += a.step) {
                projectHF(a, time, row);
                for (int j = 0; j < a.hidden; j++) {
                    long state = (long) row * a.hidden + j;
                    float input = sigmoidHF(a.work.getAtIndex(JAVA_FLOAT, j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + j));
                    float forget = sigmoidHF(a.work.getAtIndex(JAVA_FLOAT, a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + a.hidden + j));
                    float candidate = (float) StrictMath.tanh(a.work.getAtIndex(JAVA_FLOAT, 2 * a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + 2 * a.hidden + j));
                    float output = sigmoidHF(a.work.getAtIndex(JAVA_FLOAT, 3 * a.hidden + j) + a.work.getAtIndex(JAVA_FLOAT, a.packed + 3 * a.hidden + j));
                    float cell = forget * Float16Bits.toFloat(a.c.getAtIndex(JAVA_SHORT, state)) + input * candidate;
                    short storedCell = Float16Bits.fromFloat(cell);
                    a.c.setAtIndex(JAVA_SHORT, state, storedCell);
                    float consumedCell = Float16Bits.toFloat(storedCell);
                    float next = output * (float) StrictMath.tanh(consumedCell);
                    short stored = Float16Bits.fromFloat(next);
                    a.h.setAtIndex(JAVA_SHORT, state, stored);
                    a.y.setAtIndex(JAVA_SHORT, ((long) time * a.batch + row) * a.hidden + j, stored);
                }
            }
        }
    }
}
