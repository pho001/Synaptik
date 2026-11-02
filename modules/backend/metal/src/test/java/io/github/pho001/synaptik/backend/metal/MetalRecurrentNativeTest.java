package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Actual fixed-recurrent native execution against independent sequential FLOAT32 equations.
 * Prior-hidden contractions use a snapshot, so a feature-wise in-place hidden update cannot
 * accidentally satisfy the oracle. No CPU backend evaluator or recurrent kernel is called.
 */
class MetalRecurrentNativeTest {
    private static final List<MetalMpsGraphProgram.NodeKind> KINDS = List.of(
            MetalMpsGraphProgram.NodeKind.RNN_TANH,
            MetalMpsGraphProgram.NodeKind.GRU_RESET_AFTER,
            MetalMpsGraphProgram.NodeKind.LSTM);
    private static final byte POISON = (byte) 0xa5;

    @Test
    void everyFamilyBiasAndDirectionMatchesAllLengthsAndReusesItsExecutable() {
        for (var kind : KINDS) {
            for (boolean reverse : List.of(false, true)) {
                for (boolean biased : List.of(false, true)) {
                    try (var fixture = new Fixture(kind, reverse, biased, 3, 4, -1)) {
                        fixture.assertRun(new long[] {0, 1, 2, 3});
                        fixture.assertRun(new long[] {3, 2, 1, 0});
                        fixture.assertRun(new long[] {1, 3, 0, 2});
                    }
                }
            }
        }
    }

    @Test
    void inactiveRowsPreserveRawInitialStatesAndNeverReadPaddedNanInputs() {
        for (var kind : KINDS) {
            for (boolean reverse : List.of(false, true)) {
                try (var fixture = new Fixture(kind, reverse, true, 3, 2, -1)) {
                    fixture.h0[0] = -0.0f;
                    fixture.h0[1] = Float.intBitsToFloat(0x7fc1_2345);
                    fixture.h0[2] = Float.NEGATIVE_INFINITY;
                    fixture.c0[0] = Float.intBitsToFloat(0x7fc5_4321);
                    fixture.c0[1] = -0.0f;
                    fixture.c0[2] = Float.POSITIVE_INFINITY;
                    fixture.assertRun(new long[] {0, 1});
                }
            }
        }
    }

    @Test
    void saturatedAndNonfiniteActiveGatesUseTheNamedSigmoidAndTanhSemantics() {
        for (var kind : KINDS) {
            for (boolean reverse : List.of(false, true)) {
                try (var fixture = new Fixture(kind, reverse, true, 3, 2, -1)) {
                    Arrays.fill(fixture.w, 0.0f);
                    Arrays.fill(fixture.u, 0.0f);
                    for (int gate = 0; gate < fixture.gates; gate++) {
                        Arrays.fill(fixture.bias, gate * fixture.hidden, (gate + 1) * fixture.hidden,
                                (gate & 1) == 0 ? 1000.0f : -1000.0f);
                    }
                    fixture.assertRun(new long[] {3, 2});
                    fixture.bias[0] = Float.POSITIVE_INFINITY;
                    fixture.bias[1] = Float.NEGATIVE_INFINITY;
                    fixture.bias[2] = Float.NaN;
                    fixture.assertRun(new long[] {2, 3});
                }
            }
        }
    }

    @Test
    void emptyTimeAndBatchPublishOnlyRequiredInitialStateIdentities() {
        for (var kind : KINDS) {
            for (boolean reverse : List.of(false, true)) {
                for (boolean biased : List.of(false, true)) {
                    for (int[] shape : List.of(new int[] {0, 3}, new int[] {3, 0}, new int[] {0, 0})) {
                        try (var fixture = new Fixture(kind, reverse, biased, shape[0], shape[1], -1)) {
                            if (fixture.h0.length != 0) {
                                fixture.h0[0] = -0.0f;
                                fixture.c0[0] = Float.intBitsToFloat(0x7fc1_2345);
                            }
                            fixture.assertRun(new long[shape[1]]);
                        }
                    }
                }
            }
        }
    }

    @Test
    void finalStateOnlyAndSequenceOnlyTargetsStillComputeCanonicalSiblings() {
        for (var kind : KINDS) {
            int outputs = kind == MetalMpsGraphProgram.NodeKind.LSTM ? 3 : 2;
            for (int target = 0; target < outputs; target++) {
                try (var fixture = new Fixture(kind, true, true, 3, 2, target)) {
                    fixture.assertRun(new long[] {3, 1});
                }
            }
        }
    }

    @Test
    void everyLengthIsValidatedBeforeAnyOutputWriteAndFailureDoesNotPoisonReuse() {
        for (var kind : KINDS) {
            for (boolean reverse : List.of(false, true)) {
                for (boolean biased : List.of(false, true)) {
                    try (var fixture = new Fixture(kind, reverse, biased, 3, 3, -1)) {
                        for (long invalid : new long[] {-1, 4, Long.MIN_VALUE, Long.MAX_VALUE}) {
                            fixture.prepare(new long[] {3, 2, invalid});
                            List<byte[]> before = fixture.snapshot();
                            assertThrows(RuntimeException.class, fixture::invoke);
                            List<byte[]> after = fixture.snapshot();
                            for (int value = 0; value < before.size(); value++) {
                                assertArrayEquals(before.get(value), after.get(value),
                                        kind + " invalid length mutated value " + value);
                            }
                        }
                        fixture.assertRun(new long[] {3, 2, 1});
                    }
                }
            }
        }
        try (var fixture = new Fixture(MetalMpsGraphProgram.NodeKind.LSTM, false, false, 0, 2, -1)) {
            fixture.prepare(new long[] {0, 1});
            List<byte[]> before = fixture.snapshot();
            assertThrows(RuntimeException.class, fixture::invoke);
            List<byte[]> after = fixture.snapshot();
            for (int value = 0; value < before.size(); value++) {
                assertArrayEquals(before.get(value), after.get(value), "T=0 length validation");
            }
        }
    }

    /** Owns one reusable native executable and independent buffers for every canonical role. */
    private static final class Fixture implements AutoCloseable {
        final MetalMpsGraphProgram.NodeKind kind;
        final boolean reverse, biased;
        final int time, batch, input = 2, hidden = 3, gates, readCount;
        final float[] x, h0, c0, w, u, bias;
        final List<MetalMpsGraphProgram.ValueDescriptor> values = new ArrayList<>();
        final List<MetalNativeApi.Handle> buffers = new ArrayList<>();
        final int[] targets;
        final MetalNativeApi api;
        MetalNativeApi.Handle context, executable;

        Fixture(MetalMpsGraphProgram.NodeKind kind, boolean reverse, boolean biased,
                int time, int batch, int selectedOutput) {
            String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
            assumeTrue(configured != null && !configured.isBlank(),
                    "SYNAPTIK_METAL_TEST_LIBRARY is not set");
            this.kind = kind;
            this.reverse = reverse;
            this.biased = biased;
            this.time = time;
            this.batch = batch;
            gates = kind == MetalMpsGraphProgram.NodeKind.RNN_TANH ? 1
                    : kind == MetalMpsGraphProgram.NodeKind.GRU_RESET_AFTER ? 3 : 4;
            x = pattern(time * batch * input, 0.13f);
            h0 = pattern(batch * hidden, 0.09f);
            c0 = pattern(batch * hidden, -0.07f);
            w = pattern(gates * hidden * input, 0.11f);
            u = pattern(gates * hidden * hidden, -0.08f);
            bias = pattern(gates * hidden, 0.05f);
            values.add(typed(DataType.FLOAT32, time, batch, input));
            values.add(typed(DataType.INT64, batch));
            values.add(typed(DataType.FLOAT32, batch, hidden));
            if (gates == 4) values.add(typed(DataType.FLOAT32, batch, hidden));
            values.add(typed(DataType.FLOAT32, gates * hidden, input));
            values.add(typed(DataType.FLOAT32, gates * hidden, hidden));
            if (biased) values.add(typed(DataType.FLOAT32, gates * hidden));
            readCount = values.size();
            values.add(typed(DataType.FLOAT32, time, batch, hidden));
            values.add(typed(DataType.FLOAT32, batch, hidden));
            if (gates == 4) values.add(typed(DataType.FLOAT32, batch, hidden));
            int[] feeds = new int[readCount];
            for (int i = 0; i < feeds.length; i++) feeds[i] = i;
            int[] outputs = new int[gates == 4 ? 3 : 2];
            for (int i = 0; i < outputs.length; i++) outputs[i] = readCount + i;
            targets = selectedOutput < 0 ? outputs : new int[] {outputs[selectedOutput]};
            var node = MetalMpsGraphProgram.Node.generic(kind, feeds, outputs,
                    MetalMpsGraphProgram.AttributeKind.RECURRENT_DIRECTION,
                    new long[] {reverse ? 2L : 1L});
            api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
            try {
                context = api.createContext();
                executable = api.createProgramExecutable(context, List.copyOf(values),
                        new MetalMpsGraphProgram(List.of(node)), feeds, targets,
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            } catch (RuntimeException | Error failure) {
                close();
                throw failure;
            }
        }

        void prepare(long[] lengths) {
            float[] padded = x.clone();
            for (int t = 0; t < time; t++) {
                for (int b = 0; b < batch; b++) {
                    if (t >= lengths[b]) Arrays.fill(padded,
                            (t * batch + b) * input, (t * batch + b + 1) * input, Float.NaN);
                }
            }
            var reads = new ArrayList<byte[]>();
            reads.add(words(padded));
            reads.add(longs(lengths));
            reads.add(words(h0));
            if (gates == 4) reads.add(words(c0));
            reads.add(words(w));
            reads.add(words(u));
            if (biased) reads.add(words(bias));
            for (int i = 0; i < reads.size(); i++) upload(buffers.get(i), reads.get(i));
            for (int i = readCount; i < buffers.size(); i++) {
                byte[] poison = new byte[Math.toIntExact(values.get(i).byteCount())];
                Arrays.fill(poison, POISON);
                upload(buffers.get(i), poison);
            }
        }

        void assertRun(long[] lengths) {
            prepare(lengths);
            List<byte[]> before = snapshot();
            invoke();
            List<byte[]> after = snapshot();
            for (int i = 0; i < readCount; i++) {
                assertArrayEquals(before.get(i), after.get(i), "input mutation " + kind + " role " + i);
            }
            float[][] expected = oracle(this, lengths);
            for (int target : targets) {
                int output = target - readCount;
                assertResult(expected[output], after.get(target), this, lengths, output);
            }
        }

        void invoke() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int i = 0; i < buffers.size(); i++) {
                    addresses.setAtIndex(ADDRESS, i, buffers.get(i).carrier());
                }
                MemorySegment outputAddresses = arena.allocate(ADDRESS, targets.length);
                for (int i = 0; i < targets.length; i++) {
                    outputAddresses.setAtIndex(ADDRESS, i, buffers.get(targets[i]).carrier());
                }
                api.runExecutable(executable, buffers.size(), addresses, targets.length, outputAddresses);
            }
        }

        List<byte[]> snapshot() {
            var result = new ArrayList<byte[]>();
            for (int i = 0; i < values.size(); i++) {
                int size = Math.toIntExact(values.get(i).byteCount());
                if (size == 0) {
                    result.add(new byte[0]);
                } else try (Arena arena = Arena.ofConfined()) {
                    MemorySegment bytes = arena.allocate(size, 1);
                    api.download(buffers.get(i), 0L, bytes, size);
                    result.add(bytes.toArray(JAVA_BYTE));
                }
            }
            return result;
        }

        private void upload(MetalNativeApi.Handle buffer, byte[] bytes) {
            if (bytes.length == 0) return;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment source = arena.allocate(bytes.length, 1);
                source.copyFrom(MemorySegment.ofArray(bytes));
                api.upload(buffer, 0L, source, bytes.length);
            }
        }

        @Override
        public void close() {
            for (int i = buffers.size(); i-- > 0;) api.releaseBuffer(buffers.get(i));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    /** Separate projected vectors and hidden snapshots preserve the clean sequential equations. */
    private static float[][] oracle(Fixture f, long[] lengths) {
        float[] y = new float[f.time * f.batch * f.hidden];
        float[] h = f.h0.clone(), c = f.c0.clone();
        for (int row = 0; row < f.batch; row++) {
            for (int visit = 0; visit < lengths[row]; visit++) {
                int time = f.reverse ? (int) lengths[row] - 1 - visit : visit;
                float[] previous = Arrays.copyOfRange(h, row * f.hidden, (row + 1) * f.hidden);
                float[] a = new float[f.gates * f.hidden];
                float[] b = new float[a.length];
                for (int gate = 0; gate < a.length; gate++) {
                    for (int k = 0; k < f.input; k++) {
                        a[gate] += f.x[(time * f.batch + row) * f.input + k]
                                * f.w[gate * f.input + k];
                    }
                    for (int k = 0; k < f.hidden; k++) {
                        b[gate] += previous[k] * f.u[gate * f.hidden + k];
                    }
                    if (f.biased) a[gate] += f.bias[gate];
                }
                for (int j = 0; j < f.hidden; j++) {
                    int state = row * f.hidden + j;
                    float next;
                    if (f.gates == 1) {
                        next = tanh(a[j] + b[j]);
                    } else {
                        float gate0 = sigmoid(a[j] + b[j]);
                        float gate1 = sigmoid(a[f.hidden + j] + b[f.hidden + j]);
                        int candidateIndex = 2 * f.hidden + j;
                        if (f.gates == 3) {
                            float candidate = tanh(a[candidateIndex] + gate0 * b[candidateIndex]);
                            next = candidate + gate1 * (previous[j] - candidate);
                        } else {
                            float candidate = tanh(a[candidateIndex] + b[candidateIndex]);
                            float output = sigmoid(a[3 * f.hidden + j] + b[3 * f.hidden + j]);
                            c[state] = gate1 * c[state] + gate0 * candidate;
                            next = output * tanh(c[state]);
                        }
                    }
                    h[state] = next;
                    y[(time * f.batch + row) * f.hidden + j] = next;
                }
            }
        }
        return f.gates == 4 ? new float[][] {y, h, c} : new float[][] {y, h};
    }

    private static void assertResult(float[] expected, byte[] bytes, Fixture f,
            long[] lengths, int output) {
        ByteBuffer actual = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        assertEquals(expected.length * Float.BYTES, bytes.length);
        for (int i = 0; i < expected.length; i++) {
            int word = actual.getInt();
            boolean exact = output == 0
                    ? i / (f.batch * f.hidden) >= lengths[(i / f.hidden) % f.batch]
                    : lengths[i / f.hidden] == 0;
            String site = f.kind + " reverse=" + f.reverse + " bias=" + f.biased
                    + " output=" + output + " element=" + i;
            if (exact) {
                assertEquals(Float.floatToRawIntBits(expected[i]), word, site);
            } else {
                assertEquals(expected[i], Float.intBitsToFloat(word), 8.0e-6f, site);
            }
        }
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static float[] pattern(int size, float scale) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) values[i] = ((i * 7 + 3) % 13 - 6) * scale;
        return values;
    }

    private static float sigmoid(float value) {
        float e = (float) StrictMath.exp(value >= 0 ? -value : value);
        float denominator = 1.0f + e;
        return value >= 0 ? 1.0f / denominator : e / denominator;
    }

    private static float tanh(float value) {
        return (float) StrictMath.tanh(value);
    }

    private static byte[] words(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putInt(Float.floatToRawIntBits(value));
        return bytes.array();
    }

    private static byte[] longs(long[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (long value : values) bytes.putLong(value);
        return bytes.array();
    }
}
