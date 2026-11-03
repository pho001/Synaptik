package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
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
 * Actual fixed-recurrent native execution against independent sequential FLOAT32 working equations.
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
        for (var kind : KINDS) for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean reverse : List.of(false, true)) {
                for (boolean biased : List.of(false, true)) {
                    try (var fixture = new Fixture(kind, reverse, biased, 3, 4, -1, 2, 3, type)) {
                        fixture.assertRun(new long[] {0, 1, 2, 3});
                        fixture.assertRun(new long[] {3, 2, 1, 0});
                        fixture.assertRun(new long[] {1, 3, 0, 2});
                    }
                }
            }
        }
    }

    @Test
    void singletonContractionsRetainEachCarriersSignedZeroSeedSemantics() {
        for (var kind : KINDS) for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean reverse : List.of(false, true)) {
                for (boolean biased : List.of(false, true)) {
                    try (var fixture = new Fixture(kind, reverse, biased, 1, 1, -1, 1, 1, type)) {
                        Arrays.fill(fixture.w, 1.0f);
                        Arrays.fill(fixture.u, 1.0f);
                        for (float zero : new float[] {0.0f, -0.0f}) {
                            Arrays.fill(fixture.x, zero);
                            Arrays.fill(fixture.h0, zero);
                            Arrays.fill(fixture.c0, zero);
                            Arrays.fill(fixture.bias, zero);
                            fixture.assertRun(new long[] {1});
                            if (kind == MetalMpsGraphProgram.NodeKind.RNN_TANH) {
                                List<byte[]> actual = fixture.snapshot();
                                for (int target : fixture.targets) {
                                    ByteBuffer bytes = ByteBuffer.wrap(actual.get(target)).order(ByteOrder.nativeOrder());
                                    int word = type == DataType.FLOAT32 ? bytes.getInt()
                                            : Short.toUnsignedInt(bytes.getShort());
                                    assertEquals(type == DataType.FLOAT32 ? Float.floatToRawIntBits(zero)
                                                    : Short.toUnsignedInt(lowBits(zero, type)),
                                            word, type + " RNN singleton zero at public target " + target);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void lowSingletonNegativeZeroRemainsObservableThroughReciprocal() {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            try (var fixture = new Fixture(MetalMpsGraphProgram.NodeKind.RNN_TANH,
                    false, false, 1, 1, -1, 1, 1, type, true)) {
                Arrays.fill(fixture.x, -0.0f);
                Arrays.fill(fixture.h0, -0.0f);
                Arrays.fill(fixture.w, 1.0f);
                Arrays.fill(fixture.u, 1.0f);
                fixture.prepare(new long[] {1});
                fixture.invoke();
                List<byte[]> actual = fixture.snapshot();
                for (int output = 0; output < 2; output++) {
                    int bits = Short.toUnsignedInt(ByteBuffer.wrap(actual.get(fixture.readCount + output))
                            .order(ByteOrder.nativeOrder()).getShort());
                    assertEquals(0x8000, bits, type + " fixed recurrent negative zero output " + output);
                }
                int reciprocal = Short.toUnsignedInt(ByteBuffer.wrap(actual.get(fixture.values.size() - 1))
                        .order(ByteOrder.nativeOrder()).getShort());
                assertEquals(Short.toUnsignedInt(lowBits(Float.NEGATIVE_INFINITY, type)), reciprocal,
                        type + " reciprocal must observe the recurrent negative zero sign");
            }
        }
    }

    @Test
    void inactiveRowsPreserveRawInitialStatesAndNeverReadPaddedNanInputs() {
        for (var kind : KINDS) for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean reverse : List.of(false, true)) {
                try (var fixture = new Fixture(kind, reverse, true, 3, 2, -1, 2, 3, type)) {
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
        for (var kind : KINDS) for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean reverse : List.of(false, true)) {
                try (var fixture = new Fixture(kind, reverse, true, 3, 2, -1, 2, 3, type)) {
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

    @Test
    void lowCarriersPreserveRawInactiveStatePayloadsAndEmptyDomains() {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) for (var kind : KINDS) {
            for (boolean reverse : List.of(false, true)) for (boolean biased : List.of(false, true)) {
                for (int[] shape : List.of(new int[] {0, 2}, new int[] {3, 2}, new int[] {3, 0},
                        new int[] {0, 0})) {
                    try (var fixture = new Fixture(kind, reverse, biased, shape[0], shape[1], -1,
                            2, 3, type)) {
                        fixture.prepare(new long[shape[1]]);
                        ByteBuffer raw = ByteBuffer.allocate(shape[1] * 3 * Short.BYTES)
                                .order(ByteOrder.nativeOrder());
                        short[] payloads = {(short) 0x8000, (short) 0x7fa5, (short) 0xff91,
                                (short) 0x0001, (short) 0xfc00, (short) 0x7e53};
                        for (int i = 0; i < shape[1] * 3; i++) raw.putShort(payloads[i % payloads.length]);
                        fixture.upload(fixture.buffers.get(2), raw.array());
                        if (fixture.gates == 4) fixture.upload(fixture.buffers.get(3), raw.array());
                        fixture.invoke();
                        var actual = fixture.snapshot();
                        assertArrayEquals(new byte[shape[0] * shape[1] * 3 * Short.BYTES],
                                actual.get(fixture.readCount));
                        assertArrayEquals(raw.array(), actual.get(fixture.readCount + 1));
                        if (fixture.gates == 4) {
                            assertArrayEquals(raw.array(), actual.get(fixture.readCount + 2));
                        }
                    }
                }
            }
        }
    }

    @Test
    void lowLengthPreflightRejectsBeforeWritesAndAllowsReuse() {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) for (var kind : KINDS) {
            try (var fixture = new Fixture(kind, true, true, 3, 2, -1, 2, 3, type)) {
                for (long invalid : new long[] {-1, 4, Long.MIN_VALUE, Long.MAX_VALUE}) {
                    fixture.prepare(new long[] {3, invalid});
                    var before = fixture.snapshot();
                    assertThrows(RuntimeException.class, fixture::invoke);
                    var after = fixture.snapshot();
                    for (int value = 0; value < before.size(); value++) {
                        assertArrayEquals(before.get(value), after.get(value));
                    }
                }
                fixture.assertRun(new long[] {3, 1});
            }
        }
    }

    /** Owns one reusable native executable and independent buffers for every canonical role. */
    private static final class Fixture implements AutoCloseable {
        final MetalMpsGraphProgram.NodeKind kind;
        final DataType type;
        final boolean reverse, biased;
        final int time, batch, input, hidden, gates, readCount;
        final float[] x, h0, c0, w, u, bias;
        final List<MetalMpsGraphProgram.ValueDescriptor> values = new ArrayList<>();
        final List<MetalNativeApi.Handle> buffers = new ArrayList<>();
        final int[] targets;
        final MetalNativeApi api;
        MetalNativeApi.Handle context, executable;

        Fixture(MetalMpsGraphProgram.NodeKind kind, boolean reverse, boolean biased,
                int time, int batch, int selectedOutput) {
            this(kind, reverse, biased, time, batch, selectedOutput, 2, 3);
        }

        Fixture(MetalMpsGraphProgram.NodeKind kind, boolean reverse, boolean biased,
                int time, int batch, int selectedOutput, int input, int hidden) {
            this(kind, reverse, biased, time, batch, selectedOutput, input, hidden, DataType.FLOAT32);
        }

        Fixture(MetalMpsGraphProgram.NodeKind kind, boolean reverse, boolean biased,
                int time, int batch, int selectedOutput, int input, int hidden, DataType type) {
            this(kind, reverse, biased, time, batch, selectedOutput, input, hidden, type, false);
        }

        Fixture(MetalMpsGraphProgram.NodeKind kind, boolean reverse, boolean biased,
                int time, int batch, int selectedOutput, int input, int hidden, DataType type,
                boolean reciprocal) {
            String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
            assumeTrue(configured != null && !configured.isBlank(),
                    "SYNAPTIK_METAL_TEST_LIBRARY is not set");
            this.kind = kind;
            this.type = type;
            this.reverse = reverse;
            this.biased = biased;
            this.time = time;
            this.batch = batch;
            this.input = input;
            this.hidden = hidden;
            gates = kind == MetalMpsGraphProgram.NodeKind.RNN_TANH ? 1
                    : kind == MetalMpsGraphProgram.NodeKind.GRU_RESET_AFTER ? 3 : 4;
            x = pattern(time * batch * input, 0.13f);
            h0 = pattern(batch * hidden, 0.09f);
            c0 = pattern(batch * hidden, -0.07f);
            w = pattern(gates * hidden * input, 0.11f);
            u = pattern(gates * hidden * hidden, -0.08f);
            bias = pattern(gates * hidden, 0.05f);
            values.add(typed(type, time, batch, input));
            values.add(typed(DataType.INT64, batch));
            values.add(typed(type, batch, hidden));
            if (gates == 4) values.add(typed(type, batch, hidden));
            values.add(typed(type, gates * hidden, input));
            values.add(typed(type, gates * hidden, hidden));
            if (biased) values.add(typed(type, gates * hidden));
            readCount = values.size();
            values.add(typed(type, time, batch, hidden));
            values.add(typed(type, batch, hidden));
            if (gates == 4) values.add(typed(type, batch, hidden));
            int[] feeds = new int[readCount];
            for (int i = 0; i < feeds.length; i++) feeds[i] = i;
            int[] outputs = new int[gates == 4 ? 3 : 2];
            for (int i = 0; i < outputs.length; i++) outputs[i] = readCount + i;
            var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
            if (reciprocal) {
                int target = values.size();
                values.add(typed(type, batch, hidden));
                targets = new int[] {outputs[0], outputs[1], target};
                nodes.add(MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.RECIPROCAL,
                        new int[] {outputs[1]}, new int[] {target},
                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0]));
            } else targets = selectedOutput < 0 ? outputs : new int[] {outputs[selectedOutput]};
            var node = MetalMpsGraphProgram.Node.generic(kind, feeds, outputs,
                    MetalMpsGraphProgram.AttributeKind.RECURRENT_DIRECTION,
                    new long[] {reverse ? 2L : 1L});
            nodes.addFirst(node);
            api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
            try {
                context = api.createContext();
                executable = api.createProgramExecutable(context, List.copyOf(values),
                        new MetalMpsGraphProgram(List.copyOf(nodes)), feeds, targets,
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
            reads.add(words(padded, type));
            reads.add(longs(lengths));
            reads.add(words(h0, type));
            if (gates == 4) reads.add(words(c0, type));
            reads.add(words(w, type));
            reads.add(words(u, type));
            if (biased) reads.add(words(bias, type));
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
        float[] h = stored(f.h0, f.type), c = stored(f.c0, f.type);
        float[] x = stored(f.x, f.type), w = stored(f.w, f.type), u = stored(f.u, f.type);
        float[] bias = stored(f.bias, f.type);
        for (int row = 0; row < f.batch; row++) {
            for (int visit = 0; visit < lengths[row]; visit++) {
                int time = f.reverse ? (int) lengths[row] - 1 - visit : visit;
                float[] previous = Arrays.copyOfRange(h, row * f.hidden, (row + 1) * f.hidden);
                float[] a = new float[f.gates * f.hidden];
                float[] b = new float[a.length];
                for (int gate = 0; gate < a.length; gate++) {
                    int inputBase = (time * f.batch + row) * f.input;
                    a[gate] = x[inputBase] * w[gate * f.input];
                    for (int k = 1; k < f.input; k++) {
                        a[gate] += x[inputBase + k] * w[gate * f.input + k];
                    }
                    b[gate] = previous[0] * u[gate * f.hidden];
                    for (int k = 1; k < f.hidden; k++) {
                        b[gate] += previous[k] * u[gate * f.hidden + k];
                    }
                    if (f.biased) a[gate] += bias[gate];
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
                            c[state] = stored(gate1 * c[state] + gate0 * candidate, f.type);
                            next = output * tanh(c[state]);
                        }
                    }
                    h[state] = stored(next, f.type);
                    y[(time * f.batch + row) * f.hidden + j] = h[state];
                }
            }
        }
        return f.gates == 4 ? new float[][] {y, h, c} : new float[][] {y, h};
    }

    private static void assertResult(float[] expected, byte[] bytes, Fixture f,
            long[] lengths, int output) {
        ByteBuffer actual = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        assertEquals(expected.length * f.type.byteWidth(), bytes.length);
        for (int i = 0; i < expected.length; i++) {
            int word = f.type == DataType.FLOAT32 ? actual.getInt() : Short.toUnsignedInt(actual.getShort());
            boolean exact = output == 0
                    ? i / (f.batch * f.hidden) >= lengths[(i / f.hidden) % f.batch]
                    : lengths[i / f.hidden] == 0;
            String site = f.type + " " + f.kind + " reverse=" + f.reverse + " bias=" + f.biased
                    + " output=" + output + " element=" + i;
            if (f.type != DataType.FLOAT32) {
                assertEquals(Short.toUnsignedInt(lowBits(expected[i], f.type)), word, site);
            } else if (exact || expected[i] == 0.0f) {
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

    private static short lowBits(float value, DataType type) {
        return type == DataType.BFLOAT16 ? BFloat16Bits.fromFloat(value) : Float16Bits.fromFloat(value);
    }

    private static float stored(float value, DataType type) {
        if (type == DataType.FLOAT32) return value;
        short bits = lowBits(value, type);
        return type == DataType.BFLOAT16 ? BFloat16Bits.toFloat(bits) : Float16Bits.toFloat(bits);
    }

    private static float[] stored(float[] values, DataType type) {
        float[] result = values.clone();
        for (int i = 0; i < result.length; i++) result[i] = stored(result[i], type);
        return result;
    }

    private static byte[] words(float[] values, DataType type) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * type.byteWidth()).order(ByteOrder.nativeOrder());
        for (float value : values) {
            if (type == DataType.FLOAT32) bytes.putInt(Float.floatToRawIntBits(value));
            else bytes.putShort(lowBits(value, type));
        }
        return bytes.array();
    }

    private static byte[] longs(long[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (long value : values) bytes.putLong(value);
        return bytes.array();
    }
}
