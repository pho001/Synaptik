package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class MetalRandomDropoutNativeTest {
    private static final long KEY_BIAS = 0x9e37_79b9_7f4a_7c15L;
    private static final long M1 = 0xbf58_476d_1ce4_e5b9L;
    private static final long M2 = 0x94d0_49bb_1331_11ebL;
    private static final int[] EDGE_INPUT = {
        0x8000_0000,
        Float.floatToRawIntBits(1.0f),
        Float.floatToRawIntBits(-2.0f),
        0x7fc1_2345,
        0x7f80_0000,
        Float.floatToRawIntBits(0.25f),
        0x8000_0000,
        Float.floatToRawIntBits(3.0f)
    };

    @Test
    void initialStateIsBitExactInBothProfilesWithZeroFeedsAndRepeatedRuns() {
        long key = 0xfedc_ba98_7654_3210L;
        long counter = 0x0123_4567_89ab_cdefL;
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(initial(0, key, counter)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(typed(DataType.INT64, 2));
        for (NumericalProfile profile : NumericalProfile.values()) {
            List<List<byte[]>> runs = execute(
                    profile, program, values, new int[0], new int[] {0}, List.of(), 2);
            assertArrayEquals(longWords(key, counter), runs.get(0).getFirst(), profile.toString());
            assertArrayEquals(runs.get(0).getFirst(), runs.get(1).getFirst(),
                    profile + " repeated execution");
        }
    }

    @Test
    void dropoutMatchesSplitMixOracleAtEveryProbabilityBoundaryAndWrapsState() {
        double[] probabilities = {
            -0.0d,
            Double.MIN_VALUE,
            0x1.0p-54,
            0x1.0p-53,
            0x1.0000000000001p-53,
            0.5d,
            Math.nextDown(1.0d)
        };
        long key = 0L;
        long counter = -3L;
        for (double probability : probabilities) {
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                    initial(1, key, counter), dropout(0, 1, 2, 3, 4, probability)));
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    typed(DataType.FLOAT32, EDGE_INPUT.length),
                    typed(DataType.INT64, 2),
                    typed(DataType.FLOAT32, EDGE_INPUT.length),
                    typed(DataType.BOOL, EDGE_INPUT.length),
                    typed(DataType.INT64, 2));
            List<List<byte[]>> runs = execute(
                    NumericalProfile.ACCELERATOR,
                    program,
                    values,
                    new int[] {0},
                    new int[] {2, 3, 4},
                    List.of(intWords(EDGE_INPUT)),
                    2);
            Expected expected = oracle(EDGE_INPUT, key, counter, probability);
            for (List<byte[]> actual : runs) {
                assertOutput(
                        expected, actual.get(0), EDGE_INPUT, Double.toHexString(probability));
                assertArrayEquals(expected.mask(), actual.get(1), Double.toHexString(probability));
                assertArrayEquals(expected.nextState(), actual.get(2),
                        Double.toHexString(probability));
            }
        }
    }

    @Test
    void nestedFanoutBranchChainingAndTargetSubsetsRemainOrdered() {
        int[] input = {
            Float.floatToRawIntBits(1.0f), Float.floatToRawIntBits(-2.0f),
            Float.floatToRawIntBits(3.0f), Float.floatToRawIntBits(-4.0f),
            Float.floatToRawIntBits(5.0f), Float.floatToRawIntBits(-6.0f),
            Float.floatToRawIntBits(7.0f), Float.floatToRawIntBits(-8.0f)
        };
        long key = 0x0123_4567_89ab_cdefL;
        long counter = 0xfedc_ba98_7654_3210L;
        double probability = 0.25d;
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.neg(0, 1),
                initial(2, key, counter),
                dropout(1, 2, 3, 4, 5, probability),
                dropout(1, 2, 6, 7, 8, probability),
                dropout(3, 5, 9, 10, 11, probability)));
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        values.add(typed(DataType.FLOAT32, input.length));
        values.add(typed(DataType.FLOAT32, input.length));
        values.add(typed(DataType.INT64, 2));
        values.add(typed(DataType.FLOAT32, input.length));
        values.add(typed(DataType.BOOL, input.length));
        values.add(typed(DataType.INT64, 2));
        values.add(typed(DataType.FLOAT32, input.length));
        values.add(typed(DataType.BOOL, input.length));
        values.add(typed(DataType.INT64, 2));
        values.add(typed(DataType.FLOAT32, input.length));
        values.add(typed(DataType.BOOL, input.length));
        values.add(typed(DataType.INT64, 2));

        List<byte[]> result = execute(
                NumericalProfile.ACCELERATOR,
                program,
                List.copyOf(values),
                new int[] {0},
                new int[] {3, 7, 9, 11},
                List.of(intWords(input)),
                1).getFirst();
        int[] negated = new int[input.length];
        for (int index = 0; index < input.length; index++) {
            negated[index] = Float.floatToRawIntBits(-Float.intBitsToFloat(input[index]));
        }
        Expected first = oracle(negated, key, counter, probability);
        int[] firstOutput = ints(first.output());
        Expected chained = oracle(firstOutput, key, counter + input.length, probability);
        assertArrayEquals(first.output(), result.get(0), "nested primary output");
        assertArrayEquals(first.mask(), result.get(1), "same-state fanout mask");
        assertArrayEquals(chained.output(), result.get(2), "chained primary output");
        assertArrayEquals(chained.nextState(), result.get(3), "chained next state");
    }

    @Test
    void concurrentContextsAndExecutablesHaveNoSharedRandomState() throws Exception {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        double probability = 0.5d;
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                initial(1, 0L, 0L), dropout(0, 1, 2, 3, 4, probability)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, EDGE_INPUT.length), typed(DataType.INT64, 2),
                typed(DataType.FLOAT32, EDGE_INPUT.length), typed(DataType.BOOL, EDGE_INPUT.length),
                typed(DataType.INT64, 2));
        Callable<List<byte[]>> session = () -> execute(
                NumericalProfile.ACCELERATOR,
                program,
                values,
                new int[] {0},
                new int[] {2, 3, 4},
                List.of(intWords(EDGE_INPUT)),
                1).getFirst();
        try (var executor = Executors.newFixedThreadPool(4)) {
            var futures = executor.invokeAll(java.util.Collections.nCopies(8, session));
            List<byte[]> first = futures.getFirst().get();
            for (var future : futures) {
                List<byte[]> actual = future.get();
                for (int output = 0; output < first.size(); output++) {
                    assertArrayEquals(first.get(output), actual.get(output));
                }
            }
        }
    }

    private static MetalMpsGraphProgram.Node initial(
            int output, long key, long counter) {
        return MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.INITIAL_STATE,
                new int[0],
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.GRAPH_RNG_STATE,
                new long[] {key, counter});
    }

    private static MetalMpsGraphProgram.Node dropout(
            int input,
            int state,
            int output,
            int mask,
            int nextState,
            double probability) {
        return MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.DROPOUT,
                new int[] {input, state},
                new int[] {output, mask, nextState},
                MetalMpsGraphProgram.AttributeKind.DROPOUT,
                new long[] {Double.doubleToRawLongBits(probability)});
    }

    private static Expected oracle(
            int[] input, long key, long counter, double probability) {
        long threshold = MetalMpsGraphProgram.dropoutThreshold(
                Double.doubleToRawLongBits(probability));
        float complement = Float.intBitsToFloat(MetalMpsGraphProgram.dropoutComplementBits(
                Double.doubleToRawLongBits(probability)));
        float scale = 1.0f / complement;
        ByteBuffer output = ByteBuffer.allocate(input.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        byte[] mask = new byte[input.length];
        long keyOffset = mix64(key + KEY_BIAS);
        for (int index = 0; index < input.length; index++) {
            long word = mix64(counter + index + keyOffset);
            boolean keep = (word >>> 11) >= threshold;
            mask[index] = keep ? (byte) 1 : 0;
            int result = keep
                    ? Float.floatToRawIntBits(Float.intBitsToFloat(input[index]) * scale)
                    : 0;
            output.putInt(result);
        }
        return new Expected(output.array(), mask, longWords(key, counter + input.length));
    }

    private static long mix64(long word) {
        word = (word ^ (word >>> 30)) * M1;
        word = (word ^ (word >>> 27)) * M2;
        return word ^ (word >>> 31);
    }

    private static List<List<byte[]>> execute(
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> inputs,
            int runCount) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        assertEquals(feeds.length, inputs.size());
        MetalNativeApi api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createMpsGraphExecutable(
                    context, profile, values, program, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            for (int index = 0; index < feeds.length; index++) {
                upload(api, buffers.get(feeds[index]), inputs.get(index));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    addresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++) {
                    outputs.setAtIndex(ADDRESS, index, buffers.get(targets[index]).carrier());
                }
                var runs = new ArrayList<List<byte[]>>(runCount);
                for (int run = 0; run < runCount; run++) {
                    api.runExecutable(
                            executable, buffers.size(), addresses, targets.length, outputs);
                    var result = new ArrayList<byte[]>(targets.length);
                    for (int target : targets) {
                        result.add(download(api, buffers.get(target), values.get(target).byteCount()));
                    }
                    runs.add(List.copyOf(result));
                }
                return List.copyOf(runs);
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1);
            source.copyFrom(MemorySegment.ofArray(bytes));
            api.upload(buffer, 0L, source, bytes.length);
        }
    }

    private static byte[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long byteCount) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(byteCount, 1);
            api.download(buffer, 0L, target, byteCount);
            return target.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        }
    }

    private static byte[] intWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putInt(word);
        return bytes.array();
    }

    private static int[] ints(byte[] bytes) {
        int[] result = new int[bytes.length / Integer.BYTES];
        ByteBuffer source = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        for (int index = 0; index < result.length; index++) result[index] = source.getInt();
        return result;
    }

    private static void assertOutput(
            Expected expected, byte[] actualBytes, int[] input, String message) {
        int[] expectedWords = ints(expected.output());
        int[] actualWords = ints(actualBytes);
        for (int index = 0; index < input.length; index++) {
            if (expected.mask()[index] != 0
                    && Float.isNaN(Float.intBitsToFloat(input[index]))) {
                assertTrue(
                        Float.isNaN(Float.intBitsToFloat(actualWords[index])),
                        message + " kept NaN at " + index);
            } else {
                assertEquals(expectedWords[index], actualWords[index], message + " at " + index);
            }
        }
    }

    private static byte[] longWords(long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        for (long word : words) bytes.putLong(word);
        return bytes.array();
    }

    private record Expected(byte[] output, byte[] mask, byte[] nextState) {}
}
