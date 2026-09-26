package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphMatmulNativeTest {
    @Test
    void realRankTwoMatmulRunsDirectAndEveryAuthenticatedTransposeFormAcrossReuse() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        try {
            context = api.createContext();
            runEveryForm(api, context);
            runK1NumericalSentinels(api, context);
        } finally {
            if (context != null) {
                api.releaseContext(context);
            }
            api.close();
        }
    }

    private static void runEveryForm(
            MetalNativeApi api, MetalNativeApi.Handle context) {
        int[][] inputBits = {
            bits(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f),
            bits(
                    7.0f, 8.0f, 9.0f, 10.0f,
                    11.0f, 12.0f, 13.0f, 14.0f,
                    15.0f, 16.0f, 17.0f, 18.0f),
            bits(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f),
            bits(
                    7.0f, 11.0f, 15.0f,
                    8.0f, 12.0f, 16.0f,
                    9.0f, 13.0f, 17.0f,
                    10.0f, 14.0f, 18.0f)
        };
        int[] expected = bits(
                74.0f, 80.0f, 86.0f, 92.0f,
                173.0f, 188.0f, 203.0f, 218.0f);
        int[] ranks = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
        long[] dimensions = dimensions(new long[][] {
            {2, 3}, {3, 4}, {3, 2}, {4, 3},
            {2, 4}, {2, 4}, {2, 4}, {2, 4},
            {2, 3}, {3, 4}
        });
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(2, 8, List.of(1, 0)),
                MetalMpsGraphProgram.Node.permutation(3, 9, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(0, 1, 4),
                MetalMpsGraphProgram.Node.matmul(8, 1, 5),
                MetalMpsGraphProgram.Node.matmul(0, 9, 6),
                MetalMpsGraphProgram.Node.matmul(8, 9, 7)));
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        try {
            executable = api.createMpsGraphExecutable(context, NumericalProfile.ACCELERATOR, MetalTestProgram.descriptors(ranks, dimensions, program), program, new int[] {0, 1, 2, 3}, new int[] {4, 5, 6, 7});
            for (int[] bits : inputBits) {
                MetalNativeApi.Handle buffer = api.createBuffer(
                        context, Math.multiplyExact((long) bits.length, Integer.BYTES));
                inputs.add(buffer);
                upload(api, buffer, bits);
            }
            for (int output = 0; output < 4; output++) {
                outputs.add(api.createBuffer(
                        context, Math.multiplyExact((long) expected.length, Integer.BYTES)));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                for (int index = 0; index < outputs.size(); index++) {
                    outputAddresses.setAtIndex(ADDRESS, index, outputs.get(index).carrier());
                }
                for (int repetition = 0; repetition < 2; repetition++) {
                    for (MetalNativeApi.Handle output : outputs) {
                        upload(api, output, new int[] {
                            0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
                            0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef
                        });
                    }
                    api.runExecutable(
                            executable,
                            inputs.size(),
                            inputAddresses,
                            outputs.size(),
                            outputAddresses);
                    for (int output = 0; output < outputs.size(); output++) {
                        assertArrayEquals(
                                expected,
                                download(api, outputs.get(output), expected.length),
                                "output=" + output + " repetition=" + repetition);
                    }
                    for (int input = 0; input < inputs.size(); input++) {
                        assertArrayEquals(
                                inputBits[input],
                                download(api, inputs.get(input), inputBits[input].length),
                                "input=" + input + " repetition=" + repetition);
                    }
                }
            }
        } finally {
            for (int index = outputs.size(); index-- > 0;) {
                api.releaseBuffer(outputs.get(index));
            }
            for (int index = inputs.size(); index-- > 0;) {
                api.releaseBuffer(inputs.get(index));
            }
            if (executable != null) {
                api.releaseExecutable(executable);
            }
        }
    }

    private static void runK1NumericalSentinels(
            MetalNativeApi api, MetalNativeApi.Handle context) {
        float precisionInput = 1.0009765625f;
        int[][] inputBits = {
            bits(precisionInput, 0.0f, Float.intBitsToFloat(0x00000001)),
            bits(precisionInput, -1.0f)
        };
        int precisionProduct = Float.floatToRawIntBits(precisionInput * precisionInput);
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2)));
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        MetalNativeApi.Handle output = null;
        try {
            executable = api.createMpsGraphExecutable(context, NumericalProfile.ACCELERATOR, MetalTestProgram.descriptors(new int[] {2, 2, 2}, dimensions(new long[][] {{3, 1}, {1, 2}, {3, 2}}), program), program, new int[] {0, 1}, new int[] {2});
            for (int[] bits : inputBits) {
                MetalNativeApi.Handle buffer = api.createBuffer(
                        context, Math.multiplyExact((long) bits.length, Integer.BYTES));
                inputs.add(buffer);
                upload(api, buffer, bits);
            }
            output = api.createBuffer(context, 6L * Integer.BYTES);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, 1);
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                outputAddresses.setAtIndex(ADDRESS, 0, output.carrier());
                for (int repetition = 0; repetition < 2; repetition++) {
                    upload(api, output, new int[] {
                        0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
                        0xdeadbeef, 0xdeadbeef, 0xdeadbeef
                    });
                    api.runExecutable(
                            executable, 2, inputAddresses, 1, outputAddresses);
                    int[] actual = download(api, output, 6);
                    assertEquals(precisionProduct, actual[0],
                            "reduced-precision sentinel repetition=" + repetition);
                    assertEquals(Float.floatToRawIntBits(-precisionInput), actual[1]);
                    assertTrue(isEitherZero(actual[2]));
                    assertTrue(isEitherZero(actual[3]));
                    assertTrue(
                            actual[4] == 0x00000001 || isEitherZero(actual[4]),
                            "positive DAZ/FTZ result=0x"
                                    + Integer.toHexString(actual[4]));
                    assertTrue(
                            actual[5] == 0x80000001 || isEitherZero(actual[5]),
                            "negative DAZ/FTZ result=0x"
                                    + Integer.toHexString(actual[5]));
                    for (int input = 0; input < inputs.size(); input++) {
                        assertArrayEquals(
                                inputBits[input],
                                download(api, inputs.get(input), inputBits[input].length));
                    }
                }
            }
        } finally {
            if (output != null) {
                api.releaseBuffer(output);
            }
            for (int index = inputs.size(); index-- > 0;) {
                api.releaseBuffer(inputs.get(index));
            }
            if (executable != null) {
                api.releaseExecutable(executable);
            }
        }
    }

    private static boolean isEitherZero(int bits) {
        return (bits & 0x7fff_ffff) == 0;
    }

    private static int[] bits(float... values) {
        int[] bits = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            bits[index] = Float.floatToRawIntBits(values[index]);
        }
        return bits;
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] bits) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(
                    Math.multiplyExact((long) bits.length, Integer.BYTES), Integer.BYTES);
            for (int index = 0; index < bits.length; index++) {
                source.setAtIndex(JAVA_INT, index, bits[index]);
            }
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static int[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(
                    Math.multiplyExact((long) count, Integer.BYTES), Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            int[] result = new int[count];
            for (int index = 0; index < count; index++) {
                result[index] = target.getAtIndex(JAVA_INT, index);
            }
            return result;
        }
    }

    private static long[] dimensions(long[][] shapes) {
        long[] dimensions = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, dimensions, value * 16, shapes[value].length);
        }
        return dimensions;
    }
}
