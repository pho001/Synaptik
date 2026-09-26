package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
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
    @Test
    void customGeneralFloatingDomainRunsVectorBatchBroadcastAndBothMixedOrders() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.matmul(3, 4, 5),
                MetalMpsGraphProgram.Node.matmul(6, 7, 8),
                MetalMpsGraphProgram.Node.matmul(9, 10, 11)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 3),
                typed(DataType.FLOAT32, 3),
                typed(DataType.FLOAT32),
                typed(DataType.FLOAT32, 2, 2, 3),
                typed(DataType.FLOAT32, 1, 3, 2),
                typed(DataType.FLOAT32, 2, 2, 2),
                typed(DataType.BFLOAT16, 2, 2),
                typed(DataType.FLOAT32, 2, 1),
                typed(DataType.FLOAT32, 2, 1),
                typed(DataType.FLOAT32, 2, 2),
                typed(DataType.BFLOAT16, 2, 1),
                typed(DataType.FLOAT32, 2, 1));
        List<byte[]> actual = executeCustom(
                library,
                NumericalProfile.ACCELERATOR,
                program,
                values,
                new int[] {0, 1, 3, 4, 6, 7, 9, 10},
                new int[] {2, 5, 8, 11},
                List.of(
                        floatBytes(1, 2, 3),
                        floatBytes(4, 5, 6),
                        floatBytes(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                        floatBytes(1, 0, 0, 1, 1, 1),
                        bfloatBytes(1, 2, 3, 4),
                        floatBytes(5, 6),
                        floatBytes(1, 2, 3, 4),
                        bfloatBytes(5, 6)));
        assertArrayEquals(floatBytes(32), actual.get(0));
        assertArrayEquals(floatBytes(4, 5, 10, 11, 16, 17, 22, 23), actual.get(1));
        assertArrayEquals(floatBytes(17, 39), actual.get(2));
        assertArrayEquals(floatBytes(17, 39), actual.get(3));
    }

    @Test
    void customIntegralDomainRunsAllPromotionsWithModularOverflowUnderStrictProfile() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.matmul(3, 4, 5),
                MetalMpsGraphProgram.Node.matmul(6, 7, 8),
                MetalMpsGraphProgram.Node.matmul(9, 10, 11)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.INT32, 1, 2), typed(DataType.INT32, 2, 1),
                typed(DataType.INT32, 1, 1),
                typed(DataType.INT32, 1, 2), typed(DataType.INT64, 2, 1),
                typed(DataType.INT64, 1, 1),
                typed(DataType.INT64, 1, 2), typed(DataType.INT32, 2, 1),
                typed(DataType.INT64, 1, 1),
                typed(DataType.INT64, 1, 2), typed(DataType.INT64, 2, 1),
                typed(DataType.INT64, 1, 1));
        List<byte[]> actual = executeCustom(
                library,
                NumericalProfile.STRICT_IEEE,
                program,
                values,
                new int[] {0, 1, 3, 4, 6, 7, 9, 10},
                new int[] {2, 5, 8, 11},
                List.of(
                        intBytes(Integer.MAX_VALUE, 2), intBytes(2, 3),
                        intBytes(-2, 3), longBytes(Long.MAX_VALUE, 3),
                        longBytes(-2, 3), intBytes(Integer.MAX_VALUE, 3),
                        longBytes(Long.MAX_VALUE, 2), longBytes(2, 3)));
        assertArrayEquals(intBytes(4), actual.get(0));
        assertArrayEquals(longBytes(11), actual.get(1));
        assertArrayEquals(longBytes(-4_294_967_285L), actual.get(2));
        assertArrayEquals(longBytes(4), actual.get(3));
    }

    @Test
    void customRankThreeMatmulLoadsAuthenticatedTransposeFromPhysicalSource() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 2, List.of(0, 2, 1)),
                MetalMpsGraphProgram.Node.matmul(2, 1, 3)));
        Shape transposeShape = Shape.of(2, 2, 3);
        var affine = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                transposeShape.toLongArray(),
                Optional.of(LayoutDescriptor.of(
                        transposeShape, new long[] {6, 1, 2}, 0L, true)),
                false,
                true);
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 2, 3, 2),
                typed(DataType.FLOAT32, 3, 1),
                affine,
                typed(DataType.FLOAT32, 2, 2, 1));
        List<byte[]> actual = executeCustom(
                library,
                NumericalProfile.ACCELERATOR,
                program,
                values,
                new int[] {0, 1},
                new int[] {3},
                List.of(
                        floatBytes(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                        floatBytes(1, 10, 100)));
        assertArrayEquals(floatBytes(531, 642, 1197, 1308), actual.getFirst());
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
            executable = api.createMpsGraphExecutable(context, NumericalProfile.ACCELERATOR, MetalTestProgram.descriptors(ranks, dimensions, program), program, new int[] {0, 1, 2, 3}, new int[] {4, 5, 6, 7}, MetalPreparedRoute.MPSGRAPH);
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
            executable = api.createMpsGraphExecutable(context, NumericalProfile.ACCELERATOR, MetalTestProgram.descriptors(new int[] {2, 2, 2}, dimensions(new long[][] {{3, 1}, {1, 2}, {3, 2}}), program), program, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH);
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

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static List<byte[]> executeCustom(
            Path library,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> feedBytes) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createMpsGraphExecutable(
                    context, profile, values, program, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) {
                MetalNativeApi.Handle buffer = api.createBuffer(context, value.byteCount());
                buffers.add(buffer);
                byte[] poison = new byte[Math.toIntExact(value.byteCount())];
                java.util.Arrays.fill(poison, (byte) 0xa5);
                uploadBytes(api, buffer, poison);
            }
            for (int index = 0; index < feeds.length; index++) {
                assertEquals(values.get(feeds[index]).byteCount(), feedBytes.get(index).length);
                uploadBytes(api, buffers.get(feeds[index]), feedBytes.get(index));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    addresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++) {
                    outputs.setAtIndex(
                            ADDRESS, index, buffers.get(targets[index]).carrier());
                }
                api.runExecutable(
                        executable, buffers.size(), addresses, targets.length, outputs);
            }
            var result = new ArrayList<byte[]>(targets.length);
            for (int target : targets) {
                result.add(downloadBytes(
                        api, buffers.get(target), values.get(target).byteCount()));
            }
            return List.copyOf(result);
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void uploadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1);
            source.copyFrom(MemorySegment.ofArray(bytes));
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static byte[] downloadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long byteCount) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(byteCount, 1);
            api.download(buffer, 0L, target, target.byteSize());
            return target.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        }
    }

    private static byte[] floatBytes(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putInt(Float.floatToRawIntBits(value));
        return bytes.array();
    }

    private static byte[] bfloatBytes(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : values) {
            bytes.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
        }
        return bytes.array();
    }

    private static byte[] intBytes(int... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int value : values) bytes.putInt(value);
        return bytes.array();
    }

    private static byte[] longBytes(long... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        for (long value : values) bytes.putLong(value);
        return bytes.array();
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
