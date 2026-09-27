package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalConvolutionPoolingNativeTest {
    @Test
    void groupedMixedCarrierConvolutionAndPaddingMultiplicationAreLiteral() {
        var grouped = node(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {0, 1, 2}, new int[] {3},
                MetalMpsGraphProgram.AttributeKind.CONV_2D,
                1, 1, 0, 0, 1, 1, 2);
        List<MetalMpsGraphProgram.ValueDescriptor> groupedValues = List.of(
                typed(DataType.FLOAT32, 1, 2, 2, 2),
                typed(DataType.BFLOAT16, 2, 1, 1, 1),
                typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, 1, 2, 2, 2));
        List<byte[]> actual = execute(
                new MetalMpsGraphProgram(List.of(grouped)), groupedValues,
                new int[] {0, 1, 2}, new int[] {3},
                List.of(
                        floatWords(1, 2, 3, 4, 5, 6, 7, 8),
                        shortWords(0x4000, 0xbf80),
                        floatWords(0.5f, 1.0f)));
        assertArrayEquals(floatWords(2.5f, 4.5f, 6.5f, 8.5f, -4, -5, -6, -7),
                actual.getFirst());
        var biasFirst = node(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {0, 1, 2}, new int[] {3},
                MetalMpsGraphProgram.AttributeKind.CONV_2D,
                1, 1, 0, 0, 1, 1, 1);
        List<byte[]> biasFirstActual = execute(
                new MetalMpsGraphProgram(List.of(biasFirst)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 1, 2),
                        typed(DataType.FLOAT32, 1, 1, 1, 2),
                        typed(DataType.FLOAT32, 1),
                        typed(DataType.FLOAT32, 1, 1, 1, 1)),
                new int[] {0, 1, 2}, new int[] {3},
                List.of(
                        floatWords(1, 1),
                        floatWords(0x1.0p24f, -0x1.0p24f),
                        floatWords(1)));
        assertArrayEquals(floatWords(0), biasFirstActual.getFirst(),
                "bias is the initial promoted accumulator");


        var padded = node(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {0, 1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.CONV_2D,
                1, 1, 1, 1, 1, 1, 1);
        int[] weights = new int[9];
        weights[0] = 0x7f800000;
        List<byte[]> paddedActual = execute(
                new MetalMpsGraphProgram(List.of(padded)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 1, 1),
                        typed(DataType.FLOAT32, 1, 1, 3, 3),
                        typed(DataType.FLOAT32, 1, 1, 1, 1)),
                new int[] {0, 1}, new int[] {2},
                List.of(floatWords(1), intWords(weights)));
        assertTrue(Float.isNaN(scalarFloat(paddedActual.getFirst())),
                "conceptual +0 times infinity must participate in convolution arithmetic");

        var conv3d = node(
                MetalMpsGraphProgram.NodeKind.CONV3D,
                new int[] {0, 1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.CONV_3D,
                1, 1, 1, 0, 0, 0, 1, 1, 1, 1);
        List<byte[]> conv3dActual = execute(
                new MetalMpsGraphProgram(List.of(conv3d)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 2, 2, 2),
                        typed(DataType.FLOAT32, 1, 1, 2, 2, 2),
                        typed(DataType.FLOAT32, 1, 1, 1, 1, 1)),
                new int[] {0, 1}, new int[] {2},
                List.of(floatWords(1, 2, 3, 4, 5, 6, 7, 8),
                        floatWords(1, 1, 1, 1, 1, 1, 1, 1)));
        assertArrayEquals(floatWords(36), conv3dActual.getFirst());
    }

    @Test
    void maximumPoolingPreservesWinningBitsAndAllPaddingIdentity() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            long firstNaN = switch (type) {
                case FLOAT64 -> 0x7ff0_0000_0000_1234L;
                case FLOAT32 -> 0x7fa1_2345L;
                case BFLOAT16 -> 0x7f81L;
                default -> throw new AssertionError(type);
            };
            long secondNaN = switch (type) {
                case FLOAT64 -> 0xfff8_0000_0000_5678L;
                case FLOAT32 -> 0xffc5_4321L;
                case BFLOAT16 -> 0xffc2L;
                default -> throw new AssertionError(type);
            };
            var max2d = node(
                    MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                    new int[] {0}, new int[] {1},
                    MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                    2, 2, 1, 1, 0, 0, 1, 1, 0);
            List<byte[]> result = execute(
                    new MetalMpsGraphProgram(List.of(max2d)),
                    List.of(typed(type, 1, 1, 2, 2), typed(type, 1, 1, 1, 1)),
                    new int[] {0}, new int[] {1},
                    List.of(rawWords(type, firstNaN, secondNaN, 0, negativeZero(type))));
            assertArrayEquals(rawWords(type, firstNaN), result.getFirst(), type.toString());
        }

        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            var numericMax = node(
                    MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                    new int[] {0}, new int[] {1},
                    MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                    1, 2, 1, 2, 0, 0, 1, 1, 0);
            List<byte[]> result = execute(
                    NumericalProfile.STRICT_IEEE,
                    new MetalMpsGraphProgram(List.of(numericMax)),
                    List.of(typed(type, 1, 1, 1, 10), typed(type, 1, 1, 1, 5)),
                    new int[] {0}, new int[] {1},
                    List.of(rawWords(
                            type,
                            negativeInfinity(type), negativeOne(type),
                            positiveInfinity(type), positiveOne(type),
                            negativeZero(type), 0,
                            0, negativeZero(type),
                            negativeOne(type), positiveOne(type))));
            assertArrayEquals(
                    rawWords(
                            type,
                            negativeOne(type),
                            positiveInfinity(type),
                            0,
                            0,
                            positiveOne(type)),
                    result.getFirst(),
                    type + " production numeric comparator");
        }

        var max3d = node(
                MetalMpsGraphProgram.NodeKind.MAX_POOL3D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 1, 1, 1);
        List<byte[]> allPadding = execute(
                new MetalMpsGraphProgram(List.of(max3d)),
                List.of(
                        typed(DataType.BFLOAT16, 1, 1, 1, 1, 1),
                        typed(DataType.BFLOAT16, 1, 1, 1, 1, 3)),
                new int[] {0}, new int[] {1}, List.of(shortWords(0x3f80)));
        assertArrayEquals(shortWords(0xff80, 0x3f80, 0xff80), allPadding.getFirst());
    }

    @Test
    void averagePoolingUsesFixedDivisorAndPreservesAllNegativeZeroOnly() {
        var average2d = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                1, 2, 1, 1, 0, 1, 1, 1, 0);
        List<byte[]> padded = execute(
                new MetalMpsGraphProgram(List.of(average2d)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 1, 2),
                        typed(DataType.FLOAT32, 1, 1, 1, 3)),
                new int[] {0}, new int[] {1}, List.of(floatWords(2, 4)));
        assertArrayEquals(floatWords(1, 3, 2), padded.getFirst());

        var negativeZero = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                1, 2, 1, 1, 0, 0, 1, 1, 0);
        List<byte[]> zero = execute(
                new MetalMpsGraphProgram(List.of(negativeZero)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 1, 2),
                        typed(DataType.FLOAT32, 1, 1, 1, 1)),
                new int[] {0}, new int[] {1}, List.of(intWords(0x80000000, 0x80000000)));
        assertArrayEquals(intWords(0x80000000), zero.getFirst());

        var average3d = node(
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL3D,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0);
        List<byte[]> infinity = execute(
                new MetalMpsGraphProgram(List.of(average3d)),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 2, 1, 1),
                        typed(DataType.FLOAT32, 1, 1, 1, 1, 1)),
                new int[] {0}, new int[] {1},
                List.of(intWords(0x7f800000, 0xff800000)));
        assertTrue(Float.isNaN(scalarFloat(infinity.getFirst())));
    }

    @Test
    void exactLocalSingletonHeightExpansionFeedsTwoDimensionalPoolWithoutCopy() {
        Shape viewShape = Shape.of(1, 1, 1, 4);
        var expanded = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                viewShape.toLongArray(),
                Optional.of(LayoutDescriptor.of(
                        viewShape, new long[] {4, 4, 4, 1}, 0, true)),
                false,
                true);
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 0, 1, 2),
                node(
                        MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                        new int[] {1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                        1, 2, 1, 1, 0, 0, 1, 1, 0)));
        List<byte[]> result = execute(
                program,
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 4),
                        expanded,
                        typed(DataType.FLOAT32, 1, 1, 1, 3)),
                new int[] {0}, new int[] {2}, List.of(floatWords(1, 2, 3, 4)));
        assertArrayEquals(floatWords(2, 3, 4), result.getFirst());
    }

    private static long negativeZero(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0x8000_0000_0000_0000L;
            case FLOAT32 -> 0x8000_0000L;
            case BFLOAT16 -> 0x8000L;
            default -> throw new AssertionError(type);
        };
    }

    private static long positiveInfinity(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0x7ff0_0000_0000_0000L;
            case FLOAT32 -> 0x7f80_0000L;
            case BFLOAT16 -> 0x7f80L;
            default -> throw new AssertionError(type);
        };
    }

    private static long negativeInfinity(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0xfff0_0000_0000_0000L;
            case FLOAT32 -> 0xff80_0000L;
            case BFLOAT16 -> 0xff80L;
            default -> throw new AssertionError(type);
        };
    }

    private static long positiveOne(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0x3ff0_0000_0000_0000L;
            case FLOAT32 -> 0x3f80_0000L;
            case BFLOAT16 -> 0x3f80L;
            default -> throw new AssertionError(type);
        };
    }

    private static long negativeOne(DataType type) {
        return switch (type) {
            case FLOAT64 -> 0xbff0_0000_0000_0000L;
            case FLOAT32 -> 0xbf80_0000L;
            case BFLOAT16 -> 0xbf80L;
            default -> throw new AssertionError(type);
        };
    }

    private static MetalMpsGraphProgram.Node node(
            MetalMpsGraphProgram.NodeKind kind,
            int[] inputs,
            int[] outputs,
            MetalMpsGraphProgram.AttributeKind attributeKind,
            long... words) {
        return MetalMpsGraphProgram.Node.generic(kind, inputs, outputs, attributeKind, words);
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static List<byte[]> execute(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> inputs) {
        return execute(
                NumericalProfile.ACCELERATOR, program, values, feeds, targets, inputs);
    }

    private static List<byte[]> execute(
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> inputs) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        assertEquals(feeds.length, inputs.size());
        Path library = Path.of(configured).toAbsolutePath().normalize();
        MetalNativeApi api = MetalNativeApi.open(library);
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
                api.runExecutable(executable, buffers.size(), addresses, targets.length, outputs);
                var result = new ArrayList<byte[]>(targets.length);
                for (int target : targets) {
                    result.add(download(api, buffers.get(target), values.get(target).byteCount()));
                }
                return List.copyOf(result);
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void upload(MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] bytes) {
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

    private static byte[] rawWords(DataType type, long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(Math.multiplyExact(words.length, type.byteWidth()))
                .order(ByteOrder.nativeOrder());
        for (long word : words) {
            switch (type) {
                case FLOAT64 -> bytes.putLong(word);
                case FLOAT32 -> bytes.putInt((int) word);
                case BFLOAT16 -> bytes.putShort((short) word);
                default -> throw new AssertionError(type);
            }
        }
        return bytes.array();
    }

    private static byte[] floatWords(float... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float word : words) bytes.putFloat(word);
        return bytes.array();
    }

    private static byte[] intWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putInt(word);
        return bytes.array();
    }

    private static byte[] shortWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putShort((short) word);
        return bytes.array();
    }

    private static float scalarFloat(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).getFloat();
    }
}
