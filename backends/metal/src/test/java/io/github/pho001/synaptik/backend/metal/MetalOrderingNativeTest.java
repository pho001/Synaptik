package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalOrderingNativeTest {
    @Test
    void exactFloatOrderingTopKAndArgExtremaPublishEveryOutputAcrossReuse() {
        var program = new MetalMpsGraphProgram(List.of(
                node(MetalMpsGraphProgram.NodeKind.SORT, new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.SORT, 1, 0),
                node(MetalMpsGraphProgram.NodeKind.ARGSORT, new int[] {0}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.SORT, 1, 0),
                node(MetalMpsGraphProgram.NodeKind.TOP_K, new int[] {0}, new int[] {3, 4},
                        MetalMpsGraphProgram.AttributeKind.TOP_K, 1, 2, 1, 1),
                node(MetalMpsGraphProgram.NodeKind.ARG_MAX, new int[] {0}, new int[] {5},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 1, 0, 1),
                node(MetalMpsGraphProgram.NodeKind.ARG_MIN, new int[] {0}, new int[] {6},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 1, 0, 2)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 2, 4),
                typed(DataType.FLOAT32, 2, 4),
                typed(DataType.INT64, 2, 4),
                typed(DataType.FLOAT32, 2, 2),
                typed(DataType.INT64, 2, 2),
                typed(DataType.INT64, 2),
                typed(DataType.INT64, 2));
        int nanA = 0x7fa12345;
        int nanB = 0xffc54321;
        byte[] input = intWords(nanA, 0x80000000, 0x00000000, 0xbf800000,
                0x40000000, 0x40000000, nanB, 0x3f800000);
        List<byte[]> actual = execute(
                program, values, new int[] {0}, new int[] {1, 2, 3, 4, 5, 6}, input);
        assertArrayEquals(intWords(0xbf800000, 0x80000000, 0x00000000, nanA,
                0x3f800000, 0x40000000, 0x40000000, nanB), actual.get(0));
        assertArrayEquals(longWords(3, 1, 2, 0, 3, 0, 1, 2), actual.get(1));
        assertArrayEquals(intWords(0x00000000, 0x80000000, 0x40000000, 0x40000000),
                actual.get(2));
        assertArrayEquals(longWords(2, 1, 0, 1), actual.get(3));
        assertArrayEquals(longWords(0, 2), actual.get(4));
        assertArrayEquals(longWords(0, 2), actual.get(5));
    }

    @Test
    void everyCarrierHonorsDescendingStabilityUnsortedCompactionAndArgTiePolicies() {
        for (DataType type : DataType.values()) {
            boolean numeric = type != DataType.BOOL;
            var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
            nodes.add(node(MetalMpsGraphProgram.NodeKind.SORT, new int[] {0}, new int[] {1},
                    MetalMpsGraphProgram.AttributeKind.SORT, 0, 1));
            nodes.add(node(MetalMpsGraphProgram.NodeKind.ARGSORT, new int[] {0}, new int[] {2},
                    MetalMpsGraphProgram.AttributeKind.SORT, 0, 1));
            nodes.add(node(MetalMpsGraphProgram.NodeKind.TOP_K, new int[] {0}, new int[] {3, 4},
                    MetalMpsGraphProgram.AttributeKind.TOP_K, 0, 3, 1, 0));
            nodes.add(node(MetalMpsGraphProgram.NodeKind.TOP_K, new int[] {0}, new int[] {5, 6},
                    MetalMpsGraphProgram.AttributeKind.TOP_K, 0, 3, 0, 1));
            if (numeric) {
                nodes.add(node(MetalMpsGraphProgram.NodeKind.ARG_MAX,
                        new int[] {0}, new int[] {7},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 0, 1));
                nodes.add(node(MetalMpsGraphProgram.NodeKind.ARG_MAX,
                        new int[] {0}, new int[] {8},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 0, 2));
                nodes.add(node(MetalMpsGraphProgram.NodeKind.ARG_MIN,
                        new int[] {0}, new int[] {9},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 0, 1));
                nodes.add(node(MetalMpsGraphProgram.NodeKind.ARG_MIN,
                        new int[] {0}, new int[] {10},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 0, 2));
            }
            var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
            values.add(typed(type, 6));
            values.add(typed(type, 6));
            values.add(typed(DataType.INT64, 6));
            values.add(typed(type, 3));
            values.add(typed(DataType.INT64, 3));
            values.add(typed(type, 3));
            values.add(typed(DataType.INT64, 3));
            if (numeric) {
                for (int index = 0; index < 4; index++) {
                    values.add(typed(DataType.INT64));
                }
            }
            int[] targets = new int[numeric ? 10 : 6];
            for (int index = 0; index < targets.length; index++) targets[index] = index + 1;
            CarrierCase fixture = carrierCase(type);
            List<byte[]> actual = execute(new MetalMpsGraphProgram(nodes), values,
                    new int[] {0}, targets, fixture.input());
            assertArrayEquals(fixture.descendingValues(), actual.get(0), type + " sort");
            assertArrayEquals(fixture.descendingIndices(), actual.get(1), type + " argsort");
            assertArrayEquals(fixture.unsortedLargestValues(), actual.get(2),
                    type + " unsorted TOP_K values");
            assertArrayEquals(fixture.unsortedLargestIndices(), actual.get(3),
                    type + " unsorted TOP_K indices");
            assertArrayEquals(fixture.sortedSmallestValues(), actual.get(4),
                    type + " smallest TOP_K values");
            assertArrayEquals(fixture.sortedSmallestIndices(), actual.get(5),
                    type + " smallest TOP_K indices");
            if (numeric) {
                assertEquals(fixture.argMaxFirst(), scalarLong(actual.get(6)),
                        type + " arg-max first");
                assertEquals(fixture.argMaxLast(), scalarLong(actual.get(7)),
                        type + " arg-max last");
                assertEquals(fixture.argMinFirst(), scalarLong(actual.get(8)),
                        type + " arg-min first");
                assertEquals(fixture.argMinLast(), scalarLong(actual.get(9)),
                        type + " arg-min last");
            }
        }
    }

    @Test
    void int32SameSignMagnitudesDriveEveryNativeOrderingPath() {
        var program = new MetalMpsGraphProgram(List.of(
                node(MetalMpsGraphProgram.NodeKind.SORT, new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.SORT, 0, 0),
                node(MetalMpsGraphProgram.NodeKind.SORT, new int[] {0}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.SORT, 0, 1),
                node(MetalMpsGraphProgram.NodeKind.ARGSORT, new int[] {0}, new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.SORT, 0, 0),
                node(MetalMpsGraphProgram.NodeKind.ARGSORT, new int[] {0}, new int[] {4},
                        MetalMpsGraphProgram.AttributeKind.SORT, 0, 1),
                node(MetalMpsGraphProgram.NodeKind.TOP_K, new int[] {0}, new int[] {5, 6},
                        MetalMpsGraphProgram.AttributeKind.TOP_K, 0, 4, 1, 1),
                node(MetalMpsGraphProgram.NodeKind.TOP_K, new int[] {0}, new int[] {7, 8},
                        MetalMpsGraphProgram.AttributeKind.TOP_K, 0, 4, 0, 0),
                node(MetalMpsGraphProgram.NodeKind.ARG_MAX, new int[] {0}, new int[] {9},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 1, 1),
                node(MetalMpsGraphProgram.NodeKind.ARG_MAX, new int[] {0}, new int[] {10},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 1, 2),
                node(MetalMpsGraphProgram.NodeKind.ARG_MIN, new int[] {0}, new int[] {11},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 1, 1),
                node(MetalMpsGraphProgram.NodeKind.ARG_MIN, new int[] {0}, new int[] {12},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 1, 2)));
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        values.add(typed(DataType.INT32, 6));
        values.add(typed(DataType.INT32, 6));
        values.add(typed(DataType.INT32, 6));
        values.add(typed(DataType.INT64, 6));
        values.add(typed(DataType.INT64, 6));
        for (int output = 0; output < 2; output++) {
            values.add(typed(DataType.INT32, 4));
            values.add(typed(DataType.INT64, 4));
        }
        for (int output = 0; output < 4; output++) {
            values.add(typed(DataType.INT64, 1));
        }
        int[] targets = new int[12];
        for (int index = 0; index < targets.length; index++) targets[index] = index + 1;

        List<byte[]> actual = execute(
                program,
                values,
                new int[] {0},
                targets,
                intWords(7, 3, -2, -9, 7, -9));

        assertArrayEquals(intWords(-9, -9, -2, 3, 7, 7), actual.get(0));
        assertArrayEquals(intWords(7, 7, 3, -2, -9, -9), actual.get(1));
        assertArrayEquals(longWords(3, 5, 2, 1, 0, 4), actual.get(2));
        assertArrayEquals(longWords(0, 4, 1, 2, 3, 5), actual.get(3));
        assertArrayEquals(intWords(7, 7, 3, -2), actual.get(4));
        assertArrayEquals(longWords(0, 4, 1, 2), actual.get(5));
        assertArrayEquals(intWords(3, -2, -9, -9), actual.get(6));
        assertArrayEquals(longWords(1, 2, 3, 5), actual.get(7));
        assertArrayEquals(longWords(0), actual.get(8));
        assertArrayEquals(longWords(4), actual.get(9));
        assertArrayEquals(longWords(3), actual.get(10));
        assertArrayEquals(longWords(5), actual.get(11));
    }

    private static CarrierCase carrierCase(DataType type) {
        if (type.isFloating()) {
            long nanA;
            long negativeZero;
            long positiveZero = 0L;
            long negativeOne;
            long positiveOne;
            long nanB;
            switch (type) {
                case FLOAT64 -> {
                    nanA = 0x7ff0_0000_0000_1234L;
                    negativeZero = 0x8000_0000_0000_0000L;
                    negativeOne = 0xbff0_0000_0000_0000L;
                    positiveOne = 0x3ff0_0000_0000_0000L;
                    nanB = 0xfff8_0000_0000_5678L;
                }
                case FLOAT32 -> {
                    nanA = 0x7fa1_2345L;
                    negativeZero = 0x8000_0000L;
                    negativeOne = 0xbf80_0000L;
                    positiveOne = 0x3f80_0000L;
                    nanB = 0xffc5_4321L;
                }
                case BFLOAT16 -> {
                    nanA = 0x7f81L;
                    negativeZero = 0x8000L;
                    negativeOne = 0xbf80L;
                    positiveOne = 0x3f80L;
                    nanB = 0xffc2L;
                }
                case FLOAT16 -> {
                    nanA = 0x7c01L;
                    negativeZero = 0x8000L;
                    negativeOne = 0xbc00L;
                    positiveOne = 0x3c00L;
                    nanB = 0xfe02L;
                }
                default -> throw new AssertionError(type);
            }
            return new CarrierCase(
                    rawWords(type, nanA, negativeZero, positiveZero, negativeOne, positiveOne, nanB),
                    rawWords(type, positiveOne, positiveZero, negativeZero, negativeOne, nanA, nanB),
                    longWords(4, 2, 1, 3, 0, 5),
                    rawWords(type, negativeZero, positiveZero, positiveOne),
                    longWords(1, 2, 4),
                    rawWords(type, negativeOne, negativeZero, positiveZero),
                    longWords(3, 1, 2),
                    0, 5, 0, 5);
        }
        if (type == DataType.BOOL) {
            return new CarrierCase(
                    rawWords(type, 1, 0, 0, 1, 0, 1),
                    rawWords(type, 1, 1, 1, 0, 0, 0),
                    longWords(0, 3, 5, 1, 2, 4),
                    rawWords(type, 1, 1, 1),
                    longWords(0, 3, 5),
                    rawWords(type, 0, 0, 0),
                    longWords(1, 2, 4),
                    -1, -1, -1, -1);
        }
        return new CarrierCase(
                rawWords(type, 5, 0, 0, -1, 1, 5),
                rawWords(type, 5, 5, 1, 0, 0, -1),
                longWords(0, 5, 4, 1, 2, 3),
                rawWords(type, 5, 1, 5),
                longWords(0, 4, 5),
                rawWords(type, -1, 0, 0),
                longWords(3, 1, 2),
                0, 5, 3, 3);
    }

    private static byte[] rawWords(DataType type, long... words) {
        int width = type.byteWidth();
        ByteBuffer bytes = ByteBuffer.allocate(Math.multiplyExact(words.length, width))
                .order(ByteOrder.nativeOrder());
        for (long word : words) {
            switch (type) {
                case FLOAT64, INT64 -> bytes.putLong(word);
                case FLOAT32, INT32 -> bytes.putInt((int) word);
                case BFLOAT16, FLOAT16 -> bytes.putShort((short) word);
                case BOOL -> bytes.put((byte) word);
            }
        }
        return bytes.array();
    }

    private static long scalarLong(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).getLong();
    }

    private record CarrierCase(
            byte[] input,
            byte[] descendingValues,
            byte[] descendingIndices,
            byte[] unsortedLargestValues,
            byte[] unsortedLargestIndices,
            byte[] sortedSmallestValues,
            byte[] sortedSmallestIndices,
            long argMaxFirst,
            long argMaxLast,
            long argMinFirst,
            long argMinLast) {}

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
            byte[] input) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context, values, program, feeds, targets,
            MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) {
                buffers.add(api.createBuffer(context, value.byteCount()));
            }
            upload(api, buffers.getFirst(), input);
            List<byte[]> first = null;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    addresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++) {
                    outputs.setAtIndex(ADDRESS, index, buffers.get(targets[index]).carrier());
                }
                for (int repetition = 0; repetition < 2; repetition++) {
                    api.runExecutable(executable, buffers.size(), addresses, targets.length, outputs);
                    var current = new ArrayList<byte[]>(targets.length);
                    for (int target : targets) {
                        current.add(download(api, buffers.get(target), values.get(target).byteCount()));
                    }
                    if (first == null) {
                        first = List.copyOf(current);
                    } else {
                        for (int index = 0; index < first.size(); index++) {
                            assertArrayEquals(first.get(index), current.get(index));
                        }
                    }
                }
            }
            assertArrayEquals(input, download(api, buffers.getFirst(), input.length));
            return first;
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

    private static byte[] intWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putInt(word);
        return bytes.array();
    }

    private static byte[] longWords(long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        for (long word : words) bytes.putLong(word);
        return bytes.array();
    }
}
