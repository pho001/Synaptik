package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Pins current real-device Metal CUSTOM_PROGRAM observations for stored FLOAT32 and low-precision
 * subnormals. These tests qualify raw, cast, stored-value and arithmetic sites separately.
 */
class MetalStoredSubnormalProviderNativeTest {
    private static final int MIN_SUBNORMAL = 0x0000_0001;
    private static final int POSITIVE_ZERO = 0x0000_0000;

    @Test
    void customProgramDistinguishesBothOrdersOfStoredSubnormalAndZero() {
        var program = new MetalMpsGraphProgram(List.of(
                binary(MetalMpsGraphProgram.NodeKind.GT, 0, 1, 2),
                binary(MetalMpsGraphProgram.NodeKind.EQ, 0, 1, 3),
                binary(MetalMpsGraphProgram.NodeKind.TENSOR_MIN, 0, 1, 4),
                binary(MetalMpsGraphProgram.NodeKind.TENSOR_MAX, 0, 1, 5),
                scalar(MetalMpsGraphProgram.NodeKind.SCALAR_MIN, 0, 6, POSITIVE_ZERO),
                scalar(MetalMpsGraphProgram.NodeKind.SCALAR_MAX, 0, 7, POSITIVE_ZERO),
                scalar(MetalMpsGraphProgram.NodeKind.SCALAR_MIN, 0, 8, MIN_SUBNORMAL),
                scalar(MetalMpsGraphProgram.NodeKind.SCALAR_MAX, 0, 9, MIN_SUBNORMAL),
                binary(MetalMpsGraphProgram.NodeKind.GE, 0, 1, 10),
                binary(MetalMpsGraphProgram.NodeKind.LT, 0, 1, 11),
                binary(MetalMpsGraphProgram.NodeKind.LE, 0, 1, 12),
                binary(MetalMpsGraphProgram.NodeKind.NE, 0, 1, 13)));
        var values = List.of(typed(DataType.FLOAT32, 2), typed(DataType.FLOAT32, 2),
                typed(DataType.BOOL, 2), typed(DataType.BOOL, 2),
                typed(DataType.FLOAT32, 2), typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, 2), typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, 2), typed(DataType.FLOAT32, 2),
                typed(DataType.BOOL, 2), typed(DataType.BOOL, 2),
                typed(DataType.BOOL, 2), typed(DataType.BOOL, 2));
        List<byte[]> actual = execute(program, values, List.of(
                intWords(MIN_SUBNORMAL, POSITIVE_ZERO),
                intWords(POSITIVE_ZERO, MIN_SUBNORMAL)),
                new int[] {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
        assertArrayEquals(new byte[] {1, 0}, actual.get(0), "GT raw BOOL bytes");
        assertArrayEquals(new byte[] {0, 0}, actual.get(1), "EQ raw BOOL bytes");
        assertArrayEquals(intWords(POSITIVE_ZERO, POSITIVE_ZERO), actual.get(2),
                "tensor MIN raw bits");
        assertArrayEquals(intWords(MIN_SUBNORMAL, MIN_SUBNORMAL), actual.get(3),
                "tensor MAX raw bits");
        assertArrayEquals(intWords(POSITIVE_ZERO, POSITIVE_ZERO), actual.get(4),
                "scalar MIN(+0) raw bits");
        assertArrayEquals(intWords(MIN_SUBNORMAL, POSITIVE_ZERO), actual.get(5),
                "scalar MAX(+0) raw bits");
        assertArrayEquals(intWords(MIN_SUBNORMAL, POSITIVE_ZERO), actual.get(6),
                "scalar MIN(min-subnormal) raw bits");
        assertArrayEquals(intWords(MIN_SUBNORMAL, MIN_SUBNORMAL), actual.get(7),
                "scalar MAX(min-subnormal) raw bits");
        assertArrayEquals(new byte[] {1, 0}, actual.get(8), "GE raw BOOL bytes");
        assertArrayEquals(new byte[] {0, 1}, actual.get(9), "LT raw BOOL bytes");
        assertArrayEquals(new byte[] {0, 1}, actual.get(10), "LE raw BOOL bytes");
        assertArrayEquals(new byte[] {1, 1}, actual.get(11), "NE raw BOOL bytes");
    }

    @Test
    void customProgramArgMaxDistinguishesUniqueWinnerFromTrueEqualTies() {
        var program = new MetalMpsGraphProgram(List.of(
                argMax(0, 2, 1), argMax(0, 3, 2),
                argMax(1, 4, 1), argMax(1, 5, 2)));
        var values = List.of(typed(DataType.FLOAT32, 3),
                typed(DataType.FLOAT32, 2), typed(DataType.INT64),
                typed(DataType.INT64), typed(DataType.INT64), typed(DataType.INT64));
        List<byte[]> actual = execute(program, values, List.of(
                intWords(POSITIVE_ZERO, MIN_SUBNORMAL, POSITIVE_ZERO),
                intWords(MIN_SUBNORMAL, MIN_SUBNORMAL)), new int[] {2, 3, 4, 5});
        assertArrayEquals(longWords(1), actual.get(0), "unique winner FIRST index bytes");
        assertArrayEquals(longWords(1), actual.get(1), "unique winner LAST index bytes");
        assertArrayEquals(longWords(0), actual.get(2), "true-equal FIRST index bytes");
        assertArrayEquals(longWords(1), actual.get(3), "true-equal LAST index bytes");
    }

    @Test
    void customProgramKeepsLowSubnormalCastAndMovementSeparateFromArithmetic() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            int floatBits = type == DataType.BFLOAT16 ? 0x0001_0000 : 0x3380_0000;
            int halfBits = type == DataType.BFLOAT16 ? 0x3f00 : 0x3800;
            long oneBits = type == DataType.BFLOAT16 ? 0x3f80L : 0x3c00L;
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.contiguous(0, 4),
                    MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.CAST,
                            new int[] {2}, new int[] {5},
                            MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                            new long[] {MetalMpsGraphProgram.dataTypeWire(type)}),
                    binary(MetalMpsGraphProgram.NodeKind.MUL, 0, 3, 6),
                    binary(MetalMpsGraphProgram.NodeKind.GT, 0, 1, 7),
                    binary(MetalMpsGraphProgram.NodeKind.EQ, 0, 1, 8),
                    binary(MetalMpsGraphProgram.NodeKind.TENSOR_MIN, 0, 1, 9),
                    binary(MetalMpsGraphProgram.NodeKind.TENSOR_MAX, 0, 1, 10),
                    MetalMpsGraphProgram.Node.scalarValue(
                            MetalMpsGraphProgram.NodeKind.SCALAR_MUL, 0, 11, type, oneBits)));
            var low = typed(type, 2);
            var values = List.of(low, low, typed(DataType.FLOAT32, 2), low,
                    low, low, low, typed(DataType.BOOL, 2), typed(DataType.BOOL, 2),
                    low, low, low);
            List<byte[]> actual = execute(program, values, List.of(
                    shortWords(0x0001, 0x0000), shortWords(0x0000, 0x0001),
                    intWords(floatBits, floatBits), shortWords(halfBits, halfBits)),
                    new int[] {4, 5, 6, 7, 8, 9, 10, 11});

            assertArrayEquals(shortWords(0x0001, 0x0000), actual.get(0), type + " raw CONTIGUOUS");
            assertArrayEquals(shortWords(0x0001, 0x0001), actual.get(1), type + " exact FLOAT32 cast");
            assertArrayEquals(shortWords(0x0000, 0x0000), actual.get(2), type + " arithmetic underflow");
            assertArrayEquals(new byte[] {1, 0}, actual.get(3), type + " GT");
            assertArrayEquals(new byte[] {0, 0}, actual.get(4), type + " EQ");
            assertArrayEquals(shortWords(0x0000, 0x0000), actual.get(5), type + " MIN");
            assertArrayEquals(shortWords(0x0001, 0x0001), actual.get(6), type + " MAX");
            assertArrayEquals(shortWords(type == DataType.BFLOAT16 ? 0x0000 : 0x0001, 0x0000),
                    actual.get(7), type + " min-subnormal times one");
        }
    }

    @Test
    void customProgramBfloat16MinSubnormalTimes128ProducesZero() {
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_MUL, 0, 1,
                        DataType.BFLOAT16, 0x4300L)));
        var low = typed(DataType.BFLOAT16, 1);
        List<byte[]> actual = execute(program, List.of(low, low),
                List.of(shortWords(0x0001)), new int[] {1});
        assertArrayEquals(shortWords(0x0000), actual.get(0),
                "BFLOAT16 min-subnormal times 128 CUSTOM_PROGRAM raw bits");
    }

    private static MetalMpsGraphProgram.Node binary(MetalMpsGraphProgram.NodeKind kind,
            int left, int right, int output) {
        return MetalMpsGraphProgram.Node.generic(kind, new int[] {left, right},
                new int[] {output}, MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.Node scalar(MetalMpsGraphProgram.NodeKind kind,
            int input, int output, int bits) {
        return MetalMpsGraphProgram.Node.generic(kind, new int[] {input},
                new int[] {output}, MetalMpsGraphProgram.AttributeKind.SCALAR_VALUE,
                new long[] {1L, Integer.toUnsignedLong(bits)});
    }

    private static MetalMpsGraphProgram.Node argMax(int input, int output, int tieWire) {
        return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.ARG_MAX,
                new int[] {input}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA,
                new long[] {0, 0, tieWire});
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static List<byte[]> execute(MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values, List<byte[]> inputs,
            int[] targets) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "Missing packaged Metal test library: SYNAPTIK_METAL_TEST_LIBRARY is unset");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "Configured Metal package library is missing: "
                + library);
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext(); // A configured run must reach a real device or fail.
            int[] feeds = new int[inputs.size()];
            for (int index = 0; index < feeds.length; index++) feeds[index] = index;
            executable = api.createProgramExecutable(context, values, program, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            for (int index = 0; index < inputs.size(); index++) {
                upload(api, buffers.get(index), inputs.get(index));
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
            }
            var actual = new ArrayList<byte[]>(targets.length);
            for (int target : targets) {
                actual.add(download(api, buffers.get(target), values.get(target).byteCount()));
            }
            for (int index = 0; index < inputs.size(); index++) {
                assertArrayEquals(inputs.get(index), download(api, buffers.get(index),
                        values.get(index).byteCount()), "stored input " + index + " raw bits");
            }
            return actual;
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

    private static byte[] download(MetalNativeApi api, MetalNativeApi.Handle buffer,
            long byteCount) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(byteCount, 1);
            api.download(buffer, 0L, target, byteCount);
            return target.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        }
    }

    private static byte[] intWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(4 * words.length).order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putInt(word);
        return bytes.array();
    }

    private static byte[] shortWords(int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(2 * words.length).order(ByteOrder.nativeOrder());
        for (int word : words) bytes.putShort((short) word);
        return bytes.array();
    }

    private static byte[] longWords(long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(8 * words.length).order(ByteOrder.nativeOrder());
        for (long word : words) bytes.putLong(word);
        return bytes.array();
    }
}
