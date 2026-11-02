package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
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

/** Actual custom-route regressions for raw FLOAT64 and modular integral operations. */
class MetalExactNumericNativeTest {
    private static final long SIGN = Long.MIN_VALUE;
    private static final long ONE = 0x3ff0000000000000L;
    private static final long[] FLOAT64_WORDS = {
        0, SIGN, 1, SIGN | 1,
        0x000fffffffffffffL, 0x800fffffffffffffL,
        0x0010000000000000L, 0x8010000000000000L,
        0x3fefffffffffffffL, 0xbfefffffffffffffL,
        ONE, SIGN | ONE, 0x3ff0000000000001L, 0xbff0000000000001L,
        0x3ff8000000000000L, 0xbff8000000000000L,
        0x432fffffffffffffL, 0xc32fffffffffffffL,
        0x4330000000000000L, 0xc330000000000000L,
        0x7fefffffffffffffL, 0xffefffffffffffffL,
        0x7ff0000000000000L, 0xfff0000000000000L,
        0x7ff8000000001234L, 0xfff8000000005678L,
        0x7ff0000000001234L, 0xfff0000000005678L
    };

    @Test
    void float64ExactUnarySitesPreserveRepresentedSubnormalsAndSpecialClasses() {
        var kinds = List.of(MetalMpsGraphProgram.NodeKind.NEG,
                MetalMpsGraphProgram.NodeKind.ABS, MetalMpsGraphProgram.NodeKind.FLOOR,
                MetalMpsGraphProgram.NodeKind.CEIL, MetalMpsGraphProgram.NodeKind.SIGN,
                MetalMpsGraphProgram.NodeKind.RELU);
        var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        values.add(typed(DataType.FLOAT64, FLOAT64_WORDS.length));
        int[] targets = new int[kinds.size()];
        for (int i = 0; i < kinds.size(); i++) {
            targets[i] = i + 1;
            nodes.add(node(kinds.get(i), new int[] {0}, i + 1));
            values.add(typed(DataType.FLOAT64, FLOAT64_WORDS.length));
        }
        var actual = execute(new MetalMpsGraphProgram(nodes), values, new int[] {0}, targets,
                raw(DataType.FLOAT64, FLOAT64_WORDS));
        for (int k = 0; k < kinds.size(); k++) {
            long[] words = longWords(actual.get(k));
            for (int i = 0; i < FLOAT64_WORDS.length; i++) {
                double value = Double.longBitsToDouble(FLOAT64_WORDS[i]);
                if (Double.isNaN(value)) {
                    assertTrue(Double.isNaN(Double.longBitsToDouble(words[i])), kinds.get(k).name());
                    continue;
                }
                double expected = switch (kinds.get(k)) {
                    case NEG -> -value;
                    case ABS -> Math.abs(value);
                    case FLOOR -> Math.floor(value);
                    case CEIL -> Math.ceil(value);
                    case SIGN -> Math.signum(value);
                    case RELU -> value > 0 ? value : 0.0;
                    default -> throw new AssertionError();
                };
                assertEquals(Double.doubleToRawLongBits(expected), words[i], kinds.get(k) + ":" + i);
            }
        }
    }

    @Test
    void float64TensorExtremaRetainSubnormalsSignedZeroAndNaNClasses() {
        long[] left = {SIGN, 0, 1, SIGN | 1, 0x7fefffffffffffffL,
            0xfff0000000000000L, 0x7ff0000000001234L, ONE};
        long[] right = {0, SIGN, 0, SIGN, 0x7ff0000000000000L,
            0xffefffffffffffffL, ONE, 0xfff8000000004321L};
        var program = new MetalMpsGraphProgram(List.of(
                node(MetalMpsGraphProgram.NodeKind.TENSOR_MIN, new int[] {0, 1}, 2),
                node(MetalMpsGraphProgram.NodeKind.TENSOR_MAX, new int[] {0, 1}, 3)));
        var values = List.of(typed(DataType.FLOAT64, left.length), typed(DataType.FLOAT64, left.length),
                typed(DataType.FLOAT64, left.length), typed(DataType.FLOAT64, left.length));
        var outputs = execute(program, values, new int[] {0, 1}, new int[] {2, 3},
                raw(DataType.FLOAT64, left), raw(DataType.FLOAT64, right));
        for (int k = 0; k < 2; k++) {
            long[] actual = longWords(outputs.get(k));
            for (int i = 0; i < left.length; i++) {
                double a = Double.longBitsToDouble(left[i]), b = Double.longBitsToDouble(right[i]);
                assertDouble(k == 0 ? Math.min(a, b) : Math.max(a, b), actual[i]);
            }
        }
    }

    @Test
    void float64ExtremaWidenOtherFloatingCarriersExactlyUnderBroadcasting() {
        for (DataType other : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            long unit = switch (other) {
                case FLOAT32 -> 0x3f800000L;
                case BFLOAT16 -> 0x3f80L;
                case FLOAT16 -> 0x3c00L;
                default -> throw new AssertionError();
            };
            long negativeZero = other == DataType.FLOAT32 ? 0x80000000L : 0x8000L;
            double subnormal = switch (other) {
                case FLOAT32 -> Math.scalb(1.0, -149);
                case BFLOAT16 -> Math.scalb(1.0, -133);
                case FLOAT16 -> Math.scalb(1.0, -24);
                default -> throw new AssertionError();
            };
            var program = new MetalMpsGraphProgram(List.of(
                    node(MetalMpsGraphProgram.NodeKind.TENSOR_MIN, new int[] {0, 1}, 2),
                    node(MetalMpsGraphProgram.NodeKind.TENSOR_MAX, new int[] {1, 0}, 3)));
            var values = List.of(typed(DataType.FLOAT64, 2, 1), typed(other, 3),
                    typed(DataType.FLOAT64, 2, 3), typed(DataType.FLOAT64, 2, 3));
            var outputs = execute(program, values, new int[] {0, 1}, new int[] {2, 3},
                    raw(DataType.FLOAT64, 0, SIGN | ONE), raw(other, negativeZero, 1, unit));
            double[] a = {0.0, -1.0}, b = {-0.0, subnormal, 1.0};
            for (int k = 0; k < 2; k++) {
                long[] actual = longWords(outputs.get(k));
                for (int i = 0; i < 6; i++)
                    assertDouble(k == 0 ? Math.min(a[i / 3], b[i % 3])
                            : Math.max(a[i / 3], b[i % 3]), actual[i]);
            }
        }
    }

    @Test
    void float64ScalarExtremaAndClampObserveOrderedBoundsAndNaNs() {
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_MIN,
                        0, 1, DataType.FLOAT64, SIGN),
                MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_MAX,
                        0, 2, DataType.FLOAT64, 0),
                MetalMpsGraphProgram.Node.clamp(0, 3, DataType.FLOAT64, SIGN, 0),
                MetalMpsGraphProgram.Node.clamp(0, 4, DataType.FLOAT64, ONE, ONE),
                MetalMpsGraphProgram.Node.clamp(0, 5, DataType.FLOAT64, 0x7ff8000000004321L, ONE),
                MetalMpsGraphProgram.Node.clamp(0, 6, DataType.FLOAT64, SIGN | ONE, 0x7ff0000000001234L)));
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        for (int i = 0; i < 7; i++) values.add(typed(DataType.FLOAT64, FLOAT64_WORDS.length));
        var outputs = execute(program, values, new int[] {0}, new int[] {1, 2, 3, 4, 5, 6},
                raw(DataType.FLOAT64, FLOAT64_WORDS));
        for (int k = 0; k < outputs.size(); k++) {
            long[] actual = longWords(outputs.get(k));
            for (int i = 0; i < FLOAT64_WORDS.length; i++) {
                double value = Double.longBitsToDouble(FLOAT64_WORDS[i]);
                double expected = switch (k) {
                    case 0 -> Math.min(value, -0.0);
                    case 1 -> Math.max(value, 0.0);
                    case 2 -> Math.min(Math.max(value, -0.0), 0.0);
                    case 3 -> Math.min(Math.max(value, 1.0), 1.0);
                    default -> Double.NaN;
                };
                assertDouble(expected, actual[i]);
            }
        }
    }

    @Test
    void integerBroadcastArithmeticAndExtremaNeverRoundInt64ThroughFloatingPoint() {
        for (DataType leftType : List.of(DataType.INT32, DataType.INT64)) {
            for (DataType rightType : List.of(DataType.INT32, DataType.INT64)) {
                DataType resultType = leftType == DataType.INT64 || rightType == DataType.INT64
                        ? DataType.INT64 : DataType.INT32;
                long[] left = leftType == DataType.INT64
                        ? new long[] {Long.MIN_VALUE, 9007199254740993L, Long.MAX_VALUE}
                        : new long[] {Integer.MIN_VALUE, -1, Integer.MAX_VALUE};
                long[] right = rightType == DataType.INT64
                        ? new long[] {-1, 9007199254740992L}
                        : new long[] {-1, Integer.MAX_VALUE};
                var kinds = List.of(MetalMpsGraphProgram.NodeKind.ADD, MetalMpsGraphProgram.NodeKind.SUB,
                        MetalMpsGraphProgram.NodeKind.MUL, MetalMpsGraphProgram.NodeKind.TENSOR_MIN,
                        MetalMpsGraphProgram.NodeKind.TENSOR_MAX);
                var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
                var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
                values.add(typed(leftType, 3, 1)); values.add(typed(rightType, 2));
                int[] targets = new int[kinds.size()];
                for (int k = 0; k < kinds.size(); k++) {
                    targets[k] = k + 2;
                    nodes.add(node(kinds.get(k), new int[] {0, 1}, k + 2));
                    values.add(typed(resultType, 3, 2));
                }
                var outputs = execute(new MetalMpsGraphProgram(nodes), values, new int[] {0, 1}, targets,
                        raw(leftType, left), raw(rightType, right));
                for (int k = 0; k < kinds.size(); k++) {
                    long[] expected = new long[6];
                    for (int i = 0; i < 6; i++) {
                        long a = left[i / 2], b = right[i % 2];
                        expected[i] = switch (k) {
                            case 0 -> a + b;
                            case 1 -> a - b;
                            case 2 -> a * b;
                            case 3 -> Math.min(a, b);
                            case 4 -> Math.max(a, b);
                            default -> throw new AssertionError();
                        };
                    }
                    assertArrayEquals(raw(resultType, expected), outputs.get(k),
                            leftType + "/" + rightType + "/" + kinds.get(k));
                }
            }
        }
    }

    @Test
    void integerScalarOperationsRetainWidthModuloAndExactBoundsIncludingRankZero() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            long lower = type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
            long upper = type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
            long lowerBits = type == DataType.INT32 ? Integer.toUnsignedLong((int) lower) : lower;
            long negativeOneBits = type == DataType.INT32 ? 0xffff_ffffL : -1L;
            for (long[] shape : List.of(new long[0], new long[] {1})) {
                var program = new MetalMpsGraphProgram(List.of(
                        MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_ADD, 0, 1, type, 1),
                        MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_SUB, 0, 2, type, negativeOneBits),
                        MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_MUL, 0, 3, type, negativeOneBits),
                        MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_MIN, 0, 4, type, lowerBits),
                        MetalMpsGraphProgram.Node.scalarValue(MetalMpsGraphProgram.NodeKind.SCALAR_MAX, 0, 5, type, lowerBits),
                        MetalMpsGraphProgram.Node.clamp(0, 6, type, lowerBits, upper)));
                var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
                for (int i = 0; i < 7; i++) values.add(typed(type, shape));
                var outputs = execute(program, values, new int[] {0}, new int[] {1, 2, 3, 4, 5, 6}, raw(type, upper));
                long[] expected = {upper + 1, upper + 1, -upper, lower, upper, upper};
                for (int i = 0; i < expected.length; i++) assertArrayEquals(raw(type, expected[i]), outputs.get(i));
            }
        }
    }

    private static void assertDouble(double expected, long actual) {
        if (Double.isNaN(expected)) assertTrue(Double.isNaN(Double.longBitsToDouble(actual)));
        else assertEquals(Double.doubleToRawLongBits(expected), actual);
    }

    private static MetalMpsGraphProgram.Node node(
            MetalMpsGraphProgram.NodeKind kind, int[] inputs, int output) {
        return MetalMpsGraphProgram.Node.generic(kind, inputs, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(DataType type, long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape, false);
    }

    private static byte[] raw(DataType type, long... words) {
        ByteBuffer bytes = ByteBuffer.allocate(words.length * type.byteWidth()).order(ByteOrder.nativeOrder());
        for (long word : words) {
            switch (type) {
                case FLOAT64, INT64 -> bytes.putLong(word);
                case FLOAT32, INT32 -> bytes.putInt((int) word);
                case BFLOAT16, FLOAT16 -> bytes.putShort((short) word);
                default -> throw new AssertionError(type);
            }
        }
        return bytes.array();
    }

    private static long[] longWords(byte[] raw) {
        ByteBuffer bytes = ByteBuffer.wrap(raw).order(ByteOrder.nativeOrder());
        long[] result = new long[raw.length / Long.BYTES];
        for (int i = 0; i < result.length; i++) result[i] = bytes.getLong();
        return result;
    }

    private static List<byte[]> execute(MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values, int[] feeds, int[] targets,
            byte[]... inputs) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
        MetalNativeApi.Handle context = null, executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context, values, program, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            try (Arena arena = Arena.ofConfined()) {
                for (int i = 0; i < feeds.length; i++) {
                    MemorySegment source = arena.allocate(inputs[i].length, 1);
                    source.copyFrom(MemorySegment.ofArray(inputs[i]));
                    api.upload(buffers.get(feeds[i]), 0, source, inputs[i].length);
                }
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int i = 0; i < buffers.size(); i++) addresses.setAtIndex(ADDRESS, i, buffers.get(i).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int i = 0; i < targets.length; i++) outputs.setAtIndex(ADDRESS, i, buffers.get(targets[i]).carrier());
                List<byte[]> first = null;
                for (int repetition = 0; repetition < 2; repetition++) {
                    api.runExecutable(executable, buffers.size(), addresses, targets.length, outputs);
                    var current = new ArrayList<byte[]>();
                    for (int target : targets) current.add(download(api, buffers.get(target), values.get(target).byteCount()));
                    if (first == null) first = List.copyOf(current);
                    else for (int i = 0; i < first.size(); i++) assertArrayEquals(first.get(i), current.get(i));
                }
                for (int i = 0; i < feeds.length; i++)
                    assertArrayEquals(inputs[i], download(api, buffers.get(feeds[i]), inputs[i].length));
                return first;
            }
        } finally {
            for (int i = buffers.size(); i-- > 0;) api.releaseBuffer(buffers.get(i));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static byte[] download(MetalNativeApi api, MetalNativeApi.Handle buffer, long count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(count, 1);
            api.download(buffer, 0, target, count);
            return target.toArray(JAVA_BYTE);
        }
    }
}
