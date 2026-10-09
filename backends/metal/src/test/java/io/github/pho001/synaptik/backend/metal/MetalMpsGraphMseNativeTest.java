package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class MetalMpsGraphMseNativeTest {
    private static final int POSITIVE_MIN_SUBNORMAL = 0x00000001;
    private static final int NEGATIVE_MIN_SUBNORMAL = 0x80000001;
    private static final int POSITIVE_MIN_NORMAL = 0x00800000;
    private static final int MAX_FINITE = 0x7f7fffff;
    private static final int POSITIVE_INFINITY = 0x7f800000;
    private static final int QUIET_NAN = 0x7fc12345;
    private static final int SIGNALING_NAN = 0x7f800123;

    @Test
    void JavaPreflightClosesMseShapeTypeAndGradientMetadata() {
        Shape tensor = Shape.of(2, 3);
        for (long reduction = 1L; reduction <= 3L; reduction++) {
            Shape outputShape = reduction == 1L ? tensor : Shape.scalar();
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(
                    List.of(mseNode(0, 1, 2, reduction)));
            for (boolean predictionGrad : List.of(false, true)) {
                for (boolean targetGrad : List.of(false, true)) {
                    List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                            descriptor(DataType.FLOAT32, tensor, predictionGrad),
                            descriptor(DataType.FLOAT32, tensor, targetGrad),
                            descriptor(
                                    DataType.FLOAT32,
                                    outputShape,
                                    predictionGrad || targetGrad));
                    MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            values,
                            program,
                            new int[] {0, 1},
                            new int[] {2},
                            MetalPreparedRoute.MPSGRAPH);
                    List<MetalMpsGraphProgram.ValueDescriptor> wrongGradient = List.of(
                            values.get(0),
                            values.get(1),
                            descriptor(
                                    DataType.FLOAT32,
                                    outputShape,
                                    !(predictionGrad || targetGrad)));
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                    wrongGradient,
                                    program,
                                    new int[] {0, 1},
                                    new int[] {2},
                                    MetalPreparedRoute.MPSGRAPH));
                }
            }
        }

        MetalMpsGraphProgram none = new MetalMpsGraphProgram(
                List.of(mseNode(0, 1, 2, 1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        List.of(
                                descriptor(DataType.FLOAT32, Shape.of(2), false),
                                descriptor(DataType.FLOAT32, Shape.of(3), false),
                                descriptor(DataType.FLOAT32, Shape.of(2), false)),
                        none,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.MPSGRAPH));
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        List.of(
                                descriptor(DataType.FLOAT32, Shape.scalar(), false),
                                descriptor(DataType.FLOAT32, Shape.scalar(), false),
                                descriptor(DataType.FLOAT32, Shape.scalar(), false)),
                        none,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.MPSGRAPH));
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        List.of(
                                descriptor(DataType.FLOAT32, Shape.of(2), false),
                                descriptor(DataType.FLOAT64, Shape.of(2), false),
                                descriptor(DataType.FLOAT32, Shape.of(2), false)),
                        none,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.MPSGRAPH));
        assertThrows(
                IllegalArgumentException.class,
                () -> mseNode(0, 1, 2, 4L));

        MetalMpsGraphProgram repeated = new MetalMpsGraphProgram(
                List.of(mseNode(0, 0, 1, 1L)));
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(2), true),
                        descriptor(DataType.FLOAT32, Shape.of(2), true)),
                repeated,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.MPSGRAPH);
    }

    @Test
    void realMseCompositionStaysInsidePrimitiveOracleAcrossReuseAndIsolation() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(
                configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(
                Path.of(configured).toAbsolutePath().normalize());
        try {
            int[] specialPrediction = {
                0x00000000,
                POSITIVE_MIN_SUBNORMAL,
                POSITIVE_MIN_NORMAL,
                MAX_FINITE,
                POSITIVE_INFINITY,
                QUIET_NAN,
                SIGNALING_NAN,
                0x80000000
            };
            int[] specialTarget = {
                0x80000000,
                NEGATIVE_MIN_SUBNORMAL,
                0x00000000,
                0xff7fffff,
                POSITIVE_INFINITY,
                Float.floatToRawIntBits(1.0f),
                Float.floatToRawIntBits(-1.0f),
                0x00000000
            };
            int[] special = runScenario(
                    api, Shape.of(2, 4), specialPrediction, specialTarget);
            int[] ordinary = runScenario(
                    api,
                    Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8),
                    bits(1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f),
                    bits(0.5f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f));
            assertFalse(Arrays.equals(special, ordinary),
                    "independent executable sessions must publish their own inputs");
        } finally {
            api.close();
        }
    }

    @Test
    void realCustomProgramPreservesBothFeedsForNestedMse() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(
                configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(
                Path.of(configured).toAbsolutePath().normalize());
        Shape shape = Shape.of(2, 2);
        int[] predictionBits = bits(1.0f, -2.0f, 3.0f, -4.0f);
        int[] targetBits = bits(0.0f, -1.0f, 5.0f, -8.0f);
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                mseNode(0, 1, 2, 1L),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.RELU,
                        new int[] {2},
                        new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.NONE,
                        new long[0])));
        List<MetalMpsGraphProgram.ValueDescriptor> descriptors = List.of(
                descriptor(DataType.FLOAT32, shape, false),
                descriptor(DataType.FLOAT32, shape, false),
                descriptor(DataType.FLOAT32, shape, false),
                descriptor(DataType.FLOAT32, shape, false));
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context,
            descriptors,
            program,
            new int[] {0, 1},
            new int[] {3},
            MetalPreparedRoute.CUSTOM_PROGRAM);
            for (int index = 0; index < descriptors.size(); index++) {
                buffers.add(api.createBuffer(context, 4L * Integer.BYTES));
            }
            upload(api, buffers.get(0), predictionBits);
            upload(api, buffers.get(1), targetBits);
            upload(api, buffers.get(2), poison(4));
            upload(api, buffers.get(3), poison(4));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputs = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    inputs.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment outputs = arena.allocate(ADDRESS, 1);
                outputs.setAtIndex(ADDRESS, 0, buffers.get(3).carrier());
                api.runExecutable(executable, buffers.size(), inputs, 1, outputs);
            }
            int[] expected = bits(1.0f, 1.0f, 4.0f, 16.0f);
            assertArrayEquals(expected, download(api, buffers.get(2), 4));
            assertArrayEquals(expected, download(api, buffers.get(3), 4));
            assertArrayEquals(predictionBits, download(api, buffers.get(0), 4));
            assertArrayEquals(targetBits, download(api, buffers.get(1), 4));
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static int[] runScenario(
            MetalNativeApi api,
            Shape tensorShape,
            int[] predictionBits,
            int[] targetBits) {
        if (predictionBits.length != targetBits.length) {
            throw new IllegalArgumentException("MSE fixture lengths differ");
        }
        int count = predictionBits.length;
        long shapeElements = 1L;
        for (long dimension : tensorShape.toLongArray()) {
            shapeElements = Math.multiplyExact(shapeElements, dimension);
        }
        if (shapeElements != count) {
            throw new IllegalArgumentException("MSE fixture Shape does not match its data");
        }
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                mseNode(0, 1, 2, 1L),
                mseNode(0, 1, 3, 2L),
                mseNode(0, 1, 4, 3L),
                mseNode(0, 0, 5, 1L)));
        List<MetalMpsGraphProgram.ValueDescriptor> descriptors = List.of(
                descriptor(DataType.FLOAT32, tensorShape, false),
                descriptor(DataType.FLOAT32, tensorShape, false),
                descriptor(DataType.FLOAT32, tensorShape, false),
                descriptor(DataType.FLOAT32, Shape.scalar(), false),
                descriptor(DataType.FLOAT32, Shape.scalar(), false),
                descriptor(DataType.FLOAT32, tensorShape, false));
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context,
            descriptors,
            program,
            new int[] {0, 1},
            new int[] {2, 3, 4, 5},
            MetalPreparedRoute.MPSGRAPH);
            for (int elements : new int[] {count, count, count, 1, 1, count}) {
                buffers.add(api.createBuffer(context, (long) elements * Integer.BYTES));
            }
            upload(api, buffers.get(0), predictionBits);
            upload(api, buffers.get(1), targetBits);
            Allowed[] lanes = new Allowed[count];
            for (int lane = 0; lane < count; lane++) {
                lanes[lane] = squaredDifference(predictionBits[lane], targetBits[lane]);
            }
            Allowed unreleasedSum = aggregate(lanes);
            Allowed sum = withFinalZeroSignFreedom(unreleasedSum);
            Allowed mean = withFinalZeroSignFreedom(
                    divideByExactPositiveCount(unreleasedSum, count));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputs = arena.allocate(ADDRESS, 2);
                inputs.setAtIndex(ADDRESS, 0, buffers.get(0).carrier());
                inputs.setAtIndex(ADDRESS, 1, buffers.get(1).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, 4);
                for (int output = 0; output < 4; output++) {
                    outputs.setAtIndex(ADDRESS, output, buffers.get(output + 2).carrier());
                }
                int[] published = null;
                for (int repetition = 0; repetition < 2; repetition++) {
                    upload(api, buffers.get(2), poison(count));
                    upload(api, buffers.get(3), poison(1));
                    upload(api, buffers.get(4), poison(1));
                    upload(api, buffers.get(5), poison(count));
                    api.runExecutable(executable, 2, inputs, 4, outputs);
                    int[] none = download(api, buffers.get(2), count);
                    int[] repeated = download(api, buffers.get(5), count);
                    for (int lane = 0; lane < count; lane++) {
                        assertAllowed(
                                lanes[lane],
                                none[lane],
                                "NONE repetition=" + repetition + " lane=" + lane);
                        assertAllowed(
                                squaredDifference(predictionBits[lane], predictionBits[lane]),
                                repeated[lane],
                                "repeated input repetition=" + repetition + " lane=" + lane);
                    }
                    assertAllowed(
                            sum,
                            download(api, buffers.get(3), 1)[0],
                            "SUM repetition=" + repetition);
                    assertAllowed(
                            mean,
                            download(api, buffers.get(4), 1)[0],
                            "MEAN repetition=" + repetition);
                    assertArrayEquals(
                            predictionBits,
                            download(api, buffers.get(0), count),
                            "prediction preservation repetition=" + repetition);
                    assertArrayEquals(
                            targetBits,
                            download(api, buffers.get(1), count),
                            "target preservation repetition=" + repetition);
                    published = none;
                }
                return published;
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
        }
    }

    private static MetalMpsGraphProgram.Node mseNode(
            int prediction, int target, int output, long reduction) {
        return MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                new int[] {prediction, target},
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.MSE,
                new long[] {reduction});
    }

    private static MetalMpsGraphProgram.ValueDescriptor descriptor(
            DataType type, Shape shape, boolean requiresGrad) {
        return new MetalMpsGraphProgram.ValueDescriptor(
                type,
                shape.toLongArray(),
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad,
                false);
    }

    private static Allowed squaredDifference(int prediction, int target) {
        Allowed difference =
                primitive(Binary.SUB, Allowed.single(prediction), Allowed.single(target));
        Set<Integer> bits = new HashSet<>();
        boolean nan = difference.nan();
        for (int declared : difference.bits()) {
            for (int selectedLeft : dazChoices(declared)) {
                for (int selectedRight : dazChoices(declared)) {
                    int result = evaluate(Binary.MUL, selectedLeft, selectedRight);
                    if (isNaN(result)) {
                        nan = true;
                    } else {
                        bits.add(result);
                        addFtzChoices(bits, result);
                    }
                }
            }
        }
        return new Allowed(Set.copyOf(bits), nan);
    }

    private static Allowed aggregate(Allowed[] contributors) {
        if (contributors.length == 0 || contributors.length > 20) {
            throw new IllegalArgumentException("bounded aggregate contributor count");
        }
        int all = (1 << contributors.length) - 1;
        Map<Integer, Allowed> partials = new HashMap<>();
        for (int index = 0; index < contributors.length; index++) {
            partials.put(1 << index, contributors[index]);
        }
        for (int size = 2; size <= contributors.length; size++) {
            for (int mask = 1; mask <= all; mask++) {
                if (Integer.bitCount(mask) != size) continue;
                Allowed merged = Allowed.empty();
                for (int leftMask = (mask - 1) & mask;
                        leftMask != 0;
                        leftMask = (leftMask - 1) & mask) {
                    int rightMask = mask ^ leftMask;
                    if (rightMask == 0) continue;
                    Allowed left = partials.get(leftMask);
                    Allowed right = partials.get(rightMask);
                    if (left != null && right != null) {
                        merged = merged.union(primitive(Binary.ADD, left, right));
                    }
                }
                partials.put(mask, merged);
            }
        }
        return partials.get(all);
    }

    private static Allowed divideByExactPositiveCount(Allowed sum, int count) {
        return primitive(
                Binary.DIV,
                sum,
                Allowed.single(Float.floatToRawIntBits((float) count)));
    }

    private static Allowed withFinalZeroSignFreedom(Allowed allowed) {
        if (!allowed.bits().contains(0x00000000)
                && !allowed.bits().contains(0x80000000)) {
            return allowed;
        }
        var bits = new HashSet<>(allowed.bits());
        bits.add(0x00000000);
        bits.add(0x80000000);
        return new Allowed(Set.copyOf(bits), allowed.nan());
    }

    private static Allowed primitive(Binary operation, Allowed left, Allowed right) {
        Set<Integer> bits = new HashSet<>();
        boolean nan = left.nan() || right.nan();
        for (int declaredLeft : left.bits()) {
            for (int declaredRight : right.bits()) {
                for (int selectedLeft : dazChoices(declaredLeft)) {
                    for (int selectedRight : dazChoices(declaredRight)) {
                        int result = evaluate(operation, selectedLeft, selectedRight);
                        if (isNaN(result)) {
                            nan = true;
                        } else {
                            bits.add(result);
                            addFtzChoices(bits, result);
                        }
                    }
                }
            }
        }
        return new Allowed(Set.copyOf(bits), nan);
    }

    private static void addFtzChoices(Set<Integer> bits, int result) {
        if (isFiniteSubnormal(result)) {
            bits.add(0x00000000);
            bits.add(0x80000000);
        }
    }

    private static int[] dazChoices(int bits) {
        return isFiniteSubnormal(bits)
                ? new int[] {bits, bits & 0x80000000}
                : new int[] {bits};
    }

    private static int evaluate(Binary operation, int leftBits, int rightBits) {
        float left = Float.intBitsToFloat(leftBits);
        float right = Float.intBitsToFloat(rightBits);
        float value = switch (operation) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
        };
        return Float.floatToRawIntBits(value);
    }

    private static void assertAllowed(Allowed allowed, int observed, String context) {
        boolean accepted = isNaN(observed) ? allowed.nan() : allowed.bits().contains(observed);
        assertTrue(
                accepted,
                context + " observed=0x" + Integer.toHexString(observed)
                        + " allowed=" + allowed.bits()
                        + " nan=" + allowed.nan());
    }

    private static boolean isFiniteSubnormal(int bits) {
        return (bits & 0x7f800000) == 0 && (bits & 0x007fffff) != 0;
    }

    private static boolean isNaN(int bits) {
        return (bits & 0x7f800000) == 0x7f800000
                && (bits & 0x007fffff) != 0;
    }

    private static int[] bits(float... values) {
        int[] result = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = Float.floatToRawIntBits(values[index]);
        }
        return result;
    }

    private static int[] poison(int count) {
        int[] result = new int[count];
        Arrays.fill(result, 0xdeadbeef);
        return result;
    }

    private static void upload(MetalNativeApi api, MetalNativeApi.Handle buffer, int[] bits) {
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

    private enum Binary {
        ADD,
        SUB,
        MUL,
        DIV
    }

    private record Allowed(Set<Integer> bits, boolean nan) {
        private static Allowed empty() {
            return new Allowed(Set.of(), false);
        }

        private static Allowed single(int bits) {
            return isNaN(bits)
                    ? new Allowed(Set.of(), true)
                    : new Allowed(Set.of(bits), false);
        }

        private Allowed union(Allowed other) {
            var merged = new HashSet<>(bits);
            merged.addAll(other.bits);
            return new Allowed(Set.copyOf(merged), nan || other.nan);
        }
    }
}
