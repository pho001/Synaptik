package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Source-matched real-device qualification of the FLOAT32 sign-guarded SIGMOID route. */
final class MetalSigmoidNativeTest {
    private static final double MAX_RELATIVE_ERROR = 2e-6;
    private static final double SUBNORMAL_ABSOLUTE_FLOOR = 4.0 * Float.MIN_VALUE;

    @Test
    void preflightAdmitsOnlyCanonicalNoGradientFloat32OnFixedRoutes() {
        Shape shape = Shape.of(7);
        var valid = List.of(value(DataType.FLOAT32, shape, false),
                value(DataType.FLOAT32, shape, false));
        assertThrows(IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(valid, program(),
                        new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM),
                "a standalone FLOAT32 node does not justify a custom partition route");
        for (MetalPreparedRoute route : List.of(MetalPreparedRoute.MPSGRAPH)) {
            MetalNativeApi.ProgramExecutableAbi.validateCreate(valid, program(),
                    new int[] {0}, new int[] {1}, route);
            for (DataType wrong : List.of(DataType.BFLOAT16, DataType.FLOAT16,
                    DataType.FLOAT64)) {
                assertThrows(IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                List.of(value(wrong, shape, false),
                                        value(wrong, shape, false)),
                                program(), new int[] {0}, new int[] {1}, route));
            }
            for (var invalid : List.of(
                    List.of(value(DataType.FLOAT32, shape, true), valid.get(1)),
                    List.of(valid.get(0), value(DataType.FLOAT32, shape, true)),
                    List.of(valid.get(0), value(DataType.FLOAT32, Shape.of(8), false)),
                    List.of(value(DataType.FLOAT32, Shape.scalar(), false),
                            value(DataType.FLOAT32, Shape.scalar(), false)),
                    List.of(new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                            shape.toLongArray(), Optional.empty(), false, false), valid.get(1)))) {
                assertThrows(IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(invalid,
                                program(), new int[] {0}, new int[] {1}, route));
            }
            Shape huge = Shape.of(1_073_741_824L);
            assertThrows(IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            List.of(value(DataType.FLOAT32, huge, false),
                                    value(DataType.FLOAT32, huge, false)),
                            program(), new int[] {0}, new int[] {1}, route));
        }
    }

    @Test
    void explicitLowCastPermitsFloat32BoundaryStepWithoutLowSigmoid() {
        Shape shape = Shape.of(4);
        var cast = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.CAST,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                new long[] {MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)});
        var sigmoid = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SIGMOID,
                new int[] {1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        var composite = new MetalMpsGraphProgram(List.of(cast, sigmoid));
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var values = List.of(value(low, shape, false),
                    value(DataType.FLOAT32, shape, false),
                    value(DataType.FLOAT32, shape, false));
            MetalNativeApi.ProgramExecutableAbi.validateCreate(values, composite,
                    new int[] {0}, new int[] {2}, MetalPreparedRoute.CUSTOM_PROGRAM);
            assertThrows(IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(values, composite,
                            new int[] {0}, new int[] {2}, MetalPreparedRoute.MPSGRAPH));
        }
    }

    @Test
    void deviceQualifiesSpecialFiniteBoundaryAndRepeatedRuns() {
        Path library = Path.of(System.getenv("SYNAPTIK_METAL_TEST_LIBRARY"))
                .toAbsolutePath().normalize();
        try (MetalNativeApi api = MetalNativeApi.open(library)) {
            MetalNativeApi.Handle context = api.createContext();
            try {
                runCase(api, context, new float[] {0.25f});
                runCase(api, context, new float[] {
                        -Float.MAX_VALUE, -110.0f, -104.0f, -103.3f, -103.0f,
                        -100.0f, -90.0f, -87.4f, -87.3f,
                        Float.intBitsToFloat(0x80000001), Float.intBitsToFloat(0x80000002),
                        -0.0f, 0.0f, Float.intBitsToFloat(1), Float.intBitsToFloat(2),
                        -1e-7f, 1e-7f, -1.0f, 1.0f, -17.0f, 17.0f,
                        18.0f, 80.0f, Float.MAX_VALUE,
                        Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY,
                        Float.NaN, Float.intBitsToFloat(0x7f812345)});
                float[] ordinary = new float[8191];
                for (int index = 0; index < ordinary.length; index++) {
                    ordinary[index] = ((index * 67) % 1601 - 800) * 0.02f;
                }
                runCase(api, context, ordinary);
            } finally {
                api.releaseContext(context);
            }
        }
    }

    private static void runCase(MetalNativeApi api, MetalNativeApi.Handle context,
            float[] samples) {
        Shape shape = Shape.of(samples.length);
        MetalNativeApi.Handle executable = api.createProgramExecutable(context,
                List.of(value(DataType.FLOAT32, shape, false),
                        value(DataType.FLOAT32, shape, false)),
                program(), new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
        long bytes = Math.multiplyExact((long) samples.length, Float.BYTES);
        MetalNativeApi.Handle input = api.createBuffer(context, bytes);
        MetalNativeApi.Handle output = api.createBuffer(context, bytes);
        int[] original = new int[samples.length];
        for (int index = 0; index < samples.length; index++) {
            original[index] = Float.floatToRawIntBits(samples[index]);
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes, Integer.BYTES);
            MemorySegment destination = arena.allocate(bytes, Integer.BYTES);
            MemorySegment feeds = arena.allocate(ADDRESS);
            MemorySegment targets = arena.allocate(ADDRESS);
            for (int index = 0; index < samples.length; index++) {
                source.setAtIndex(JAVA_INT, index, original[index]);
                destination.setAtIndex(JAVA_INT, index, 0xdeadbeef);
            }
            api.upload(input, 0L, source, bytes);
            api.upload(output, 0L, destination, bytes);
            feeds.setAtIndex(ADDRESS, 0, input.carrier());
            targets.setAtIndex(ADDRESS, 0, input.carrier());
            assertThrows(RuntimeException.class,
                    () -> api.runExecutable(executable, 1, feeds, 1, targets));
            api.download(input, 0L, destination, bytes);
            for (int index = 0; index < samples.length; index++) {
                assertEquals(original[index], destination.getAtIndex(JAVA_INT, index),
                        "rejected alias input " + index);
            }
            api.download(output, 0L, destination, bytes);
            for (int index = 0; index < samples.length; index++) {
                assertEquals(0xdeadbeef, destination.getAtIndex(JAVA_INT, index),
                        "rejected alias output " + index);
            }
            targets.setAtIndex(ADDRESS, 0, output.carrier());
            for (int repeat = 0; repeat < 3; repeat++) {
                api.runExecutable(executable, 1, feeds, 1, targets);
                api.download(output, 0L, destination, bytes);
                for (int index = 0; index < samples.length; index++) {
                    assertSigmoid(samples[index], destination.getAtIndex(JAVA_INT, index));
                }
                api.download(input, 0L, destination, bytes);
                int[] preserved = new int[samples.length];
                for (int index = 0; index < samples.length; index++) {
                    preserved[index] = destination.getAtIndex(JAVA_INT, index);
                }
                assertArrayEquals(original, preserved);
            }
        } finally {
            api.releaseBuffer(output);
            api.releaseBuffer(input);
            api.releaseExecutable(executable);
        }
    }

    private static void assertSigmoid(float input, int actualBits) {
        float actual = Float.intBitsToFloat(actualBits);
        if (Float.isNaN(input)) {
            assertTrue(Float.isNaN(actual), "NaN class input=" + input);
            return;
        }
        if (input == 0.0f) {
            assertEquals(0x3f000000, actualBits, "signed zero input=" + input);
            return;
        }
        if (input == Float.NEGATIVE_INFINITY) {
            assertEquals(0, actualBits, "negative infinity");
            return;
        }
        if (input == Float.POSITIVE_INFINITY) {
            assertEquals(0x3f800000, actualBits, "positive infinity");
            return;
        }
        double exponent = StrictMath.exp(input < 0.0f ? (double) input : -(double) input);
        double reference = input < 0.0f
                ? exponent / (1.0 + exponent) : 1.0 / (1.0 + exponent);
        if (reference < Float.MIN_NORMAL) {
            assertTrue(actualBits == 0 || actualBits > 0
                            && actual < Float.MIN_NORMAL
                            && Math.abs(actual - reference)
                                    <= Math.max(SUBNORMAL_ABSOLUTE_FLOOR,
                                            MAX_RELATIVE_ERROR * reference),
                    "underflow input=" + input + " actual=" + actual
                            + " reference=" + reference);
            return;
        }
        if (1.0 - reference <= 2.0 * Math.ulp(1.0f)) {
            assertTrue(actual > 0.0f && actual <= 1.0f
                            && Math.abs(actual - reference) <= 2.0 * Math.ulp(1.0f),
                    "saturation input=" + input + " actual=" + actual
                            + " reference=" + reference);
            return;
        }
        assertTrue(Float.isFinite(actual) && actual > 0.0f && actual < 1.0f,
                "ordinary class input=" + input + " actual=" + actual);
        double error = Math.abs(actual - reference) / reference;
        assertTrue(error <= MAX_RELATIVE_ERROR,
                "finite input=" + input + " actual=" + actual
                        + " reference=" + reference + " relativeError=" + error);
    }

    private static MetalMpsGraphProgram program() {
        return new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SIGMOID,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0])));
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(
            DataType type, Shape shape, boolean grad) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape.toLongArray(),
                Optional.of(LayoutDescriptor.contiguous(shape)), grad, false);
    }
}
