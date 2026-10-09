package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Real-device qualification for direct FLOAT32 and independent custom low EXP routes. */
final class MetalExpNativeTest {
    private static final double MAX_RELATIVE_ERROR = 2e-6;
    private static final double BF16_MAX_RELATIVE_ERROR = 0.0040;
    private static final double F16_MAX_RELATIVE_ERROR = 0.00055;

    @Test
    void javaPreflightClosesTypeShapeLayoutGradientAndRoute() {
        Shape shape = Shape.of(7);
        MetalMpsGraphProgram program = program();
        List<MetalMpsGraphProgram.ValueDescriptor> valid =
                List.of(value(DataType.FLOAT32, shape, false),
                        value(DataType.FLOAT32, shape, false));
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                valid, program, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
        List<List<MetalMpsGraphProgram.ValueDescriptor>> rejectedCases = List.of(
                List.of(value(DataType.BFLOAT16, shape, false), valid.get(1)),
                List.of(value(DataType.FLOAT16, shape, false), valid.get(1)),
                List.of(value(DataType.FLOAT64, shape, false), valid.get(1)),
                List.of(valid.get(0), value(DataType.FLOAT32, Shape.of(8), false)),
                List.of(value(DataType.FLOAT32, Shape.scalar(), false),
                        value(DataType.FLOAT32, Shape.scalar(), false)),
                List.of(value(DataType.FLOAT32, shape, true), valid.get(1)),
                List.of(valid.get(0), value(DataType.FLOAT32, shape, true)),
                List.of(new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, shape.toLongArray(), Optional.empty(), false, false),
                        valid.get(1)),
                List.of(valid.get(0), new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, shape.toLongArray(), Optional.empty(), false, false)));
        for (int rejectedCase = 0; rejectedCase < rejectedCases.size(); rejectedCase++) {
            List<MetalMpsGraphProgram.ValueDescriptor> rejected = rejectedCases.get(rejectedCase);
            assertThrows(IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            rejected, program, new int[] {0}, new int[] {1},
                            MetalPreparedRoute.MPSGRAPH), "negative case " + rejectedCase);
        }
        assertThrows(IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        valid, program, new int[] {0}, new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM),
                "standalone FLOAT32 EXP cannot claim a custom whole-partition route");
        Shape huge = Shape.of(1_073_741_824L);
        assertThrows(IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        List.of(value(DataType.FLOAT32, huge, false),
                                value(DataType.FLOAT32, huge, false)),
                        program, new int[] {0}, new int[] {1},
                        MetalPreparedRoute.MPSGRAPH));
    }

    @Test
    void javaPreflightAllowsExplicitLowCastThenFloat32ExpInCustomPartition() {
        Shape shape = Shape.of(4);
        var cast = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CAST,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                new long[] {MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)});
        var exp = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.EXP,
                new int[] {1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        var composite = new MetalMpsGraphProgram(List.of(cast, exp));
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    value(low, shape, false),
                    value(DataType.FLOAT32, shape, false),
                    value(DataType.FLOAT32, shape, false));
            MetalNativeApi.ProgramExecutableAbi.validateCreate(
                    values, composite, new int[] {0}, new int[] {2},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertThrows(IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            values, composite, new int[] {0}, new int[] {2},
                            MetalPreparedRoute.MPSGRAPH), low.toString());
        }
    }

    @Test
    void javaPreflightAdmitsOnlySameTypeCanonicalLowCustomExp() {
        Shape shape = Shape.of(7);
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var value = value(low, shape, false);
            var valid = List.of(value, value);
            MetalNativeApi.ProgramExecutableAbi.validateCreate(valid, program(),
                    new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM);
            assertThrows(IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(valid, program(),
                            new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH));
            DataType other = low == DataType.BFLOAT16
                    ? DataType.FLOAT16 : DataType.BFLOAT16;
            for (var invalid : List.of(
                    List.of(value, value(other, shape, false)),
                    List.of(value, value(DataType.FLOAT32, shape, false)),
                    List.of(value(low, shape, true), value),
                    List.of(value, value(low, shape, true)),
                    List.of(value, value(low, Shape.of(8), false)),
                    List.of(value(low, Shape.scalar(), false),
                            value(low, Shape.scalar(), false)),
                    List.of(new MetalMpsGraphProgram.ValueDescriptor(low,
                            shape.toLongArray(), Optional.empty(), false, false), value))) {
                assertThrows(IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(invalid,
                                program(), new int[] {0}, new int[] {1},
                                MetalPreparedRoute.CUSTOM_PROGRAM), low + " " + invalid);
            }
            Shape huge = Shape.of(2_147_483_648L);
            assertThrows(IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            List.of(value(low, huge, false), value(low, huge, false)),
                            program(), new int[] {0}, new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
        }
    }

    @Test
    void lowCustomExponentQualifiesBothTypesOnRealDevice() {
        Path library = Path.of(System.getenv("SYNAPTIK_METAL_TEST_LIBRARY"))
                .toAbsolutePath().normalize();
        try (MetalNativeApi api = MetalNativeApi.open(library)) {
            MetalNativeApi.Handle context = api.createContext();
            try {
                for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    short[] boundary = low == DataType.BFLOAT16
                            ? lowBits(low, new float[] {
                                    0.0f, -0.0f, Float.NEGATIVE_INFINITY,
                                    Float.POSITIVE_INFINITY, Float.NaN,
                                    -100.0f, -93.5f, -93.0f, -92.5f, -90.0f,
                                    -88.0f, 12.0f, 88.0f, 88.5f, 89.0f,
                                    -1.0f, 1.0f})
                            : lowBits(low, new float[] {
                                    0.0f, -0.0f, Float.NEGATIVE_INFINITY,
                                    Float.POSITIVE_INFINITY, Float.NaN,
                                    -20.0f, -18.0f, -17.5f, -17.0f, -12.0f,
                                    12.0f, 11.0f, 11.0625f, 11.09375f,
                                    -1.0f, 1.0f});
                    boundary[4] = (short) (low == DataType.BFLOAT16
                            ? 0x7fc1 : 0x7e01);
                    runLowCase(api, context, low, boundary);
                    runLowCase(api, context, low, new short[] {1, (short) 0x8001});
                    short[] ordinary = new short[8191];
                    for (int index = 0; index < ordinary.length; index++) {
                        float input = ((index * 67) % 1601 - 800) * 0.01f;
                        ordinary[index] = toLow(low, input);
                    }
                    runLowCase(api, context, low, ordinary);
                    short[] multidimensional = new short[7 * 17];
                    for (int index = 0; index < multidimensional.length; index++) {
                        multidimensional[index] = toLow(low,
                                ((index * 23) % 101 - 50) * 0.125f);
                    }
                    runLowCase(api, context, low, Shape.of(7, 17), multidimensional);
                }
            } finally {
                api.releaseContext(context);
            }
        }
    }

    private static short[] lowBits(DataType low, float[] inputs) {
        short[] result = new short[inputs.length];
        for (int index = 0; index < result.length; index++) {
            result[index] = toLow(low, inputs[index]);
        }
        return result;
    }

    private static short toLow(DataType low, float input) {
        return low == DataType.BFLOAT16
                ? BFloat16Bits.fromFloat(input) : Float16Bits.fromFloat(input);
    }

    private static float widen(DataType low, short input) {
        return low == DataType.BFLOAT16
                ? BFloat16Bits.toFloat(input) : Float16Bits.toFloat(input);
    }

    private static void runLowCase(MetalNativeApi api, MetalNativeApi.Handle context,
            DataType low, short[] samples) {
        runLowCase(api, context, low, Shape.of(samples.length), samples);
    }

    private static void runLowCase(MetalNativeApi api, MetalNativeApi.Handle context,
            DataType low, Shape shape, short[] samples) {
        MetalNativeApi.Handle executable = api.createProgramExecutable(context,
                List.of(value(low, shape, false), value(low, shape, false)),
                program(), new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM);
        long bytes = Math.multiplyExact((long) samples.length, Short.BYTES);
        MetalNativeApi.Handle input = api.createBuffer(context, bytes);
        MetalNativeApi.Handle output = api.createBuffer(context, bytes);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes, Short.BYTES);
            MemorySegment destination = arena.allocate(bytes, Short.BYTES);
            MemorySegment inputs = arena.allocate(ADDRESS, 2);
            MemorySegment outputs = arena.allocate(ADDRESS);
            for (int index = 0; index < samples.length; index++) {
                source.setAtIndex(JAVA_SHORT, index, samples[index]);
                destination.setAtIndex(JAVA_SHORT, index, (short) 0xdead);
            }
            api.upload(input, 0L, source, bytes);
            api.upload(output, 0L, destination, bytes);
            inputs.setAtIndex(ADDRESS, 0, input.carrier());
            inputs.setAtIndex(ADDRESS, 1, input.carrier());
            outputs.setAtIndex(ADDRESS, 0, input.carrier());
            assertThrows(RuntimeException.class,
                    () -> api.runExecutable(executable, 2, inputs, 1, outputs),
                    low + " alias must fail before mutation");
            api.download(input, 0L, destination, bytes);
            for (int index = 0; index < samples.length; index++) {
                assertEquals(samples[index], destination.getAtIndex(JAVA_SHORT, index));
            }
            api.download(output, 0L, destination, bytes);
            for (int index = 0; index < samples.length; index++) {
                assertEquals((short) 0xdead, destination.getAtIndex(JAVA_SHORT, index),
                        low + " untouched output " + index);
            }
            inputs.setAtIndex(ADDRESS, 1, output.carrier());
            outputs.setAtIndex(ADDRESS, 0, output.carrier());
            double worstError = 0.0;
            float worstInput = 0.0f;
            for (int repeat = 0; repeat < 2; repeat++) {
                api.runExecutable(executable, 2, inputs, 1, outputs);
                api.download(output, 0L, destination, bytes);
                for (int index = 0; index < samples.length; index++) {
                    float sample = widen(low, samples[index]);
                    short actual = destination.getAtIndex(JAVA_SHORT, index);
                    double error = assertLowExp(low, sample, actual);
                    if (repeat == 0 && (samples.length == 16 || samples.length == 17)) {
                        short rounded = toLow(low,
                                (float) StrictMath.exp((double) sample));
                        System.out.println("LOW_EXP_BOUNDARY " + low
                                + " input=" + sample
                                + " actualBits=0x" + Integer.toHexString(
                                        Short.toUnsignedInt(actual))
                                + " roundedBits=0x" + Integer.toHexString(
                                        Short.toUnsignedInt(rounded)));
                    }
                    if (error > worstError) {
                        worstError = error;
                        worstInput = sample;
                    }
                }
            }
            System.out.println("LOW_EXP " + low + " lanes=" + samples.length
                    + " worstOrdinaryRelativeError=" + worstError
                    + " input=" + worstInput);
            api.download(input, 0L, destination, bytes);
            for (int index = 0; index < samples.length; index++) {
                assertEquals(samples[index], destination.getAtIndex(JAVA_SHORT, index),
                        low + " input " + index);
            }
        } finally {
            api.releaseBuffer(output);
            api.releaseBuffer(input);
            api.releaseExecutable(executable);
        }
    }

    private static double assertLowExp(DataType low, float input, short actualBits) {
        int bits = Short.toUnsignedInt(actualBits);
        float actual = widen(low, actualBits);
        if (Float.isNaN(input)) {
            assertTrue(Float.isNaN(actual), low + " NaN class");
            return 0.0;
        }
        if (input == 0.0f) {
            assertEquals(Short.toUnsignedInt(toLow(low, 1.0f)), bits,
                    low + " signed-zero input");
            return 0.0;
        }
        if (input == Float.NEGATIVE_INFINITY) {
            assertEquals(0, bits, low + " negative infinity");
            return 0.0;
        }
        if (input == Float.POSITIVE_INFINITY) {
            assertEquals(low == DataType.BFLOAT16 ? 0x7f80 : 0x7c00, bits,
                    low + " positive infinity");
            return 0.0;
        }
        double reference = StrictMath.exp((double) input);
        short expectedBits = toLow(low, (float) reference);
        float expected = widen(low, expectedBits);
        double q = low == DataType.BFLOAT16
                ? BFloat16Bits.toFloat((short) 1) : Float16Bits.toFloat((short) 1);
        float minimumNormal = low == DataType.BFLOAT16
                ? BFloat16Bits.toFloat((short) 0x0080)
                : Float16Bits.toFloat((short) 0x0400);
        if (Float.isInfinite(expected)) {
            assertEquals(low == DataType.BFLOAT16 ? 0x7f80 : 0x7c00, bits,
                    low + " overflow input=" + input);
            return 0.0;
        }
        if (expected == 0.0f) {
            boolean boundary = reference >= 0.4 * q && reference < 0.5 * q;
            assertTrue(bits == 0 || boundary && bits == 1,
                    low + " underflow input=" + input + " reference=" + reference
                            + " actualBits=" + bits);
            return 0.0;
        }
        if (expected < minimumNormal) {
            assertTrue(bits == 0 && low == DataType.BFLOAT16
                            || actual > 0.0f && actual < minimumNormal
                                    && Math.abs((double) actual - expected) <= q,
                    low + " subnormal input=" + input + " reference=" + reference
                            + " expected=" + expected + " actual=" + actual);
            if (low == DataType.FLOAT16 && input == -12.0f) {
                assertTrue(bits > 0 && bits < 0x0400,
                        "FLOAT16 EXP(-12) must remain positive subnormal");
            }
            return 0.0;
        }
        assertTrue(Float.isFinite(actual) && actual >= minimumNormal,
                low + " ordinary class input=" + input + " actual=" + actual);
        double error = Math.abs((double) actual - reference) / reference;
        assertTrue(error <= (low == DataType.BFLOAT16
                        ? BF16_MAX_RELATIVE_ERROR : F16_MAX_RELATIVE_ERROR),
                low + " ordinary error input=" + input + " actual=" + actual
                        + " reference=" + reference + " relativeError=" + error);
        return error;
    }

    @Test
    void directExponentQualifiesFiniteAndSpecialClassesAcrossSizesAndReuse() {
        Path library = Path.of(System.getenv("SYNAPTIK_METAL_TEST_LIBRARY"))
                .toAbsolutePath().normalize();
        try (MetalNativeApi api = MetalNativeApi.open(library)) {
            MetalNativeApi.Handle context = api.createContext();
            try {
                runCase(api, context, new float[] {0.25f});
                runCase(api, context, new float[] {
                        -2.0f, -1.0f, -0.25f, 0.125f, 0.5f, 1.0f, 3.0f});
                float[] medium = new float[257];
                float[] large = new float[8191];
                for (int index = 0; index < medium.length; index++) {
                    medium[index] = (index - 128) * 0.625f;
                }
                for (int index = 0; index < large.length; index++) {
                    large[index] = ((index * 67) % 1601 - 800) * 0.1f;
                }
                runCase(api, context, medium);
                runCase(api, context, large);
                runCase(api, context, new float[] {
                        0.0f, -0.0f, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY,
                        Float.NaN, Float.intBitsToFloat(0x7f812345),
                        -110.0f, -104.0f, -103.3f, -103.0f, -100.0f,
                        -90.0f, -87.4f, -87.3f, 88.7f, 88.7228f, 88.73f, 90.0f,
                        Float.intBitsToFloat(1), -Float.intBitsToFloat(1)});
            } finally {
                api.releaseContext(context);
            }
        }
    }

    private static void runCase(
            MetalNativeApi api, MetalNativeApi.Handle context, float[] samples) {
        Shape shape = Shape.of(samples.length);
        MetalNativeApi.Handle executable = api.createProgramExecutable(
                context,
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
            MemorySegment inputs = arena.allocate(ADDRESS);
            MemorySegment outputs = arena.allocate(ADDRESS);
            for (int index = 0; index < samples.length; index++) {
                source.setAtIndex(JAVA_INT, index, original[index]);
                destination.setAtIndex(JAVA_INT, index, 0xdeadbeef);
            }
            api.upload(input, 0L, source, bytes);
            api.upload(output, 0L, destination, bytes);
            inputs.setAtIndex(ADDRESS, 0, input.carrier());
            outputs.setAtIndex(ADDRESS, 0, input.carrier());
            assertThrows(RuntimeException.class,
                    () -> api.runExecutable(executable, 1, inputs, 1, outputs));
            api.download(input, 0L, destination, bytes);
            for (int index = 0; index < samples.length; index++) {
                assertEquals(original[index], destination.getAtIndex(JAVA_INT, index));
            }
            outputs.setAtIndex(ADDRESS, 0, output.carrier());
            for (int repeat = 0; repeat < 3; repeat++) {
                api.runExecutable(executable, 1, inputs, 1, outputs);
                api.download(output, 0L, destination, bytes);
                for (int index = 0; index < samples.length; index++) {
                    assertExp(samples[index], destination.getAtIndex(JAVA_INT, index));
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

    private static void assertExp(float input, int actualBits) {
        float actual = Float.intBitsToFloat(actualBits);
        if (Float.isNaN(input)) {
            assertTrue(Float.isNaN(actual), "EXP NaN class");
            return;
        }
        if (input == 0.0f) {
            assertEquals(0x3f800000, actualBits, "EXP signed-zero input");
            return;
        }
        if (input == Float.NEGATIVE_INFINITY) {
            assertEquals(0, actualBits, "EXP negative infinity");
            return;
        }
        if (input == Float.POSITIVE_INFINITY) {
            assertEquals(0x7f800000, actualBits, "EXP positive infinity");
            return;
        }
        float reference = (float) StrictMath.exp((double) input);
        if (Float.isInfinite(reference)) {
            assertEquals(0x7f800000, actualBits, "EXP genuine overflow " + input);
        } else if (reference == 0.0f) {
            assertTrue(actualBits >= 0 && actualBits <= 4,
                    "EXP underflow " + input + " actual=" + actual);
        } else if (reference < Float.MIN_NORMAL) {
            assertTrue(actualBits == 0
                            || actualBits > 0 && actualBits < 0x00800000
                                    && Math.abs((long) actualBits
                                            - Float.floatToRawIntBits(reference)) <= 4,
                    "EXP subnormal " + input + " actual=" + actual);
        } else {
            assertTrue(Float.isFinite(actual) && actual > 0.0f,
                    "EXP ordinary class " + input + " actual=" + actual);
            assertTrue(Math.abs((double) actual - reference) / Math.abs(reference)
                            <= MAX_RELATIVE_ERROR,
                    "EXP ordinary finite relative error " + input + " actual=" + actual
                            + " reference=" + reference);
        }
    }

    private static MetalMpsGraphProgram program() {
        return new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.EXP, new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0])));
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(
            DataType type, Shape shape, boolean grad) {
        return MetalMpsGraphProgram.ValueDescriptor.from(new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }
}
