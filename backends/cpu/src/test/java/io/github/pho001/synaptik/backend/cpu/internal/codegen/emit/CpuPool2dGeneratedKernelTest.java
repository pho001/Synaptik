package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuPool2dReferenceKernel;
import io.github.pho001.synaptik.model.datatype.*;
import java.lang.classfile.*;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.*;
import org.junit.jupiter.api.Test;

class CpuPool2dGeneratedKernelTest {
    @Test
    void arraysPreserveAllTypeAndSpecialValueSemantics() throws Throwable {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64))
            for (CpuPool2dIr.Kind kind : CpuPool2dIr.Kind.values()) checkArray(type, kind);
    }

    @Test
    void segmentsAndMixedCarriersHonorGeneralStridesAndSplitRanges() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
                for (CpuPool2dIr.Kind kind : CpuPool2dIr.Kind.values()) {
                    for (List<CarrierAccess> carriers :
                            List.of(
                                    List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT),
                                    List.of(arrayAccess(type), CarrierAccess.MEMORY_SEGMENT),
                                    List.of(CarrierAccess.MEMORY_SEGMENT, arrayAccess(type)))) {
                        var f = fixture(type, kind, true, carriers.get(0), carriers.get(1));
                        Object input = carrier(type, carriers.get(0), arena, 80);
                        Object actual = carrier(type, carriers.get(1), arena, 96);
                        Object expected = carrier(type, carriers.get(1), arena, 96);
                        fillCarrier(type, input, 80);
                        CpuPool2dReferenceKernel.evaluate(
                                f.geometry, input, expected, 0, f.geometry.outputCount());
                        long middle = f.geometry.outputCount() / 2;
                        f.handle.invokeWithArguments(input, actual, f.geometry.pack(0, 0), 0L, 0L);
                        f.handle.invokeWithArguments(input, actual, f.geometry.pack(0, 0), 0L, middle);
                        f.handle.invokeWithArguments(
                                input, actual, f.geometry.pack(0, 0), middle, f.geometry.outputCount());
                        assertCarrierEquals(type, expected, actual, kind + " " + carriers);
                    }
                }
            }
        }
    }

    @Test
    void maxDistinguishesBfloatSignedZeroAndProducesAllPaddingInfinity() throws Throwable {
        var f =
                fixture(
                        DataType.BFLOAT16,
                        CpuPool2dIr.Kind.MAX,
                        false,
                        CarrierAccess.SHORT_ARRAY,
                        CarrierAccess.SHORT_ARRAY);
        short[] input = new short[20];
        Arrays.fill(input, BFloat16Bits.fromFloat(-0.0f));
        input[1] = BFloat16Bits.fromFloat(+0.0f);
        short[] output = new short[15];
        f.handle.invokeExact(input, output, f.geometry.pack(0, 0), 0L, 15L);
        assertEquals(BFloat16Bits.fromFloat(+0.0f), output[1]);

        var empty = new CpuPool2dLowering.Layout(new long[] {1, 1, 0, 0}, 0, new long[] {0, 0, 1, 1});
        var singleton =
                new CpuPool2dLowering.Layout(new long[] {1, 1, 1, 1}, 0, new long[] {1, 1, 1, 1});
        var geometry =
                new CpuPool2dLowering.Geometry(
                        CpuPool2dIr.Kind.MAX,
                        DataType.BFLOAT16,
                        empty,
                        singleton,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1);
        short[] padded = new short[1];
        f.handle.invokeExact(new short[0], padded, geometry.pack(0, 0), 0L, 1L);
        assertEquals(BFloat16Bits.fromFloat(Float.NEGATIVE_INFINITY), padded[0]);
    }

    @Test
    void averageUsesNegativeZeroOnlyForAnAllNegativeZeroWindow() throws Throwable {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            CarrierAccess access = arrayAccess(type);
            var fixture = fixture(type, CpuPool2dIr.Kind.AVERAGE, false, access, access);
            Object input = array(type, 20);
            Object paddedOutput = array(type, 15);
            switch (type) {
                case BFLOAT16 -> Arrays.fill((short[]) input, BFloat16Bits.fromFloat(-0.0f));
                case FLOAT32 -> Arrays.fill((float[]) input, -0.0f);
                case FLOAT64 -> Arrays.fill((double[]) input, -0.0d);
                default -> throw new AssertionError(type);
            }

            fixture.handle.invokeWithArguments(
                    input, paddedOutput, fixture.geometry.pack(0, 0), 0L, fixture.geometry.outputCount());
            var unpadded =
                    new CpuPool2dLowering.Geometry(
                            CpuPool2dIr.Kind.AVERAGE,
                            type,
                            new CpuPool2dLowering.Layout(new long[] {1, 1, 1, 2}, 0, new long[] {2, 2, 2, 1}),
                            new CpuPool2dLowering.Layout(new long[] {1, 1, 1, 1}, 0, new long[] {1, 1, 1, 1}),
                            1,
                            2,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            2,
                            1);
            Object unpaddedInput = array(type, 2);
            Object unpaddedOutput = array(type, 1);
            switch (type) {
                case BFLOAT16 -> Arrays.fill((short[]) unpaddedInput, BFloat16Bits.fromFloat(-0.0f));
                case FLOAT32 -> Arrays.fill((float[]) unpaddedInput, -0.0f);
                case FLOAT64 -> Arrays.fill((double[]) unpaddedInput, -0.0d);
                default -> throw new AssertionError(type);
            }
            fixture.handle.invokeWithArguments(
                    unpaddedInput, unpaddedOutput, unpadded.pack(0, 0), 0L, 1L);

            switch (type) {
                case BFLOAT16 -> {
                    assertEquals(BFloat16Bits.fromFloat(+0.0f), ((short[]) paddedOutput)[0]);
                    assertEquals(BFloat16Bits.fromFloat(-0.0f), ((short[]) unpaddedOutput)[0]);
                }
                case FLOAT32 -> {
                    assertEquals(0, Float.floatToRawIntBits(((float[]) paddedOutput)[0]));
                    assertEquals(Integer.MIN_VALUE, Float.floatToRawIntBits(((float[]) unpaddedOutput)[0]));
                }
                case FLOAT64 -> {
                    assertEquals(0L, Double.doubleToRawLongBits(((double[]) paddedOutput)[0]));
                    assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(((double[]) unpaddedOutput)[0]));
                }
                default -> throw new AssertionError(type);
            }
        }
    }

    @Test
    void zeroInputStridesBroadcastOnePhysicalElement() throws Throwable {
        var fixture =
                fixture(
                        DataType.FLOAT64,
                        CpuPool2dIr.Kind.AVERAGE,
                        true,
                        CarrierAccess.DOUBLE_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY);
        var geometry =
                new CpuPool2dLowering.Geometry(
                        CpuPool2dIr.Kind.AVERAGE,
                        DataType.FLOAT64,
                        new CpuPool2dLowering.Layout(new long[] {1, 1, 2, 2}, 0, new long[] {0, 0, 0, 0}),
                        new CpuPool2dLowering.Layout(new long[] {1, 1, 2, 2}, 0, new long[] {4, 4, 2, 1}),
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        1,
                        4);
        double[] output = new double[4];
        fixture.handle.invokeExact(new double[] {3.5}, output, geometry.pack(0, 0), 0L, 4L);
        assertArrayEquals(new double[] {3.5, 3.5, 3.5, 3.5}, output);
    }

    @Test
    void classIsFinalFieldFreeAndHasNoSynaptikHotHelper() throws Throwable {
        var f =
                fixture(
                        DataType.FLOAT32,
                        CpuPool2dIr.Kind.AVERAGE,
                        false,
                        CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY);
        var model = ClassFile.of().parse(f.bytes);
        assertTrue(model.flags().has(java.lang.reflect.AccessFlag.FINAL));
        assertTrue(model.fields().isEmpty());
        assertEquals(
                1,
                model.methods().stream()
                        .filter(
                                method -> method.methodName().stringValue().equals(CpuGeneratorSchema.ENTRY_NAME))
                        .count());
        String refs =
                java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                        .filter(MemberRefEntry.class::isInstance)
                        .map(MemberRefEntry.class::cast)
                        .map(entry -> entry.owner().asInternalName())
                        .reduce("", (left, right) -> left + '\n' + right);
        assertFalse(refs.contains("io/github/pho001/synaptik"));
        assertFalse(refs.contains("java/util/Map"));
        assertFalse(refs.contains("java/lang/reflect"));
    }

    private static void checkArray(DataType type, CpuPool2dIr.Kind kind) throws Throwable {
        CarrierAccess access = arrayAccess(type);
        var f = fixture(type, kind, false, access, access);
        Object input = array(type, 20), actual = array(type, 15), expected = array(type, 15);
        fillArray(type, input);
        CpuPool2dReferenceKernel.evaluate(f.geometry, input, expected, 0, 15);
        f.handle.invokeWithArguments(input, actual, f.geometry.pack(0, 0), 0L, 15L);
        switch (type) {
            case BFLOAT16 -> assertArrayEquals((short[]) expected, (short[]) actual);
            case FLOAT32 -> assertArrayEquals(bits((float[]) expected), bits((float[]) actual));
            case FLOAT64 -> assertArrayEquals(bits((double[]) expected), bits((double[]) actual));
            default -> throw new AssertionError(type);
        }
    }

    private static Fixture fixture(
            DataType type,
            CpuPool2dIr.Kind kind,
            boolean general,
            CarrierAccess inputCarrier,
            CarrierAccess outputCarrier)
            throws Throwable {
        long[] ie = {1, 1, 4, 5}, oe = {1, 1, 3, 5};
        long[] is = general ? new long[] {80, 80, 13, 2} : new long[] {20, 20, 5, 1};
        long[] os = general ? new long[] {96, 96, 17, 3} : new long[] {15, 15, 5, 1};
        var geometry =
                new CpuPool2dLowering.Geometry(
                        kind,
                        type,
                        new CpuPool2dLowering.Layout(ie, 0, is),
                        new CpuPool2dLowering.Layout(oe, 0, os),
                        3,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        1,
                        6,
                        15);
        var regime =
                general ? CpuAccessPlan.Regime.GENERAL_ODOMETER : CpuAccessPlan.Regime.DENSE_LINEAR;
        var roles =
                general
                        ? Collections.nCopies(4, CpuAccessPlan.AxisRole.STRIDED)
                        : Collections.nCopies(4, CpuAccessPlan.AxisRole.CONTIGUOUS);
        var pool =
                new CpuPool2dIr(
                        kind,
                        type,
                        CpuPool2dIr.Realization.DIRECT_SCALAR,
                        new CpuAccessPlan(CpuAccessPlan.AccessKind.READ, regime, 4, roles, general ? 0 : 4),
                        new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, regime, 4, roles, general ? 0 : 4));
        var ir = pool.encodedKernelIr();
        var specialization =
                new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        List.of(type, type),
                        List.of(inputCarrier, outputCarrier),
                        0,
                        -1,
                        List.of(),
                        false,
                        55);
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(specialization, ir);
        return new Fixture(
                geometry, generator.defineClassBytes(specialization, bytes).entryPoint(), bytes);
    }

    private static CarrierAccess arrayAccess(DataType type) {
        return switch (type) {
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
            case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
            default -> throw new AssertionError(type);
        };
    }

    private static Object array(DataType type, int n) {
        return switch (type) {
            case BFLOAT16 -> new short[n];
            case FLOAT32 -> new float[n];
            case FLOAT64 -> new double[n];
            default -> throw new AssertionError(type);
        };
    }

    private static void fillArray(DataType type, Object carrier) {
        for (int i = 0; i < 20; i++) {
            double v =
                    i == 7
                            ? Double.NaN
                            : i == 8
                                    ? Double.POSITIVE_INFINITY
                                    : i == 9 ? Double.NEGATIVE_INFINITY : i % 7 - 3.25;
            switch (type) {
                case BFLOAT16 -> ((short[]) carrier)[i] = BFloat16Bits.fromFloat((float) v);
                case FLOAT32 -> ((float[]) carrier)[i] = (float) v;
                case FLOAT64 -> ((double[]) carrier)[i] = v;
                default -> throw new AssertionError(type);
            }
        }
    }

    private static void fillSegment(DataType type, MemorySegment segment, int n) {
        for (int i = 0; i < n; i++) {
            long offset = (long) i * type.byteWidth();
            double v = i % 11 - 5.25;
            switch (type) {
                case BFLOAT16 ->
                        segment.set(
                                ValueLayout.JAVA_SHORT_UNALIGNED, offset, BFloat16Bits.fromFloat((float) v));
                case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, (float) v);
                case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset, v);
                default -> throw new AssertionError(type);
            }
        }
    }

    private static Object carrier(
            DataType type, CarrierAccess access, Arena arena, int elementCount) {
        return access == CarrierAccess.MEMORY_SEGMENT
                ? arena.allocate((long) elementCount * type.byteWidth(), type.byteWidth())
                : array(type, elementCount);
    }

    private static void fillCarrier(DataType type, Object carrier, int elementCount) {
        if (carrier instanceof MemorySegment segment) {
            fillSegment(type, segment, elementCount);
            return;
        }
        for (int index = 0; index < elementCount; index++) {
            double value = index % 11 - 5.25;
            switch (type) {
                case BFLOAT16 -> ((short[]) carrier)[index] = BFloat16Bits.fromFloat((float) value);
                case FLOAT32 -> ((float[]) carrier)[index] = (float) value;
                case FLOAT64 -> ((double[]) carrier)[index] = value;
                default -> throw new AssertionError(type);
            }
        }
    }

    private static void assertCarrierEquals(
            DataType type, Object expected, Object actual, String message) {
        if (expected instanceof MemorySegment expectedSegment) {
            assertEquals(-1, expectedSegment.mismatch((MemorySegment) actual), message);
            return;
        }
        switch (type) {
            case BFLOAT16 -> assertArrayEquals((short[]) expected, (short[]) actual, message);
            case FLOAT32 -> assertArrayEquals(bits((float[]) expected), bits((float[]) actual), message);
            case FLOAT64 ->
                    assertArrayEquals(bits((double[]) expected), bits((double[]) actual), message);
            default -> throw new AssertionError(type);
        }
    }

    private static int[] bits(float[] values) {
        int[] bits = new int[values.length];
        for (int i = 0; i < values.length; i++) bits[i] = Float.floatToRawIntBits(values[i]);
        return bits;
    }

    private static long[] bits(double[] values) {
        long[] bits = new long[values.length];
        for (int i = 0; i < values.length; i++) bits[i] = Double.doubleToRawLongBits(values[i]);
        return bits;
    }

    private record Fixture(CpuPool2dLowering.Geometry geometry, MethodHandle handle, byte[] bytes) {}
}
