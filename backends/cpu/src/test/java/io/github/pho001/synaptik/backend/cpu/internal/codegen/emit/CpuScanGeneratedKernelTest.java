package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CpuScanGeneratedKernelTest {
    @Test void executesBothKindsAndAllModesInLogicalAxisOrder() throws Throwable {
        assertArrayEquals(new int[]{1,3,6}, (int[]) invoke(CumulativeScanKind.CUM_SUM,
                DataType.INT32, false, false, new int[]{1,2,3}));
        assertArrayEquals(new int[]{0,1,3}, (int[]) invoke(CumulativeScanKind.CUM_SUM,
                DataType.INT32, true, false, new int[]{1,2,3}));
        assertArrayEquals(new int[]{6,5,3}, (int[]) invoke(CumulativeScanKind.CUM_SUM,
                DataType.INT32, false, true, new int[]{1,2,3}));
        assertArrayEquals(new int[]{5,3,0}, (int[]) invoke(CumulativeScanKind.CUM_SUM,
                DataType.INT32, true, true, new int[]{1,2,3}));
        assertArrayEquals(new long[]{2,6,24}, (long[]) invoke(CumulativeScanKind.CUM_PROD,
                DataType.INT64, false, false, new long[]{2,3,4}));
        assertArrayEquals(new long[]{12,4,1}, (long[]) invoke(CumulativeScanKind.CUM_PROD,
                DataType.INT64, true, true, new long[]{2,3,4}));
    }

    @Test void preservesTypedRoundingOverflowSpecialValuesAndSliceRanges() throws Throwable {
        assertArrayEquals(new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE}, (int[]) invoke(
                CumulativeScanKind.CUM_SUM, DataType.INT32, false, false,
                new int[]{Integer.MAX_VALUE, 1}));
        float[] floats = (float[]) invoke(CumulativeScanKind.CUM_PROD, DataType.FLOAT32,
                false, false, new float[]{-0.0f, Float.POSITIVE_INFINITY, Float.NaN});
        assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(floats[0]));
        assertTrue(Float.isNaN(floats[1])); assertTrue(Float.isNaN(floats[2]));
        short one = (short) 0x3f80, halfUlp = (short) 0x3b80;
        short[] bfloat = (short[]) invoke(CumulativeScanKind.CUM_SUM, DataType.BFLOAT16,
                false, false, new short[]{one, halfUlp, halfUlp});
        assertArrayEquals(new short[]{one, one, one}, bfloat);
    }

    @Test void generatedResultsMatchIndependentReferenceForFiveTypesKindsAndModes() throws Throwable {
        for (CumulativeScanKind kind : CumulativeScanKind.values())
            for (boolean exclusive : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                compare(kind, DataType.FLOAT64, exclusive, reverse, new double[]{1,-2,3,4,0,Double.POSITIVE_INFINITY});
                compare(kind, DataType.FLOAT32, exclusive, reverse, new float[]{1,-2,3,4,0,Float.NaN});
                compare(kind, DataType.BFLOAT16, exclusive, reverse,
                        new short[]{(short)0x3f80,(short)0xc000,(short)0x4040,(short)0x4080,0,(short)0x7f80});
                compare(kind, DataType.INT32, exclusive, reverse,
                        new int[]{1,-2,Integer.MAX_VALUE,4,0,7});
                compare(kind, DataType.INT64, exclusive, reverse,
                        new long[]{1,-2,Long.MAX_VALUE,4,0,7});
            }
    }

    private static void compare(CumulativeScanKind kind, DataType type, boolean exclusive,
            boolean reverse, Object input) throws Throwable {
        Object actual = invoke(kind, type, exclusive, reverse, input);
        Object expected = java.lang.reflect.Array.newInstance(input.getClass().componentType(),
                java.lang.reflect.Array.getLength(input));
        var lowered = new CpuPartitionLowering().lower(CpuScanLoweringTest.context(kind, type,
                Shape.of(2, 3), 1, exclusive, reverse));
        CpuScalarReferenceKernel.execute((CpuScanIr) lowered.portableKernelIr(),
                lowered.scanGeometry().orElseThrow(), List.of(argument(type, input, true),
                        argument(type, expected, false)));
        if (expected instanceof double[] a) assertArrayEquals(a, (double[]) actual);
        else if (expected instanceof float[] a) assertArrayEquals(a, (float[]) actual);
        else if (expected instanceof short[] a) assertArrayEquals(a, (short[]) actual);
        else if (expected instanceof int[] a) assertArrayEquals(a, (int[]) actual);
        else assertArrayEquals((long[]) expected, (long[]) actual);
    }

    private static CpuBufferArgument argument(DataType type, Object value, boolean readOnly) {
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) value, 0, ((double[]) value).length * 8L, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) value, 0, ((float[]) value).length * 4L, readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[]) value, 0, ((short[]) value).length * 2L, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) value, 0, ((int[]) value).length * 4L, readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[]) value, 0, ((long[]) value).length * 8L, readOnly);
            case BOOL -> throw new AssertionError();
        };
    }

    private static Object invoke(CumulativeScanKind kind, DataType type, boolean exclusive,
            boolean reverse, Object input) throws Throwable {
        int length = java.lang.reflect.Array.getLength(input);
        Shape shape = length == 6 ? Shape.of(2, 3) : Shape.of(length);
        int axis = length == 6 ? 1 : 0;
        var base = CpuScanLoweringTest.context(kind, type, shape, axis, exclusive, reverse);
        CarrierAccess carrier = switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> throw new AssertionError();
        };
        PrepareContext<CpuPartitionAnalysisInputs> context = new PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of(carrier, carrier)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        Object output = java.lang.reflect.Array.newInstance(input.getClass().componentType(), length);
        long[] packed = plan.scanGeometry().orElseThrow().pack(new long[2]);
        artifact.entryPoint().invokeWithArguments(input, output, packed, 0L,
                plan.scanGeometry().orElseThrow().sliceCount());
        return output;
    }
}
