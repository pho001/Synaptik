package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import org.junit.jupiter.api.Test;

class CpuFoldGeneratedKernelTest {
    @Test void axisOverlapUsesCanonicalInputOrderAndLeavesUncoveredPositiveZeros() throws Throwable {
        double[] output = {-1, -1, -1, -1, -1, -1};
        invoke(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 6, 2)), DataType.FLOAT64,
                Shape.of(2, 3), Shape.of(6)), new double[]{1, 2, 3, 4, 5, 6}, output, 0, 6);
        assertArrayEquals(new double[]{1, 2, 7, 5, 6, 0.0}, output);

        float large = 1e30f;
        float[] ordered = new float[5];
        invoke(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32,
                Shape.of(3, 3), Shape.of(5)),
                new float[]{0,0,large,0,-large,0,3,0,0}, ordered, 0, 5);
        assertEquals(Float.floatToRawIntBits(3.0f), Float.floatToRawIntBits(ordered[2]));
    }

    @Test void twoDimensionalFoldExcludesPaddingAndProducesNchwOverlap() throws Throwable {
        var window = new Window2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, false);
        Shape outputShape = Shape.of(1, 1, 2, 2);
        double[] input = new double[36]; Arrays.fill(input, 1.0);
        double[] output = new double[4];
        invoke(context(new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(outputShape, window)), DataType.FLOAT64,
                Shape.of(1, 4, 9), outputShape), input, output, 0, 4);
        assertArrayEquals(new double[]{4, 4, 4, 4}, output);

        var ceilWindow = new Window2dAttrs(2, 2, 2, 2, 1, 1, 1, 1, true);
        Shape ceilShape = Shape.of(1, 1, 3, 3);
        float[] ceilOutput = new float[9];
        float[] ceilColumns = new float[36]; Arrays.fill(ceilColumns, 1.0f);
        invoke(context(new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(ceilShape, ceilWindow)), DataType.FLOAT32,
                Shape.of(1, 4, 9), ceilShape), ceilColumns, ceilOutput, 0, 9);
        assertArrayEquals(new float[]{1,1,1,1,1,1,1,1,1}, ceilOutput);
    }

    @Test void supportsEveryRepresentedAxisTypeBfloatStepRoundingAndPartialRanges()
            throws Throwable {
        int[] intOutput = {9, 9, 9, 9, 9};
        invoke(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), DataType.INT32,
                Shape.of(3, 3), Shape.of(5)),
                new int[]{0, Integer.MAX_VALUE, 0, 1, 0, 0, 0, 0, 0}, intOutput, 0, 5);
        assertEquals(Integer.MIN_VALUE, intOutput[1]);
        assertEquals(0, intOutput[4]);

        long[] longOutput = new long[3];
        invoke(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 3, 1)), DataType.INT64,
                Shape.of(2, 2), Shape.of(3)),
                new long[]{0, Long.MAX_VALUE, 1, 0}, longOutput, 0, 3);
        assertEquals(Long.MIN_VALUE, longOutput[1]);

        short one = (short) 0x3f80, halfUlp = (short) 0x3b80;
        short[] bfloatOutput = new short[5];
        invoke(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), DataType.BFLOAT16,
                Shape.of(3, 3), Shape.of(5)),
                new short[]{0,0,one,0,halfUlp,0,halfUlp,0,0}, bfloatOutput, 0, 5);
        assertEquals(one, bfloatOutput[2]);

        float[] partial = {-7, -7, -7, -7, -7};
        var partialContext = context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32, Shape.of(3, 3), Shape.of(5));
        invoke(partialContext, new float[]{1,2,3,4,5,6,7,8,9}, partial, 1, 4);
        assertArrayEquals(new float[]{-7, 6, 15, 14, -7}, partial);
    }

    @Test void generatedResultsMatchIndependentOracleForBothFamilies() throws Throwable {
        compareWithReference(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT64,
                Shape.of(3, 3), Shape.of(5)),
                new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        Shape output = Shape.of(1, 1, 3, 3);
        compareWithReference(context(new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(output, CpuFoldLoweringTest.window(false))),
                DataType.FLOAT32, Shape.of(1, 4, 4), output),
                new float[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    }

    @Test void floatingSpecialValuesSubnormalsCancellationAndBfloatBoundariesMatchOracle()
            throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            double maximum = type == DataType.FLOAT64 ? Double.MAX_VALUE
                    : type == DataType.FLOAT32 ? Float.MAX_VALUE : Float.intBitsToFloat(0x7f7f0000);
            double normal = type == DataType.FLOAT64 ? Double.MIN_NORMAL
                    : type == DataType.FLOAT32 ? Float.MIN_NORMAL : Math.scalb(1.0, -126);
            double subnormal = type == DataType.FLOAT64 ? Double.MIN_VALUE
                    : type == DataType.FLOAT32 ? Float.MIN_VALUE : Math.scalb(1.0, -133);
            compareWithReference(context(new Operation(WindowTransformKind.FOLD_AXIS,
                            new FoldAxisAttrs(0, 5, 1)), type, Shape.of(3, 3), Shape.of(5)),
                    floating(type, maximum, normal, subnormal, -maximum, -normal, -subnormal,
                            Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN));
        }
    }

    @Test void executesZeroStrideInputAndInjectiveStridedOutputAcrossMixedCarriers()
            throws Throwable {
        Shape inputShape = Shape.of(3, 3), outputShape = Shape.of(5);
        var inputDescriptor = new TensorDescriptor(DataType.FLOAT32, inputShape,
                Optional.of(LayoutDescriptor.of(inputShape, new long[]{0, 1}, 1, true)), false);
        var outputDescriptor = new TensorDescriptor(DataType.FLOAT32, outputShape,
                Optional.of(LayoutDescriptor.of(outputShape, new long[]{2}, 1, true)), false);
        var base = CpuScatterLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), List.of(0), List.of(inputDescriptor),
                outputDescriptor);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY)));
        try (Arena arena = Arena.ofConfined()) {
            var input = arena.allocate(4L * Float.BYTES, Float.BYTES);
            input.set(ValueLayout.JAVA_FLOAT, Float.BYTES, 1.0f);
            input.set(ValueLayout.JAVA_FLOAT, 2L * Float.BYTES, 2.0f);
            input.set(ValueLayout.JAVA_FLOAT, 3L * Float.BYTES, 3.0f);
            float[] output = new float[11]; Arrays.fill(output, -7.0f);
            invoke(context, input, output, 0, 5);
            assertArrayEquals(new float[]{-7,1,-7,3,-7,6,-7,5,-7,3,-7}, output);
        }
    }

    private static void compareWithReference(PrepareContext<CpuPartitionAnalysisInputs> context,
            Object input) throws Throwable {
        var lowered = new CpuPartitionLowering().lower(context);
        int count = Math.toIntExact(lowered.elementCount());
        DataType type = ((CpuFoldIr) lowered.portableKernelIr()).dataType();
        Object expected = empty(type, count), actual = empty(type, count);
        CpuScalarReferenceKernel.execute((CpuFoldIr) lowered.portableKernelIr(),
                lowered.foldGeometry().orElseThrow(),
                List.of(argument(type, input, true), argument(type, expected, false)), 0, count);
        invoke(context, input, actual, 0, count);
        if (expected instanceof double[] values) assertArrayEquals(values, (double[]) actual);
        else if (expected instanceof float[] values) assertArrayEquals(values, (float[]) actual);
        else if (expected instanceof short[] values) assertArrayEquals(values, (short[]) actual);
        else fail("unexpected comparison type");
    }

    private static CpuPartitionPreparationPlan invoke(
            PrepareContext<CpuPartitionAnalysisInputs> context, Object input, Object output,
            long start, long end) throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long[] packed = plan.foldGeometry().orElseThrow().pack(new long[2], start, end);
        artifact.entryPoint().invokeWithArguments(input, output, packed, start, end);
        return plan;
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            DataType type, Shape input, Shape output) {
        var base = CpuFoldLoweringTest.context(operation, type, input, output);
        CarrierAccess carrier = heap(type);
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of(carrier, carrier)));
    }

    private static CarrierAccess heap(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> CarrierAccess.BYTE_ARRAY;
        };
    }

    private static Object empty(DataType type, int count) {
        return switch (type) {
            case FLOAT64 -> new double[count]; case FLOAT32 -> new float[count];
            case BFLOAT16 -> new short[count]; case INT32 -> new int[count];
            case INT64 -> new long[count]; case BOOL -> new byte[count];
        };
    }

    private static CpuBufferArgument argument(DataType type, Object carrier, boolean readOnly) {
        long bytes = Math.multiplyExact(java.lang.reflect.Array.getLength(carrier), type.byteWidth());
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) carrier, 0, bytes, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) carrier, 0, bytes, readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[]) carrier, 0, bytes, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) carrier, 0, bytes, readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[]) carrier, 0, bytes, readOnly);
            case BOOL -> new CpuBufferArgument.Bytes((byte[]) carrier, 0, bytes, readOnly);
        };
    }

    private static Object floating(DataType type, double... values) {
        if (type == DataType.FLOAT64) return values.clone();
        if (type == DataType.FLOAT32) {
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
            return result;
        }
        short[] result = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            int bits = Float.floatToRawIntBits((float) values[i]);
            if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0) {
                result[i] = (short) ((bits >>> 16) | 0x40);
            } else {
                int upper = bits >>> 16, lower = bits & 0xffff;
                if (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0) upper++;
                result[i] = (short) upper;
            }
        }
        return result;
    }
}
