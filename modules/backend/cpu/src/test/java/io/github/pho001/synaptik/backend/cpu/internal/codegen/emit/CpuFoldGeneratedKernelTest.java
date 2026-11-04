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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import org.junit.jupiter.api.Test;

class CpuFoldGeneratedKernelTest {
    @Test void volumetricFoldPreservesBroadcastInputOccurrencesAndStridedOutputRanges() throws Throwable {
        var window = new Window3dAttrs(2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, false);
        Shape image = Shape.of(1, 1, 2, 2, 2), columns = Shape.of(1, 8, 27);
        var in = new TensorDescriptor(DataType.FLOAT64, columns,
                Optional.of(LayoutDescriptor.of(columns, new long[]{0, 0, 2}, 1, true)), false);
        var out = new TensorDescriptor(DataType.FLOAT64, image,
                Optional.of(LayoutDescriptor.of(image, new long[]{0, 0, 12, 5, 2}, 3, true)), false);
        var selected = carriers(CpuScatterLoweringTest.context(
                new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(image, window)),
                List.of(0), List.of(in), out), CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY);
        double[] input = new double[54], actual = new double[23], expected = new double[23];
        for (int i = 0; i < input.length; i++) input[i] = (i % 17 - 8) * 0.125;
        Arrays.fill(actual, -17); Arrays.fill(expected, -17);
        var lowered = new CpuPartitionLowering().lower(selected);
        CpuScalarReferenceKernel.execute((CpuFoldIr) lowered.portableKernelIr(),
                lowered.foldGeometry().orElseThrow(),
                List.of(argument(DataType.FLOAT64, input, true), argument(DataType.FLOAT64, expected, false)), 1, 7);
        invoke(selected, input, actual, 1, 7);
        assertArrayEquals(expected, actual);
        assertEquals(-17, actual[3]); assertEquals(-17, actual[22]);
    }

    @Test void controlledVolumetricFoldExcludesInactiveInputRowsAndPublishesPositiveZero() throws Throwable {
        var window = new Window3dAttrs(2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, false);
        Shape image = Shape.of(2, 1, 2, 2, 2);
        var selected = context(new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(image, window)),
                DataType.FLOAT32, Shape.of(2, 8, 27), image);
        var plan = new CpuPartitionPreparer().analyze(selected).plan();
        var entry = CpuAffineCopyGeneratedKernelTest.controlledEntry(
                plan.units().getFirst().portablePlan(), List.of(
                        new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(0, 216),
                        new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(0, 8)));
        float[] input = new float[216], output = new float[16];
        Arrays.fill(input, 1); Arrays.fill(output, -99);
        entry.invokeWithArguments(input, output,
                plan.foldGeometry().orElseThrow().pack(new long[2], 0, 16),
                0L, 16L, CpuAffineCopyGeneratedKernelTest.control(new byte[]{1, 0}));
        for (int i = 0; i < 8; i++) assertEquals(8.0f, output[i]);
        for (int i = 8; i < 16; i++) assertEquals(0, Float.floatToRawIntBits(output[i]));
    }

    @Test void volumetricFoldMatchesCanonicalOracleWithOverlapDilationCeilAndPartialRanges() throws Throwable {
        var window = new Window3dAttrs(2, 2, 3, 2, 2, 2, 1, 0, 1, 2, 1, 1, true);
        Shape image = Shape.of(2, 2, 3, 4, 5), columns = Shape.of(2, 24, 12);
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var base = context(new Operation(WindowTransformKind.FOLD3D,
                    new Fold3dAttrs(image, window)), type, columns, image);
            Object input = empty(type, 576), expected = empty(type, 240);
            if (input instanceof float[] f) {
                for (int i = 0; i < f.length; i++) f[i] = (i % 31 - 15) * 0.125f;
            } else {
                var d = (double[]) input;
                for (int i = 0; i < d.length; i++) d[i] = (i % 31 - 15) * 0.125;
            }
            var lowered = new CpuPartitionLowering().lower(base);
            CpuScalarReferenceKernel.execute((CpuFoldIr) lowered.portableKernelIr(),
                    lowered.foldGeometry().orElseThrow(),
                    List.of(argument(type, input, true), argument(type, expected, false)), 0, 240);
            Object direct = empty(type, 240);
            long[] cells = ((io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLowering.SpatialGeometry)
                    lowered.foldGeometry().orElseThrow().mapping()).spatial().cells();
            if (input instanceof float[] f)
                CpuSpatialWindowCleanJavaOracle.fold(f, (float[]) direct, 2, 3, 4, 5, cells, 0, 240);
            else CpuSpatialWindowCleanJavaOracle.fold((double[]) input, (double[]) direct, 2, 3, 4, 5, cells, 0, 240);
            assertCarrierEquals(type, expected, direct);
            for (var pattern : List.of(List.of(heap(type), heap(type)),
                    List.of(CarrierAccess.MEMORY_SEGMENT, heap(type)),
                    List.of(heap(type), CarrierAccess.MEMORY_SEGMENT),
                    List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT))) {
                Object actual = empty(type, 240);
                var selected = carriers(base, pattern.getFirst(), pattern.getLast());
                Object in = pattern.getFirst() == CarrierAccess.MEMORY_SEGMENT ? segment(input) : input;
                Object out = pattern.getLast() == CarrierAccess.MEMORY_SEGMENT ? segment(actual) : actual;
                for (long[] range : List.of(new long[]{0, 0}, new long[]{0, 17},
                        new long[]{17, 137}, new long[]{137, 240}))
                    invoke(selected, in, out, range[0], range[1]);
                assertCarrierEquals(type, expected, actual);
            }
        }
    }

    @Test void volumetricFoldLeavesPositiveZeroGapsAndPreservesContributorOrder() throws Throwable {
        Shape image = Shape.of(1, 1, 3, 3, 3);
        var sparse = new Window3dAttrs(1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, false);
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            Object input = floating(type, 1, 2, 3, 4, 5, 6, 7, 8), output = empty(type, 27);
            invoke(context(new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(image, sparse)),
                    type, Shape.of(1, 1, 8), image), input, output, 0, 27);
            for (int d = 0; d < 3; d++) for (int h = 0; h < 3; h++) for (int w = 0; w < 3; w++) {
                int index = (d * 3 + h) * 3 + w;
                double expected = (d % 2 != 0 || h % 2 != 0 || w % 2 != 0) ? 0
                        : (d / 2 * 2 + h / 2) * 2 + w / 2 + 1;
                if (output instanceof float[] f)
                    assertEquals(Float.floatToRawIntBits((float) expected), Float.floatToRawIntBits(f[index]));
                else assertEquals(Double.doubleToRawLongBits(expected),
                        Double.doubleToRawLongBits(((double[]) output)[index]));
            }
        }
        Shape small = Shape.of(1, 1, 2, 2, 2);
        var overlap = new Window3dAttrs(2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, false);
        float[] input = new float[216], output = new float[8];
        input[26] = 1e30f; input[27 + 25] = -1e30f; input[54 + 23] = 3;
        invoke(context(new Operation(WindowTransformKind.FOLD3D, new Fold3dAttrs(small, overlap)),
                DataType.FLOAT32, Shape.of(1, 8, 27), small), input, output, 0, 8);
        assertEquals(3.0f, output[7]);
    }


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

    @Test void boundedPaddedDilatedFloatFoldPreservesPartialRangesAndTypedFallback()
            throws Throwable {
        var context = boundedFoldContext();
        float[] input = new float[4_608];
        for (int index = 0; index < input.length; index++) {
            input[index] = (index % 29 - 14) * 0.03125f;
        }
        float[] expected = new float[1_027];
        float[] actual = new float[1_027];
        Arrays.fill(expected, -17.0f);
        Arrays.fill(actual, -17.0f);
        var lowered = new CpuPartitionLowering().lower(context);
        for (long[] range : List.of(new long[]{7, 173}, new long[]{173, 389})) {
            CpuScalarReferenceKernel.execute((CpuFoldIr) lowered.portableKernelIr(),
                    lowered.foldGeometry().orElseThrow(),
                    List.of(argument(DataType.FLOAT32, input, true),
                            argument(DataType.FLOAT32, expected, false)), range[0], range[1]);
            invoke(context, input, MemorySegment.ofArray(actual), range[0], range[1]);
        }
        assertArrayEquals(expected, actual);

    }

    @Test void representedAxisTypesRetainIntegerArithmeticAndPartialRanges()
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


        float[] partial = {-7, -7, -7, -7, -7};
        var partialContext = context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32, Shape.of(3, 3), Shape.of(5));
        invoke(partialContext, new float[]{1,2,3,4,5,6,7,8,9}, partial, 1, 4);
        assertArrayEquals(new float[]{-7, 6, 15, 14, -7}, partial);
    }

    @Test void everyAxisTypeExecutesBothMixedSegmentDirectionsAndSegmentToSegment()
            throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) {
            var base = CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                    new FoldAxisAttrs(0, 5, 1)), type, Shape.of(3, 3), Shape.of(5));
            Object input = sample(type);
            Object expected = empty(type, 5);
            
            var lowered = new CpuPartitionLowering().lower(base);
            CpuScalarReferenceKernel.execute((CpuFoldIr) lowered.portableKernelIr(),
                    lowered.foldGeometry().orElseThrow(),
                    List.of(argument(type, input, true), argument(type, expected, false)), 0, 5);

            Object segmentToHeap = empty(type, 5);
            invoke(carriers(base, CarrierAccess.MEMORY_SEGMENT, heap(type)), segment(input),
                    segmentToHeap, 0, 5);
            assertCarrierEquals(type, expected, segmentToHeap);

            Object heapToSegment = empty(type, 5);
            invoke(carriers(base, heap(type), CarrierAccess.MEMORY_SEGMENT), input,
                    segment(heapToSegment), 0, 5);
            assertCarrierEquals(type, expected, heapToSegment);

            Object segmentToSegment = empty(type, 5);
            invoke(carriers(base, CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT),
                    segment(input), segment(segmentToSegment), 0, 5);
            assertCarrierEquals(type, expected, segmentToSegment);
        }
    }

    @Test void denseRankOneSpecializationHonorsNonzeroInputAndOutputBases() throws Throwable {
        Shape inputShape = Shape.of(3, 3), outputShape = Shape.of(5);
        var inputDescriptor = new TensorDescriptor(DataType.FLOAT32, inputShape,
                Optional.of(LayoutDescriptor.of(inputShape, new long[]{3, 1}, 2, true)), false);
        var outputDescriptor = new TensorDescriptor(DataType.FLOAT32, outputShape,
                Optional.of(LayoutDescriptor.of(outputShape, new long[]{1}, 3, true)), false);
        var base = CpuScatterLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), List.of(0), List.of(inputDescriptor),
                outputDescriptor);
        float[] input = {-9, -9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        float[] output = new float[8];
        Arrays.fill(output, -7.0f);
        invoke(carriers(base, CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY), input,
                output, 0, 5);
        assertArrayEquals(new float[]{-7, -7, -7, 1, 6, 15, 14, 9}, output);
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

    @Test void higherRankAxisAndPaddedDilatedMultiBatchFoldMatchIndependentOracle()
            throws Throwable {
        float[] axis = new float[2 * 3 * 4 * 3];
        for (int i = 0; i < axis.length; i++) axis[i] = (i % 13 - 6) * 0.25f;
        compareWithReference(context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(1, 5, 1)), DataType.FLOAT32,
                Shape.of(2, 3, 4, 3), Shape.of(2, 5, 4)), axis);

        Shape output = Shape.of(2, 2, 5, 5);
        var window = new Window2dAttrs(2, 2, 1, 1, 1, 1, 2, 2, false);
        double[] columns = new double[2 * 8 * 25];
        for (int i = 0; i < columns.length; i++) columns[i] = (i % 17 - 8) * 0.125;
        compareWithReference(context(new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(output, window)), DataType.FLOAT64,
                Shape.of(2, 8, 25), output), columns);
    }

    @Test void oneGeneratedArtifactProducesBitwiseEqualScalarAndPartitionedRanges()
            throws Throwable {
        var fold = context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32, Shape.of(3, 3), Shape.of(5));
        float[] input = {1,2,3,4,5,6,7,8,9};
        float[] scalar = new float[5], partitioned = new float[5];
        invoke(fold, input, scalar, 0, 5);
        var plan = new CpuPartitionPreparer().analyze(fold).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        for (long[] range : List.of(new long[]{0, 2}, new long[]{2, 3}, new long[]{3, 5})) {
            long[] packed = plan.foldGeometry().orElseThrow().pack(new long[2], range[0], range[1]);
            artifact.entryPoint().invokeWithArguments(input, partitioned, packed,
                    range[0], range[1]);
        }
        assertArrayEquals(scalar, partitioned);
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
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
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

        var denseOutput = CpuScatterLoweringTest.desc(DataType.FLOAT32, outputShape);
        var heapBase = CpuScatterLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), List.of(0), List.of(inputDescriptor),
                denseOutput);
        float[] heapOutput = new float[5];
        invoke(carriers(heapBase, CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY),
                new float[]{99, 1, 2, 3}, heapOutput, 0, 5);
        assertArrayEquals(new float[]{1, 3, 6, 5, 3}, heapOutput);
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
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long[] packed = plan.foldGeometry().orElseThrow().pack(new long[2], start, end);
        artifact.entryPoint().invokeWithArguments(input, output, packed, start, end);
        return plan;
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            DataType type, Shape input, Shape output) {
        var base = CpuFoldLoweringTest.context(operation, type, input, output);
        CarrierAccess carrier = heap(type);
        return carriers(base, carrier, carrier);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> boundedFoldContext() {
        Shape inputShape = Shape.of(1, 9, 512);
        Shape outputShape = Shape.of(1, 1, 16, 32);
        var input = new TensorDescriptor(DataType.FLOAT32, inputShape,
                Optional.of(LayoutDescriptor.of(inputShape, new long[]{4_608, 512, 1}, 0, true)),
                false);
        var output = new TensorDescriptor(DataType.FLOAT32, outputShape,
                Optional.of(LayoutDescriptor.of(outputShape,
                        new long[]{1_024, 1_024, 64, 2}, 3, true)), false);
        var base = CpuScatterLoweringTest.context(new Operation(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(outputShape,
                                new Window2dAttrs(3, 3, 1, 1, 2, 2, 2, 2, false))),
                List.of(0), List.of(input), output);
        return carriers(base, CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> carriers(
            PrepareContext<CpuPartitionAnalysisInputs> base, CarrierAccess input,
            CarrierAccess output) {
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(input, output)));
    }

    private static CarrierAccess heap(DataType type) {
        return switch (type) {
            case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16, FLOAT16 -> CarrierAccess.SHORT_ARRAY; case INT32 -> CarrierAccess.INT_ARRAY;
            case FLOAT8_E4M3, FLOAT8_E5M2 -> CarrierAccess.BYTE_ARRAY;
            case INT64 -> CarrierAccess.LONG_ARRAY; case BOOL -> throw new AssertionError();
        };
    }

    private static Object empty(DataType type, int count) {
        return switch (type) {
            case FLOAT64 -> new double[count]; case FLOAT32 -> new float[count];
            case BFLOAT16 -> new short[count]; case INT32 -> new int[count];
            case INT64 -> new long[count]; default -> throw new AssertionError();
        };
    }

    private static Object sample(DataType type) {
        return switch (type) {
            case FLOAT64 -> new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
            case FLOAT32 -> new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
            case BFLOAT16 -> new short[]{0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0,
                    0x40c0, 0x40e0, 0x4100, 0x4110};
            case INT32 -> new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
            case INT64 -> new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
            default -> throw new AssertionError();
        };
    }

    private static MemorySegment segment(Object carrier) {
        return switch (carrier) {
            case double[] values -> MemorySegment.ofArray(values);
            case float[] values -> MemorySegment.ofArray(values);
            case short[] values -> MemorySegment.ofArray(values);
            case int[] values -> MemorySegment.ofArray(values);
            case long[] values -> MemorySegment.ofArray(values);
            default -> throw new AssertionError("unexpected fold carrier");
        };
    }

    private static void assertCarrierEquals(DataType type, Object expected, Object actual) {
        switch (type) {
            case FLOAT64 -> assertArrayEquals((double[]) expected, (double[]) actual);
            case FLOAT32 -> assertArrayEquals((float[]) expected, (float[]) actual);
            case BFLOAT16 -> assertArrayEquals((short[]) expected, (short[]) actual);
            case INT32 -> assertArrayEquals((int[]) expected, (int[]) actual);
            case INT64 -> assertArrayEquals((long[]) expected, (long[]) actual);
            default -> throw new AssertionError();
        }
    }

    private static CpuBufferArgument argument(DataType type, Object carrier, boolean readOnly) {
        long bytes = Math.multiplyExact(java.lang.reflect.Array.getLength(carrier), type.byteWidth());
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) carrier, 0, bytes, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) carrier, 0, bytes, readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[]) carrier, 0, bytes, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) carrier, 0, bytes, readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[]) carrier, 0, bytes, readOnly);
            default -> throw new AssertionError();
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
