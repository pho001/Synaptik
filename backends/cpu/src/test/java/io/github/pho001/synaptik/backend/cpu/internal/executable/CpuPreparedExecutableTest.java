package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.Arena;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import java.lang.foreign.Arena;

class CpuPreparedExecutableTest {
    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfInt INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfLong LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());

    @Test void batchNormalizationExecutesDirectParallelRangesAndValidatesOverlapFirst() {
        var base = CpuBatchNormInferenceLoweringTest.context(
                java.util.Collections.nCopies(5, DataType.FLOAT32), Shape.of(8, 3, 4), 1,
                List.of(0, 1, 2, 3, 4));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        java.util.Collections.nCopies(6, CarrierAccess.FLOAT_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        assertAll(() -> assertEquals(4, analysis.plan().selectedRangeCount()),
                () -> assertTrue(analysis.plan().workspaceDeclaration().isEmpty()),
                () -> assertTrue(analysis.plan().materialization().isEmpty()));
        try (var workers = new CpuWorkerGroup(4)) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            float[] input = new float[96];
            for (int index = 0; index < input.length; index++) input[index] = index * .125f - 4f;
            float[] scale = {.5f, 1.25f, -2f}, bias = {.125f, -.5f, 2f};
            float[] mean = {-1f, .25f, 3f}, variance = {4f, .5f, -2f};
            float[] output = new float[98]; java.util.Arrays.fill(output, -77f);
            var run = state(executable, List.of(borrow(input, 0, 96), borrow(scale, 0, 3),
                    borrow(bias, 0, 3), borrow(mean, 0, 3), borrow(variance, 0, 3),
                    borrow(output, 1, 96)));
            try {
                var bound = executable.bind(run); bound.execute();
                for (int index = 0; index < input.length; index++) {
                    int channel = index / 4 % 3;
                    float centered = input[index] - mean[channel];
                    float denominator = (float) Math.sqrt(variance[channel] + 1e-5f);
                    float expected = centered / denominator * scale[channel] + bias[channel];
                    assertEquals(Float.floatToRawIntBits(expected),
                            Float.floatToRawIntBits(output[index + 1]), "index " + index);
                }
                assertEquals(-77f, output[0]); assertEquals(-77f, output[97]);
                float[] first = output.clone(); bound.execute(); assertArrayEquals(first, output);
            } finally { run.close(); }

            float[] sharedInputOutput = new float[110];
            java.util.Arrays.fill(sharedInputOutput, 19f);
            float[] unchanged = sharedInputOutput.clone();
            var overlap = state(executable, List.of(borrow(sharedInputOutput, 0, 96),
                    borrow(scale, 0, 3), borrow(bias, 0, 3), borrow(mean, 0, 3),
                    borrow(variance, 0, 3), borrow(sharedInputOutput, 5, 96)));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                assertArrayEquals(unchanged, sharedInputOutput);
            } finally { overlap.close(); }

            float[] sharedVectors = {2f, 3f, 4f};
            float[] aliasOutput = new float[96];
            var inputAlias = state(executable, List.of(borrow(input, 0, 96),
                    borrow(sharedVectors, 0, 3), borrow(sharedVectors, 0, 3),
                    borrow(sharedVectors, 0, 3), borrow(sharedVectors, 0, 3),
                    borrow(aliasOutput, 0, 96)));
            try { assertDoesNotThrow(() -> executable.bind(inputAlias).execute()); }
            finally { inputAlias.close(); }
        }
    }

    @Test void layerNormalizationUsesPrivateRangeScratchAndRejectsOverlapBeforeWrites() {
        var base = CpuTrailingNormalizationLoweringTest.context(true, false, DataType.FLOAT64,
                Shape.of(8, 4), Shape.of(4), List.of(0));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 2, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var declaration = analysis.plan().workspaceDeclaration().orElseThrow();
        assertAll(() -> assertTrue(analysis.plan().selectedRangeCount() >= 2),
                () -> assertEquals(analysis.plan().trailingNormalizationGeometry().orElseThrow()
                        .workspaceBytes(analysis.plan().selectedRangeCount()),
                        declaration.byteSize()));
        try (var workers = new CpuWorkerGroup(4)) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            double[] input = new double[32];
            for (int row = 0; row < 8; row++) for (int column = 0; column < 4; column++)
                input[row * 4 + column] = column - 2;
            double[] output = new double[34]; java.util.Arrays.fill(output, -77);
            var workspace = CpuContiguousWorkspace.allocate(declaration.byteSize(),
                    declaration.byteAlignment());
            var run = state(executable, List.of(borrow(input, 0, 32), borrow(output, 1, 32)),
                    List.of(workspace));
            try {
                var bound = executable.bind(run); bound.execute();
                double root = StrictMath.sqrt(1.25 + 1e-5);
                for (int row = 0; row < 8; row++) for (int column = 0; column < 4; column++)
                    assertEquals((column - 1.5) / root, output[1 + row * 4 + column], 2e-15);
                assertEquals(-77, output[0]); assertEquals(-77, output[33]);
                double[] first = output.clone(); bound.execute(); assertArrayEquals(first, output);
            } finally { run.close(); }

            double[] shared = new double[40]; java.util.Arrays.fill(shared, 19);
            double[] unchanged = shared.clone();
            var overlapWorkspace = CpuContiguousWorkspace.allocate(declaration.byteSize(),
                    declaration.byteAlignment());
            var overlap = state(executable, List.of(borrow(shared, 0, 32),
                    borrow(shared, 3, 32)), List.of(overlapWorkspace));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                assertArrayEquals(unchanged, shared);
            } finally { overlap.close(); }
        }
    }

    @Test void rmsNormalizationNeedsNoWorkspaceAndKeepsLargeFiniteRootsFinite() {
        var base = CpuTrailingNormalizationLoweringTest.context(false, false, DataType.FLOAT64,
                Shape.of(1, 2), Shape.of(2), List.of(0));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        assertTrue(analysis.plan().workspaceDeclaration().isEmpty());
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        double scale = Double.MAX_VALUE / 2.0;
        double[] input = {scale, scale / 2.0}, output = {-9, -9};
        var run = state(executable, List.of(borrow(input, 0, 2), borrow(output, 0, 2)));
        try {
            executable.bind(run).execute();
            double root = StrictMath.hypot(scale * StrictMath.sqrt(1.25 / 2.0),
                    StrictMath.sqrt(1e-5));
            assertAll(() -> assertTrue(Double.isFinite(root)),
                    () -> assertEquals(input[0] / root, output[0], 0.0),
                    () -> assertEquals(input[1] / root, output[1], 0.0));
        } finally { run.close(); }
    }

    @Test void softmaxValidatesBeforeWritesAndParallelizesOnlyCompleteSlices() {
        var base = CpuSoftmaxLoweringTest.context(SoftmaxKind.SOFTMAX, DataType.FLOAT64,
                Shape.of(8, 4), 1);
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        try (var workers = new CpuWorkerGroup(4)) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            double[] input = new double[32];
            for (int index = 0; index < input.length; index++) input[index] = index % 4 - 2;
            double[] output = new double[34]; java.util.Arrays.fill(output, -77);
            var run = state(executable, List.of(borrow(input, 0, 32), borrow(output, 1, 32)));
            try {
                var bound = executable.bind(run); bound.execute();
                for (int row = 0; row < 8; row++) assertEquals(1.0,
                        output[1 + row * 4] + output[2 + row * 4]
                                + output[3 + row * 4] + output[4 + row * 4], 2e-15);
                assertEquals(-77, output[0]); assertEquals(-77, output[33]);
                double[] first = output.clone(); bound.execute(); assertArrayEquals(first, output);
            } finally { run.close(); }

            double[] invalidInput = input.clone(); invalidInput[31] = Double.NaN;
            double[] untouched = new double[34]; java.util.Arrays.fill(untouched, -31);
            var invalid = state(executable, List.of(borrow(invalidInput, 0, 32),
                    borrow(untouched, 1, 32)));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(invalid));
                double[] expected = new double[34]; java.util.Arrays.fill(expected, -31);
                assertArrayEquals(expected, untouched);
            } finally { invalid.close(); }

            double[] shared = new double[40]; java.util.Arrays.fill(shared, 19);
            double[] unchanged = shared.clone();
            var overlap = state(executable, List.of(borrow(shared, 0, 32),
                    borrow(shared, 3, 32)));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                assertArrayEquals(unchanged, shared);
            } finally { overlap.close(); }
        }
    }

    @Test void advancedStatisticsOwnWholeParallelCellsAndRejectOverlapBeforeWrites() {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.VARIANCE,
                DataType.FLOAT64, Shape.of(8, 4),
                new StatisticalReductionAttrs(List.of(1), false, 1), Shape.of(8));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var declaration = analysis.plan().workspaceDeclaration().orElseThrow();
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            double[] input = new double[32];
            for (int row = 0; row < 8; row++) for (int column = 0; column < 4; column++)
                input[row * 4 + column] = row + column;
            double[] output = new double[10]; java.util.Arrays.fill(output, -77);
            var run = state(executable, List.of(borrow(input, 0, input.length),
                    borrow(output, 1, 8)), List.of(CpuContiguousWorkspace.allocate(
                            declaration.byteSize(), declaration.byteAlignment())));
            try {
                var bound = executable.bind(run); bound.execute();
                for (int row = 0; row < 8; row++) assertEquals(5.0 / 3.0, output[row + 1]);
                assertEquals(-77, output[0]); assertEquals(-77, output[9]);
                double[] first = output.clone(); bound.execute(); assertArrayEquals(first, output);
            } finally { run.close(); }

            double[] shared = new double[40]; java.util.Arrays.fill(shared, 19);
            double[] unchanged = shared.clone();
            var overlap = state(executable, List.of(borrow(shared, 0, 32),
                    borrow(shared, 3, 8)), List.of(CpuContiguousWorkspace.allocate(
                            declaration.byteSize(), declaration.byteAlignment())));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                assertArrayEquals(unchanged, shared);
            } finally { overlap.close(); }
        } finally { workers.close(); }
    }

    @Test void maskedReductionParallelRangesAreDeterministicAndColdFailuresPrecedeWrites() {
        var base = CpuMaskedReductionLoweringTest.context(AggregateReductionKind.MEAN,
                DataType.FLOAT64, Shape.of(8, 4), Shape.of(4), 1);
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.BYTE_ARRAY,
                                CarrierAccess.DOUBLE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var declaration = analysis.plan().workspaceDeclaration().orElseThrow();
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            double[] data = new double[32];
            for (int row = 0; row < 8; row++) {
                data[row * 4] = 0x1p53;
                data[row * 4 + 1] = row + 1;
                data[row * 4 + 2] = -0x1p53;
                data[row * 4 + 3] = Double.NaN;
            }
            byte[] mask = {1, 1, 1, 0};
            double[] output = new double[10]; java.util.Arrays.fill(output, -77);
            var run = state(executable, List.of(borrow(data, 0, data.length), borrow(mask),
                    borrow(output, 1, 8)), List.of(CpuContiguousWorkspace.allocate(
                            declaration.byteSize(), declaration.byteAlignment())));
            try {
                var bound = executable.bind(run);
                bound.execute();
                for (int row = 0; row < 8; row++) assertEquals((row + 1) / 3.0,
                        output[row + 1]);
                assertEquals(-77, output[0]); assertEquals(-77, output[9]);
                double[] first = output.clone();
                bound.execute();
                assertArrayEquals(first, output);
            } finally { run.close(); }

            byte[] invalidMask = {1, 2, 1, 0};
            double[] untouched = new double[10]; java.util.Arrays.fill(untouched, -31);
            var invalid = state(executable, List.of(borrow(data, 0, data.length),
                    borrow(invalidMask), borrow(untouched, 1, 8)), List.of(
                            CpuContiguousWorkspace.allocate(declaration.byteSize(),
                                    declaration.byteAlignment())));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(invalid));
                double[] expected = new double[10]; java.util.Arrays.fill(expected, -31);
                assertArrayEquals(expected, untouched);
            } finally { invalid.close(); }

            double[] shared = new double[40]; java.util.Arrays.fill(shared, 19);
            double[] unchanged = shared.clone();
            var overlap = state(executable, List.of(borrow(shared, 0, 32), borrow(mask),
                    borrow(shared, 3, 8)), List.of(CpuContiguousWorkspace.allocate(
                            declaration.byteSize(), declaration.byteAlignment())));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                assertArrayEquals(unchanged, shared);
            } finally { overlap.close(); }
        } finally { workers.close(); }
    }

    @Test void zeroInputInitializerExecutesItsSinglePrologueWithNoWorkspace() {
        var base = CpuRandomLoweringTest.initialContext(Long.MIN_VALUE, Long.MAX_VALUE);
        var analysis = new CpuPartitionPreparer().analyze(
                new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(),
                        base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                        new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.LONG_ARRAY))));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        long[] stateWords = new long[2];
        var run = state(executable, List.of(borrow(stateWords)));
        try { executable.bind(run).execute(); } finally { run.close(); }
        assertAll(() -> assertArrayEquals(new long[] {Long.MIN_VALUE, Long.MAX_VALUE}, stateWords),
                () -> assertTrue(executable.memoryPlan().workspaces().isEmpty()));
    }

    @Test void dropoutColdBindingExecutesPrologueAndParallelRangesBitwise() {
        int count = 32;
        var base = CpuRandomLoweringTest.dropoutContext(DataType.FLOAT32, Shape.of(count), .25);
        var carriers = List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY,
                CarrierAccess.FLOAT_ARRAY, CarrierAccess.BYTE_ARRAY, CarrierAccess.LONG_ARRAY);
        float[] input = new float[count]; for (int i = 0; i < count; i++) input[i] = i - 7.5f;
        long[] rng = {0x1234, Long.MAX_VALUE - 5};
        float[] scalarOutput = new float[count], parallelOutput = new float[count];
        byte[] scalarMask = new byte[count], parallelMask = new byte[count];
        long[] scalarNext = new long[2], parallelNext = new long[2];

        var scalarContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers));
        var scalar = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(scalarContext), Optional.empty());
        var scalarRun = state(scalar, List.of(borrow(input, 0, count), borrow(rng),
                borrow(scalarOutput, 0, count), borrow(scalarMask), borrow(scalarNext)));
        try { scalar.bind(scalarRun).execute(); } finally { scalarRun.close(); }

        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var parallelContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers, config));
        var analysis = new CpuPartitionPreparer().analyze(parallelContext);
        try (var workers = new CpuWorkerGroup(4)) {
            var parallel = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty(),
                    Optional.of(workers));
            var run = state(parallel, List.of(borrow(input, 0, count), borrow(rng),
                    borrow(parallelOutput, 0, count), borrow(parallelMask), borrow(parallelNext)));
            try { parallel.bind(run).execute(); } finally { run.close(); }
        }
        assertAll(() -> assertArrayEquals(scalarMask, parallelMask),
                () -> assertArrayEquals(java.util.stream.IntStream.range(0, count)
                                .map(i -> Float.floatToRawIntBits(scalarOutput[i])).toArray(),
                        java.util.stream.IntStream.range(0, count)
                                .map(i -> Float.floatToRawIntBits(parallelOutput[i])).toArray()),
                () -> assertArrayEquals(scalarNext, parallelNext),
                () -> assertArrayEquals(new long[] {rng[0], rng[1] + count}, scalarNext),
                () -> assertTrue(analysis.plan().workspaceDeclaration().isEmpty()));
    }

    @Test void dropoutRejectsEveryInputOutputAndOutputPairOverlapBeforeMutation() {
        var base = CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64, Shape.of(3), .5);
        var carriers = java.util.Collections.nCopies(5, CarrierAccess.MEMORY_SEGMENT);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());
        try (Arena arena = Arena.ofConfined()) {
            for (int left = 0; left < 5; left++) for (int right = left + 1; right < 5; right++) {
                if (left < 2 && right < 2) continue;
                MemorySegment shared = arena.allocate(256, 8); shared.fill((byte) 0x5a);
                var resources = new ArrayList<io.github.pho001.synaptik.runtime.resource.BufferRepresentation>();
                for (int index = 0; index < 5; index++) {
                    DataType type = List.of(DataType.FLOAT64, DataType.INT64, DataType.FLOAT64,
                            DataType.BOOL, DataType.INT64).get(index);
                    long elements = index == 1 || index == 4 ? 2 : 3;
                    long offset = index == left || index == right ? 0 : 64 + index * 32L;
                    resources.add(borrowed(type, elements,
                            shared.asSlice(offset, elements * type.byteWidth())));
                }
                var run = state(executable, resources);
                try {
                    assertThrows(IllegalArgumentException.class, () -> executable.bind(run),
                            left + ":" + right);
                    assertEquals((byte) 0x5a, shared.get(ValueLayout.JAVA_BYTE, 0));
                } finally { run.close(); }
            }
        }
    }

    @Test void dropoutPermitsPhysicalInputInputOverlapAndExecutesCoherently() {
        var base = CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64, Shape.of(2), 0.0d);
        var carriers = java.util.Collections.nCopies(5, CarrierAccess.MEMORY_SEGMENT);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sharedInputs = arena.allocate(16, 8);
            long key = Double.doubleToRawLongBits(1.5);
            long counter = Double.doubleToRawLongBits(-2.25);
            sharedInputs.set(ValueLayout.JAVA_LONG, 0, key);
            sharedInputs.set(ValueLayout.JAVA_LONG, 8, counter);
            MemorySegment output = arena.allocate(16, 8);
            MemorySegment mask = arena.allocate(2, 1);
            MemorySegment nextState = arena.allocate(16, 8);
            var run = state(executable, List.of(
                    borrowed(DataType.FLOAT64, 2, sharedInputs),
                    borrowed(DataType.INT64, 2, sharedInputs),
                    borrowed(DataType.FLOAT64, 2, output),
                    borrowed(DataType.BOOL, 2, mask),
                    borrowed(DataType.INT64, 2, nextState)));
            try { executable.bind(run).execute(); } finally { run.close(); }
            assertAll(() -> assertEquals(Double.doubleToRawLongBits(1.5),
                            Double.doubleToRawLongBits(output.get(DOUBLE, 0))),
                    () -> assertEquals(Double.doubleToRawLongBits(-2.25),
                            Double.doubleToRawLongBits(output.get(DOUBLE, 8))),
                    () -> assertEquals((byte) 1, mask.get(ValueLayout.JAVA_BYTE, 0)),
                    () -> assertEquals((byte) 1, mask.get(ValueLayout.JAVA_BYTE, 1)),
                    () -> assertEquals(key, nextState.get(ValueLayout.JAVA_LONG, 0)),
                    () -> assertEquals(counter + 2, nextState.get(ValueLayout.JAVA_LONG, 8)));
        }
    }

    @Test void topKColdBindingExecutesBothOutputsAndRejectsAllOverlapBeforeMutation() {
        var base = CpuOrderingLoweringTest.context(new Operation(
                io.github.pho001.synaptik.model.operation.ordering.TopKKind.TOP_K,
                new io.github.pho001.synaptik.model.operation.ordering.TopKAttrs(1, 3, true, false)),
                DataType.FLOAT32, Shape.of(2, 5), Shape.of(2, 3), true);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT)));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(10L * 4, 8);
            float[] source = {3, 1, 4, 2, 0, 9, 7, 8, 6, 5};
            for (int i = 0; i < source.length; i++) input.set(FLOAT, i * 4L, source[i]);
            MemorySegment values = arena.allocate(6L * 4, 8);
            MemorySegment indices = arena.allocate(6L * 8, 8);
            var workspaceEntry = executable.memoryPlan().workspaces().getFirst();
            var workspace = CpuContiguousWorkspace.allocate(workspaceEntry.byteSize(), 8);
            var run = state(executable, List.of(borrowed(DataType.FLOAT32, 10, input),
                    borrowed(DataType.FLOAT32, 6, values), borrowed(DataType.INT64, 6, indices)),
                    List.of(workspace));
            try {
                executable.bind(run).execute();
                assertArrayEquals(new long[]{0, 2, 3, 0, 1, 2}, new long[]{
                        indices.get(ValueLayout.JAVA_LONG, 0), indices.get(ValueLayout.JAVA_LONG, 8),
                        indices.get(ValueLayout.JAVA_LONG, 16), indices.get(ValueLayout.JAVA_LONG, 24),
                        indices.get(ValueLayout.JAVA_LONG, 32), indices.get(ValueLayout.JAVA_LONG, 40)});
            } finally { run.close(); }

            MemorySegment shared = arena.allocate(48, 8); shared.fill((byte) 0x5a);
            var overlapWorkspace = CpuContiguousWorkspace.allocate(workspaceEntry.byteSize(), 8);
            var overlap = state(executable, List.of(borrowed(DataType.FLOAT32, 10, input),
                    borrowed(DataType.FLOAT32, 6, shared.asSlice(0, 24)),
                    borrowed(DataType.INT64, 6, shared.asSlice(0, 48))), List.of(overlapWorkspace));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                assertEquals((byte) 0x5a, shared.get(ValueLayout.JAVA_BYTE, 0));
            } finally { overlap.close(); }
        }
    }

    @Test void sortAndArgsortRejectOverlapOutsideTheSliceOrdinalPrefixBeforeMutation() {
        for (var family : List.of(
                io.github.pho001.synaptik.model.operation.ordering.OrderingKind.SORT,
                io.github.pho001.synaptik.model.operation.ordering.OrderingKind.ARGSORT)) {
            var base = CpuOrderingLoweringTest.context(new Operation(family,
                            new io.github.pho001.synaptik.model.operation.ordering.SortAttrs(1, false)),
                    DataType.INT64, Shape.of(4, 5), Shape.of(4, 5), false);
            var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                    base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                    base.constants(), new CpuPartitionAnalysisInputs(false,
                            List.of(CarrierAccess.LONG_ARRAY, CarrierAccess.LONG_ARRAY)));
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                    new CpuPartitionPreparer().analyze(context), Optional.empty());
            long[] shared = new long[30];
            java.util.Arrays.fill(shared, 0x5a5a5a5a5a5a5a5aL);
            long[] unchanged = shared.clone();
            var workspaceEntry = executable.memoryPlan().workspaces().getFirst();
            var workspace = CpuContiguousWorkspace.allocate(workspaceEntry.byteSize(),
                    workspaceEntry.byteAlignment());
            var run = state(executable, List.of(borrow(shared, 0, 20),
                    borrow(shared, 10, 20)), List.of(workspace));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(run),
                        family.name());
                assertArrayEquals(unchanged, shared, family.name());
            } finally { run.close(); }
        }
    }

    @Test void orderingParallelSlicesMatchScalarBitwiseWithDisjointScratch() {
        var operation = new Operation(io.github.pho001.synaptik.model.operation.ordering.OrderingKind.SORT,
                new io.github.pho001.synaptik.model.operation.ordering.SortAttrs(1, true));
        var base = CpuOrderingLoweringTest.context(operation, DataType.FLOAT32,
                Shape.of(4, 5), Shape.of(4, 5), false);
        var parallel = new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY), parallel));
        var analysis = new CpuPartitionPreparer().analyze(context);
        float[] input = {1, 5, 3, 2, 4, -0.0f, +0.0f, Float.NaN, 9, 9,
                -4, -2, -3, -1, -5, Float.POSITIVE_INFINITY, 7, 8, 6, Float.NEGATIVE_INFINITY};
        float[] actual = new float[20];
        try (var workers = new CpuWorkerGroup(2)) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            var entry = executable.memoryPlan().workspaces().getFirst();
            var workspace = CpuContiguousWorkspace.allocate(entry.byteSize(), entry.byteAlignment());
            var run = state(executable, List.of(borrow(input, 0, input.length),
                    borrow(actual, 0, actual.length)), List.of(workspace));
            try { executable.bind(run).execute(); } finally { run.close(); }
        }
        float[] expected = {5,4,3,2,1, 9,9,+0.0f,-0.0f,Float.NaN,
                -1,-2,-3,-4,-5, Float.POSITIVE_INFINITY,8,7,6,Float.NEGATIVE_INFINITY};
        assertArrayEquals(java.util.stream.IntStream.range(0, expected.length)
                        .map(i -> Float.floatToRawIntBits(expected[i])).toArray(),
                java.util.stream.IntStream.range(0, actual.length)
                        .map(i -> Float.floatToRawIntBits(actual[i])).toArray());
    }

    @Test void executesParallelVectorChunksWithArbitraryBoundsAndScalarTails() {
        int count = DoubleVector.SPECIES_PREFERRED.length() * 4 + 3;
        Shape shape = Shape.of(count);
        var descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var heap = List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY);
        var config = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 3, 3,
                DoubleVector.SPECIES_PREFERRED.length());
        var analysis = CpuPartitionPreparerTest.analyze(descriptor, descriptor, descriptor,
                descriptor, new CpuPartitionAnalysisInputs(false, heap, config));
        try (var workers = new CpuWorkerGroup(3)) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers)).forRange(1, count - 1);
            double[] a = new double[count], b = new double[count], c = new double[count];
            double[] output = new double[count];
            java.util.Arrays.fill(output, 123.0);
            for (int index = 0; index < count; index++) {
                a[index] = index * 0.25 - 3.0; b[index] = 0.75; c[index] = -1.5;
            }
            var state = state(executable, List.of(borrow(a, 0, count), borrow(b, 0, count),
                    borrow(c, 0, count), borrow(output, 0, count)));
            try {
                executable.bind(state).execute();
                assertEquals(123.0, output[0]);
                assertEquals(123.0, output[count - 1]);
                for (int index = 1; index < count - 1; index++) assertEquals(
                        CpuScalarReferenceKernel.gelu(a[index] + b[index]) * c[index],
                        output[index], 2e-7 * Math.max(1.0, Math.abs(output[index])),
                        "logical index " + index);
            } finally { state.close(); }
        }
    }

    @Test void coldBindsOnceAndExecutesThroughDirectPartitionHandle() {
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(Shape.of(4), Optional.empty());
        var buffers = new ArrayList<CpuNativeBuffer>();
        var bindings = new ArrayList<List<BufferRepresentationBinding>>();
        for (var entry : executable.memoryPlan().buffers()) {
            var buffer = CpuNativeBuffer.allocate(DataType.FLOAT64, entry.byteSize(), entry.byteAlignment());
            buffers.add(buffer);
            bindings.add(List.of(new BufferRepresentationBinding(buffer, RunResourceOwnership.RUN_OWNED)));
        }
        var state = new RunState(executable.memoryPlan(), bindings, List.of());
        try {
            for (int i = 0; i < 4; i++) {
                buffers.get(0).segment().set(DOUBLE, i * 8L, i - 1.0);
                buffers.get(1).segment().set(DOUBLE, i * 8L, 0.5);
                buffers.get(2).segment().set(DOUBLE, i * 8L, 3.0);
            }
            var invocation = executable.bind(state);
            invocation.execute();
            for (int i = 0; i < 4; i++) assertEquals(
                    CpuScalarReferenceKernel.gelu((i - 1.0) + 0.5) * 3.0,
                    buffers.get(3).segment().get(DOUBLE, i * 8L), 0.0);
            state.close();
            assertThrows(IllegalStateException.class, invocation::execute);
        } finally { if (!state.isClosed()) state.close(); }
    }

    @Test void copiesSelectedGeneralInputOnceIntoRunWorkspaceBeforeConsumer() {
        Shape shape = Shape.of(2, 3);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var general = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)), false);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                0, 1, 10, 1, 2, 48, 1, 1);
        var originalPattern = List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT);
        var analysis = CpuPartitionPreparerTest.analyze(dense, dense, general, dense,
                new CpuPartitionAnalysisInputs(false,
                        originalPattern,
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy));
        assertEquals(2, analysis.plan().materialization().orElseThrow().sourceBoundaryIndex());
        assertAll(
                () -> assertEquals(CarrierAccess.DOUBLE_ARRAY,
                        analysis.plan().carrierPattern().get(2)),
                () -> assertEquals(CarrierAccess.MEMORY_SEGMENT,
                        analysis.plan().generatedCarrierPattern().get(2)));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var buffers = new ArrayList<io.github.pho001.synaptik.runtime.resource.BufferRepresentation>();
        var bindings = new ArrayList<List<BufferRepresentationBinding>>();
        for (int index = 0; index < executable.memoryPlan().buffers().size(); index++) {
            var entry = executable.memoryPlan().buffers().get(index);
            var buffer = index == 2 ? borrow(new double[6], 0, 6)
                    : CpuNativeBuffer.allocate(DataType.FLOAT64, entry.byteSize(), entry.byteAlignment());
            buffers.add(buffer);
            bindings.add(List.of(new BufferRepresentationBinding(buffer, index == 2
                    ? RunResourceOwnership.BORROWED : RunResourceOwnership.RUN_OWNED)));
        }
        var workspaceEntry = executable.memoryPlan().workspaces().getFirst();
        var workspace = CpuContiguousWorkspace.allocate(workspaceEntry.byteSize(),
                workspaceEntry.byteAlignment());
        var state = new RunState(executable.memoryPlan(), bindings, List.of(workspace));
        try {
            for (int logical = 0; logical < 6; logical++) {
                segment(buffers.get(0)).set(DOUBLE, logical * 8L, logical - 2.0);
                segment(buffers.get(1)).set(DOUBLE, logical * 8L, 0.5);
            }
            for (int row = 0; row < 2; row++) for (int column = 0; column < 3; column++) {
                long address = row + column * 2L;
                segment(buffers.get(2)).set(DOUBLE, address * 8L, 2.0 + row + column);
            }
            executable.bind(state).execute();
            for (int row = 0; row < 2; row++) for (int column = 0; column < 3; column++) {
                int logical = row * 3 + column;
                double c = 2.0 + row + column;
                assertEquals(CpuScalarReferenceKernel.gelu(logical - 1.5) * c,
                        segment(buffers.get(3)).get(DOUBLE, logical * 8L), 0.0);
            }
        } finally { state.close(); }
        assertFalse(workspace.isAccessible());
    }

    @Test void handlesScalarAndZeroElementBindings() {
        assertEquals(1, CpuPartitionFinalizerTest.finalizeExecutable(
                Shape.scalar(), Optional.empty()).binding().elementCount());
        assertEquals(0, CpuPartitionFinalizerTest.finalizeExecutable(
                Shape.of(2, 0, 3), Optional.empty()).binding().elementCount());
    }

    @Test void zeroRangeSkipsCopyAndCopyFailurePreventsConsumerExecution() {
        Shape shape = Shape.of(2, 3);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var general = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)), false);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 2, 48, 1, 1);
        var analysis = CpuPartitionPreparerTest.analyze(general, dense, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy));

        var zero = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty())
                .forRange(3, 3);
        var zeroResources = nativeResources(zero);
        var zeroWorkspace = CpuContiguousWorkspace.allocate(48, 8);
        zeroWorkspace.writableSegment().set(DOUBLE, 0, 456.0);
        segment(zeroResources.get(3)).set(DOUBLE, 24, 789.0);
        var zeroState = state(zero, zeroResources, List.of(zeroWorkspace));
        try {
            zero.bind(zeroState).execute();
            assertAll(
                    () -> assertEquals(456.0,
                            zeroWorkspace.writableSegment().get(DOUBLE, 0)),
                    () -> assertEquals(789.0,
                            segment(zeroResources.get(3)).get(DOUBLE, 24)));
        } finally { zeroState.close(); }

        var failing = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var failureResources = nativeResources(failing);
        var failureWorkspace = CpuContiguousWorkspace.allocate(48, 8);
        segment(failureResources.get(3)).set(DOUBLE, 0, 321.0);
        var failureState = state(failing, failureResources, List.of(failureWorkspace));
        try {
            var invocation = failing.bind(failureState);
            failureResources.getFirst().close();
            assertAll(
                    () -> assertThrows(IllegalStateException.class, invocation::execute),
                    () -> assertEquals(321.0,
                            segment(failureResources.get(3)).get(DOUBLE, 0)));
        } finally { failureState.close(); }
    }

    @Test void materializationDoesNotHideOriginalSourceOutputOverlap() {
        Shape shape = Shape.of(2, 3);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var general = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)), false);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 2, 48, 1, 1);
        var pattern = List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.DOUBLE_ARRAY);
        var analysis = CpuPartitionPreparerTest.analyze(general, dense, dense, dense,
                new CpuPartitionAnalysisInputs(false, pattern,
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        double[] sharedCarrier = new double[6];
        var sharedInput = borrow(sharedCarrier, 0, 6);
        var sharedOutput = borrow(sharedCarrier, 0, 6);
        var b = CpuNativeBuffer.allocate(DataType.FLOAT64, 48, 8);
        var c = CpuNativeBuffer.allocate(DataType.FLOAT64, 48, 8);
        var workspace = CpuContiguousWorkspace.allocate(48, 8);
        var state = state(executable, List.of(sharedInput, b, c, sharedOutput), List.of(workspace));
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(state));
        } finally { state.close(); }
    }

    @Test void executesAllSixteenDirectCarrierPatternsForEveryEligibleStrategy() {
        int count = DoubleVector.SPECIES_PREFERRED.length() * 2 + 1;
        Shape shape = Shape.of(count);
        var descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var configurations = List.of(
                new PortableExecutionConfig(ComputePreference.SCALAR, 1, 1, 1),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1),
                new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 2, 2, 1));
        var strategies = List.of("scalar", "vector", "parallel-scalar", "parallel-vector");
        for (int strategy = 0; strategy < configurations.size(); strategy++) {
            for (int mask = 0; mask < 16; mask++) executeCarrierPattern(descriptor, count, mask,
                    configurations.get(strategy), strategies.get(strategy), DataType.FLOAT64);
        }
        int floatCount = FloatVector.SPECIES_PREFERRED.length() * 2 + 1;
        Shape floatShape = Shape.of(floatCount);
        var floatDescriptor = new TensorDescriptor(DataType.FLOAT32, floatShape,
                Optional.of(LayoutDescriptor.contiguous(floatShape)), false);
        var floatConfigurations = List.of(
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 2, 2, 1));
        for (int strategy = 0; strategy < floatConfigurations.size(); strategy++) {
            for (int mask = 0; mask < 16; mask++) executeCarrierPattern(floatDescriptor,
                    floatCount, mask, floatConfigurations.get(strategy),
                    strategy == 0 ? "vector" : "parallel-vector", DataType.FLOAT32);
        }
    }

    private static void executeCarrierPattern(TensorDescriptor descriptor, int count, int mask,
            PortableExecutionConfig config, String expectedStrategy, DataType dataType) {
            var pattern = new ArrayList<CarrierAccess>();
            for (int i = 0; i < 4; i++) pattern.add((mask & (1 << i)) != 0
                    ? dataType == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY
                            : CarrierAccess.DOUBLE_ARRAY : CarrierAccess.MEMORY_SEGMENT);
            var analysis = CpuPartitionPreparerTest.analyze(descriptor, descriptor, descriptor,
                    descriptor, new CpuPartitionAnalysisInputs(false, pattern, config));
            assertEquals(expectedStrategy, analysis.plan().executionStrategy().toString());
            CpuWorkerGroup workers = expectedStrategy.startsWith("parallel")
                    ? new CpuWorkerGroup(2) : null;
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty(),
                    Optional.ofNullable(workers));
            var resources = new ArrayList<io.github.pho001.synaptik.runtime.resource.BufferRepresentation>();
            var bindings = new ArrayList<List<BufferRepresentationBinding>>();
            for (int i = 0; i < 4; i++) {
                io.github.pho001.synaptik.runtime.resource.BufferRepresentation resource;
                RunResourceOwnership ownership;
                if (pattern.get(i) != CarrierAccess.MEMORY_SEGMENT) {
                    var storage = new MemorySegmentStorage(dataType, count,
                            dataType == DataType.FLOAT32 ? MemorySegment.ofArray(new float[count])
                                    : MemorySegment.ofArray(new double[count]));
                    resource = CpuBorrowedBuffer.borrow(storage);
                    ownership = RunResourceOwnership.BORROWED;
                } else {
                    resource = CpuNativeBuffer.allocate(dataType,
                            (long) count * dataType.byteWidth(), dataType.byteWidth());
                    ownership = RunResourceOwnership.RUN_OWNED;
                }
                resources.add(resource);
                bindings.add(List.of(new BufferRepresentationBinding(resource, ownership)));
            }
            var state = new RunState(executable.memoryPlan(), bindings, List.of());
            try {
                for (int i = 0; i < count; i++) {
                    if (dataType == DataType.FLOAT32) {
                        segment(resources.get(0)).set(FLOAT, i * 4L, i - 1.0f);
                        segment(resources.get(1)).set(FLOAT, i * 4L, 0.5f);
                        segment(resources.get(2)).set(FLOAT, i * 4L, 2.0f);
                    } else {
                        segment(resources.get(0)).set(DOUBLE, i * 8L, i - 1.0);
                        segment(resources.get(1)).set(DOUBLE, i * 8L, 0.5);
                        segment(resources.get(2)).set(DOUBLE, i * 8L, 2.0);
                    }
                }
                executable.bind(state).execute();
                for (int i = 0; i < count; i++) {
                    if (dataType == DataType.FLOAT32) {
                        float sum = (float) ((i - 1.0f) + 0.5f);
                        float expected = (float) ((float) CpuScalarReferenceKernel.gelu(sum) * 2.0f);
                        assertEquals(expected, segment(resources.get(3)).get(FLOAT, i * 4L),
                                Math.max(2e-5f, 2e-5f * Math.abs(expected)),
                                expectedStrategy + " carrier mask " + mask + " index " + i);
                    } else assertEquals(CpuScalarReferenceKernel.gelu(i - 0.5) * 2.0,
                            segment(resources.get(3)).get(DOUBLE, i * 8L),
                            2e-7 * Math.max(1.0, Math.abs(i - 0.5)),
                            expectedStrategy + " carrier mask " + mask + " index " + i);
                }
            } finally {
                state.close();
                if (workers != null) workers.close();
            }
    }

    @Test void rejectsConfinedSegmentsBeforeParallelExecution() {
        int count = 8;
        Shape shape = Shape.of(count);
        var descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1);
        var analysis = CpuPartitionPreparerTest.analyze(descriptor, descriptor, descriptor,
                descriptor, new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), config));
        try (var workers = new CpuWorkerGroup(2); var arena = Arena.ofConfined()) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            var resources = new ArrayList<CpuBorrowedBuffer>();
            for (int i = 0; i < 4; i++) resources.add(CpuBorrowedBuffer.borrow(
                    new MemorySegmentStorage(DataType.FLOAT64, count,
                            arena.allocate(count * 8L, 8))));
            resources.get(3).segment().set(DOUBLE, 0, 777.0);
            var state = state(executable, resources);
            try {
                var failure = assertThrows(IllegalArgumentException.class,
                        () -> executable.bind(state));
                assertAll(
                        () -> assertEquals("segment is not accessible to every CPU worker",
                                failure.getMessage()),
                        () -> assertEquals(777.0, resources.get(3).segment().get(DOUBLE, 0)));
            } finally { state.close(); }
        }
    }

    @Test void acceptsProvedDisjointSameArraySlicesAndRejectsActualOverlap() {
        Shape shape = Shape.of(4);
        var descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var heap = List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY);
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                CpuPartitionPreparerTest.analyze(descriptor, descriptor, descriptor, descriptor,
                        new CpuPartitionAnalysisInputs(false, heap)), Optional.empty());
        double[] shared = new double[8];
        var a = borrow(shared, 0); var b = borrow(new double[4], 0);
        var c = borrow(new double[4], 0); var output = borrow(shared, 4);
        var disjoint = state(executable, List.of(a, b, c, output));
        try { assertDoesNotThrow(() -> executable.bind(disjoint)); }
        finally { disjoint.close(); }
        var overlapping = state(executable, List.of(a, b, c, borrow(shared, 0)));
        try { assertThrows(IllegalArgumentException.class, () -> executable.bind(overlapping)); }
        finally { overlapping.close(); }
    }

    @Test void rejectsConcreteCarrierPatternDifferentFromPreparedSpecialization() {
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(Shape.of(4), Optional.empty());
        var state = state(executable, List.of(borrow(new double[4], 0),
                borrow(new double[4], 0), borrow(new double[4], 0),
                borrow(new double[4], 0)));
        try { assertThrows(IllegalArgumentException.class, () -> executable.bind(state)); }
        finally { state.close(); }
    }

    @Test void affineColdBindingRejectsOverlappingSourceAndResultAddresses() {
        var context = CpuAffineLayoutLoweringTest.select(DataType.INT32,
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());
        int[] shared = new int[9];
        var input = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.INT32, 9,
                MemorySegment.ofArray(shared)));
        var output = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.INT32, 8,
                MemorySegment.ofArray(shared).asSlice(0, 32)));
        var run = state(executable, List.of(input, output));
        try { assertThrows(IllegalArgumentException.class, () -> executable.bind(run)); }
        finally { run.close(); }
    }

    @Test void movementColdBindingExecutesRangesAndRejectsOutputInputOverlapBeforeWrite() {
        var base = CpuNonAffineMovementLoweringTest.context(new Operation(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int32(-7))),
                List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(2))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY)));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty()).forRange(1, 4);
        int[] input = {10, 20};
        int[] output = {99, 99, 99, 99, 99};
        var run = state(executable, List.of(borrow(input), borrow(output)));
        try {
            executable.bind(run).execute();
            assertArrayEquals(new int[]{99, 10, 20, -7, 99}, output);
        } finally { run.close(); }

        int[] shared = {10, 20, 77, 77, 77};
        var overlap = state(executable, List.of(borrow(shared, 0, 2), borrow(shared, 0, 5)));
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
            assertArrayEquals(new int[]{10, 20, 77, 77, 77}, shared);
        } finally { overlap.close(); }
    }

    @Test void windowMovementExecutesParallelChunksAndRejectsNoncanonicalBoolBeforeWrite() {
        var base = CpuNonAffineMovementLoweringTest.context(
                new Operation(WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(0, 2, 1)),
                List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(6))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5, 2)));
        var parallelContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY),
                        new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1)));
        var analysis = new CpuPartitionPreparer().analyze(parallelContext);
        try (var workers = new CpuWorkerGroup(2)) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            int[] output = new int[10];
            var run = state(executable, List.of(borrow(new int[]{1, 2, 3, 4, 5, 6}),
                    borrow(output)));
            try {
                executable.bind(run).execute();
                assertArrayEquals(new int[]{1, 2, 2, 3, 3, 4, 4, 5, 5, 6}, output);
            } finally { run.close(); }
        }

        var boolBase = CpuNonAffineMovementLoweringTest.context(
                new Operation(WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(0, 2, 1)),
                List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.BOOL, Shape.of(3))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 2)));
        var boolContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                boolBase.partition(), boolBase.nodes(), boolBase.values(),
                boolBase.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.BYTE_ARRAY, CarrierAccess.BYTE_ARRAY)));
        var boolExecutable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(boolContext), Optional.empty());
        byte[] boolOutput = {9, 9, 9, 9};
        var boolRun = state(boolExecutable,
                List.of(borrow(new byte[]{1, 2, 0}), borrow(boolOutput)));
        try {
            assertThrows(IllegalArgumentException.class, () -> boolExecutable.bind(boolRun));
            assertArrayEquals(new byte[]{9, 9, 9, 9}, boolOutput);
        } finally { boolRun.close(); }
    }

    @Test void sliceUpdateExecutesParallelMixedCarriersWithArbitraryResolvedLayouts() {
        Shape baseShape = Shape.of(3, 4), updateShape = Shape.of(2, 2);
        var baseDescriptor = new TensorDescriptor(DataType.INT32, baseShape,
                Optional.of(LayoutDescriptor.of(baseShape, new long[]{0, 2}, 1, true)), false);
        var updateDescriptor = new TensorDescriptor(DataType.INT32, updateShape,
                Optional.of(LayoutDescriptor.of(updateShape, new long[]{3, 1}, 1, true)), false);
        var outputDescriptor = new TensorDescriptor(DataType.INT32, baseShape,
                Optional.of(LayoutDescriptor.of(baseShape, new long[]{10, 2}, 2, true)), false);
        var base = CpuNonAffineMovementLoweringTest.context(
                new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(2L, 3L), List.of(2L, 2L), List.of(0, 1),
                                List.of(-2L, -2L))),
                List.of(0, 1), List.of(baseDescriptor, updateDescriptor), outputDescriptor);
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 3, 3, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                                CarrierAccess.INT_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        int[] baseBits = {99, 10, 99, 20, 99, 30, 99, 40};
        int[] baseSnapshot = baseBits.clone();
        int[] outputBits = new int[30];
        java.util.Arrays.fill(outputBits, -7);
        try (var workers = new CpuWorkerGroup(3)) {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            var update = CpuNativeBuffer.allocate(DataType.INT32,
                    executable.memoryPlan().buffers().get(1).byteSize(), Integer.BYTES);
            update.segment().set(INT, 4, 90);
            update.segment().set(INT, 8, 91);
            update.segment().set(INT, 16, 80);
            update.segment().set(INT, 20, 81);
            int outputCount = Math.toIntExact(
                    executable.memoryPlan().buffers().get(2).byteSize() / Integer.BYTES);
            var run = state(executable, List.of(borrow(baseBits), update,
                    borrow(outputBits, 0, outputCount)));
            try {
                executable.bind(run).execute();
                assertAll(
                        () -> assertArrayEquals(baseSnapshot, baseBits),
                        () -> assertEquals(90, update.segment().get(INT, 4)),
                        () -> assertEquals(91, update.segment().get(INT, 8)),
                        () -> assertEquals(80, update.segment().get(INT, 16)),
                        () -> assertEquals(81, update.segment().get(INT, 20)),
                        () -> assertEquals(10, outputBits[2]),
                        () -> assertEquals(81, outputBits[4]),
                        () -> assertEquals(30, outputBits[6]),
                        () -> assertEquals(80, outputBits[8]),
                        () -> assertEquals(10, outputBits[12]),
                        () -> assertEquals(40, outputBits[18]),
                        () -> assertEquals(10, outputBits[22]),
                        () -> assertEquals(91, outputBits[24]),
                        () -> assertEquals(30, outputBits[26]),
                        () -> assertEquals(90, outputBits[28]));
            } finally { run.close(); }
        }
    }

    @Test void sliceUpdateExecutesAllMemorySegmentCarriers() {
        var base = CpuNonAffineMovementLoweringTest.context(
                new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(4L), List.of(2L), List.of(0), List.of(-2L))),
                List.of(0, 1), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT32, Shape.of(5)),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                                CarrierAccess.MEMORY_SEGMENT)));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());
        var resources = executable.memoryPlan().buffers().stream().map(entry ->
                CpuNativeBuffer.allocate(DataType.INT32, entry.byteSize(), Integer.BYTES)).toList();
        for (int index = 0; index < 5; index++) {
            resources.get(0).segment().set(INT, index * 4L, 10 + index);
        }
        resources.get(1).segment().set(INT, 0, 90);
        resources.get(1).segment().set(INT, 4, 80);
        var run = state(executable, resources);
        try {
            executable.bind(run).execute();
            assertAll(
                    () -> assertEquals(10, resources.get(2).segment().get(INT, 0)),
                    () -> assertEquals(11, resources.get(2).segment().get(INT, 4)),
                    () -> assertEquals(80, resources.get(2).segment().get(INT, 8)),
                    () -> assertEquals(13, resources.get(2).segment().get(INT, 12)),
                    () -> assertEquals(90, resources.get(2).segment().get(INT, 16)));
        } finally { run.close(); }
    }

    @Test void sliceUpdateBindingRejectsBoolAndOutputOverlapBeforeWritingButAllowsInputAlias() {
        var base = CpuNonAffineMovementLoweringTest.context(
                new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(1L), List.of(2L), List.of(0), List.of(1L))),
                List.of(0, 1), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.BOOL, Shape.of(4)),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.BOOL, Shape.of(2))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.BOOL, Shape.of(4)));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.BYTE_ARRAY, CarrierAccess.BYTE_ARRAY,
                                CarrierAccess.BYTE_ARRAY)));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());

        byte[] invalidOutput = {7, 7, 7, 7};
        var invalid = state(executable, List.of(borrow(new byte[]{0, 2, 0, 1}),
                borrow(new byte[]{1, 0}), borrow(invalidOutput)));
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(invalid));
            assertArrayEquals(new byte[]{7, 7, 7, 7}, invalidOutput);
        } finally { invalid.close(); }

        byte[] invalidUpdateOutput = {7, 7, 7, 7};
        var invalidUpdate = state(executable, List.of(borrow(new byte[]{0, 1, 0, 1}),
                borrow(new byte[]{1, 2}), borrow(invalidUpdateOutput)));
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(invalidUpdate));
            assertArrayEquals(new byte[]{7, 7, 7, 7}, invalidUpdateOutput);
        } finally { invalidUpdate.close(); }

        byte[] sharedOutput = {0, 1, 0, 1};
        var overlap = state(executable, List.of(borrow(sharedOutput),
                borrow(new byte[]{1, 0}), borrow(sharedOutput)));
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
            assertArrayEquals(new byte[]{0, 1, 0, 1}, sharedOutput);
        } finally { overlap.close(); }

        byte[] sharedUpdateOutput = {1, 0, 7, 7};
        var updateOverlap = state(executable, List.of(borrow(new byte[]{0, 1, 0, 1}),
                borrow(sharedUpdateOutput, 0, 2), borrow(sharedUpdateOutput)));
        try {
            assertThrows(IllegalArgumentException.class, () -> executable.bind(updateOverlap));
            assertArrayEquals(new byte[]{1, 0, 7, 7}, sharedUpdateOutput);
        } finally { updateOverlap.close(); }

        byte[] sharedInputs = {0, 1, 1, 0};
        byte[] aliasedOutput = {7, 7, 7, 7};
        var aliased = state(executable, List.of(borrow(sharedInputs),
                borrow(sharedInputs, 0, 2), borrow(aliasedOutput)));
        try {
            executable.bind(aliased).execute();
            assertArrayEquals(new byte[]{0, 0, 1, 0}, aliasedOutput);
        } finally { aliased.close(); }
    }

    @Test void indexingValidatesEveryIndexBeforeAnyOutputWrite() {
        var base = CpuIndexingLoweringTest.context(
                new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)), List.of(0),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(3))),
                CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(3, 3)));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.LONG_ARRAY, CarrierAccess.BYTE_ARRAY)));
        var executable = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());
        byte[] output = new byte[9]; java.util.Arrays.fill(output, (byte) 7);
        var invalid = state(executable, List.of(borrow(new long[]{2, -1, 3}), borrow(output)));
        try {
            var failure = assertThrows(IndexOutOfBoundsException.class,
                    () -> executable.bind(invalid).execute());
            assertAll(() -> assertEquals("ONE_HOT index at logical position 1 is out of bounds: "
                            + "value=-1, depth=3", failure.getMessage()),
                    () -> assertArrayEquals(new byte[]{7,7,7,7,7,7,7,7,7}, output));
        } finally { invalid.close(); }

        byte[] validOutput = new byte[9];
        var valid = state(executable, List.of(borrow(new long[]{2, 0, 1}), borrow(validOutput)));
        try {
            executable.bind(valid).execute();
            assertArrayEquals(new byte[]{0,0,1, 1,0,0, 0,1,0}, validOutput);
        } finally { valid.close(); }
    }

    @Test void everyIndexingFamilyReportsTheFirstNegativeOrUpperFailureWithoutWrites() {
        var gather = indexingExecutable(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(3))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(3)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.INT_ARRAY));
        int[] gatherOutput = {7, 7, 7};
        var gatherRun = state(gather, List.of(borrow(new int[]{10, 20}),
                borrow(new int[]{0, -1, 2}), borrow(gatherOutput)));
        try {
            var failure = assertThrows(IndexOutOfBoundsException.class,
                    () -> gather.bind(gatherRun).execute());
            assertAll(() -> assertEquals("GATHER index at logical position 1 for data axis 0 "
                            + "is out of bounds: value=-1, extent=2", failure.getMessage()),
                    () -> assertArrayEquals(new int[]{7, 7, 7}, gatherOutput));
        } finally { gatherRun.close(); }

        var elements = indexingExecutable(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                        new IndexAxisAttrs(1)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 2)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 2)),
                List.of(CarrierAccess.LONG_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.LONG_ARRAY));
        long[] elementsOutput = {7, 7, 7, 7};
        var elementsRun = state(elements, List.of(borrow(new long[]{10, 11, 20, 21}),
                borrow(new long[]{0, 1, 2, -1}), borrow(elementsOutput)));
        try {
            var failure = assertThrows(IndexOutOfBoundsException.class,
                    () -> elements.bind(elementsRun).execute());
            assertAll(() -> assertEquals("GATHER_ELEMENTS index at logical position 2 for data "
                            + "axis 1 is out of bounds: value=2, extent=2", failure.getMessage()),
                    () -> assertArrayEquals(new long[]{7, 7, 7, 7}, elementsOutput));
        } finally { elementsRun.close(); }

        var nd = indexingExecutable(new Operation(GatherNdKind.GATHER_ND,
                        new GatherNdAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2, 3)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.INT_ARRAY));
        int[] ndOutput = {7, 7};
        var ndRun = state(nd, List.of(borrow(new int[]{0, 1, 2, 3, 4, 5}),
                borrow(new int[]{1, 3, -1, 0}), borrow(ndOutput)));
        try {
            var failure = assertThrows(IndexOutOfBoundsException.class,
                    () -> nd.bind(ndRun).execute());
            assertAll(() -> assertEquals("GATHER_ND index at logical position 1 for data axis 1 "
                            + "is out of bounds: value=3, extent=3", failure.getMessage()),
                    () -> assertArrayEquals(new int[]{7, 7}, ndOutput));
        } finally { ndRun.close(); }

        var hot = indexingExecutable(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(2)),
                List.of(0), List.of(CpuIndexingLoweringTest.descriptor(
                        DataType.INT32, Shape.of(3))),
                CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(3, 2)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.BYTE_ARRAY));
        byte[] hotOutput = {7, 7, 7, 7, 7, 7};
        var hotRun = state(hot, List.of(borrow(new int[]{1, 2, -1}), borrow(hotOutput)));
        try {
            var failure = assertThrows(IndexOutOfBoundsException.class,
                    () -> hot.bind(hotRun).execute());
            assertAll(() -> assertEquals("ONE_HOT index at logical position 1 is out of bounds: "
                            + "value=2, depth=2", failure.getMessage()),
                    () -> assertArrayEquals(new byte[]{7, 7, 7, 7, 7, 7}, hotOutput));
        } finally { hotRun.close(); }
    }

    @Test void emptyIndexAndZeroOutputValidationDomainsRemainIndependent() {
        var emptyHot = indexingExecutable(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)),
                List.of(0), List.of(CpuIndexingLoweringTest.descriptor(
                        DataType.INT32, Shape.of(0))),
                CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(0, 3)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.BYTE_ARRAY));
        var emptyRun = state(emptyHot, List.of(borrow(new int[0]), borrow(new byte[0])));
        try { assertDoesNotThrow(() -> emptyHot.bind(emptyRun).execute()); }
        finally { emptyRun.close(); }

        var zeroSuffix = indexingExecutable(new Operation(GatherNdKind.GATHER_ND,
                        new GatherNdAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(1, 1, 0)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(1, 2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(1, 0)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.INT_ARRAY));
        var invalidRun = state(zeroSuffix, List.of(borrow(new int[0]),
                borrow(new int[]{0, 1}), borrow(new int[0])));
        try {
            var failure = assertThrows(IndexOutOfBoundsException.class,
                    () -> zeroSuffix.bind(invalidRun).execute());
            assertEquals("GATHER_ND index at logical position 1 for data axis 1 is out of bounds: "
                    + "value=1, extent=1", failure.getMessage());
        } finally { invalidRun.close(); }
        var validRun = state(zeroSuffix, List.of(borrow(new int[0]),
                borrow(new int[]{0, 0}), borrow(new int[0])));
        try { assertDoesNotThrow(() -> zeroSuffix.bind(validRun).execute()); }
        finally { validRun.close(); }

        var zeroAxis = indexingExecutable(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(0)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(1))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(1)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.INT_ARRAY));
        int[] sentinel = {7};
        var zeroAxisRun = state(zeroAxis, List.of(borrow(new int[0]), borrow(new int[]{0}),
                borrow(sentinel)));
        try {
            assertThrows(IndexOutOfBoundsException.class, () -> zeroAxis.bind(zeroAxisRun).execute());
            assertArrayEquals(new int[]{7}, sentinel);
        } finally { zeroAxisRun.close(); }
    }

    @Test void indexingSupportsDeduplicationMixedSegmentsOffsetsAndRejectsOverlap() {
        var deduplicated = indexingExecutable(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(0)), List.of(0, 0),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY));
        int[] deduplicatedOutput = new int[2];
        var deduplicatedRun = state(deduplicated,
                List.of(borrow(new int[]{1, 0}), borrow(deduplicatedOutput)));
        try {
            deduplicated.bind(deduplicatedRun).execute();
            assertArrayEquals(new int[]{0, 1}, deduplicatedOutput);
        } finally { deduplicatedRun.close(); }

        var mixed = indexingExecutable(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(3)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.INT_ARRAY));
        try (var arena = Arena.ofConfined()) {
            var indexSegment = arena.allocate(2 * Long.BYTES, Long.BYTES);
            indexSegment.set(java.lang.foreign.ValueLayout.JAVA_LONG, 0, 2);
            indexSegment.set(java.lang.foreign.ValueLayout.JAVA_LONG, Long.BYTES, 0);
            int[] dataCarrier = {99, 10, 20, 30, 99};
            int[] outputCarrier = {99, 99, 99, 99};
            var mixedRun = state(mixed, List.of(borrow(dataCarrier, 1, 3),
                    CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.INT64, 2,
                            indexSegment)), borrow(outputCarrier, 1, 2)));
            try {
                mixed.bind(mixedRun).execute();
                assertArrayEquals(new int[]{99, 30, 10, 99}, outputCarrier);
            } finally { mixedRun.close(); }
        }

        var overlap = indexingExecutable(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                List.of(CarrierAccess.INT_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.INT_ARRAY));
        int[] shared = {0, 1};
        var overlapRun = state(overlap,
                List.of(borrow(shared), borrow(new long[]{1, 0}), borrow(shared)));
        try { assertThrows(IllegalArgumentException.class, () -> overlap.bind(overlapRun)); }
        finally { overlapRun.close(); }

        var boolGather = indexingExecutable(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(1))),
                CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(1)),
                List.of(CarrierAccess.BYTE_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.BYTE_ARRAY));
        byte[] boolOutput = {7};
        var boolRun = state(boolGather,
                List.of(borrow(new byte[]{1, 2}), borrow(new int[]{0}), borrow(boolOutput)));
        try {
            assertThrows(IllegalArgumentException.class, () -> boolGather.bind(boolRun));
            assertArrayEquals(new byte[]{7}, boolOutput);
        } finally { boolRun.close(); }
    }

    @Test void parallelIndexingStillValidatesBeforeWorkerWrites() {
        var base = CpuIndexingLoweringTest.context(
                new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(2)), List.of(0),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(16))),
                CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(16, 2)));
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.LONG_ARRAY, CarrierAccess.BYTE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            long[] indices = new long[16]; indices[1] = 2;
            byte[] output = new byte[32]; java.util.Arrays.fill(output, (byte) 7);
            var run = state(executable, List.of(borrow(indices), borrow(output)));
            try {
                assertThrows(IndexOutOfBoundsException.class,
                        () -> executable.bind(run).execute());
                byte[] expected = new byte[32]; java.util.Arrays.fill(expected, (byte) 7);
                assertArrayEquals(expected, output);
            } finally { run.close(); }
        } finally { workers.close(); }
    }

    @Test void scatterBoundsPrecedeDuplicatesAndEveryFailureLeavesOutputUntouched() {
        var executable=scatterExecutable(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0,ScatterReduction.NONE)),
                List.of(CpuScatterLoweringTest.desc(DataType.INT32,Shape.of(3)),
                        CpuScatterLoweringTest.desc(DataType.INT32,Shape.of(3)),
                        CpuScatterLoweringTest.desc(DataType.INT32,Shape.of(3))),
                CpuScatterLoweringTest.desc(DataType.INT32,Shape.of(3)));
        int[] firstOutput={7,7,7};
        var bounds=state(executable,List.of(borrow(new int[]{1,2,3}),borrow(new int[]{0,0,3}),
                borrow(new int[]{9,8,7}),borrow(firstOutput)));
        try{
            var failure=assertThrows(IndexOutOfBoundsException.class,()->executable.bind(bounds).execute());
            assertAll(()->assertEquals("SCATTER_ELEMENTS index at logical position 2 for data axis 0 is out of bounds: value=3, extent=3",failure.getMessage()),
                    ()->assertArrayEquals(new int[]{7,7,7},firstOutput));
        }finally{bounds.close();}
        int[] duplicateOutput={6,6,6};
        var duplicate=state(executable,List.of(borrow(new int[]{1,2,3}),borrow(new int[]{0,0,1}),
                borrow(new int[]{9,8,7}),borrow(duplicateOutput)));
        try{
            var failure=assertThrows(IllegalArgumentException.class,()->executable.bind(duplicate).execute());
            assertAll(()->assertEquals("SCATTER_ELEMENTS duplicate target at logical update position 1; first addressed at logical update position 0",failure.getMessage()),
                    ()->assertArrayEquals(new int[]{6,6,6},duplicateOutput));
        }finally{duplicate.close();}
    }

    @Test void foldRejectsOverlapBeforeWritesAndParallelRangesRepeatDeterministically() {
        var base = CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 16, 1)), DataType.INT32, Shape.of(15, 2), Shape.of(16));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            int[] input = new int[30]; java.util.Arrays.fill(input, 1);
            int[] output = new int[16]; java.util.Arrays.fill(output, -7);
            var run = state(executable, List.of(borrow(input), borrow(output)));
            try {
                var bound = executable.bind(run);
                bound.execute();
                int[] expected = new int[16]; java.util.Arrays.fill(expected, 2);
                expected[0] = 1; expected[15] = 1;
                assertArrayEquals(expected, output);
                bound.execute();
                assertArrayEquals(expected, output);
                int[] original = new int[30]; java.util.Arrays.fill(original, 1);
                assertArrayEquals(original, input);
            } finally { run.close(); }

            int[] shared = new int[30]; java.util.Arrays.fill(shared, 9);
            var overlap = state(executable, List.of(borrow(shared, 0, 30),
                    borrow(shared, 0, 16)));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                int[] untouched = new int[30]; java.util.Arrays.fill(untouched, 9);
                assertArrayEquals(untouched, shared);
            } finally { overlap.close(); }
        } finally { workers.close(); }
    }

    @Test void scanRejectsCompleteOverlapBeforeWritesAndParallelizesOnlyWholeSlices() {
        var base = CpuScanLoweringTest.context(CumulativeScanKind.CUM_SUM, DataType.INT32,
                Shape.of(8, 3), 1, false, false);
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            int[] input = new int[24]; java.util.Arrays.fill(input, 1);
            int[] output = new int[24]; java.util.Arrays.fill(output, -7);
            var run = state(executable, List.of(borrow(input), borrow(output)));
            try {
                executable.bind(run).execute();
                for (int i = 0; i < output.length; i++) assertEquals(i % 3 + 1, output[i]);
            } finally { run.close(); }
            int[] shared = new int[30]; java.util.Arrays.fill(shared, 9);
            var overlap = state(executable, List.of(borrow(shared, 0, 24), borrow(shared, 3, 24)));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                int[] untouched = new int[30]; java.util.Arrays.fill(untouched, 9);
                assertArrayEquals(untouched, shared);
            } finally { overlap.close(); }
        } finally { workers.close(); }
    }

    @Test void sumToShapeRejectsCompleteOverlapBeforeWritesAndParallelizesWholeOutputCells() {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.INT32,
                Shape.of(2,8,3), new SumToShapeAttrs(Shape.of(8,1)), Shape.of(8,1));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 2, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        assertAll(() -> assertEquals(2, analysis.plan().bufferDeclarations().size()),
                () -> assertTrue(analysis.plan().workspaceDeclaration().isEmpty()),
                () -> assertEquals(2, analysis.plan().selectedRangeCount()));
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            int[] input = new int[48];
            for (int cell = 0; cell < 8; cell++) {
                input[cell * 3] = cell; input[cell * 3 + 1] = -cell; input[cell * 3 + 2] = cell + 10;
                int second = (8 + cell) * 3;
                input[second] = 1; input[second + 1] = 2; input[second + 2] = 3;
            }
            int[] output = new int[8]; java.util.Arrays.fill(output, -7);
            var run = state(executable, List.of(borrow(input), borrow(output)));
            try {
                executable.bind(run).execute();
                assertArrayEquals(new int[]{16,17,18,19,20,21,22,23}, output);
            } finally { run.close(); }
            int[] shared = new int[56]; java.util.Arrays.fill(shared, 9);
            var overlap = state(executable, List.of(borrow(shared,0,48), borrow(shared,3,8)));
            try {
                assertThrows(IllegalArgumentException.class, () -> executable.bind(overlap));
                int[] untouched = new int[56]; java.util.Arrays.fill(untouched, 9);
                assertArrayEquals(untouched, shared);
            } finally { overlap.close(); }
        } finally { workers.close(); }
    }

    @Test void exactFloatingSumToShapeUsesDisjointRunOwnedSlicesAndResetsOnConcurrentReuse()
            throws InterruptedException {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2,8,3), new SumToShapeAttrs(Shape.of(8,1)), Shape.of(8,1));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var declaration = analysis.plan().workspaceDeclaration().orElseThrow();
        assertEquals(analysis.plan().aggregateGeometry().orElseThrow().workspaceBytes(4),
                declaration.byteSize());
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            double[] input = new double[48];
            for (int cell = 0; cell < 8; cell++) {
                input[cell * 3] = 0x1p53; input[cell * 3 + 1] = cell + 1;
                input[cell * 3 + 2] = -0x1p53;
            }
            double[] output = new double[8]; java.util.Arrays.fill(output, -7);
            var workspace = CpuContiguousWorkspace.allocate(declaration.byteSize(),
                    declaration.byteAlignment());
            var run = state(executable, List.of(borrow(input, 0, input.length),
                    borrow(output, 0, output.length)), List.of(workspace));
            try {
                var bound = executable.bind(run); bound.execute();
                assertArrayEquals(new double[]{1,2,3,4,5,6,7,8}, output);
                java.util.Arrays.fill(output, -9); bound.execute();
                assertArrayEquals(new double[]{1,2,3,4,5,6,7,8}, output);
            } finally { run.close(); }

            double[] firstOutput = new double[8], secondOutput = new double[8];
            var firstRun = state(executable, List.of(borrow(input, 0, input.length),
                    borrow(firstOutput, 0, firstOutput.length)), List.of(
                        CpuContiguousWorkspace.allocate(declaration.byteSize(),
                                declaration.byteAlignment())));
            var secondRun = state(executable, List.of(borrow(input, 0, input.length),
                    borrow(secondOutput, 0, secondOutput.length)), List.of(
                        CpuContiguousWorkspace.allocate(declaration.byteSize(),
                                declaration.byteAlignment())));
            try {
                var firstBound = executable.bind(firstRun);
                var secondBound = executable.bind(secondRun);
                Thread first = Thread.ofVirtual().start(firstBound::execute);
                Thread second = Thread.ofVirtual().start(secondBound::execute);
                first.join(); second.join();
                assertAll(
                        () -> assertArrayEquals(new double[]{1,2,3,4,5,6,7,8}, firstOutput),
                        () -> assertArrayEquals(new double[]{1,2,3,4,5,6,7,8}, secondOutput));
            } finally {
                firstRun.close(); secondRun.close();
            }
        } finally { workers.close(); }
    }

    @Test void parallelArgExtremaUsesMixedCarriersRejectsOverlapAndSupportsConcurrentReuse()
            throws InterruptedException {
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = argExecutable(AggregateReductionKind.ARG_MAX, DataType.FLOAT32,
                    Shape.of(8, 3), 1, true, ArgExtremaTiePolicy.LAST_INDEX,
                    List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT), workers);
            float[] input = new float[24];
            for (int cell = 0; cell < 8; cell++) {
                input[cell * 3] = -0.0f;
                input[cell * 3 + 1] = Float.intBitsToFloat(0x7f800001 + cell);
                input[cell * 3 + 2] = Float.intBitsToFloat(0x7fc00100 + cell);
            }
            try (var arena = Arena.ofShared()) {
                MemorySegment first = arena.allocate(64, 8), second = arena.allocate(64, 8);
                var firstRun = state(executable, List.of(borrow(input, 0, input.length),
                        borrowed(DataType.INT64, 8, first)));
                var secondRun = state(executable, List.of(borrow(input, 0, input.length),
                        borrowed(DataType.INT64, 8, second)));
                try {
                    Thread a = Thread.ofVirtual().start(executable.bind(firstRun)::execute);
                    Thread b = Thread.ofVirtual().start(executable.bind(secondRun)::execute);
                    a.join(); b.join();
                    for (int index = 0; index < 8; index++) {
                        assertEquals(2, first.getAtIndex(LONG, index));
                        assertEquals(2, second.getAtIndex(LONG, index));
                    }
                    assertTrue(workers.isOpen());
                } finally { firstRun.close(); secondRun.close(); }
            }

            var overlapExecutable = argExecutable(AggregateReductionKind.ARG_MIN, DataType.INT64,
                    Shape.of(2, 3), 1, false, ArgExtremaTiePolicy.FIRST_INDEX,
                    List.of(CarrierAccess.LONG_ARRAY, CarrierAccess.LONG_ARRAY), workers);
            long[] shared = {3, 2, 1, 7, 8, 9, 55};
            long[] unchanged = shared.clone();
            var overlap = state(overlapExecutable,
                    List.of(borrow(shared, 0, 6), borrow(shared, 1, 2)));
            try {
                assertThrows(IllegalArgumentException.class,
                        () -> overlapExecutable.bind(overlap));
                assertArrayEquals(unchanged, shared);
            } finally { overlap.close(); }

            var zero = argExecutable(AggregateReductionKind.ARG_MIN, DataType.INT32,
                    Shape.of(0, 3), 1, false, ArgExtremaTiePolicy.FIRST_INDEX,
                    List.of(CarrierAccess.INT_ARRAY, CarrierAccess.LONG_ARRAY), workers);
            var zeroRun = state(zero, List.of(borrow(new int[0]), borrow(new long[0])));
            try { assertDoesNotThrow(() -> zero.bind(zeroRun).execute()); }
            finally { zeroRun.close(); }
            assertTrue(workers.isOpen());
        } finally { workers.close(); }
    }

    @Test void scatterNdRejectsFirstLaterDuplicateTupleEvenWithEmptySuffix() {
        var executable=scatterExecutable(new Operation(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(0,ScatterReduction.NONE)),
                List.of(CpuScatterLoweringTest.desc(DataType.INT64,Shape.of(2,0)),
                        CpuScatterLoweringTest.desc(DataType.INT32,Shape.of(3,1)),
                        CpuScatterLoweringTest.desc(DataType.INT64,Shape.of(3,0))),
                CpuScatterLoweringTest.desc(DataType.INT64,Shape.of(2,0)));
        var run=state(executable,List.of(borrow(new long[0]),borrow(new int[]{1,0,1}),
                borrow(new long[0]),borrow(new long[0])));
        try{
            var failure=assertThrows(IllegalArgumentException.class,()->executable.bind(run).execute());
            assertEquals("SCATTER_ND duplicate target tuple at logical tuple position 2; first addressed at logical tuple position 0",failure.getMessage());
        }finally{run.close();}
    }

    @Test void zeroOutputScatterStillCompletesBoundsValidationBeforeExecution() {
        var executable = scatterExecutable(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.ADD)),
                List.of(CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(0)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(1)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(1))),
                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(0)));
        var run = state(executable, List.of(borrow(new int[0]), borrow(new int[]{0}),
                borrow(new int[]{9}), borrow(new int[0])));
        try {
            var failure = assertThrows(IndexOutOfBoundsException.class,
                    () -> executable.bind(run).execute());
            assertEquals("SCATTER_ELEMENTS index at logical position 0 for data axis 0 is out "
                    + "of bounds: value=0, extent=0", failure.getMessage());
        } finally { run.close(); }
    }

    @Test void replacementUniquenessDistinguishesNonAxisCoordinatesAndNdBatches() {
        var elements = scatterExecutable(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                List.of(CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 2)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 2)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 2))),
                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 2)));
        int[] elementsOutput = new int[4];
        var elementsRun = state(elements, List.of(borrow(new int[]{1, 2, 3, 4}),
                borrow(new int[]{0, 1, 0, 1}), borrow(new int[]{9, 8, 7, 6}),
                borrow(elementsOutput)));
        try {
            elements.bind(elementsRun).execute();
            assertArrayEquals(new int[]{9, 8, 7, 6}, elementsOutput);
        } finally { elementsRun.close(); }

        var nd = scatterExecutable(new Operation(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(1, ScatterReduction.NONE)),
                List.of(CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(2, 2)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 1, 1)),
                        CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(2, 1))),
                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(2, 2)));
        long[] ndOutput = new long[4];
        var ndRun = state(nd, List.of(borrow(new long[]{1, 2, 3, 4}),
                borrow(new int[]{0, 0}), borrow(new long[]{9, 8}), borrow(ndOutput)));
        try {
            nd.bind(ndRun).execute();
            assertArrayEquals(new long[]{9, 2, 8, 4}, ndOutput);
        } finally { ndRun.close(); }
    }

    @Test void scatterRejectsOutputOverlapAndSupportsExactInputOccurrenceDeduplication() {
        var overlap = scatterExecutable(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.ADD)),
                List.of(CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2))),
                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2)));
        int[] shared = {2, 3};
        var overlapRun = state(overlap, List.of(borrow(shared), borrow(new int[]{0, 1}),
                borrow(new int[]{4, 5}), borrow(shared)));
        try {
            assertThrows(IllegalArgumentException.class, () -> overlap.bind(overlapRun));
            assertArrayEquals(new int[]{2, 3}, shared);
        } finally { overlapRun.close(); }

        var descriptors = List.of(CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2)),
                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2)));
        var base = CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.ADD)), List.of(0, 1, 0),
                descriptors, CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2)));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false, List.of(
                        CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.INT_ARRAY)));
        var deduplicated = CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());
        int[] output = new int[2];
        var dedupRun = state(deduplicated, List.of(borrow(new int[]{2, 3}),
                borrow(new int[]{1, 0}), borrow(output)));
        try {
            deduplicated.bind(dedupRun).execute();
            assertArrayEquals(new int[]{5, 5}, output);
        } finally { dedupRun.close(); }
    }

    @Test void parallelScatterProductUsesMixedCarriersDisjointScratchAndRepeatsDeterministically() {
        int outputCount = 8, updateCount = 16;
        var inputs = List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64,
                        Shape.of(outputCount)),
                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(updateCount)),
                CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(updateCount)));
        var base = CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MUL)), List.of(0, 1, 2),
                inputs, CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(outputCount)));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.INT_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        assertEquals(4, analysis.plan().selectedRangeCount());
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            var data = CpuNativeBuffer.allocate(DataType.FLOAT64,
                    outputCount * (long) Double.BYTES, Double.BYTES);
            for (int i = 0; i < outputCount; i++) data.segment().set(ValueLayout.JAVA_DOUBLE,
                    i * (long) Double.BYTES, 2.0);
            int[] indices = new int[updateCount];
            double[] updates = new double[updateCount];
            for (int i = 0; i < outputCount; i++) {
                indices[2 * i] = i; indices[2 * i + 1] = i;
                updates[2 * i] = 3.0; updates[2 * i + 1] = 4.0;
            }
            double[] output = new double[outputCount]; java.util.Arrays.fill(output, -7.0);
            var declaration = analysis.plan().workspaceDeclaration().orElseThrow();
            var workspace = CpuContiguousWorkspace.allocate(declaration.byteSize(),
                    declaration.byteAlignment());
            var run = state(executable, List.of(data, borrow(indices), borrow(updates, 0,
                    updateCount), borrow(output, 0, outputCount)), List.of(workspace));
            try {
                var bound = executable.bind(run);
                bound.execute();
                assertArrayEquals(new double[]{24, 24, 24, 24, 24, 24, 24, 24}, output);
                bound.execute();
                assertArrayEquals(new double[]{24, 24, 24, 24, 24, 24, 24, 24}, output);
                for (int i = 0; i < outputCount; i++) assertEquals(2.0,
                        data.segment().get(ValueLayout.JAVA_DOUBLE, i * (long) Double.BYTES));
            } finally { run.close(); }
        } finally { workers.close(); }
    }

    @Test void parallelScatterValidationCompletesBeforeAnyWorkerCanWrite() {
        int count = 16;
        var inputs = List.of(CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(count)),
                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(count)),
                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(count)));
        var base = CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.NONE)), List.of(0, 1, 2),
                inputs, CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(count)));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 4, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false, List.of(
                        CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY, CarrierAccess.INT_ARRAY,
                        CarrierAccess.INT_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var workers = new CpuWorkerGroup(4);
        try {
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis,
                    Optional.empty(), Optional.of(workers));
            int[] indices = new int[count];
            for (int i = 0; i < count; i++) indices[i] = i;
            indices[3] = count;
            int[] output = new int[count]; java.util.Arrays.fill(output, 73);
            var run = state(executable, List.of(borrow(new int[count]), borrow(indices),
                    borrow(new int[count]), borrow(output)));
            try {
                var failure = assertThrows(IndexOutOfBoundsException.class,
                        () -> executable.bind(run).execute());
                assertEquals("SCATTER_ELEMENTS index at logical position 3 for data axis 0 is "
                        + "out of bounds: value=16, extent=16", failure.getMessage());
                int[] expected = new int[count]; java.util.Arrays.fill(expected, 73);
                assertArrayEquals(expected, output);
            } finally { run.close(); }
        } finally { workers.close(); }
    }

    private static CpuPreparedExecutable scatterExecutable(Operation operation,
            List<TensorDescriptor> inputs,TensorDescriptor output){
        var base=CpuScatterLoweringTest.context(operation,List.of(0,1,2),inputs,output);
        var carriers=new ArrayList<CarrierAccess>();for(var input:inputs)carriers.add(switch(input.dataType()){case FLOAT64->CarrierAccess.DOUBLE_ARRAY;case FLOAT32->CarrierAccess.FLOAT_ARRAY;case BFLOAT16->CarrierAccess.SHORT_ARRAY;case INT32->CarrierAccess.INT_ARRAY;case INT64->CarrierAccess.LONG_ARRAY;case BOOL->CarrierAccess.BYTE_ARRAY;});carriers.add(switch(output.dataType()){case FLOAT64->CarrierAccess.DOUBLE_ARRAY;case FLOAT32->CarrierAccess.FLOAT_ARRAY;case BFLOAT16->CarrierAccess.SHORT_ARRAY;case INT32->CarrierAccess.INT_ARRAY;case INT64->CarrierAccess.LONG_ARRAY;case BOOL->CarrierAccess.BYTE_ARRAY;});
        var context=new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(),base.nodes(),base.values(),base.memoryRequirements(),base.constants(),new CpuPartitionAnalysisInputs(false,carriers));
        return CpuPartitionFinalizerTest.finalizeExecutable(new CpuPartitionPreparer().analyze(context),Optional.empty());
    }

    private static CpuPreparedExecutable indexingExecutable(Operation operation,
            List<Integer> occurrences,
            List<io.github.pho001.synaptik.model.tensor.TensorDescriptor> inputs,
            io.github.pho001.synaptik.model.tensor.TensorDescriptor output,
            List<CarrierAccess> carriers) {
        var base = CpuIndexingLoweringTest.context(operation, occurrences, inputs, output);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        return CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty());
    }

    private static CpuPreparedExecutable argExecutable(AggregateReductionKind kind, DataType type,
            Shape shape, int axis, boolean keep, ArgExtremaTiePolicy tie,
            List<CarrierAccess> carriers, CpuWorkerGroup workers) {
        var base = CpuArgExtremaLoweringTest.context(kind, type, shape, axis, keep, tie);
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 2, 1);
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false, carriers, config));
        return CpuPartitionFinalizerTest.finalizeExecutable(
                new CpuPartitionPreparer().analyze(context), Optional.empty(),
                Optional.of(workers));
    }

    private static CpuBorrowedBuffer borrow(double[] carrier, int elementOffset) {
        return borrow(carrier, elementOffset, 4);
    }

    private static CpuBorrowedBuffer borrow(double[] carrier, int elementOffset, int count) {
        var segment = MemorySegment.ofArray(carrier).asSlice(elementOffset * 8L, count * 8L);
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64, count, segment));
    }

    private static CpuBorrowedBuffer borrow(int[] carrier) {
        return borrow(carrier, 0, carrier.length);
    }

    private static CpuBorrowedBuffer borrow(float[] carrier, int elementOffset, int count) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32, count,
                MemorySegment.ofArray(carrier).asSlice(elementOffset * 4L, count * 4L)));
    }

    private static CpuBorrowedBuffer borrow(int[] carrier, int elementOffset, int count) {
        var segment = MemorySegment.ofArray(carrier).asSlice(elementOffset * 4L, count * 4L);
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.INT32, count, segment));
    }

    private static CpuBorrowedBuffer borrow(byte[] carrier) {
        return borrow(carrier, 0, carrier.length);
    }

    private static CpuBorrowedBuffer borrow(byte[] carrier, int elementOffset, int count) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.BOOL, count,
                MemorySegment.ofArray(carrier).asSlice(elementOffset, count)));
    }

    private static CpuBorrowedBuffer borrow(long[] carrier) {
        return borrow(carrier, 0, carrier.length);
    }

    private static CpuBorrowedBuffer borrow(long[] carrier, int elementOffset, int count) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.INT64, count,
                MemorySegment.ofArray(carrier).asSlice(elementOffset * 8L, count * 8L)));
    }

    private static CpuBorrowedBuffer borrowed(DataType type, long elements, MemorySegment segment) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, elements, segment));
    }

    private static RunState state(CpuPreparedExecutable executable,
            List<? extends io.github.pho001.synaptik.runtime.resource.BufferRepresentation> resources) {
        return state(executable, resources, List.of());
    }

    private static RunState state(CpuPreparedExecutable executable,
            List<? extends io.github.pho001.synaptik.runtime.resource.BufferRepresentation> resources,
            List<? extends io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation>
                    workspaces) {
        var bindings = resources.stream().map(resource -> List.of(
                new BufferRepresentationBinding(resource, resource instanceof CpuNativeBuffer
                        ? RunResourceOwnership.RUN_OWNED
                        : RunResourceOwnership.BORROWED))).toList();
        var workspaceSnapshot = new ArrayList<
                io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation>();
        workspaceSnapshot.addAll(workspaces);
        return new RunState(executable.memoryPlan(), bindings, workspaceSnapshot);
    }

    private static List<CpuNativeBuffer> nativeResources(CpuPreparedExecutable executable) {
        return executable.memoryPlan().buffers().stream().map(entry -> CpuNativeBuffer.allocate(
                DataType.FLOAT64, entry.byteSize(), entry.byteAlignment())).toList();
    }

    private static MemorySegment segment(
            io.github.pho001.synaptik.runtime.resource.BufferRepresentation resource) {
        return ((io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation)
                resource).segment();
    }
}
