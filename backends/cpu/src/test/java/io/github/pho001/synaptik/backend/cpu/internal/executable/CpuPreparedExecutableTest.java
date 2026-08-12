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
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.INT64, carrier.length,
                MemorySegment.ofArray(carrier)));
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
