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
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;

class CpuPreparedExecutableTest {
    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder());

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
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.BOOL, carrier.length,
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
