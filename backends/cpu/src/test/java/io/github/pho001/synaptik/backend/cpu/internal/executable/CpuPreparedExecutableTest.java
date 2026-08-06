package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBorrowedBuffer;
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
import java.util.Optional;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import jdk.incubator.vector.DoubleVector;

class CpuPreparedExecutableTest {
    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());

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

    @Test void handlesScalarAndZeroElementBindings() {
        assertEquals(1, CpuPartitionFinalizerTest.finalizeExecutable(
                Shape.scalar(), Optional.empty()).binding().elementCount());
        assertEquals(0, CpuPartitionFinalizerTest.finalizeExecutable(
                Shape.of(2, 0, 3), Optional.empty()).binding().elementCount());
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
                    configurations.get(strategy), strategies.get(strategy));
        }
    }

    private static void executeCarrierPattern(TensorDescriptor descriptor, int count, int mask,
            PortableExecutionConfig config, String expectedStrategy) {
            var pattern = new ArrayList<CarrierAccess>();
            for (int i = 0; i < 4; i++) pattern.add((mask & (1 << i)) != 0
                    ? CarrierAccess.DOUBLE_ARRAY : CarrierAccess.MEMORY_SEGMENT);
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
                if (pattern.get(i) == CarrierAccess.DOUBLE_ARRAY) {
                    var storage = new MemorySegmentStorage(DataType.FLOAT64, count,
                            MemorySegment.ofArray(new double[count]));
                    resource = CpuBorrowedBuffer.borrow(storage);
                    ownership = RunResourceOwnership.BORROWED;
                } else {
                    resource = CpuNativeBuffer.allocate(DataType.FLOAT64, count * 8L, 8);
                    ownership = RunResourceOwnership.RUN_OWNED;
                }
                resources.add(resource);
                bindings.add(List.of(new BufferRepresentationBinding(resource, ownership)));
            }
            var state = new RunState(executable.memoryPlan(), bindings, List.of());
            try {
                for (int i = 0; i < count; i++) {
                    segment(resources.get(0)).set(DOUBLE, i * 8L, i - 1.0);
                    segment(resources.get(1)).set(DOUBLE, i * 8L, 0.5);
                    segment(resources.get(2)).set(DOUBLE, i * 8L, 2.0);
                }
                executable.bind(state).execute();
                for (int i = 0; i < count; i++) assertEquals(
                        CpuScalarReferenceKernel.gelu(i - 0.5) * 2.0,
                        segment(resources.get(3)).get(DOUBLE, i * 8L),
                        2e-7 * Math.max(1.0, Math.abs(i - 0.5)),
                        expectedStrategy + " carrier mask " + mask + " index " + i);
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

    private static CpuBorrowedBuffer borrow(double[] carrier, int elementOffset) {
        return borrow(carrier, elementOffset, 4);
    }

    private static CpuBorrowedBuffer borrow(double[] carrier, int elementOffset, int count) {
        var segment = MemorySegment.ofArray(carrier).asSlice(elementOffset * 8L, count * 8L);
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64, count, segment));
    }

    private static RunState state(CpuPreparedExecutable executable,
            List<? extends io.github.pho001.synaptik.runtime.resource.BufferRepresentation> resources) {
        var bindings = resources.stream().map(resource -> List.of(
                new BufferRepresentationBinding(resource, RunResourceOwnership.BORROWED))).toList();
        return new RunState(executable.memoryPlan(), bindings, List.of());
    }

    private static MemorySegment segment(
            io.github.pho001.synaptik.runtime.resource.BufferRepresentation resource) {
        return ((io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation)
                resource).segment();
    }
}
