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
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuPreparedExecutableTest {
    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());

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

    @Test void executesAllSixteenDirectCarrierPatternsOnDemand() {
        Shape shape = Shape.of(4);
        var descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        for (int mask = 0; mask < 16; mask++) {
            var pattern = new ArrayList<CarrierAccess>();
            for (int i = 0; i < 4; i++) pattern.add((mask & (1 << i)) != 0
                    ? CarrierAccess.DOUBLE_ARRAY : CarrierAccess.MEMORY_SEGMENT);
            var analysis = CpuPartitionPreparerTest.analyze(descriptor, descriptor, descriptor,
                    descriptor, new CpuPartitionAnalysisInputs(false, pattern));
            var executable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
            var resources = new ArrayList<io.github.pho001.synaptik.runtime.resource.BufferRepresentation>();
            var bindings = new ArrayList<List<BufferRepresentationBinding>>();
            for (int i = 0; i < 4; i++) {
                io.github.pho001.synaptik.runtime.resource.BufferRepresentation resource;
                RunResourceOwnership ownership;
                if (pattern.get(i) == CarrierAccess.DOUBLE_ARRAY) {
                    var storage = new MemorySegmentStorage(DataType.FLOAT64, 4,
                            MemorySegment.ofArray(new double[4]));
                    resource = CpuBorrowedBuffer.borrow(storage);
                    ownership = RunResourceOwnership.BORROWED;
                } else {
                    resource = CpuNativeBuffer.allocate(DataType.FLOAT64, 32, 8);
                    ownership = RunResourceOwnership.RUN_OWNED;
                }
                resources.add(resource);
                bindings.add(List.of(new BufferRepresentationBinding(resource, ownership)));
            }
            var state = new RunState(executable.memoryPlan(), bindings, List.of());
            try {
                for (int i = 0; i < 4; i++) {
                    segment(resources.get(0)).set(DOUBLE, i * 8L, i - 1.0);
                    segment(resources.get(1)).set(DOUBLE, i * 8L, 0.5);
                    segment(resources.get(2)).set(DOUBLE, i * 8L, 2.0);
                }
                executable.bind(state).execute();
                for (int i = 0; i < 4; i++) assertEquals(
                        CpuScalarReferenceKernel.gelu(i - 0.5) * 2.0,
                        segment(resources.get(3)).get(DOUBLE, i * 8L), 0.0);
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
        var segment = MemorySegment.ofArray(carrier).asSlice(elementOffset * 8L, 32);
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64, 4, segment));
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
