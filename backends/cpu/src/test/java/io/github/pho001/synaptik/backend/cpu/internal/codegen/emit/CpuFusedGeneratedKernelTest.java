package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import java.util.List;
import java.util.ArrayList;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuNativeBuffer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;

class CpuFusedGeneratedKernelTest {
    private static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());

    @Test void executesArbitraryHalfOpenBoundsWithoutMaterializedIntermediates() throws Throwable {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(5)).plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (Arena arena = Arena.ofConfined()) {
            var a = arena.allocate(40, 8); var b = arena.allocate(40, 8);
            var c = arena.allocate(40, 8); var output = arena.allocate(40, 8);
            for (int i = 0; i < 5; i++) {
                a.set(DOUBLE, i * 8L, i - 2.0); b.set(DOUBLE, i * 8L, 0.25);
                c.set(DOUBLE, i * 8L, 2.0); output.set(DOUBLE, i * 8L, -99.0);
            }
            long[] geometry = {5, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            artifact.entryPoint().invokeExact(a, b, c, output, geometry, 1L, 4L);
            assertEquals(-99.0, output.get(DOUBLE, 0));
            for (int i = 1; i < 4; i++) assertEquals(
                    CpuScalarReferenceKernel.gelu((i - 2.0) + 0.25) * 2.0,
                    output.get(DOUBLE, i * 8L), 0.0);
            assertEquals(-99.0, output.get(DOUBLE, 32));
        }
    }

    @Test void executesGeneralOdometerFromColdStartingCoordinates() throws Throwable {
        Shape shape = Shape.of(2, 3);
        var general = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[]{1, 2}, 0, true)), false);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var scalar = new TensorDescriptor(DataType.FLOAT64, Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), false);
        var route = CpuPartitionPreparerTest.analyze(general, dense, scalar, dense,
                CpuPartitionAnalysisInputs.DEFAULT).plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (Arena arena = Arena.ofConfined()) {
            var a = arena.allocate(48, 8); var b = arena.allocate(48, 8);
            var c = arena.allocate(8, 8); var output = arena.allocate(48, 8);
            for (int i = 0; i < 6; i++) { a.set(DOUBLE, i * 8L, i); b.set(DOUBLE, i * 8L, 0); output.set(DOUBLE, i * 8L, -9); }
            c.set(DOUBLE, 0, 1);
            long[] geometry = {2,3, 0,1, 2,1,0,1, 1,2, 3,1, 0,0, 3,1};
            artifact.entryPoint().invokeExact(a,b,c,output,geometry,1L,5L);
            double[] logical = {0,2,4,1,3,5};
            assertEquals(-9, output.get(DOUBLE, 0));
            for (int i = 1; i < 5; i++) assertEquals(CpuScalarReferenceKernel.gelu(logical[i]),
                    output.get(DOUBLE, i * 8L), 0);
            assertEquals(-9, output.get(DOUBLE, 40));
        }
    }

    @Test void zeroRangeReturnsBeforeAnyCarrierAccessOrAddressFormation() throws Throwable {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(0)).plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        long[] geometry = {0,0,0,0,0,0,1,1,1,1};
        artifact.entryPoint().invokeExact(java.lang.foreign.MemorySegment.NULL,
                java.lang.foreign.MemorySegment.NULL, java.lang.foreign.MemorySegment.NULL,
                java.lang.foreign.MemorySegment.NULL, geometry, 0L, 0L);
    }

    @Test void generatedAndReferenceExecutionAgreeAcrossAllFiveRegimes() {
        Shape denseShape = Shape.of(2, 3);
        executeCase(descriptor(denseShape, LayoutDescriptor.contiguous(denseShape)), denseShape);
        executeCase(descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())), denseShape);
        executeCase(descriptor(Shape.of(3), LayoutDescriptor.contiguous(Shape.of(3))), denseShape);
        Shape blockSource = Shape.of(2, 1, 3), blockTarget = Shape.of(2, 4, 3);
        executeCase(descriptor(blockSource, LayoutDescriptor.contiguous(blockSource)), blockTarget);
        executeCase(descriptor(denseShape,
                LayoutDescriptor.of(denseShape, new long[]{1, 2}, 0, true)), denseShape);
    }

    private static void executeCase(TensorDescriptor first, Shape target) {
        var dense = descriptor(target, LayoutDescriptor.contiguous(target));
        var scalar = descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()));
        var analysis = CpuPartitionPreparerTest.analyze(first, dense, scalar, dense,
                CpuPartitionAnalysisInputs.DEFAULT);
        var fullExecutable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty());
        var executable = fullExecutable.forRange(1, fullExecutable.binding().elementCount() - 1);
        var buffers = new ArrayList<CpuNativeBuffer>();
        var runtimeBindings = new ArrayList<List<BufferRepresentationBinding>>();
        for (var entry : executable.memoryPlan().buffers()) {
            var buffer = CpuNativeBuffer.allocate(DataType.FLOAT64, entry.byteSize(), 8);
            buffers.add(buffer);
            runtimeBindings.add(List.of(new BufferRepresentationBinding(buffer,
                    RunResourceOwnership.RUN_OWNED)));
        }
        for (int value = 0; value < 3; value++) for (long offset = 0;
                offset < buffers.get(value).byteSize(); offset += 8) {
            buffers.get(value).segment().set(DOUBLE, offset, value + offset / 8.0 + 0.25);
        }
        var state = new RunState(executable.memoryPlan(), runtimeBindings, List.of());
        try {
            executable.bind(state).execute();
            int outputLength = Math.toIntExact(buffers.get(3).byteSize() / 8);
            double[] generated = new double[outputLength];
            for (int i = 0; i < outputLength; i++) {
                generated[i] = buffers.get(3).segment().get(DOUBLE, i * 8L);
                buffers.get(3).segment().set(DOUBLE, i * 8L, 0);
            }
            CpuScalarReferenceKernel.execute(buffers.stream().map(buffer -> buffer.argument()).toList(),
                    executable.accessBindings(), executable.binding().start(),
                    executable.binding().end());
            for (int i = 0; i < outputLength; i++) assertEquals(generated[i],
                    buffers.get(3).segment().get(DOUBLE, i * 8L), 0);
        } finally { state.close(); }
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }
}
