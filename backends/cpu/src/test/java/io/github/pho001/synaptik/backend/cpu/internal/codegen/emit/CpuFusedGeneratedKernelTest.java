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
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import jdk.incubator.vector.DoubleVector;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;

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

    @Test void vectorExecutionAgreesAcrossEveryAdmittedAccessRegime() {
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        Shape denseShape = Shape.of(2, lanes + 1);
        executeVectorCase(descriptor(denseShape, LayoutDescriptor.contiguous(denseShape)), denseShape);
        executeVectorCase(descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                denseShape);
        executeVectorCase(descriptor(Shape.of(lanes + 1),
                LayoutDescriptor.contiguous(Shape.of(lanes + 1))), denseShape);
        Shape source = Shape.of(2, 1, lanes + 1), target = Shape.of(2, 3, lanes + 1);
        executeVectorCase(descriptor(source, LayoutDescriptor.contiguous(source)), target);
    }

    @Test void vectorGeluPreservesSpecialClassificationsAndThresholdNeighborhoods() {
        double[] values = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                -0.0, 0.0, Math.nextDown(Math.sqrt(2.0)), Math.sqrt(2.0),
                Math.nextUp(Math.sqrt(2.0)), Math.nextDown(8.0 * Math.sqrt(2.0)),
                8.0 * Math.sqrt(2.0), Math.nextUp(8.0 * Math.sqrt(2.0)),
                Double.MIN_VALUE, -Double.MIN_VALUE, -12.0, 12.0};
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        for (int offset = 0; offset < values.length; offset += lanes) {
            double[] input = new double[lanes];
            for (int lane = 0; lane < lanes; lane++) input[lane] = values[
                    Math.min(values.length - 1, offset + lane)];
            double[] output = new double[lanes];
            CpuVectorEmitter.gelu(DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, input, 0))
                    .intoArray(output, 0);
            for (int lane = 0; lane < lanes && offset + lane < values.length; lane++) {
                double expected = CpuScalarReferenceKernel.gelu(input[lane]);
                if (Double.isNaN(expected)) assertTrue(Double.isNaN(output[lane]));
                else if (expected == 0.0) assertEquals(Double.doubleToRawLongBits(expected),
                        Double.doubleToRawLongBits(output[lane]));
                else assertEquals(expected, output[lane],
                        2e-7 * Math.max(1.0, Math.abs(expected)));
            }
        }
    }

    @Test void selectedCopiesMatchReferenceForEverySourceRegimeAndStrategy() {
        var configs = List.of(
                new PortableExecutionConfig(ComputePreference.SCALAR, 1, 1, 1),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1),
                new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 2, 2, 1));
        Shape generalShape = Shape.of(2, 3);
        var general = descriptor(generalShape,
                LayoutDescriptor.of(generalShape, new long[] {1, 2}, 0, true));
        var denseGeneral = descriptor(generalShape, LayoutDescriptor.contiguous(generalShape));
        Shape blockTarget = Shape.of(2, 4, 3);
        var block = descriptor(Shape.of(2, 1, 3),
                LayoutDescriptor.contiguous(Shape.of(2, 1, 3)));
        var denseBlock = descriptor(blockTarget, LayoutDescriptor.contiguous(blockTarget));
        for (var config : configs) {
            executeMaterialized(general, denseGeneral, denseGeneral, denseGeneral, config, 0);
            executeMaterialized(denseGeneral,
                    descriptor(Shape.of(3), LayoutDescriptor.contiguous(Shape.of(3))),
                    denseGeneral, denseGeneral, config, 1);
            executeMaterialized(denseBlock, denseBlock, block, denseBlock, config, 2);
        }
    }

    private static void executeMaterialized(TensorDescriptor a, TensorDescriptor b,
            TensorDescriptor c, TensorDescriptor output, PortableExecutionConfig config,
            int expectedSource) {
        long bytes = Math.multiplyExact(output.shape().knownElementCount().orElseThrow(), 8);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 2, bytes, 1, 1);
        var analysis = CpuPartitionPreparerTest.analyze(a, b, c, output,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), config, policy));
        assertEquals(expectedSource,
                analysis.plan().materialization().orElseThrow().sourceBoundaryIndex());
        executeAnalysis(analysis);
    }

    private static void executeVectorCase(TensorDescriptor first, Shape target) {
        var dense = descriptor(target, LayoutDescriptor.contiguous(target));
        var scalar = descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()));
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1));
        var analysis = CpuPartitionPreparerTest.analyze(first, dense, scalar, dense, inputs);
        assertEquals("vector", analysis.plan().executionStrategy().toString());
        executeAnalysis(analysis);
    }

    private static void executeCase(TensorDescriptor first, Shape target) {
        var dense = descriptor(target, LayoutDescriptor.contiguous(target));
        var scalar = descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()));
        var analysis = CpuPartitionPreparerTest.analyze(first, dense, scalar, dense,
                CpuPartitionAnalysisInputs.DEFAULT);
        executeAnalysis(analysis);
    }

    private static void executeAnalysis(
            io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan>
                    analysis) {
        CpuWorkerGroup workers = analysis.plan().selectedRangeCount() >= 2
                ? new CpuWorkerGroup(analysis.plan().selectedRangeCount()) : null;
        var fullExecutable = CpuPartitionFinalizerTest.finalizeExecutable(analysis, Optional.empty(),
                Optional.ofNullable(workers));
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
        List<io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation> workspaces =
                executable.memoryPlan().workspaces().isEmpty() ? List.of()
                : List.of(CpuContiguousWorkspace.allocate(
                        executable.memoryPlan().workspaces().getFirst().byteSize(),
                        executable.memoryPlan().workspaces().getFirst().byteAlignment()));
        var state = new RunState(executable.memoryPlan(), runtimeBindings, workspaces);
        try {
            executable.bind(state).execute();
            int outputLength = Math.toIntExact(buffers.get(3).byteSize() / 8);
            double[] generated = new double[outputLength];
            for (int i = 0; i < outputLength; i++) {
                generated[i] = buffers.get(3).segment().get(DOUBLE, i * 8L);
                buffers.get(3).segment().set(DOUBLE, i * 8L, 0);
            }
            var referenceBindings = new ArrayList<>(executable.accessBindings());
            analysis.plan().materialization().ifPresent(copy -> referenceBindings.set(
                    copy.sourceBoundaryIndex(), ranged(copy.sourceBinding(),
                            executable.binding().start(), executable.binding().end())));
            CpuScalarReferenceKernel.execute(buffers.stream().map(buffer -> buffer.argument()).toList(),
                    referenceBindings, executable.binding().start(),
                    executable.binding().end());
            for (int i = 0; i < outputLength; i++) assertEquals(
                    buffers.get(3).segment().get(DOUBLE, i * 8L), generated[i],
                    2e-7 * Math.max(1.0, Math.abs(generated[i])));
        } finally {
            state.close();
            if (workers != null) workers.close();
        }
    }

    private static CpuAccessPlan.Binding ranged(CpuAccessPlan.Binding source,
            long start, long end) {
        return CpuAccessPlan.Binding.create(source.plan(), source.extents().stream()
                        .mapToLong(Long::longValue).toArray(), source.baseElementOffset(),
                source.effectiveStrides().stream().mapToLong(Long::longValue).toArray(),
                source.elementCount(), start, end, source.referencedElementSpan());
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }
}
