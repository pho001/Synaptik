package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;

class CpuShapePolymorphicArtifactTest {
    @Test void twoCompatibleExtentsShareBytesAndLoadedCompatibilityIdentity() {
        var first = CpuPartitionPreparerTest.analyze(Shape.of(3)).plan().units().getFirst().portablePlan();
        var second = CpuPartitionPreparerTest.analyze(Shape.of(17)).plan().units().getFirst().portablePlan();
        var zero = CpuPartitionPreparerTest.analyze(Shape.of(0)).plan().units().getFirst().portablePlan();
        var store = new CpuGeneratedKernelArtifactStore();
        var a = store.loadOrGenerate(first.specialization(), first.kernelIr());
        var b = store.loadOrGenerate(second.specialization(), second.kernelIr());
        assertAll(
                () -> assertEquals(first.kernelIr().structuralKey(), second.kernelIr().structuralKey()),
                () -> assertEquals(first.kernelIr().structuralKey(), zero.kernelIr().structuralKey()),
                () -> assertArrayEquals(a.classBytes(), b.classBytes()),
                () -> assertSame(a.hiddenClass(), b.hiddenClass()),
                () -> assertSame(a, b));
    }

    @Test void materializedClassesRemainShapePolymorphicButDifferFromDirectClasses() {
        var first = materialized(Shape.of(2, 3), new long[] {1, 2});
        var second = materialized(Shape.of(4, 5), new long[] {1, 4});
        var direct = CpuPartitionPreparerTest.analyze(first.descriptor, first.dense, first.dense,
                first.dense, CpuPartitionAnalysisInputs.DEFAULT).plan().units().getFirst()
                .portablePlan();
        var firstRoute = first.route;
        var secondRoute = second.route;
        var store = new CpuGeneratedKernelArtifactStore();
        var firstArtifact = store.loadOrGenerate(firstRoute.specialization(), firstRoute.kernelIr());
        var secondArtifact = store.loadOrGenerate(secondRoute.specialization(), secondRoute.kernelIr());
        assertAll(
                () -> assertEquals(firstRoute.specialization(), secondRoute.specialization()),
                () -> assertArrayEquals(firstArtifact.classBytes(), secondArtifact.classBytes()),
                () -> assertNotEquals(direct.specialization().structuralKey(),
                        firstRoute.specialization().structuralKey()),
                () -> assertEquals(0, firstRoute.specialization().materializedSourcePosition()));
    }

    private static Materialized materialized(Shape shape, long[] strides) {
        var source = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0, true)), false);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 0, 1, 20, 1, 2, Long.MAX_VALUE, 1, 1);
        var inputs = new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy);
        var ordinary = CpuPartitionPreparerTest.analyze(source, dense, dense, dense, inputs);
        assertTrue(ordinary.plan().materializations().isEmpty());
        var route = CpuPartitionPreparerTest.explicitRepresentationCandidate(ordinary, policy, 0)
                .plan().representationUnits().getFirst().portablePlan();
        return new Materialized(source, dense, route);
    }

    private record Materialized(TensorDescriptor descriptor, TensorDescriptor dense,
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route) { }
}
