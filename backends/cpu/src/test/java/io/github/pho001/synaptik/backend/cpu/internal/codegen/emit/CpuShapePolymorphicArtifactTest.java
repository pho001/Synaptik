package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStore;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import org.junit.jupiter.api.Test;

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
}
