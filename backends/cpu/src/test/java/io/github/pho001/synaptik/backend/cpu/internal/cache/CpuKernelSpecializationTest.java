package io.github.pho001.synaptik.backend.cpu.internal.cache;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import org.junit.jupiter.api.Test;

class CpuKernelSpecializationTest {
    @Test void excludesCompatibleExtentsButIncludesNumericalAndStrategyFacts() {
        var one = CpuPartitionPreparerTest.analyze(Shape.of(1)).plan().units().getFirst()
                .portablePlan().specialization();
        var many = CpuPartitionPreparerTest.analyze(Shape.of(99)).plan().units().getFirst()
                .portablePlan().specialization();
        assertAll(
                () -> assertEquals(one, many),
                () -> assertEquals(one.structuralKey(), many.structuralKey()),
                () -> assertSame(CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        one.numericalMode()),
                () -> assertArrayEquals(one.compatibilityBytes(), many.compatibilityBytes()));
    }
}
