package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import org.junit.jupiter.api.Test;

class CpuGeneratedKernelShapeTest {
    @Test void loadedArtifactRetainsExactHandleAndDefensiveBytes() {
        var route = CpuPartitionPreparerTest.analyze(Shape.scalar()).plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var artifact = generator.defineClassBytes(route.specialization(), bytes);
        bytes[0] = 0;
        assertAll(
                () -> assertSame(artifact.hiddenClass(), artifact.hiddenLookup().lookupClass()),
                () -> assertEquals(route.specialization().entryType(), artifact.entryPoint().type()),
                () -> assertNotEquals(0, artifact.classBytes()[0]));
    }
}
