package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

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
            artifact.entryPoint().invokeExact(a, b, c, output, 1L, 4L);
            assertEquals(-99.0, output.get(DOUBLE, 0));
            for (int i = 1; i < 4; i++) assertEquals(
                    CpuScalarReferenceKernel.gelu((i - 2.0) + 0.25) * 2.0,
                    output.get(DOUBLE, i * 8L), 0.0);
            assertEquals(-99.0, output.get(DOUBLE, 32));
        }
    }
}
