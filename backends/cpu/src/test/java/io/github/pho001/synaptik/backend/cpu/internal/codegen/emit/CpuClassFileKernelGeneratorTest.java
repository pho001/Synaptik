package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.classfile.ClassFile;
import org.junit.jupiter.api.Test;

class CpuClassFileKernelGeneratorTest {
    @Test void emitsDeterministicVerifiedJava26BytesWithPrimitiveBounds() {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(7)).plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var model = ClassFile.of().parse(first);
        assertAll(
                () -> assertArrayEquals(first, second),
                () -> assertEquals(ClassFile.JAVA_26_VERSION, model.majorVersion()),
                () -> assertEquals("(Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;"
                                + "Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;JJ)V",
                        model.methods().getFirst().methodType().stringValue()));
    }
}
