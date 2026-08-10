package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.classfile.ClassFile;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import jdk.incubator.vector.FloatVector;

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
                                + "Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;[JJJ)V",
                        model.methods().getFirst().methodType().stringValue()));
    }

    @Test void generatedCodeShapeDistinguishesEveryOrderedAccessRegime() {
        Shape target = Shape.of(2, 4, 3);
        int dense = codeLength(descriptor(target, LayoutDescriptor.contiguous(target)), target);
        int scalar = codeLength(descriptor(Shape.scalar(),
                LayoutDescriptor.contiguous(Shape.scalar())), target);
        int bias = codeLength(descriptor(Shape.of(3),
                LayoutDescriptor.contiguous(Shape.of(3))), target);
        Shape blockShape = Shape.of(2, 1, 3);
        int block = codeLength(descriptor(blockShape,
                LayoutDescriptor.contiguous(blockShape)), target);
        int general = codeLength(descriptor(target,
                LayoutDescriptor.of(target, new long[]{20, 4, 2}, 0, true)), target);
        assertAll(
                () -> assertEquals(5, java.util.Set.of(dense, scalar, bias, block, general).size()),
                () -> assertTrue(dense < general),
                () -> assertTrue(scalar < general));
    }

    @Test void emitsDeterministicVerifiedPreferredFloatVectorBytes() {
        Shape shape = Shape.of(FloatVector.SPECIES_PREFERRED.length() + 1);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var analysis = CpuPartitionPreparerTest.analyze(descriptor, descriptor, descriptor,
                descriptor, new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1)));
        var route = analysis.plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertAll(
                () -> assertEquals("vector", analysis.plan().executionStrategy().toString()),
                () -> assertArrayEquals(first, second),
                () -> assertDoesNotThrow(() -> generator.defineClassBytes(
                        route.specialization(), first)));
    }

    private static int codeLength(TensorDescriptor first, Shape target) {
        var dense = descriptor(target, LayoutDescriptor.contiguous(target));
        var scalar = descriptor(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()));
        var route = CpuPartitionPreparerTest.analyze(first, dense, scalar, dense,
                CpuPartitionAnalysisInputs.DEFAULT).plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        return ((java.lang.classfile.attribute.CodeAttribute) ClassFile.of().parse(
                generator.generateClassBytes(route.specialization(), route.kernelIr()))
                .methods().getFirst().code().orElseThrow()).codeLength();
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }
}
