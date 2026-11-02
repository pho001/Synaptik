package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.util.List;
import java.util.Optional;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

class CpuClassFileKernelGeneratorTest {
    @Test void emitsAndDefinesAScalarBfloat16PointwiseChain() {
        io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertNumericalOutcome(DataType.BFLOAT16, () -> {
        Shape shape = Shape.of(7);
        var descriptor = new TensorDescriptor(DataType.BFLOAT16, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var analysis = new CpuPartitionPreparer().analyze(CpuPartitionPreparerTest.context(
                descriptor, descriptor, descriptor, descriptor,
                new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.SHORT_ARRAY,
                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.SHORT_ARRAY,
                        CarrierAccess.MEMORY_SEGMENT))));
        var route = analysis.plan().units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());

        assertAll(
                () -> assertEquals("scalar", analysis.plan().executionStrategy().toString()),
                () -> assertEquals(59, route.specialization().classIdentitySchema()),
                () -> assertDoesNotThrow(() -> generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes)),
                () -> assertThrows(IllegalArgumentException.class, () -> generator.generateClassBytes(
                        new io.github.pho001.synaptik.backend.cpu.internal.cache
                                .CpuKernelSpecialization(route.specialization().loweringFingerprint(),
                                route.specialization().executionStrategy(),
                                route.specialization().boundaryDataTypes(),
                                route.specialization().carrierPattern(), 0, -1,
                                route.specialization().scalarPowerRealizations(), false, 52),
                        route.kernelIr())));
        });
    }

    @Test void emitsExactRandomEntryDescriptorsForRepresentativeCarrierAndAddressForms() {
        var heapInitial = randomRoute(CpuRandomLoweringTest.initialContext(1, 2),
                List.of(CarrierAccess.LONG_ARRAY));
        var segmentInitial = randomRoute(CpuRandomLoweringTest.initialContext(1, 2),
                List.of(CarrierAccess.MEMORY_SEGMENT));
        var denseFloat64 = randomRoute(CpuRandomLoweringTest.dropoutContext(
                        DataType.FLOAT64, Shape.of(3), .5),
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.DOUBLE_ARRAY, CarrierAccess.BYTE_ARRAY,
                        CarrierAccess.LONG_ARRAY));
        var denseFloat32 = randomRoute(CpuRandomLoweringTest.dropoutContext(
                        DataType.FLOAT32, Shape.of(3), .5),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.LONG_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.BYTE_ARRAY,
                        CarrierAccess.LONG_ARRAY));
        var allSegmentFloat64 = randomRoute(CpuRandomLoweringTest.dropoutContext(
                        DataType.FLOAT64, Shape.of(3), .5),
                java.util.Collections.nCopies(5, CarrierAccess.MEMORY_SEGMENT));
        Shape generalShape = Shape.of(2, 2);
        Shape stateShape = Shape.of(2);
        var mixedGeneralFloat32 = randomRoute(CpuRandomLoweringTest.dropoutContext(
                        DataType.FLOAT32, generalShape, .5,
                        List.of(LayoutDescriptor.of(generalShape, new long[]{0, 2}, 1, true),
                                LayoutDescriptor.of(stateShape, new long[]{2}, 1, true),
                                LayoutDescriptor.of(generalShape, new long[]{7, 2}, 1, true),
                                LayoutDescriptor.of(generalShape, new long[]{8, 3}, 2, true),
                                LayoutDescriptor.of(stateShape, new long[]{3}, 2, true))),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.LONG_ARRAY));

        assertAll(
                () -> assertRandomDescriptor(heapInitial, "([J[JJJ)V"),
                () -> assertRandomDescriptor(segmentInitial,
                        "(Ljava/lang/foreign/MemorySegment;[JJJ)V"),
                () -> assertRandomDescriptor(denseFloat64, "([D[J[D[B[J[JJJ)V"),
                () -> assertRandomDescriptor(denseFloat32, "([F[J[F[B[J[JJJ)V"),
                () -> assertRandomDescriptor(allSegmentFloat64,
                        "(Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;"
                                + "Ljava/lang/foreign/MemorySegment;"
                                + "Ljava/lang/foreign/MemorySegment;"
                                + "Ljava/lang/foreign/MemorySegment;[JJJ)V"),
                () -> assertRandomDescriptor(mixedGeneralFloat32,
                        "([FLjava/lang/foreign/MemorySegment;[F"
                                + "Ljava/lang/foreign/MemorySegment;[J[JJJ)V"));
    }
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
                () -> assertDoesNotThrow(() -> generator.defineClassBytes(route.specialization(), route.kernelIr(), first)));
    }

    @Test void controlledOrdinalsGuardNonleadingRowsBeforeOutOfBoundsOperandLoads() throws Throwable {
        double[] input = {10, 11, Double.NaN, Double.NaN, Double.NaN, Double.NaN, 16, 17};
        double[] output = new double[12];
        java.util.Arrays.fill(output, -7.0);
        invokeRowGuardWitness(false, 2, input, output, new long[] {2, 3, 2}, 12);
        assertArrayEquals(new double[] {10, 11, -7, -7, -7, -7, 16, 17, -7, -7, -7, -7}, output);
    }

    @Test void controlledSliceOrdinalsRetainRowsWhenTheOmittedFeatureExtentIsZero() throws Throwable {
        double[] input = {10, Double.NaN, Double.NaN, 13};
        double[] output = {-7, -7, -7, -7, -7, -7};
        invokeRowGuardWitness(true, 0, input, output, new long[] {2, 3, 0}, 6);
        assertArrayEquals(new double[] {10, -7, -7, 13, -7, -7}, output);
    }

    private static void invokeRowGuardWitness(boolean sliced, long logicalRowStride,
            double[] input, double[] output, long[] geometry, long count) throws Throwable {
        var segment = java.lang.constant.ClassDesc.of("java.lang.foreign.MemorySegment");
        var type = java.lang.constant.MethodTypeDesc.of(java.lang.constant.ConstantDescs.CD_void,
                java.lang.constant.ConstantDescs.CD_double.arrayType(),
                java.lang.constant.ConstantDescs.CD_double.arrayType(),
                java.lang.constant.ConstantDescs.CD_long.arrayType(),
                java.lang.constant.ConstantDescs.CD_long, java.lang.constant.ConstantDescs.CD_long, segment);
        var row = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, logicalRowStride);
        var facts = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(
                List.of(row, row), List.of(row, row));
        byte[] bytes = CpuCarrierEmitter.withControlFacts(facts, () -> ClassFile.of().build(
                java.lang.constant.ClassDesc.of(CpuClassFileKernelGeneratorTest.class.getPackageName()
                        + ".ControlledRowGuardWitness"),
                owner -> owner.withMethodBody("apply", type, ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                        code -> CpuCarrierEmitter.withDescriptor(type, true, () -> {
                            int ordinal = code.allocateLocal(java.lang.classfile.TypeKind.LONG);
                            var loop = code.newLabel(); var done = code.newLabel(); var advance = code.newLabel();
                            code.lload(3).lstore(ordinal).labelBinding(loop);
                            code.lload(ordinal).lload(5).lcmp().branch(java.lang.classfile.Opcode.IFGE, done);
                            if (sliced) CpuLoopEmitter.emitActiveSliceGuard(code, 1, ordinal,
                                    java.lang.classfile.TypeKind.LONG, 2, 0,
                                    new boolean[] {false, false, true}, advance);
                            else CpuLoopEmitter.emitActiveOrdinalGuard(code, 1, ordinal,
                                    java.lang.classfile.TypeKind.LONG, advance);
                            code.aload(1).lload(ordinal).l2i().aload(0).lload(ordinal).l2i().daload().dastore();
                            code.labelBinding(advance).lload(ordinal).loadConstant(1L).ladd().lstore(ordinal)
                                    .branch(java.lang.classfile.Opcode.GOTO, loop).labelBinding(done).return_();
                        }))));
        var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes, true);
        var method = lookup.findStatic(lookup.lookupClass(), "apply", java.lang.invoke.MethodType.methodType(
                void.class, double[].class, double[].class, long[].class, long.class, long.class,
                java.lang.foreign.MemorySegment.class));
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var control = arena.allocate(19, Long.BYTES);
            control.fill((byte) 0);
            control.set(java.lang.foreign.ValueLayout.JAVA_BYTE, 0, (byte) 1);
            control.set(java.lang.foreign.ValueLayout.JAVA_LONG, 8, 3L);
            control.set(java.lang.foreign.ValueLayout.JAVA_BYTE, 16, (byte) 1);
            method.invokeExact(input, output, geometry, 0L, count, control);
        }
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

    private static CpuPortableRoutePlan randomRoute(
            PrepareContext<CpuPartitionAnalysisInputs> base, List<CarrierAccess> carriers) {
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }

    private static void assertRandomDescriptor(CpuPortableRoutePlan route, String expected) {
        var generator = new CpuClassFileKernelGenerator();
        var model = ClassFile.of().parse(generator.generateClassBytes(
                route.specialization(), route.kernelIr()));
        assertAll(
                () -> assertEquals(expected, route.specialization().entryType().descriptorString()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals(expected,
                        model.methods().getFirst().methodType().stringValue()));
    }
}
