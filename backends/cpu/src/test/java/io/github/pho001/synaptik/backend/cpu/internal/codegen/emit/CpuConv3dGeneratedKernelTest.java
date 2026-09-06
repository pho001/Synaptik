package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv3dReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.AccessFlag;
import java.nio.ByteOrder;
import java.util.List;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

class CpuConv3dGeneratedKernelTest {
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());

    @Test void emitsDeterministicGroupedBiasedFloatBody() throws Throwable {
        var attrs = new Conv3dAttrs(1, 1, 1, 1, 1, 1, 1, 1, 1, 2);
        var base = CpuConv3dLoweringTest.context(
                List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                Shape.of(1, 4, 2, 2, 2), Shape.of(4, 2, 2, 2, 2),
                Shape.of(1, 4, 3, 3, 3), attrs, null);
        var context = withInputs(base, 4, CarrierAccess.FLOAT_ARRAY);
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var unit = plan.units().getFirst();
        var route = unit.portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(bytes,
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        var model = ClassFile.of().parse(bytes);
        StringBuilder members = new StringBuilder();
        java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast)
                .forEach(member -> members.append(member.owner().asInternalName()).append('.')
                        .append(member.name().stringValue()).append('\n'));
        assertAll(() -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertFalse(members.toString().contains("synaptik")),
                () -> assertFalse(members.toString().contains("java/lang/reflect")));
        float[] input = new float[32];
        float[] weight = new float[64];
        float[] bias = {.25f, -.5f, 1f, -2f};
        float[] output = new float[108];
        for (int i = 0; i < input.length; i++) {
            input[i] = i * .125f - 2f;
        }
        for (int i = 0; i < weight.length; i++) {
            weight[i] = (i % 7 - 3) * .25f;
        }
        var handle = generator.defineClassBytes(route.specialization(), bytes).entryPoint();
        handle.invokeExact(input, weight, bias, output,
                unit.conv3dGeometry().orElseThrow().pack(new long[4]), 0L, 108L);
        double[] expected = CpuConv3dReferenceKernel.evaluate(
                List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                DataType.FLOAT32, new double[][] {doubles(input), doubles(weight), doubles(bias)},
                new long[][] {{1, 4, 2, 2, 2}, {4, 2, 2, 2, 2}, {4}},
                new long[] {0, 0, 0},
                new long[][] {{32, 8, 4, 2, 1}, {16, 8, 4, 2, 1}, {1}},
                new long[] {1, 4, 3, 3, 3}, attrs);
        for (int i = 0; i < output.length; i++) {
            assertEquals(Float.floatToRawIntBits((float) expected[i]),
                    Float.floatToRawIntBits(output[i]), "cell " + i);
        }
        assertAll(() -> assertTrue(plan.workspaceDeclaration().isEmpty()),
                () -> assertTrue(plan.materialization().isEmpty()));
    }

    @Test void conceptualPaddingMultipliesZeroByInfinity() throws Throwable {
        var base = CpuConv3dLoweringTest.context(List.of(DataType.FLOAT64, DataType.FLOAT64),
                Shape.of(1, 1, 1, 1, 1), Shape.of(1, 1, 1, 1, 1),
                Shape.of(1, 1, 3, 3, 3),
                new Conv3dAttrs(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), null);
        var plan = new CpuPartitionPreparer().analyze(
                withInputs(base, 3, CarrierAccess.DOUBLE_ARRAY)).plan();
        var unit = plan.units().getFirst();
        var route = unit.portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var handle = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        double[] output = new double[27];
        handle.invokeExact(new double[] {2}, new double[] {Double.POSITIVE_INFINITY}, output,
                unit.conv3dGeometry().orElseThrow().pack(new long[3]), 0L, 27L);
        assertAll(() -> assertTrue(Double.isNaN(output[0])),
                () -> assertEquals(Double.POSITIVE_INFINITY, output[13]),
                () -> assertTrue(Double.isNaN(output[26])));
    }

    @Test void provisionalNestedFloatVectorMatchesGroupedBiasedPaddingSubrange() throws Throwable {
        var base = CpuConv3dLoweringTest.context(
                List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                Shape.of(2, 4, 3, 3, 18), Shape.of(6, 2, 2, 2, 3),
                Shape.of(2, 6, 4, 4, 18),
                new Conv3dAttrs(1, 1, 1, 1, 1, 1, 1, 1, 1, 2), null);
        var plan = new CpuPartitionPreparer().analyze(
                withInputs(base, 4, CarrierAccess.FLOAT_ARRAY)).plan();
        var route = plan.units().getFirst().portablePlan();
        var vector = vector(route.specialization(),
                FloatVector.SPECIES_PREFERRED.vectorBitSize());
        var generator = new CpuClassFileKernelGenerator();
        var scalar = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        byte[] bytes = generator.generateClassBytes(vector, route.kernelIr());
        var actual = generator.defineClassBytes(vector, bytes).entryPoint();
        float[] input = new float[2 * 4 * 3 * 3 * 18];
        float[] weight = new float[6 * 2 * 2 * 2 * 3];
        float[] bias = {.25f, -.5f, 1f, -2f, .125f, -.75f};
        float[] expected = new float[2 * 6 * 4 * 4 * 18];
        float[] output = new float[expected.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = (i % 37 - 18) * .03125f;
        }
        for (int i = 0; i < weight.length; i++) {
            weight[i] = (i % 17 - 8) * .0625f;
        }
        java.util.Arrays.fill(expected, 17f);
        java.util.Arrays.fill(output, 17f);
        long[] geometry = plan.units().getFirst().conv3dGeometry().orElseThrow()
                .pack(new long[4]);
        scalar.invokeExact(input, weight, bias, expected, geometry, 5L,
                (long) expected.length - 7);
        actual.invokeExact(input, weight, bias, output, geometry, 5L,
                (long) output.length - 7);
        assertArrayEquals(raw(expected), raw(output));
        assertVectorMembers(bytes, "FloatVector");
    }

    @Test void provisionalNestedDoubleVectorMatchesNonWidthStrideDilationBatchSubrange() throws Throwable {
        var base = CpuConv3dLoweringTest.context(List.of(DataType.FLOAT64, DataType.FLOAT64),
                Shape.of(2, 3, 9, 9, 18), Shape.of(4, 3, 3, 3, 3),
                Shape.of(2, 4, 4, 4, 18),
                new Conv3dAttrs(2, 2, 1, 1, 1, 1, 2, 2, 1, 1), null);
        var plan = new CpuPartitionPreparer().analyze(
                withInputs(base, 3, CarrierAccess.DOUBLE_ARRAY)).plan();
        var route = plan.units().getFirst().portablePlan();
        var vector = vector(route.specialization(),
                DoubleVector.SPECIES_PREFERRED.vectorBitSize());
        var generator = new CpuClassFileKernelGenerator();
        var scalar = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        byte[] bytes = generator.generateClassBytes(vector, route.kernelIr());
        var actual = generator.defineClassBytes(vector, bytes).entryPoint();
        double[] input = new double[2 * 3 * 9 * 9 * 18];
        double[] weight = new double[4 * 3 * 3 * 3 * 3];
        double[] expected = new double[2 * 4 * 4 * 4 * 18];
        double[] output = new double[expected.length];
        for (int i = 0; i < input.length; i++) {
            input[i] = (i % 41 - 20) * .015625;
        }
        for (int i = 0; i < weight.length; i++) {
            weight[i] = (i % 19 - 9) * .03125;
        }
        java.util.Arrays.fill(expected, -17d);
        java.util.Arrays.fill(output, -17d);
        long[] geometry = plan.units().getFirst().conv3dGeometry().orElseThrow()
                .pack(new long[3]);
        scalar.invokeExact(input, weight, expected, geometry, 3L,
                (long) expected.length - 5);
        actual.invokeExact(input, weight, output, geometry, 3L,
                (long) output.length - 5);
        assertArrayEquals(raw(expected), raw(output));
        assertVectorMembers(bytes, "DoubleVector");
    }

    @Test void provisionalNestedFloatVectorKeepsBiasImmutableAcrossSegmentScalarCells()
            throws Throwable {
        Shape inputShape = Shape.of(1, 2, 3, 3, 18);
        Shape weightShape = Shape.of(3, 2, 2, 2, 3);
        Shape outputShape = Shape.of(1, 3, 4, 4, 18);
        var layouts = List.of(offsetLayout(inputShape, 3), offsetLayout(weightShape, 2),
                offsetLayout(Shape.of(3), 1), offsetLayout(outputShape, 4));
        var base = CpuConv3dLoweringTest.context(
                List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32), inputShape,
                weightShape, outputShape,
                new Conv3dAttrs(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), layouts);
        var plan = new CpuPartitionPreparer().analyze(withInputs(base, 4,
                CarrierAccess.MEMORY_SEGMENT)).plan();
        var route = plan.units().getFirst().portablePlan();
        var vector = vector(route.specialization(), FloatVector.SPECIES_PREFERRED.vectorBitSize());
        var generator = new CpuClassFileKernelGenerator();
        var scalarHandle = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        byte[] vectorBytes = generator.generateClassBytes(vector, route.kernelIr());
        var vectorHandle = generator.defineClassBytes(vector, vectorBytes).entryPoint();
        try (Arena arena = Arena.ofConfined()) {
            var input = arena.allocate((3 + 1 * 2 * 3 * 3 * 18L) * Float.BYTES, Float.BYTES);
            var weight = arena.allocate((2 + 3 * 2 * 2 * 2 * 3L) * Float.BYTES, Float.BYTES);
            var bias = arena.allocate((1 + 3L) * Float.BYTES, Float.BYTES);
            var scalar = arena.allocate((4 + 1 * 3 * 4 * 4 * 18L) * Float.BYTES, Float.BYTES);
            var actual = arena.allocate(scalar.byteSize(), Float.BYTES);
            for (long i = 0; i < input.byteSize() / Float.BYTES; i++) {
                input.set(FLOAT, i * Float.BYTES, (i % 29 - 14) * .03125f);
            }
            for (long i = 0; i < weight.byteSize() / Float.BYTES; i++) {
                weight.set(FLOAT, i * Float.BYTES, (i % 13 - 6) * .0625f);
            }
            for (long i = 0; i < bias.byteSize() / Float.BYTES; i++) {
                bias.set(FLOAT, i * Float.BYTES, (i - 2) * .25f);
            }
            byte[] immutableBias = bias.toArray(ValueLayout.JAVA_BYTE);
            scalar.fill((byte) 0x5a);
            actual.fill((byte) 0x5a);
            long[] geometry = plan.units().getFirst().conv3dGeometry().orElseThrow()
                    .pack(new long[4]);
            // The first padded planes and rows contain consecutive scalar cells. A later
            // interior row begins with a scalar left border and then reaches a full vector block.
            long start = 0;
            long end = 3L * 4 * 4 * 18 - 5;
            scalarHandle.invokeExact(input, weight, bias, scalar, geometry, start, end);
            vectorHandle.invokeExact(input, weight, bias, actual, geometry, start, end);
            assertArrayEquals(scalar.toArray(ValueLayout.JAVA_BYTE),
                    actual.toArray(ValueLayout.JAVA_BYTE));
            assertArrayEquals(immutableBias, bias.toArray(ValueLayout.JAVA_BYTE));
        }
        assertVectorMembers(vectorBytes, "FloatVector");
    }

    private static double[] doubles(float[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
    }

    private static LayoutDescriptor offsetLayout(Shape shape, long offset) {
        long[] extents = shape.toLongArray();
        long[] strides = new long[extents.length];
        long stride = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, extents[axis]);
        }
        return LayoutDescriptor.of(shape, strides, offset, true);
    }
    private static PrepareContext<CpuPartitionAnalysisInputs> withInputs(
            PrepareContext<CpuPartitionAnalysisInputs> base, int count, CarrierAccess access) {
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                java.util.Collections.nCopies(count, access)));
    }

    private static CpuKernelSpecialization vector(CpuKernelSpecialization scalar, int bits) {
        return new CpuKernelSpecialization(scalar.loweringFingerprint(), scalar.numericalMode(),
                io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR,
                scalar.boundaryDataTypes(), scalar.carrierPattern(), bits, -1,
                scalar.scalarPowerRealizations(), false, 63);
    }

    private static int[] raw(float[] values) {
        int[] result = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = Float.floatToRawIntBits(values[i]);
        }
        return result;
    }

    private static long[] raw(double[] values) {
        long[] result = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = Double.doubleToRawLongBits(values[i]);
        }
        return result;
    }

    private static void assertVectorMembers(byte[] bytes, String vector) {
        String members = java.util.stream.StreamSupport.stream(
                        ClassFile.of().parse(bytes).constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance)
                .map(MemberRefEntry.class::cast)
                .map(member -> member.owner().asInternalName() + '.'
                        + member.name().stringValue())
                .collect(java.util.stream.Collectors.joining("\n"));
        assertAll(() -> assertTrue(members.contains(vector + ".fromArray")
                        || members.contains(vector + ".fromMemorySegment")),
                () -> assertTrue(members.contains(vector + ".broadcast")),
                () -> assertTrue(members.contains(vector + ".mul")),
                () -> assertTrue(members.contains(vector + ".add")),
                () -> assertTrue(members.contains(vector + ".intoArray")
                        || members.contains(vector + ".intoMemorySegment")),
                () -> assertFalse(members.contains("synaptik")));
    }
}
