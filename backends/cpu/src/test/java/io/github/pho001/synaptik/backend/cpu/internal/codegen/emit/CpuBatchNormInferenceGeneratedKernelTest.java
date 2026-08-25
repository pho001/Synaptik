package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.reflect.AccessFlag;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuBatchNormInferenceGeneratedKernelTest {
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());

    @Test void emitsDirectTypedChannelHoistedFloatBody() throws Throwable {
        var base = CpuBatchNormInferenceLoweringTest.context(
                java.util.Collections.nCopies(5, DataType.FLOAT32), Shape.of(2, 3, 4), 1,
                List.of(0, 1, 2, 3, 4));
        var context = withInputs(base, new CpuPartitionAnalysisInputs(false,
                java.util.Collections.nCopies(6, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, second);
        var model = ClassFile.of().parse(first);
        assertAll(() -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals("invoke", model.methods().getFirst().methodName().stringValue()));
        StringBuilder members = new StringBuilder();
        java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .forEach(member -> members.append(member.owner().asInternalName()).append('.')
                        .append(member.name().stringValue()).append('\n'));
        assertAll(() -> assertTrue(members.toString().contains("java/lang/Math.sqrt")),
                () -> assertFalse(members.toString().contains("synaptik")),
                () -> assertFalse(members.toString().contains("java/nio/ByteOrder")),
                () -> assertFalse(members.toString().contains("ValueLayout.withOrder")));

        var handle = generator.defineClassBytes(route.specialization(), first).entryPoint();
        float[] input = new float[24];
        for (int index = 0; index < input.length; index++) input[index] = index * .25f - 2f;
        float[] scale = {.5f, 1.25f, -2f};
        float[] bias = {.125f, -.5f, 2f};
        float[] mean = {-1f, .25f, 3f};
        float[] variance = {4f, .5f, -2f};
        float[] output = new float[24];
        long[] geometry = plan.batchNormInferenceGeometry().orElseThrow().pack(new long[6]);
        handle.invokeExact(input, scale, bias, mean, variance, output, geometry, 0L, 3L);
        for (int index = 0; index < output.length; index++) {
            int channel = index / 4 % 3;
            float centered = input[index] - mean[channel];
            float radicand = variance[channel] + 1e-5f;
            float denominator = (float) Math.sqrt(radicand);
            float standardized = centered / denominator;
            float scaled = standardized * scale[channel];
            float expected = scaled + bias[channel];
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(output[index]),
                    "index " + index);
        }
    }

    @Test void parallelPreparationBakesLargerNonChannelDomainAndThresholdUnits() {
        var base = CpuBatchNormInferenceLoweringTest.context(
                java.util.Collections.nCopies(5, DataType.FLOAT32), Shape.of(32, 64, 256), 1,
                List.of(0, 1, 2, 3, 4));
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                4, 4, 4096);
        var plan = new CpuPartitionPreparer().analyze(withInputs(base,
                new CpuPartitionAnalysisInputs(false,
                        java.util.Collections.nCopies(6, CarrierAccess.FLOAT_ARRAY), config))).plan();
        assertAll(() -> assertEquals(CpuBatchNormInferenceIr.RangeForm.NON_CHANNEL_RANGE,
                        plan.batchNormInferenceGeometry().orElseThrow().rangeForm()),
                () -> assertEquals(8192, plan.elementCount()),
                () -> assertEquals(64, plan.minimumElementsPerWorker()),
                () -> assertEquals(4, plan.selectedRangeCount()),
                () -> assertTrue(plan.workspaceDeclaration().isEmpty()),
                () -> assertTrue(plan.materialization().isEmpty()));
    }

    @Test void mixedCarriersAndGeneralLayoutsPreserveExactTypedFormula() throws Throwable {
        Shape shape = Shape.of(2, 3, 4), vector = Shape.of(3);
        var layouts = List.of(
                LayoutDescriptor.of(shape, new long[] {20, 6, 1}, 2, true),
                LayoutDescriptor.of(vector, new long[] {2}, 3, true),
                LayoutDescriptor.of(vector, new long[] {0}, 5, true),
                LayoutDescriptor.of(vector, new long[] {3}, 7, true),
                LayoutDescriptor.of(vector, new long[] {1}, 11, true),
                LayoutDescriptor.of(shape, new long[] {30, 8, 2}, 3, true));
        var types = List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16,
                DataType.FLOAT32, DataType.FLOAT64);
        var base = CpuBatchNormInferenceLoweringTest.context(types, shape, 1,
                List.of(0, 1, 2, 3, 4), layouts);
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                4, 4, 1);
        var carriers = List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.SHORT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.DOUBLE_ARRAY, CarrierAccess.MEMORY_SEGMENT);
        var plan = new CpuPartitionPreparer().analyze(withInputs(base,
                new CpuPartitionAnalysisInputs(false, carriers, config))).plan();
        assertEquals(CpuBatchNormInferenceIr.RangeForm.NON_CHANNEL_RANGE,
                plan.batchNormInferenceGeometry().orElseThrow().rangeForm());
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        var handle = generator.defineClassBytes(route.specialization(), bytes).entryPoint();
        float[] input = new float[40];
        double[] variance = new double[14];
        short[] bias = new short[6];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment scale = arena.allocate(9 * 8L, 8);
            MemorySegment mean = arena.allocate(14 * 4L, 4);
            MemorySegment output = arena.allocate(56 * 8L, 8);
            for (long prefix = 0; prefix < 2; prefix++) for (long channel = 0; channel < 3;
                    channel++) for (long suffix = 0; suffix < 4; suffix++) {
                long address = 2 + prefix * 20 + channel * 6 + suffix;
                input[(int) address] = (float) (prefix * 2 + channel * .25 + suffix * .125 - 1);
            }
            double[] scaleValues = {.5, 1.25, -2};
            float[] meanValues = {-1, .25f, 3};
            double[] varianceValues = {4, .5, -2};
            for (int channel = 0; channel < 3; channel++) {
                scale.set(DOUBLE, (3 + channel * 2L) * 8L, scaleValues[channel]);
                mean.set(FLOAT, (7 + channel * 3L) * 4L, meanValues[channel]);
                variance[11 + channel] = varianceValues[channel];
            }
            bias[5] = (short) (Float.floatToRawIntBits(.5f) >>> 16);
            long[] geometry = plan.batchNormInferenceGeometry().orElseThrow()
                    .pack(new long[6]);
            handle.invokeExact(input, scale, bias, mean, variance, output, geometry, 0L, 8L);
            for (long prefix = 0; prefix < 2; prefix++) for (long channel = 0; channel < 3;
                    channel++) for (long suffix = 0; suffix < 4; suffix++) {
                long inputAddress = 2 + prefix * 20 + channel * 6 + suffix;
                long outputAddress = 3 + prefix * 30 + channel * 8 + suffix * 2;
                double expected = ((input[(int) inputAddress] - meanValues[(int) channel])
                        / Math.sqrt(varianceValues[(int) channel] + 1e-5))
                        * scaleValues[(int) channel] + .5;
                assertEquals(Double.doubleToRawLongBits(expected),
                        Double.doubleToRawLongBits(output.get(DOUBLE, outputAddress * 8L)));
            }
        }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> withInputs(
            PrepareContext<CpuPartitionAnalysisInputs> base, CpuPartitionAnalysisInputs inputs) {
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), inputs);
    }
}
