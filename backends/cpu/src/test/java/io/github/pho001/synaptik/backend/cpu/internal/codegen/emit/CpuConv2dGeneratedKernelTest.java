package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.lang.reflect.AccessFlag;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuConv2dGeneratedKernelTest {
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    @Test void emitsDeterministicDirectGroupedFloatBody() throws Throwable {
        var base = CpuConv2dLoweringTest.context(
                List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                Shape.of(1, 4, 3, 4), Shape.of(4, 2, 2, 2), Shape.of(1, 4, 4, 5),
                new Conv2dAttrs(1, 1, 1, 1, 1, 1, 2), null);
        var context = withInputs(base, new CpuPartitionAnalysisInputs(false,
                java.util.Collections.nCopies(4, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, generator.generateClassBytes(route.specialization(), route.kernelIr()));
        var model = ClassFile.of().parse(first);
        StringBuilder members = new StringBuilder();
        java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .forEach(member -> members.append(member.owner().asInternalName()).append('.')
                        .append(member.name().stringValue()).append('\n'));
        assertAll(() -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertFalse(members.toString().contains("synaptik")),
                () -> assertFalse(members.toString().contains("java/lang/reflect")));

        float[] input = new float[48];
        float[] weight = new float[32];
        float[] bias = {.25f, -.5f, 1f, -2f};
        for (int i = 0; i < input.length; i++) input[i] = i * .125f - 2f;
        for (int i = 0; i < weight.length; i++) weight[i] = (i % 7 - 3) * .25f;
        float[] output = new float[80];
        long[] geometry = plan.conv2dGeometry().orElseThrow().pack(new long[4]);
        var handle = generator.defineClassBytes(route.specialization(), first).entryPoint();
        handle.invokeExact(input, weight, bias, output, geometry, 0L, 80L);
        for (int oc = 0; oc < 4; oc++) for (int oh = 0; oh < 4; oh++) for (int ow = 0; ow < 5; ow++) {
            float expected = bias[oc];
            int groupBase = oc / 2 * 2;
            for (int ic = 0; ic < 2; ic++) for (int kh = 0; kh < 2; kh++) for (int kw = 0; kw < 2; kw++) {
                int ih = oh - 1 + kh, iw = ow - 1 + kw;
                float x = ih < 0 || ih >= 3 || iw < 0 || iw >= 4 ? 0f
                        : input[(groupBase + ic) * 12 + ih * 4 + iw];
                expected = (float) (expected + x * weight[oc * 8 + ic * 4 + kh * 2 + kw]);
            }
            int index = oc * 20 + oh * 5 + ow;
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(output[index]),
                    "output " + index);
        }
        assertAll(() -> assertEquals(0, plan.workspaceDeclaration().stream().count()),
                () -> assertTrue(plan.materialization().isEmpty()));
    }

    @Test void mixedGeneralCarriersAndBfloat16UseFloatAccumulation() throws Throwable {
        var inputShape = Shape.of(1, 1, 2, 2);
        var weightShape = Shape.of(1, 1, 2, 2);
        var outputShape = Shape.of(1, 1, 1, 1);
        var layouts = List.of(
                io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(inputShape,
                        new long[] {12, 12, 3, 1}, 2, true),
                io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(weightShape,
                        new long[] {10, 10, 3, 1}, 1, true),
                io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(outputShape,
                        new long[] {2, 2, 2, 2}, 3, true));
        var base = CpuConv2dLoweringTest.context(
                List.of(DataType.BFLOAT16, DataType.FLOAT32), inputShape, weightShape,
                outputShape, Conv2dAttrs.defaults(), layouts);
        var context = withInputs(base, new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var handle = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        short[] input = new short[10];
        input[2] = bf16(1f); input[3] = bf16(2f); input[5] = bf16(-3f); input[6] = bf16(.5f);
        float[] output = new float[6];
        try (Arena arena = Arena.ofConfined()) {
            var weight = arena.allocate(12L * Float.BYTES, Float.BYTES);
            weight.set(FLOAT, 1L * Float.BYTES, .25f);
            weight.set(FLOAT, 2L * Float.BYTES, -2f);
            weight.set(FLOAT, 4L * Float.BYTES, 1.5f);
            weight.set(FLOAT, 5L * Float.BYTES, 4f);
            long[] geometry = plan.conv2dGeometry().orElseThrow().pack(new long[3]);
            handle.invokeExact(input, weight, output, geometry, 0L, 1L);
        }
        float expected = 0f;
        expected = (float) (expected + 1f * .25f);
        expected = (float) (expected + 2f * -2f);
        expected = (float) (expected + -3f * 1.5f);
        expected = (float) (expected + .5f * 4f);
        assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(output[3]));
    }

    @Test void conceptualPaddingMultipliesPositiveZeroByInfinity() throws Throwable {
        var base = CpuConv2dLoweringTest.context(List.of(DataType.FLOAT64, DataType.FLOAT64),
                Shape.of(1, 1, 1, 1), Shape.of(1, 1, 1, 1), Shape.of(1, 1, 3, 3),
                new Conv2dAttrs(1, 1, 1, 1, 1, 1, 1), null);
        var plan = new CpuPartitionPreparer().analyze(withInputs(base,
                new CpuPartitionAnalysisInputs(false, java.util.Collections.nCopies(3,
                        CarrierAccess.DOUBLE_ARRAY)))).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var handle = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        double[] output = new double[9];
        handle.invokeExact(new double[] {2}, new double[] {Double.POSITIVE_INFINITY}, output,
                plan.conv2dGeometry().orElseThrow().pack(new long[3]), 0L, 9L);
        assertAll(() -> assertTrue(Double.isNaN(output[0])),
                () -> assertEquals(Double.POSITIVE_INFINITY, output[4]),
                () -> assertTrue(Double.isNaN(output[8])));
    }

    @Test void fusesExternalAddAndReluAfterRepresentedConvolutionBoundary() throws Throwable {
        var base = fusedContext();
        var context = withInputs(base, new CpuPartitionAnalysisInputs(false,
                java.util.Collections.nCopies(5, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr.Epilogue.ADD_RELU,
                ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr)
                        plan.units().getFirst().portablePlan().portableKernelIr()).epilogue());
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var handle = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
        float[] input = {1, -2, 3, 4};
        float[] weight = {2};
        float[] bias = {.5f};
        float[] external = {-3, 2, -10, 1};
        float[] output = new float[4];
        handle.invokeExact(input, weight, bias, external, output,
                plan.conv2dGeometry().orElseThrow().pack(new long[5]), 0L, 4L);
        assertArrayEquals(new float[] {0, 0, 0, 9.5f}, output);
        assertTrue(plan.workspaceDeclaration().isEmpty());
    }

    static PrepareContext<CpuPartitionAnalysisInputs> fusedContext() {
        Shape shape = Shape.of(1, 1, 2, 2), weightShape = Shape.of(1, 1, 1, 1);
        TensorDescriptor tensor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        TensorDescriptor weight = new TensorDescriptor(DataType.FLOAT32, weightShape,
                Optional.of(LayoutDescriptor.contiguous(weightShape)), false);
        TensorDescriptor bias = new TensorDescriptor(DataType.FLOAT32, Shape.of(1),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(1))), false);
        List<ValueId> ids = java.util.stream.LongStream.range(0, 7)
                .mapToObj(ValueId::new).toList();
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(Conv2dKind.CONV2D,
                        Conv2dAttrs.defaults()), ids.subList(0, 3), List.of(ids.get(4))),
                new CompiledNode(new NodeId(1), new Operation(BinaryArithmeticKind.ADD,
                        NoOperationAttrs.INSTANCE), List.of(ids.get(4), ids.get(3)),
                        List.of(ids.get(5))),
                new CompiledNode(new NodeId(2), new Operation(UnaryElementwiseKind.RELU,
                        NoOperationAttrs.INSTANCE), List.of(ids.get(5)), List.of(ids.get(6))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var descriptors = List.of(tensor, weight, bias, tensor, tensor, tensor, tensor);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < ids.size(); i++) {
            values.add(new GraphValue(ids.get(i), descriptors.get(i)));
            boolean produced = i >= 4;
            boolean published = i == 6;
            memory.add(new LogicalMemoryRequirement(ids.get(i), descriptors.get(i),
                    produced ? Optional.of(partition) : Optional.empty(),
                    published ? List.of() : List.of(partition), published));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static short bf16(float value) {
        return (short) (Float.floatToRawIntBits(value) >>> 16);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> withInputs(
            PrepareContext<CpuPartitionAnalysisInputs> base, CpuPartitionAnalysisInputs inputs) {
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), inputs);
    }
}
