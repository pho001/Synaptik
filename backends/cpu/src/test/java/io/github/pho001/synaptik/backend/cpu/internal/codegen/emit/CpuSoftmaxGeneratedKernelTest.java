package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuSoftmaxReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.AccessFlag;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuSoftmaxGeneratedKernelTest {
    @Test void retainsThreeCurrentSchemaUnchangedFamilyControlsWhenEvidenceIsRequested()
            throws Exception {
        var reductionCarriers = List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY);
        var log = controlPlan(CpuAggregateLoweringTest.context(AggregateReductionKind.LOG_SUM_EXP,
                DataType.FLOAT32, Shape.of(128, 2048),
                new MultiAxisReductionAttrs(List.of(1), false), Shape.of(128)), reductionCarriers);
        var sum = controlPlan(CpuAggregateLoweringTest.context(AggregateReductionKind.SUM,
                DataType.FLOAT32, Shape.of(128, 2048),
                new MultiAxisReductionAttrs(List.of(1), false), Shape.of(128)), reductionCarriers);
        var expBase = CpuScatterLoweringTest.context(new Operation(UnaryElementwiseKind.EXP,
                NoOperationAttrs.INSTANCE), List.of(0),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(128, 2048))),
                CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(128, 2048)));
        var exp = controlPlan(expBase, reductionCarriers);
        retainControl("control-log-sum-exp", log);
        retainControl("control-sum", sum);
        retainControl("control-exp", exp);
        assertAll(() -> assertEquals(2, log.bufferDeclarations().size()),
                () -> assertEquals(2, sum.bufferDeclarations().size()),
                () -> assertEquals(2, exp.bufferDeclarations().size()));
    }

    @Test void frozenGeneralSegmentToHeapLogSoftmaxUsesDirectTypedBody() throws Throwable {
        Shape shape = Shape.of(2, 3);
        var input = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(DataType.FLOAT32,
                shape, Optional.of(LayoutDescriptor.of(shape, new long[] {4, 1}, 1, true)), false);
        var output = CpuScatterLoweringTest.desc(DataType.FLOAT32, shape);
        var base = CpuScatterLoweringTest.context(new Operation(SoftmaxKind.LOG_SOFTMAX,
                new SoftmaxAttrs(1)), List.of(0), List.of(input), output);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        retainEvidence(SoftmaxKind.LOG_SOFTMAX, DataType.FLOAT32, 1, route, bytes);
        float[] source = {99, 1, 2, 3, 99, -3, -2, -1}; float[] result = new float[6];
        generator.defineClassBytes(route.specialization(), bytes).entryPoint().invokeWithArguments(
                MemorySegment.ofArray(source), result,
                plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 2L);
        assertEquals(1.0, StrictMath.exp(result[0]) + StrictMath.exp(result[1])
                + StrictMath.exp(result[2]), 2e-6);
    }

    @Test void arbitraryLayoutsAndMiddleAxisPreserveLogicalSliceCoordinates() throws Throwable {
        Shape shape = Shape.of(2, 3, 4);
        var inputDescriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 4, 1}, 0, true)), false);
        var outputDescriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {20, 5, 1}, 2, true)), false);
        var base = CpuScatterLoweringTest.context(new Operation(SoftmaxKind.LOG_SOFTMAX,
                new SoftmaxAttrs(1)), List.of(0), List.of(inputDescriptor), outputDescriptor);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        double[] input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        double[] output = new double[40]; java.util.Arrays.fill(output, -77);
        artifact.entryPoint().invokeWithArguments(input, output,
                plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 8L);
        double[] expected = CpuSoftmaxReferenceKernel.evaluate(SoftmaxKind.LOG_SOFTMAX,
                DataType.FLOAT64, input, new long[] {2, 3, 4}, 0,
                new long[] {0, 4, 1}, 1);
        for (int first = 0; first < 2; first++) for (int selected = 0; selected < 3;
                selected++) for (int last = 0; last < 4; last++) {
            int logical = (first * 3 + selected) * 4 + last;
            int physical = 2 + first * 20 + selected * 5 + last;
            assertEquals(Double.doubleToRawLongBits(expected[logical]),
                    Double.doubleToRawLongBits(output[physical]));
        }
        assertEquals(-77, output[0]); assertEquals(-77, output[39]);
    }

    @Test void allTypesKindsAndHeapSegmentPairsProduceNormalizedSlices() throws Throwable {
        for (SoftmaxKind kind : SoftmaxKind.values()) for (DataType type : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) for (int pattern = 0;
                pattern < 4; pattern++) {
            var carriers = List.of((pattern & 1) == 0 ? arrayCarrier(type)
                            : CarrierAccess.MEMORY_SEGMENT,
                    (pattern & 2) == 0 ? arrayCarrier(type) : CarrierAccess.MEMORY_SEGMENT);
            var plan = plan(kind, type, Shape.of(2, 3), 1, carriers);
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(route.specialization(),
                    generator.generateClassBytes(route.specialization(), route.kernelIr()));
            retainEvidence(kind, type, pattern, route, artifact.classBytes());
            Object input = represented(type, new double[] {1, 2, 3, -3, -2, -1});
            Object output = represented(type, new double[6]);
            Object inCarrier = (pattern & 1) == 0 ? input : segment(input);
            Object outCarrier = (pattern & 2) == 0 ? output : segment(output);
            artifact.entryPoint().invokeWithArguments(inCarrier, outCarrier,
                    plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 2L);
            double[] values = decoded(type, output);
            double[] expected = CpuSoftmaxReferenceKernel.evaluate(kind, type,
                    new double[] {1, 2, 3, -3, -2, -1}, new long[] {2, 3}, 0,
                    new long[] {3, 1}, 1);
            double sum = kind == SoftmaxKind.SOFTMAX
                    ? values[0] + values[1] + values[2]
                    : StrictMath.exp(values[0]) + StrictMath.exp(values[1])
                            + StrictMath.exp(values[2]);
            assertEquals(1.0, sum, type == DataType.BFLOAT16 ? 0.02 : 2e-6,
                    kind + "/" + type + "/" + pattern);
            for (int index = 0; index < values.length; index++) assertEquals(
                    raw(type, expected[index]), raw(type, values[index]),
                    kind + "/" + type + "/" + pattern + "/" + index);
        }
    }

    @Test void directDoubleBodiesProduceStableExpectedValuesAndTypedClass() throws Throwable {
        for (SoftmaxKind kind : SoftmaxKind.values()) {
            var base = CpuSoftmaxLoweringTest.context(kind, DataType.FLOAT64, Shape.of(2, 3), 1);
            var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                    base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                    List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
            var artifact = generator.defineClassBytes(route.specialization(), bytes);
            double[] input = {1, 2, 3, -3, -2, -1}; double[] output = new double[6];
            artifact.entryPoint().invokeWithArguments(input, output,
                    plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 2L);
            double sum = kind == SoftmaxKind.SOFTMAX ? output[0] + output[1] + output[2]
                    : StrictMath.exp(output[0]) + StrictMath.exp(output[1]) + StrictMath.exp(output[2]);
            var model = ClassFile.of().parse(bytes);
            var members = java.util.stream.StreamSupport.stream(
                    model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                    .map(MemberRefEntry.class::cast).toList();
            assertAll(() -> assertEquals(1.0, sum, 2e-15),
                    () -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                    () -> assertTrue(model.fields().isEmpty()),
                    () -> assertEquals("([D[D[JJJ)V",
                            model.methods().getFirst().methodTypeSymbol().descriptorString()),
                    () -> assertEquals(kind == SoftmaxKind.LOG_SOFTMAX,
                            members.stream().anyMatch(member -> member.owner().asInternalName()
                                    .equals("java/lang/Math")
                                    && member.name().stringValue().equals("log"))),
                    () -> assertTrue(members.stream().noneMatch(member -> member.owner()
                            .asInternalName().startsWith("io/github/pho001/synaptik"))));
        }
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            plan(SoftmaxKind kind, DataType type, Shape shape, int axis,
                    List<CarrierAccess> carriers) {
        var base = CpuSoftmaxLoweringTest.context(kind, type, shape, axis);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan();
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            controlPlan(PrepareContext<CpuPartitionAnalysisInputs> base,
                    List<CarrierAccess> carriers) {
        return new CpuPartitionPreparer().analyze(new PrepareContext<>(base.partition(),
                base.nodes(), base.values(), base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, carriers))).plan();
    }

    private static void retainControl(String name,
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan)
            throws Exception {
        String root = evidenceRoot(); if (root == null) return;
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        Path generated = Path.of(root, "generated"); Files.createDirectories(generated);
        Files.write(generated.resolve(name + ".class"),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        Files.write(generated.resolve(name + ".compatibility"),
                route.specialization().compatibilityBytes());
        Files.writeString(generated.resolve(name + ".descriptor"),
                route.specialization().entryType().descriptorString() + System.lineSeparator());
        Files.writeString(generated.resolve(name + ".specialization"),
                route.specialization().toString() + System.lineSeparator());
    }

    private static CarrierAccess arrayCarrier(DataType type) {
        return type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY;
    }
    private static Object represented(DataType type, double[] values) {
        if (type == DataType.FLOAT64) return values.clone();
        if (type == DataType.FLOAT32) {
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
            return result;
        }
        short[] result = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            int bits = Float.floatToRawIntBits((float) values[i]);
            result[i] = (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
        }
        return result;
    }
    private static MemorySegment segment(Object values) {
        if (values instanceof double[] value) return MemorySegment.ofArray(value);
        if (values instanceof float[] value) return MemorySegment.ofArray(value);
        return MemorySegment.ofArray((short[]) values);
    }
    private static double[] decoded(DataType type, Object values) {
        int length = type == DataType.FLOAT64 ? ((double[]) values).length
                : type == DataType.FLOAT32 ? ((float[]) values).length : ((short[]) values).length;
        double[] result = new double[length];
        for (int i = 0; i < length; i++) result[i] = type == DataType.FLOAT64
                ? ((double[]) values)[i] : type == DataType.FLOAT32 ? ((float[]) values)[i]
                : Float.intBitsToFloat(Short.toUnsignedInt(((short[]) values)[i]) << 16);
        return result;
    }
    private static long raw(DataType type, double value) {
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(value)
                : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                        Float.floatToRawIntBits((float) value))
                : Integer.toUnsignedLong(Float.floatToRawIntBits((float) value) >>> 16);
    }

    private static void retainEvidence(SoftmaxKind kind, DataType type, int pattern,
            io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan route,
            byte[] bytes) throws Exception {
        String root = evidenceRoot();
        boolean general = route.kernelIr().values().getFirst().accessPlan().regime()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.GENERAL_ODOMETER;
        boolean target = pattern == 0 || type == DataType.FLOAT32
                && kind == SoftmaxKind.SOFTMAX && pattern == 2
                || type == DataType.FLOAT32 && kind == SoftmaxKind.LOG_SOFTMAX && pattern == 1
                    && general;
        if (root == null || !target) return;
        Path generated = Path.of(root, "generated"); Files.createDirectories(generated);
        String name = kind.name().toLowerCase() + "-" + type.name().toLowerCase()
                + (pattern == 0 ? "-heap" : pattern == 1 ? "-segment-to-heap"
                        : "-heap-to-segment");
        Files.write(generated.resolve(name + ".class"), bytes);
        Files.write(generated.resolve(name + ".compatibility"),
                route.specialization().compatibilityBytes());
        Files.writeString(generated.resolve(name + ".descriptor"),
                route.specialization().entryType().descriptorString() + System.lineSeparator());
        Files.writeString(generated.resolve(name + ".specialization"),
                route.specialization().toString() + System.lineSeparator());
    }

    private static String evidenceRoot() {
        String root = System.getProperty("synaptik.cpu.softmax.evidence");
        if (root == null) root = System.getenv("SYNAPTIK_CPU_SOFTMAX_EVIDENCE");
        return root;
    }
}
