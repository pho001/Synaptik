package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuTrailingNormalizationReferenceKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Map;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import org.junit.jupiter.api.Test;

class CpuRmsNormGeneratedKernelTest {
    @Test void allFormsAndResultTypesGenerateDirectTypedArtifacts() throws Throwable {
        for (boolean scaled : List.of(false, true)) for (DataType type : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            int inputs = scaled ? 2 : 1;
            var base = CpuTrailingNormalizationLoweringTest.context(false, scaled, type,
                    Shape.of(2, 3), Shape.of(3), scaled ? List.of(0, 1) : List.of(0));
            var carriers = new java.util.ArrayList<CarrierAccess>();
            for (int i = 0; i <= inputs; i++) carriers.add(array(type));
            var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                    base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
            var model = ClassFile.of().parse(bytes);
            assertEquals(inputs + 4, generator.defineClassBytes(route.specialization(), bytes)
                    .entryPoint().type().parameterCount());
            var members = java.util.stream.StreamSupport.stream(
                    model.constantPool().spliterator(), false).filter(MemberRefEntry.class::isInstance)
                    .map(MemberRefEntry.class::cast).toList();
            assertAll(() -> assertTrue(model.fields().isEmpty()),
                    () -> assertEquals(1, model.methods().size()),
                    () -> assertTrue(members.stream().anyMatch(member -> member.owner()
                            .asInternalName().equals("java/lang/Math")
                            && member.name().stringValue().equals("hypot"))),
                    () -> assertTrue(members.stream().noneMatch(member -> member.owner()
                            .asInternalName().startsWith("io/github/pho001/synaptik"))));
            var artifact = generator.defineClassBytes(route.specialization(), bytes);
            Object input = CpuLayerNormGeneratedKernelTest.represented(type,
                    new double[] {1, 2, 3, -3, -2, -1});
            Object output = CpuLayerNormGeneratedKernelTest.represented(type, new double[6]);
            Object scale = CpuLayerNormGeneratedKernelTest.represented(type,
                    new double[] {2, -1, .5});
            var geometry = plan.trailingNormalizationGeometry().orElseThrow();
            if (scaled) artifact.entryPoint().invokeWithArguments(input, scale, output,
                    geometry.pack(new long[3]), 0L, 2L);
            else artifact.entryPoint().invokeWithArguments(input, output,
                    geometry.pack(new long[2]), 0L, 2L);
            double[][] semanticInputs = scaled
                    ? new double[][] {{1, 2, 3, -3, -2, -1}, {2, -1, .5}}
                    : new double[][] {{1, 2, 3, -3, -2, -1}};
            double[] expected = CpuTrailingNormalizationReferenceKernel.evaluate(
                    CpuTrailingNormalizationIr.Kind.RMS, scaled
                            ? CpuTrailingNormalizationIr.Form.RMS_SCALED
                            : CpuTrailingNormalizationIr.Form.RMS,
                    java.util.Collections.nCopies(inputs, type), type,
                    CpuLayerNormGeneratedKernelTest.epsilon(type), semanticInputs,
                    scaled ? new long[][] {{2, 3}, {3}} : new long[][] {{2, 3}},
                    new long[inputs], scaled ? new long[][] {{3, 1}, {1}}
                            : new long[][] {{3, 1}}, 1);
            double[] actual = CpuLayerNormGeneratedKernelTest.decoded(type, output);
            for (int index = 0; index < actual.length; index++) assertEquals(
                    CpuLayerNormGeneratedKernelTest.raw(type, expected[index]),
                    CpuLayerNormGeneratedKernelTest.raw(type, actual[index]),
                    type + "/" + scaled + "/" + index + " expectedValue=" + expected[index]
                            + " actualValue=" + actual[index]);
            CpuLayerNormGeneratedKernelTest.retain("rms-" + (scaled ? "scaled-" : "plain-")
                    + type.name().toLowerCase(), bytes, route.specialization().compatibilityBytes(),
                    route.specialization().toString());
        }
    }

    @Test void bfloatBodyRequiresNoScratchAndPreservesSigns() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(false, false, DataType.BFLOAT16,
                Shape.of(1, 3), Shape.of(3), List.of(0));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.SHORT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        short[] input = {bits(-1), bits(0), bits(1)}, output = new short[3];
        artifact.entryPoint().invokeWithArguments(input, output,
                plan.trailingNormalizationGeometry().orElseThrow().pack(new long[2]), 0L, 1L);
        assertAll(() -> assertTrue((output[0] & 0x8000) != 0), () -> assertEquals(0, output[1]),
                () -> assertTrue((output[2] & 0x8000) == 0));
    }

    @Test void scaledFloatBodyReusesTrailingScale() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(false, true, DataType.FLOAT32,
                Shape.of(2, 3), Shape.of(3), List.of(0, 1));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        float[] input = {1, 2, 3, -3, -2, -1}, scale = {2, -1, .5f}, output = new float[6];
        artifact.entryPoint().invokeWithArguments(input, scale, output,
                plan.trailingNormalizationGeometry().orElseThrow().pack(new long[3]), 0L, 2L);
        float root = (float) StrictMath.sqrt((float) (14f / 3f) + 1e-5f);
        assertArrayEquals(new float[] {(1f / root) * 2, (2f / root) * -1, (3f / root) * .5f,
                (-3f / root) * 2, (-2f / root) * -1, (-1f / root) * .5f}, output, 2e-6f);
    }

    @Test void directTwoPassBodyUsesUncenteredPopulationRoot() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(false, false, DataType.FLOAT64,
                Shape.of(2, 3), Shape.of(3), List.of(0));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        double[] input = {1, 2, 3, -3, -2, -1}, output = new double[6];
        artifact.entryPoint().invokeWithArguments(input, output,
                plan.trailingNormalizationGeometry().orElseThrow().pack(new long[2]), 0L, 2L);
        double root = StrictMath.sqrt(14.0 / 3.0 + 1e-5);
        assertArrayEquals(new double[] {1 / root, 2 / root, 3 / root,
                -3 / root, -2 / root, -1 / root}, output, 2e-15);
    }

    @Test void zerosInfinityAndNanPreserveSpecifiedClassesAndSigns() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(false, false, DataType.FLOAT64,
                Shape.of(3, 2), Shape.of(2), List.of(0));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        double[] input = {0.0, -0.0, Double.POSITIVE_INFINITY, -2.0,
                Double.longBitsToDouble(0x7ff0000000000001L), 1.0};
        double[] output = new double[6];
        artifact.entryPoint().invokeWithArguments(input, output,
                plan.trailingNormalizationGeometry().orElseThrow().pack(new long[2]), 0L, 3L);
        assertAll(() -> assertEquals(0L, Double.doubleToRawLongBits(output[0])),
                () -> assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(output[1])),
                () -> assertTrue(Double.isNaN(output[2])),
                () -> assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(output[3])),
                () -> assertTrue(Double.isNaN(output[4])), () -> assertTrue(Double.isNaN(output[5])));
    }

    private static short bits(float value) {
        int bits = Float.floatToRawIntBits(value);
        return (short) ((bits + 0x7fff + (bits >>> 16 & 1)) >>> 16);
    }
    private static CarrierAccess array(DataType type) { return type == DataType.FLOAT64
            ? CarrierAccess.DOUBLE_ARRAY : type == DataType.FLOAT32
                ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY; }
}
