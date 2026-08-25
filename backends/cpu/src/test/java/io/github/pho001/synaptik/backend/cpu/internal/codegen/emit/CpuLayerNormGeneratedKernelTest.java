package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuTrailingNormalizationReferenceKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.AccessFlag;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CpuLayerNormGeneratedKernelTest {
    @Test void allFormsAndResultTypesGenerateDirectTypedArtifacts() throws Throwable {
        for (boolean affine : List.of(false, true)) for (DataType type : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            int inputs = affine ? 3 : 1;
            var base = CpuTrailingNormalizationLoweringTest.context(true, affine, type,
                    Shape.of(2, 3), Shape.of(3), affine ? List.of(0, 1, 2) : List.of(0));
            var carriers = new java.util.ArrayList<CarrierAccess>();
            for (int i = 0; i <= inputs; i++) carriers.add(array(type));
            var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                    base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
            byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
            var model = ClassFile.of().parse(bytes);
            assertAll(() -> assertEquals(inputs + 5,
                            generator.defineClassBytes(route.specialization(), bytes).entryPoint()
                                    .type().parameterCount()),
                    () -> assertTrue(model.flags().has(AccessFlag.FINAL)),
                    () -> assertTrue(model.fields().isEmpty()),
                    () -> assertEquals(1, model.methods().size()),
                    () -> assertTrue(java.util.stream.StreamSupport.stream(
                            model.constantPool().spliterator(), false)
                            .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                            .noneMatch(member -> member.owner().asInternalName()
                                    .startsWith("io/github/pho001/synaptik"))));
            var artifact = generator.defineClassBytes(route.specialization(), bytes);
            Object input = represented(type, new double[] {1, 2, 3, -3, -2, -1});
            Object output = represented(type, new double[6]);
            Object scale = represented(type, new double[] {2, -1, .5});
            Object bias = represented(type, new double[] {1, 2, 3});
            var geometry = plan.trailingNormalizationGeometry().orElseThrow();
            try (Arena arena = Arena.ofConfined()) {
                if (affine) artifact.entryPoint().invokeWithArguments(input, scale, bias, output,
                        arena.allocate(geometry.scratchSliceBytes(), 8),
                        geometry.pack(new long[4]), 0L, 2L);
                else artifact.entryPoint().invokeWithArguments(input, output,
                        arena.allocate(geometry.scratchSliceBytes(), 8),
                        geometry.pack(new long[2]), 0L, 2L);
            }
            double[][] semanticInputs = affine
                    ? new double[][] {{1, 2, 3, -3, -2, -1}, {2, -1, .5}, {1, 2, 3}}
                    : new double[][] {{1, 2, 3, -3, -2, -1}};
            long[][] extents = affine ? new long[][] {{2, 3}, {3}, {3}}
                    : new long[][] {{2, 3}};
            long[][] strides = affine ? new long[][] {{3, 1}, {1}, {1}}
                    : new long[][] {{3, 1}};
            double[] expected = CpuTrailingNormalizationReferenceKernel.evaluate(
                    CpuTrailingNormalizationIr.Kind.LAYER, affine
                            ? CpuTrailingNormalizationIr.Form.LAYER_AFFINE
                            : CpuTrailingNormalizationIr.Form.LAYER,
                    java.util.Collections.nCopies(inputs, type), type, epsilon(type), semanticInputs,
                    extents, new long[inputs], strides, 1);
            double[] actual = decoded(type, output);
            for (int index = 0; index < actual.length; index++) assertEquals(raw(type, expected[index]),
                    raw(type, actual[index]), type + "/" + affine + "/" + index
                            + " expectedValue=" + expected[index] + " actualValue=" + actual[index]);
            retain("layer-" + (affine ? "affine-" : "plain-") + type.name().toLowerCase(),
                    bytes, route.specialization().compatibilityBytes(), route.specialization().toString());
        }
    }

    @Test void mixedTypesCarriersOffsetsAndGeneralLayoutsMatchIndependentOracle() throws Throwable {
        Shape inputShape = Shape.of(2, 3), normalized = Shape.of(3);
        var inputDescriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.BFLOAT16, inputShape,
                java.util.Optional.of(LayoutDescriptor.of(inputShape, new long[] {4, 1}, 1, true)),
                false);
        var scaleDescriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT32, normalized,
                java.util.Optional.of(LayoutDescriptor.of(normalized, new long[] {0}, 1, true)),
                false);
        var biasDescriptor = CpuScatterLoweringTest.desc(DataType.FLOAT64, normalized);
        var outputDescriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT64, inputShape,
                java.util.Optional.of(LayoutDescriptor.of(inputShape, new long[] {5, 1}, 2, true)),
                false);
        var base = CpuScatterLoweringTest.context(new Operation(
                io.github.pho001.synaptik.model.operation.normalization.LayerNormKind.LAYER_NORM,
                new AffineLayerNormAttrs(normalized, ScalarValue.float64(1e-5))),
                List.of(0, 1, 2), List.of(inputDescriptor, scaleDescriptor, biasDescriptor),
                outputDescriptor);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        short[] input = {bits(99), bits(1), bits(2), bits(3), bits(99), bits(-3), bits(-2), bits(-1)};
        float[] scale = {99, 2}; double[] bias = {1, 2, 3};
        double[] output = new double[12]; java.util.Arrays.fill(output, -77);
        try (Arena arena = Arena.ofConfined()) {
            artifact.entryPoint().invokeWithArguments(MemorySegment.ofArray(input), scale,
                    MemorySegment.ofArray(bias), output,
                    arena.allocate(plan.trailingNormalizationGeometry().orElseThrow()
                            .scratchSliceBytes(), 8),
                    plan.trailingNormalizationGeometry().orElseThrow().pack(new long[4]), 0L, 2L);
        }
        double root = StrictMath.sqrt(2.0 / 3.0 + 1e-5);
        assertAll(() -> assertEquals((-1 / root) * 2 + 1, output[2], 2e-15),
                () -> assertEquals(2, output[3], 0),
                () -> assertEquals((1 / root) * 2 + 3, output[4], 2e-15),
                () -> assertEquals((-1 / root) * 2 + 1, output[7], 2e-15),
                () -> assertEquals(-77, output[0]), () -> assertEquals(-77, output[11]));
    }

    @Test void bfloatBodyUsesFloatMeanAndOneFinalEncoding() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(true, false, DataType.BFLOAT16,
                Shape.of(1, 3), Shape.of(3), List.of(0));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.SHORT_ARRAY, CarrierAccess.SHORT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        short[] input = {bits(1), bits(2), bits(3)}, output = new short[3];
        var geometry = plan.trailingNormalizationGeometry().orElseThrow();
        try (Arena arena = Arena.ofConfined()) {
            artifact.entryPoint().invokeWithArguments(input, output,
                    arena.allocate(geometry.scratchSliceBytes(), 8), geometry.pack(new long[2]), 0L, 1L);
        }
        assertAll(() -> assertTrue(Float.intBitsToFloat(Short.toUnsignedInt(output[0]) << 16) < 0),
                () -> assertEquals(0, output[1]),
                () -> assertTrue(Float.intBitsToFloat(Short.toUnsignedInt(output[2]) << 16) > 0));
    }

    @Test void affineFloatBodyReusesTrailingScaleAndBias() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(true, true, DataType.FLOAT32,
                Shape.of(2, 3), Shape.of(3), List.of(0, 1, 2));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY,
                        CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        float[] input = {1, 2, 3, 4, 5, 6}, scale = {2, -1, 0.5f}, bias = {1, 2, 3};
        float[] output = new float[6]; var geometry = plan.trailingNormalizationGeometry().orElseThrow();
        try (Arena arena = Arena.ofConfined()) {
            artifact.entryPoint().invokeWithArguments(input, scale, bias, output,
                    arena.allocate(geometry.scratchSliceBytes(), 8), geometry.pack(new long[4]), 0L, 2L);
        }
        float root = (float) StrictMath.sqrt((float) (2f / 3f) + 1e-5f);
        assertArrayEquals(new float[] {(-1f / root) * 2 + 1, 2, (1f / root) * .5f + 3,
                (-1f / root) * 2 + 1, 2, (1f / root) * .5f + 3}, output, 2e-6f);
    }

    @Test void directThreePassBodyNormalizesEachCompleteSlice() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(true, false, DataType.FLOAT64,
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
        var geometry = plan.trailingNormalizationGeometry().orElseThrow();
        try (Arena arena = Arena.ofConfined()) {
            artifact.entryPoint().invokeWithArguments(input, output,
                    arena.allocate(geometry.scratchSliceBytes(), 8), geometry.pack(new long[2]),
                    0L, 2L);
        }
        double root = StrictMath.sqrt(2.0 / 3.0 + 1e-5);
        assertArrayEquals(new double[] {-1 / root, 0, 1 / root, -1 / root, 0, 1 / root},
                output, 2e-15);
    }

    @Test void constantsSignedZerosNanAndInfinityUseExactSliceClasses() throws Throwable {
        var base = CpuTrailingNormalizationLoweringTest.context(true, false, DataType.FLOAT64,
                Shape.of(3, 2), Shape.of(2), List.of(0));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        double[] input = {-0.0, 0.0, Double.longBitsToDouble(0x7ff0000000000001L), 1.0,
                Double.NEGATIVE_INFINITY, 2.0};
        double[] output = new double[6];
        try (Arena arena = Arena.ofConfined()) {
            artifact.entryPoint().invokeWithArguments(input, output,
                    arena.allocate(plan.trailingNormalizationGeometry().orElseThrow()
                            .scratchSliceBytes(), 8),
                    plan.trailingNormalizationGeometry().orElseThrow().pack(new long[2]), 0L, 3L);
        }
        assertAll(() -> assertEquals(0L, Double.doubleToRawLongBits(output[0])),
                () -> assertEquals(0L, Double.doubleToRawLongBits(output[1])),
                () -> assertTrue(Double.isNaN(output[2])), () -> assertTrue(Double.isNaN(output[3])),
                () -> assertTrue(Double.isNaN(output[4])), () -> assertTrue(Double.isNaN(output[5])));
    }

    private static short bits(float value) {
        int bits = Float.floatToRawIntBits(value);
        return (short) ((bits + 0x7fff + (bits >>> 16 & 1)) >>> 16);
    }

    private static CarrierAccess array(DataType type) { return type == DataType.FLOAT64
            ? CarrierAccess.DOUBLE_ARRAY : type == DataType.FLOAT32
                ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY; }
    static Object represented(DataType type, double[] values) {
        if (type == DataType.FLOAT64) return values.clone();
        if (type == DataType.FLOAT32) {
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
            return result;
        }
        short[] result = new short[values.length];
        for (int i = 0; i < values.length; i++) result[i] = bits((float) values[i]);
        return result;
    }
    static double[] decoded(DataType type, Object values) {
        int length = type == DataType.FLOAT64 ? ((double[]) values).length
                : type == DataType.FLOAT32 ? ((float[]) values).length : ((short[]) values).length;
        double[] result = new double[length];
        for (int i = 0; i < length; i++) result[i] = type == DataType.FLOAT64
                ? ((double[]) values)[i] : type == DataType.FLOAT32 ? ((float[]) values)[i]
                : Float.intBitsToFloat(Short.toUnsignedInt(((short[]) values)[i]) << 16);
        return result;
    }
    static long raw(DataType type, double value) {
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(value)
                : type == DataType.FLOAT32
                    ? Integer.toUnsignedLong(Float.floatToRawIntBits((float) value))
                    : Integer.toUnsignedLong(Float.floatToRawIntBits((float) value) >>> 16);
    }
    static double epsilon(DataType type) { return type == DataType.FLOAT64 ? 1e-5
            : type == DataType.FLOAT32 ? (double) 1e-5f
            : Float.intBitsToFloat(0x3728 << 16); }
    static void retain(String name, byte[] bytes, byte[] compatibility, String specialization)
            throws Exception {
        String root = System.getProperty("synaptik.cpu.normalization.evidence");
        if (root == null) root = System.getenv("SYNAPTIK_CPU_NORMALIZATION_EVIDENCE");
        if (root == null) return; Path generated = Path.of(root, "generated");
        Files.createDirectories(generated); Files.write(generated.resolve(name + ".class"), bytes);
        Files.write(generated.resolve(name + ".compatibility"), compatibility);
        Files.writeString(generated.resolve(name + ".specialization"), specialization + "\n");
    }
}
