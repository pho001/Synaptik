package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuSoftmaxReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuSoftmaxGeneratedKernelTest {

    @Test void frozenGeneralSegmentToHeapLogSoftmaxUsesDirectTypedBody() throws Throwable {
        Shape shape = Shape.of(2, 3);
        var input = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(DataType.FLOAT32,
                shape, Optional.of(LayoutDescriptor.of(shape, new long[] {4, 1}, 1, true)), false);
        var output = CpuScatterLoweringTest.desc(DataType.FLOAT32, shape);
        var base = CpuScatterLoweringTest.context(new Operation(SoftmaxKind.LOG_SOFTMAX,
                new SoftmaxAttrs(1)), List.of(0), List.of(input), output);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
        List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        retainEvidence(SoftmaxKind.LOG_SOFTMAX, DataType.FLOAT32, 1, route, bytes);
        float[] source = {99, 1, 2, 3, 99, -3, -2, -1}; float[] result = new float[6];
        generator.defineClassBytes(route.specialization(), route.kernelIr(), bytes).entryPoint().invokeWithArguments(
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
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
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
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) for (int pattern = 0;
                pattern < 4; pattern++) {
            var carriers = List.of((pattern & 1) == 0 ? arrayCarrier(type)
                            : CarrierAccess.MEMORY_SEGMENT,
                    (pattern & 2) == 0 ? arrayCarrier(type) : CarrierAccess.MEMORY_SEGMENT);
            var plan = plan(kind, type, Shape.of(2, 3), 1, carriers);
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
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
            if (type != DataType.FLOAT8_E4M3 && type != DataType.FLOAT8_E5M2)
                assertEquals(1.0, sum, type == DataType.BFLOAT16 ? 0.02
                    : type == DataType.FLOAT16 ? 0.002 : 2e-6,
                    kind + "/" + type + "/" + pattern);
            for (int index = 0; index < values.length; index++) assertEquals(
                    raw(type, expected[index]), raw(type, values[index]),
                    kind + "/" + type + "/" + pattern + "/" + index);
        }
    }

    @Test void nonfiniteSlicesPreserveModelClassesAcrossCarriersAndPartialRanges()
            throws Throwable {
        double n = Double.NaN, positive = Double.POSITIVE_INFINITY,
                negative = Double.NEGATIVE_INFINITY;
        double[] samples = {negative, 0, negative, 1000, negative, -1000,
                positive, 1000, positive, positive, n, positive, negative, negative, positive, n};
        double[] softmax = {0, 1, 0, n, n, n, n, 0, n, n, n, n, n, n, n, n};
        double[] logarithmic = {negative, 0, negative, negative, n, positive,
                n, negative, n, n, n, n, n, n, n, n};
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (SoftmaxKind kind : SoftmaxKind.values()) for (int pattern = 0; pattern < 4; pattern++) {
                var carriers = List.of((pattern & 1) == 0 ? arrayCarrier(type) : CarrierAccess.MEMORY_SEGMENT,
                        (pattern & 2) == 0 ? arrayCarrier(type) : CarrierAccess.MEMORY_SEGMENT);
                var plan = plan(kind, type, Shape.of(8, 2), 1, carriers);
                var route = plan.units().getFirst().portablePlan();
                var generator = new CpuClassFileKernelGenerator();
                var kernel = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr()));
                Object input = represented(type, samples), output = represented(type, new double[16]);
                Object inputCarrier = (pattern & 1) == 0 ? input : segment(input);
                Object outputCarrier = (pattern & 2) == 0 ? output : segment(output);
                long[] geometry = plan.softmaxGeometry().orElseThrow().pack(new long[2]);
                kernel.entryPoint().invokeWithArguments(inputCarrier, outputCarrier, geometry, 0L, 3L);
                kernel.entryPoint().invokeWithArguments(inputCarrier, outputCarrier, geometry, 3L, 8L);
                double[] actual = decoded(type, output);
                double[] expected = kind == SoftmaxKind.SOFTMAX ? softmax : logarithmic;
                for (int i = 0; i < actual.length; i++) {
                    if (Double.isNaN(expected[i])) assertTrue(Double.isNaN(actual[i]), "lane " + i);
                    else assertEquals(raw(type, expected[i]), raw(type, actual[i]), "lane " + i);
                }
            }
        }
    }

    @Test void emptySelectedAxisLeavesOutputPaddingUntouched() throws Throwable {
        for (SoftmaxKind kind : SoftmaxKind.values()) {
            var plan = plan(kind, DataType.FLOAT32, Shape.of(2, 0), 1,
                    List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var kernel = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                    generator.generateClassBytes(route.specialization(), route.kernelIr()));
            float[] output = {-77};
            kernel.entryPoint().invokeWithArguments(new float[0], output,
                    plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 0L);
            assertArrayEquals(new float[]{-77}, output);
        }
    }

    @Test void stableFiniteOverflowUnderflowAndOffsetsMatchCleanJavaOracle() throws Throwable {
        for (SoftmaxKind kind : SoftmaxKind.values()) for (DataType type : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                    { double[][] examples = {{1000, 1000}, {0, -1000}, {1000, 1008}};
                    for (double[] example : examples) {
                        var plan = plan(kind, type, Shape.of(2), 0,
                                List.of(arrayCarrier(type), arrayCarrier(type)));
                        var route = plan.units().getFirst().portablePlan();
                        var generator = new CpuClassFileKernelGenerator();
                        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
                        Object input = represented(type, example), output = represented(type, new double[2]);
                        artifact.entryPoint().invokeWithArguments(input, output,
                                plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 1L);
                        double[] actual = decoded(type, output);
                        double[] expected = CpuSoftmaxReferenceKernel.evaluate(kind, type, example,
                                new long[] {2}, 0, new long[] {1}, 0);
                        for (int i = 0; i < 2; i++) assertEquals(raw(type, expected[i]),
                                raw(type, actual[i]), kind + "/" + type + "/" + java.util.Arrays.toString(example));
                        if (example[0] == 1000 && example[1] == 1000) assertEquals(
                                kind == SoftmaxKind.SOFTMAX ? 0.5 : -Math.log(2), actual[0],
                                type == DataType.BFLOAT16 ? 0.005 : type == DataType.FLOAT16 ? 0.001 : 2e-7);
                        if (example[0] == 0 && example[1] == -1000) {
                            assertEquals(kind == SoftmaxKind.SOFTMAX ? 0 : -1000, actual[1]);
                        }
                    } }
                }
    }

    @Test void logSoftmaxKeepsFiniteUnderflowDistanceAfterRoundedSoftmaxBecomesZero()
            throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32,
                DataType.BFLOAT16, DataType.FLOAT16)) {
                    { double[] results = new double[2];
                    int kindIndex = 0;
                    for (SoftmaxKind kind : SoftmaxKind.values()) {
                        var plan = plan(kind, type, Shape.of(2), 0,
                                List.of(arrayCarrier(type), arrayCarrier(type)));
                        var route = plan.units().getFirst().portablePlan();
                        var generator = new CpuClassFileKernelGenerator();
                        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
                        Object input = represented(type, new double[] {0, -1000});
                        Object output = represented(type, new double[2]);
                        artifact.entryPoint().invokeWithArguments(input, output,
                                plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 1L);
                        results[kindIndex++] = decoded(type, output)[1];
                    }
                    assertEquals(0, results[0], "rounded softmax underflows for " + type);
                    assertEquals(-1000, results[1], "log-softmax retains the finite shift for " + type);
                    assertNotEquals(Math.log(results[0]), results[1],
                            "executed log-softmax must not log the executed rounded softmax for " + type); }
                }
    }

    @Test void admittedFloat32ExtremeWorkingOverflowHasDefinedOutputClasses() throws Throwable {
        for (SoftmaxKind kind : SoftmaxKind.values()) {
            DataType type = DataType.FLOAT32;
            var plan = plan(kind, type, Shape.of(2), 0,
                    List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
            float[] input = {-Float.MAX_VALUE, Float.MAX_VALUE}, output = new float[2];
            artifact.entryPoint().invokeWithArguments(input, output,
                    plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 1L);
            if (kind == SoftmaxKind.SOFTMAX) assertArrayEquals(new float[] {0, 1}, output);
            else assertArrayEquals(new float[] {Float.NEGATIVE_INFINITY, 0}, output);
        }
    }

    @Test void compensatedSumRetainsSmallShiftedContributors() throws Throwable {
        double[] values = new double[2048];
        java.util.Arrays.fill(values, 0, 1024, 0);
        java.util.Arrays.fill(values, 1024, 2048, -10);
        var plan = plan(SoftmaxKind.SOFTMAX, DataType.FLOAT32, Shape.of(2048), 0,
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY));
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        float[] input = (float[]) represented(DataType.FLOAT32, values);
        float[] output = new float[2048];
        artifact.entryPoint().invokeWithArguments(input, output,
                plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, 1L);
        double expected = 1.0 / (1024 * (1.0 + Math.exp(-10)));
        assertEquals(expected, output[0], 2e-10);
        assertEquals(expected * Math.exp(-10), output[1024], 2e-10);
    }


    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            plan(SoftmaxKind kind, DataType type, Shape shape, int axis,
                    List<CarrierAccess> carriers) {
        var base = CpuSoftmaxLoweringTest.context(kind, type, shape, axis);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false, carriers));
        return new CpuPartitionPreparer().analyze(context).plan();
    }


    private static CarrierAccess arrayCarrier(DataType type) {
        return type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY
                : type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2
                    ? CarrierAccess.BYTE_ARRAY : CarrierAccess.SHORT_ARRAY;
    }
    private static Object represented(DataType type, double[] values) {
        if (type == DataType.FLOAT64) return values.clone();
        if (type == DataType.FLOAT32) {
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) result[i] = (float) values[i];
            return result;
        }
        if (type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2) {
            byte[] result = new byte[values.length];
            for (int i = 0; i < values.length; i++) result[i] = type == DataType.FLOAT8_E4M3
                    ? io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.fromDouble(values[i])
                    : io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.fromDouble(values[i]);
            return result;
        }
        short[] result = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            if (type == DataType.FLOAT16) {
                result[i] = Float.floatToFloat16((float) values[i]);
                continue;
            }
            int bits = Float.floatToRawIntBits((float) values[i]);
            result[i] = (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
        }
        return result;
    }
    private static MemorySegment segment(Object values) {
        if (values instanceof double[] value) return MemorySegment.ofArray(value);
        if (values instanceof float[] value) return MemorySegment.ofArray(value);
        if (values instanceof byte[] value) return MemorySegment.ofArray(value);
        return MemorySegment.ofArray((short[]) values);
    }
    private static double[] decoded(DataType type, Object values) {
        if (values instanceof byte[] bytes) {
            double[] result = new double[bytes.length];
            for (int i = 0; i < result.length; i++) result[i] = type == DataType.FLOAT8_E4M3
                    ? io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.toFloat(bytes[i])
                    : io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.toFloat(bytes[i]);
            return result;
        }
        int length = type == DataType.FLOAT64 ? ((double[]) values).length
                : type == DataType.FLOAT32 ? ((float[]) values).length : ((short[]) values).length;
        double[] result = new double[length];
        for (int i = 0; i < length; i++) result[i] = type == DataType.FLOAT64
                ? ((double[]) values)[i] : type == DataType.FLOAT32 ? ((float[]) values)[i]
                : type == DataType.FLOAT16
                    ? Float.float16ToFloat(((short[]) values)[i])
                    : Float.intBitsToFloat(Short.toUnsignedInt(((short[]) values)[i]) << 16);
        return result;
    }
    private static long raw(DataType type, double value) {
        if (type == DataType.FLOAT8_E4M3) return Byte.toUnsignedLong(
                io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.fromDouble(value));
        if (type == DataType.FLOAT8_E5M2) return Byte.toUnsignedLong(
                io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.fromDouble(value));
        return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(value)
                : type == DataType.FLOAT32 ? Integer.toUnsignedLong(
                        Float.floatToRawIntBits((float) value))
                : type == DataType.FLOAT16 ? Short.toUnsignedLong(Float.floatToFloat16((float) value))
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
