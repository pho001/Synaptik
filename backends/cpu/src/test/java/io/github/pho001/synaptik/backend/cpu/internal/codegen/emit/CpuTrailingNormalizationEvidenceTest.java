package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** Frozen CPU 0007F generated/direct evidence probe and isolated-fork driver. */
public final class CpuTrailingNormalizationEvidenceTest {
    private static final int ROWS = 128;
    private static final int WIDTH = 2048;
    private static final int ELEMENTS = ROWS * WIDTH;
    private static final long MIN_BATCH_NANOS = 25_000_000L;
    private static final double EPSILON64 = 1e-5;
    private static final float EPSILON32 = 1e-5f;
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    private static volatile long sink;

    @FunctionalInterface private interface Action { long run() throws Throwable; }
    @FunctionalInterface private interface Verification { void run() throws Throwable; }
    private record Case(String name, Action generated, Action direct, Verification verify,
            long[] generatedRounds, long[] directRounds) { }
    private record Prepared(MethodHandle handle, MemorySegment scratch, long[] geometry,
            byte[] classBytes, byte[] compatibility, String specialization) { }
    private enum ExactType {
        F64(53, -1074, 1023, 52, 1023), F32(24, -149, 127, 23, 127);
        final int precision, minimumUnitExponent, maximumExponent, fractionBits, bias;
        ExactType(int precision, int minimumUnitExponent, int maximumExponent,
                int fractionBits, int bias) {
            this.precision = precision; this.minimumUnitExponent = minimumUnitExponent;
            this.maximumExponent = maximumExponent; this.fractionBits = fractionBits;
            this.bias = bias;
        }
    }

    @Test void frozenInventoryContainsFourteenTargetsAndThreeControls() {
        assertEquals(17, frozenNames().size());
        assertEquals(14, frozenNames().stream().filter(name -> !name.startsWith("CONTROL_")).count());
        assertEquals(List.of("CONTROL_F32_VARIANCE", "CONTROL_F32_SOFTMAX", "CONTROL_F32_ADD"),
                frozenNames().subList(14, 17));
    }

    @Test void mixedSegmentSpecializationsUseFrozenNativeLayouts() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            List<Case> cases = new ArrayList<>();
            addMixedLayer(cases, arena);
            addMixedRms(cases, arena);
            assertEquals(2, cases.size());
        }
    }

    /** Runs one isolated performance fork; the first argument is its deterministic fork id. */
    public static void main(String[] arguments) throws Throwable {
        long fork = arguments.length == 0 ? 0 : Long.parseLong(arguments[0]);
        try (Arena arena = Arena.ofConfined()) {
            List<Case> cases = cases(arena);
            for (Case value : cases) { value.generated.run(); value.direct.run(); value.verify.run(); }
            int[] repetitions = new int[cases.size() * 2];
            for (int index = 0; index < cases.size(); index++) {
                repetitions[index * 2] = repetitions(cases.get(index).generated);
                repetitions[index * 2 + 1] = repetitions(cases.get(index).direct);
            }
            Random random = new Random(0x0007f20260825L ^ fork * 0x9e3779b97f4a7c15L);
            for (int round = -5; round < 9; round++) {
                List<Integer> order = new ArrayList<>();
                for (int index = 0; index < cases.size(); index++) order.add(index);
                Collections.shuffle(order, random);
                for (int index : order) {
                    Case value = cases.get(index); long generated; long direct;
                    if (random.nextBoolean()) {
                        generated = time(value.generated, repetitions[index * 2]);
                        direct = time(value.direct, repetitions[index * 2 + 1]);
                    } else {
                        direct = time(value.direct, repetitions[index * 2 + 1]);
                        generated = time(value.generated, repetitions[index * 2]);
                    }
                    if (round >= 0) {
                        value.generatedRounds[round] = generated / repetitions[index * 2];
                        value.directRounds[round] = direct / repetitions[index * 2 + 1];
                        value.verify.run();
                    }
                }
            }
            int failures = 0;
            for (Case value : cases) {
                long generated = median(value.generatedRounds), direct = median(value.directRounds);
                double ratio = (double) generated / direct;
                if (ratio > 1.15) failures++;
                System.out.printf(Locale.ROOT, "RESULT,%s,%d,%d,%.9f,%s,%s%n", value.name,
                        generated, direct, ratio, Arrays.toString(value.generatedRounds),
                        Arrays.toString(value.directRounds));
            }
            System.out.println("SINK," + sink);
            if (failures != 0) throw new AssertionError("ratio failures " + failures);
        }
    }

    private static List<String> frozenNames() {
        List<String> result = new ArrayList<>();
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            result.add(type + "_LAYER_HEAP"); result.add(type + "_LAYER_AFFINE_HEAP");
            result.add(type + "_RMS_HEAP"); result.add(type + "_RMS_SCALED_HEAP");
        }
        result.add("MIXED_LAYER_AFFINE_GENERAL"); result.add("MIXED_RMS_SCALED_GENERAL");
        result.add("CONTROL_F32_VARIANCE"); result.add("CONTROL_F32_SOFTMAX");
        result.add("CONTROL_F32_ADD"); return List.copyOf(result);
    }

    private static List<Case> cases(Arena arena) throws Throwable {
        List<Case> result = new ArrayList<>();
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            addDense(result, arena, type, true, false);
            addDense(result, arena, type, true, true);
            addDense(result, arena, type, false, false);
            addDense(result, arena, type, false, true);
        }
        addMixedLayer(result, arena); addMixedRms(result, arena);
        addVarianceControl(result, arena); addSoftmaxControl(result); addPointwiseControl(result);
        if (!result.stream().map(Case::name).toList().equals(frozenNames()))
            throw new AssertionError("frozen case order disagrees");
        return result;
    }

    private static void addDense(List<Case> cases, Arena arena, DataType type,
            boolean layer, boolean affine) throws Throwable {
        int inputCount = affine ? layer ? 3 : 2 : 1;
        List<CarrierAccess> carriers = new ArrayList<>();
        for (int index = 0; index <= inputCount; index++) carriers.add(arrayCarrier(type));
        Prepared prepared = normalization(type, layer, affine, carriers, arena,
                (layer ? "layer" : "rms") + (affine ? layer ? "-affine" : "-scaled" : "-plain") + "-"
                        + type.name().toLowerCase(Locale.ROOT));
        Object input = input(type), scale = parameter(type, false), bias = parameter(type, true);
        Object generatedOutput = output(type), directOutput = output(type);
        Action generated = denseGenerated(type, layer, affine, prepared, input, scale, bias,
                generatedOutput);
        Action direct = denseDirect(type, layer, affine, arena, input, scale, bias, directOutput);
        cases.add(new Case(type + "_" + (layer ? "LAYER" : "RMS")
                + (affine ? layer ? "_AFFINE" : "_SCALED" : "") + "_HEAP", generated, direct,
                () -> equal(type, generatedOutput, directOutput), new long[9], new long[9]));
    }

    private static Prepared normalization(DataType type, boolean layer, boolean affine,
            List<CarrierAccess> carriers, Arena arena, String retainedName) throws Exception {
        var base = CpuTrailingNormalizationLoweringTest.context(layer, affine, type,
                Shape.of(ROWS, WIDTH), Shape.of(WIDTH), affine
                        ? layer ? List.of(0, 1, 2) : List.of(0, 1) : List.of(0));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(),
                new CpuPartitionAnalysisInputs(false, carriers));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        retain(retainedName, bytes, route.specialization().compatibilityBytes(),
                route.specialization().toString(), generator.defineClassBytes(
                        route.specialization(), bytes).entryPoint().type().toString());
        long scratchBytes = plan.trailingNormalizationGeometry().orElseThrow().scratchSliceBytes();
        return new Prepared(generator.defineClassBytes(route.specialization(), bytes).entryPoint(),
                scratchBytes == 0 ? null : arena.allocate(scratchBytes, 8),
                plan.trailingNormalizationGeometry().orElseThrow().pack(new long[inputCount(layer,
                        affine) + 1]), bytes, route.specialization().compatibilityBytes(),
                route.specialization().toString());
    }

    private static int inputCount(boolean layer, boolean affine) {
        return affine ? layer ? 3 : 2 : 1;
    }

    private static Action denseGenerated(DataType type, boolean layer, boolean affine,
            Prepared prepared, Object input, Object scale, Object bias, Object output) {
        return () -> {
            validateDense(type, input, scale, bias, output, layer, affine);
            if (type == DataType.FLOAT64) {
                if (layer && affine) prepared.handle.invokeExact((double[]) input, (double[]) scale,
                        (double[]) bias, (double[]) output, prepared.scratch, prepared.geometry,
                        0L, (long) ROWS);
                else if (layer) prepared.handle.invokeExact((double[]) input, (double[]) output,
                        prepared.scratch, prepared.geometry, 0L, (long) ROWS);
                else if (affine) prepared.handle.invokeExact((double[]) input, (double[]) scale,
                        (double[]) output, prepared.geometry, 0L, (long) ROWS);
                else prepared.handle.invokeExact((double[]) input, (double[]) output,
                        prepared.geometry, 0L, (long) ROWS);
            } else if (type == DataType.FLOAT32) {
                if (layer && affine) prepared.handle.invokeExact((float[]) input, (float[]) scale,
                        (float[]) bias, (float[]) output, prepared.scratch, prepared.geometry,
                        0L, (long) ROWS);
                else if (layer) prepared.handle.invokeExact((float[]) input, (float[]) output,
                        prepared.scratch, prepared.geometry, 0L, (long) ROWS);
                else if (affine) prepared.handle.invokeExact((float[]) input, (float[]) scale,
                        (float[]) output, prepared.geometry, 0L, (long) ROWS);
                else prepared.handle.invokeExact((float[]) input, (float[]) output,
                        prepared.geometry, 0L, (long) ROWS);
            } else {
                if (layer && affine) prepared.handle.invokeExact((short[]) input, (short[]) scale,
                        (short[]) bias, (short[]) output, prepared.scratch, prepared.geometry,
                        0L, (long) ROWS);
                else if (layer) prepared.handle.invokeExact((short[]) input, (short[]) output,
                        prepared.scratch, prepared.geometry, 0L, (long) ROWS);
                else if (affine) prepared.handle.invokeExact((short[]) input, (short[]) scale,
                        (short[]) output, prepared.geometry, 0L, (long) ROWS);
                else prepared.handle.invokeExact((short[]) input, (short[]) output,
                        prepared.geometry, 0L, (long) ROWS);
            }
            return checksum(type, output);
        };
    }

    private static Action denseDirect(DataType type, boolean layer, boolean affine, Arena arena,
            Object input, Object scale, Object bias, Object output) {
        MemorySegment exact = layer ? arena.allocate(exactBytes(type), 8) : null;
        return () -> {
            validateDense(type, input, scale, bias, output, layer, affine);
            if (type == DataType.FLOAT64) direct64((double[]) input, affine ? (double[]) scale : null,
                    layer && affine ? (double[]) bias : null, (double[]) output, layer, exact);
            else if (type == DataType.FLOAT32) direct32((float[]) input,
                    affine ? (float[]) scale : null, layer && affine ? (float[]) bias : null,
                    (float[]) output, layer, exact, false);
            else directBfloat((short[]) input, affine ? (short[]) scale : null,
                    layer && affine ? (short[]) bias : null, (short[]) output, layer, exact);
            return checksum(type, output);
        };
    }

    private static void validateDense(DataType type, Object input, Object scale, Object bias,
            Object output, boolean layer, boolean affine) {
        int inputLength = arrayLength(type, input);
        int outputLength = arrayLength(type, output);
        if (inputLength != ELEMENTS || outputLength != ELEMENTS) throw new AssertionError("span");
        if (affine && arrayLength(type, scale) != WIDTH) throw new AssertionError("scale");
        if (layer && affine && arrayLength(type, bias) != WIDTH)
            throw new AssertionError("bias");
        if (type == null || input == output) throw new AssertionError("carrier/overlap");
    }

    private static int arrayLength(DataType type, Object values) {
        return type == DataType.FLOAT64 ? ((double[]) values).length
                : type == DataType.FLOAT32 ? ((float[]) values).length : ((short[]) values).length;
    }

    private static void direct64(double[] input, double[] scale, double[] bias, double[] output,
            boolean layer, MemorySegment exact) {
        for (int row = 0; row < ROWS; row++) {
            int base = row * WIDTH; double root; double mean = 0;
            if (layer) {
                boolean nan = false, infinity = false, constant = true; double first = 0;
                exactReset(exact, ExactType.F64); for (int i = 0; i < WIDTH; i++) {
                    double v = input[base + i]; if (Double.isNaN(v)) nan = true;
                    else if (v == Double.POSITIVE_INFINITY || v == Double.NEGATIVE_INFINITY) infinity = true;
                    if (i == 0) first = v; else if (v != first) constant = false; exactAdd64(exact, v); }
                mean = Double.longBitsToDouble(exactMean(exact, ExactType.F64, WIDTH));
                double deviations = 0, dc = 0, squares = 0, sc = 0;
                for (int i = 0; i < WIDTH; i++) { double d = input[base + i] - mean;
                    double y = d - dc, n = deviations + y; dc = (n - deviations) - y; deviations = n;
                    double square = d * d, q = square - sc, z = squares + q;
                    sc = (z - squares) - q; squares = z; }
                double numerator = squares - deviations * deviations / WIDTH;
                if (numerator < 0) numerator = 0; root = Math.sqrt(numerator / WIDTH + EPSILON64);
                if (nan || infinity) root = Double.NaN; else if (constant) root = Double.POSITIVE_INFINITY;
            } else {
                root = rms64(input, base);
            }
            for (int i = 0; i < WIDTH; i++) { double value = layer
                    ? (input[base + i] - mean) / root : input[base + i] / root;
                if (scale != null) value = value * scale[i]; if (bias != null) value = value + bias[i];
                output[base + i] = value; }
        }
    }

    private static void direct32(float[] input, float[] scale, float[] bias, float[] output,
            boolean layer, MemorySegment exact, boolean ignored) {
        for (int row = 0; row < ROWS; row++) {
            int base = row * WIDTH; double root; float mean = 0;
            if (layer) {
                boolean nan = false, infinity = false, constant = true; float first = 0;
                exactReset(exact, ExactType.F32); for (int i = 0; i < WIDTH; i++) {
                    float v = input[base + i]; if (Double.isNaN(v)) nan = true;
                    else if (v == Float.POSITIVE_INFINITY || v == Float.NEGATIVE_INFINITY) infinity = true;
                    if (i == 0) first = v; else if (v != first) constant = false; exactAdd32(exact, v); }
                mean = Float.intBitsToFloat((int) exactMean(exact, ExactType.F32, WIDTH));
                double deviations = 0, dc = 0, squares = 0, sc = 0;
                for (int i = 0; i < WIDTH; i++) { double d = (double) input[base + i] - mean;
                    double y = d - dc, n = deviations + y; dc = (n - deviations) - y; deviations = n;
                    double square = d * d, q = square - sc, z = squares + q;
                    sc = (z - squares) - q; squares = z; }
                double numerator = squares - deviations * deviations / WIDTH;
                if (numerator < 0) numerator = 0; root = Math.sqrt(numerator / WIDTH + EPSILON32);
                if (nan || infinity) root = Double.NaN; else if (constant) root = Double.POSITIVE_INFINITY;
            } else root = rms32(input, base);
            for (int i = 0; i < WIDTH; i++) { float value = layer
                    ? (float) (((float) (input[base + i] - mean)) / (float) root)
                    : (float) (input[base + i] / (float) root);
                if (scale != null) value = value * scale[i]; if (bias != null) value = value + bias[i];
                output[base + i] = value; }
        }
    }

    private static void directBfloat(short[] input, short[] scale, short[] bias, short[] output,
            boolean layer, MemorySegment exact) {
        for (int row = 0; row < ROWS; row++) {
            int base = row * WIDTH;
            double root; float mean = 0;
            if (layer) {
                boolean nan = false, infinity = false, constant = true; float first = 0;
                exactReset(exact, ExactType.F32); for (int i = 0; i < WIDTH; i++) {
                    float v = decode(input[base + i]); if (Double.isNaN(v)) nan = true;
                    else if (v == Float.POSITIVE_INFINITY || v == Float.NEGATIVE_INFINITY) infinity = true;
                    if (i == 0) first = v; else if (v != first) constant = false; exactAdd32(exact, v); }
                mean = Float.intBitsToFloat((int) exactMean(exact, ExactType.F32, WIDTH));
                double deviations = 0, dc = 0, squares = 0, sc = 0;
                for (int i = 0; i < WIDTH; i++) { double d = (double) decode(input[base + i]) - mean;
                    double y = d - dc, n = deviations + y; dc = (n - deviations) - y; deviations = n;
                    double square = d * d, q = square - sc, z = squares + q;
                    sc = (z - squares) - q; squares = z; }
                double numerator = squares - deviations * deviations / WIDTH;
                if (numerator < 0) numerator = 0; root = Math.sqrt(numerator / WIDTH + bfloatEpsilon());
                if (nan || infinity) root = Double.NaN; else if (constant) root = Double.POSITIVE_INFINITY;
            } else root = rmsBfloat(input, base);
            for (int i = 0; i < WIDTH; i++) { float decoded = decode(input[base + i]);
                float value = layer ? (float) (((float) (decoded - mean)) / (float) root)
                    : (float) (decoded / (float) root);
                if (scale != null) value = value * decode(scale[i]);
                if (bias != null) value = value + decode(bias[i]); output[base + i] = bfloat(value); }
        }
    }

    private static double rms64(double[] input, int base) {
        double scale = 0, squares = 0; boolean nan = false, infinity = false;
        for (int i = 0; i < WIDTH; i++) { double value = input[base + i];
            if (Double.isNaN(value)) nan = true;
            else if (value == Double.POSITIVE_INFINITY || value == Double.NEGATIVE_INFINITY) infinity = true;
            double a = Math.abs(value);
            if (a != 0 && scale < a) { double r = scale / a; squares = 1 + squares * r * r; scale = a; }
            else if (a != 0) { double r = a / scale; squares += r * r; } }
        double root = Math.hypot(scale * Math.sqrt(squares / WIDTH), Math.sqrt(EPSILON64));
        return nan ? Double.NaN : infinity ? Double.POSITIVE_INFINITY : root;
    }
    private static double rms32(float[] input, int base) { double scale = 0, squares = 0;
        boolean nan = false, infinity = false;
        for (int i = 0; i < WIDTH; i++) { double value = input[base + i];
            if (Double.isNaN(value)) nan = true;
            else if (value == Double.POSITIVE_INFINITY || value == Double.NEGATIVE_INFINITY) infinity = true;
            double a = Math.abs(value);
            if (a != 0 && scale < a) { double r = scale / a; squares = 1 + squares * r * r; scale = a; }
            else if (a != 0) { double r = a / scale; squares += r * r; } }
        double root = (float) Math.hypot(scale * Math.sqrt(squares / WIDTH), Math.sqrt(EPSILON32));
        return nan ? Double.NaN : infinity ? Double.POSITIVE_INFINITY : root; }
    private static double rmsBfloat(short[] input, int base) { double scale = 0, squares = 0;
        boolean nan = false, infinity = false;
        for (int i = 0; i < WIDTH; i++) { double value = decode(input[base + i]);
            if (Double.isNaN(value)) nan = true;
            else if (value == Double.POSITIVE_INFINITY || value == Double.NEGATIVE_INFINITY) infinity = true;
            double a = Math.abs(value);
            if (a != 0 && scale < a) { double r = scale / a; squares = 1 + squares * r * r; scale = a; }
            else if (a != 0) { double r = a / scale; squares += r * r; } }
        double root = (float) Math.hypot(scale * Math.sqrt(squares / WIDTH), Math.sqrt(bfloatEpsilon()));
        return nan ? Double.NaN : infinity ? Double.POSITIVE_INFINITY : root; }

    private static void addMixedLayer(List<Case> cases, Arena arena) throws Throwable {
        Shape inputShape = Shape.of(ROWS, WIDTH), normalized = Shape.of(WIDTH);
        int inputStride = WIDTH + 3, outputStride = WIDTH + 5;
        TensorDescriptor inputDescriptor = descriptor(DataType.BFLOAT16, inputShape,
                new long[] {inputStride, 1}, 1);
        TensorDescriptor scaleDescriptor = CpuScatterLoweringTest.desc(DataType.FLOAT32, normalized);
        TensorDescriptor biasDescriptor = descriptor(DataType.FLOAT64, normalized,
                new long[] {1}, 2);
        TensorDescriptor outputDescriptor = descriptor(DataType.FLOAT64, inputShape,
                new long[] {outputStride, 1}, 3);
        var base = CpuScatterLoweringTest.context(new Operation(LayerNormKind.LAYER_NORM,
                new AffineLayerNormAttrs(normalized, ScalarValue.float64(EPSILON64))),
                List.of(0, 1, 2), List.of(inputDescriptor, scaleDescriptor, biasDescriptor),
                outputDescriptor);
        Prepared prepared = mixedPrepared(base, List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.DOUBLE_ARRAY), arena, "mixed-layer-affine-general");
        MemorySegment input = arena.allocate((1L + ROWS * (long) inputStride) * Short.BYTES, 2);
        for (int row = 0; row < ROWS; row++) for (int i = 0; i < WIDTH; i++)
            input.set(SHORT, (1L + row * (long) inputStride + i) * Short.BYTES,
                    bfloat(value(row * WIDTH + i)));
        float[] scale = (float[]) parameter(DataType.FLOAT32, false);
        MemorySegment bias = arena.allocate((WIDTH + 2L) * Double.BYTES, 8);
        double[] biasValues = (double[]) parameter(DataType.FLOAT64, true);
        for (int i = 0; i < WIDTH; i++) bias.set(DOUBLE, (i + 2L) * Double.BYTES, biasValues[i]);
        double[] generated = new double[3 + ROWS * outputStride], direct = new double[generated.length];
        MemorySegment directScratch = arena.allocate(exactBytes(DataType.FLOAT64), 8);
        Action generatedAction = () -> { validateMixed(input, scale, bias, generated);
            prepared.handle.invokeExact(input, scale, bias, generated, prepared.scratch,
                    prepared.geometry, 0L, (long) ROWS); return checksumMixed64(generated, outputStride, 3); };
        Action directAction = () -> { validateMixed(input, scale, bias, direct);
            for (int row = 0; row < ROWS; row++) { long inputBase = 1L + row * inputStride;
                int outputBase = 3 + row * outputStride; boolean nan = false, infinity = false,
                        constant = true; double first = 0; exactReset(directScratch, ExactType.F64);
                for (int i = 0; i < WIDTH; i++) { double v = decode(input.get(SHORT,
                        (inputBase + i) * Short.BYTES)); if (Double.isNaN(v)) nan = true;
                    else if (v == Double.POSITIVE_INFINITY || v == Double.NEGATIVE_INFINITY) infinity = true;
                    if (i == 0) first = v; else if (v != first) constant = false; exactAdd64(directScratch, v); }
                double mean = Double.longBitsToDouble(exactMean(directScratch, ExactType.F64, WIDTH));
                double deviations = 0, dc = 0, squares = 0, sc = 0;
                for (int i = 0; i < WIDTH; i++) { double d = decode(input.get(SHORT,
                        (inputBase + i) * Short.BYTES)) - mean; double y = d - dc, n = deviations + y;
                    dc = (n - deviations) - y; deviations = n; double square = d * d;
                    double q = square - sc, z = squares + q; sc = (z - squares) - q; squares = z; }
                double numerator = squares - deviations * deviations / WIDTH;
                if (numerator < 0) numerator = 0; double root = Math.sqrt(numerator / WIDTH + EPSILON64);
                if (nan || infinity) root = Double.NaN; else if (constant) root = Double.POSITIVE_INFINITY;
                for (int i = 0; i < WIDTH; i++) { double standardized =
                        (decode(input.get(SHORT, (inputBase + i) * Short.BYTES)) - mean) / root;
                    direct[outputBase + i] = standardized * (double) scale[i]
                            + bias.get(DOUBLE, (i + 2L) * Double.BYTES); } }
            return checksumMixed64(direct, outputStride, 3); };
        cases.add(new Case("MIXED_LAYER_AFFINE_GENERAL", generatedAction, directAction,
                () -> equalMixed64(generated, direct, outputStride, 3), new long[9], new long[9]));
    }
    private static void addMixedRms(List<Case> cases, Arena arena) throws Throwable {
        Shape inputShape = Shape.of(ROWS, WIDTH), normalized = Shape.of(WIDTH);
        int inputStride = WIDTH + 3, outputStride = WIDTH + 5;
        TensorDescriptor inputDescriptor = descriptor(DataType.BFLOAT16, inputShape,
                new long[] {inputStride, 1}, 1);
        TensorDescriptor scaleDescriptor = CpuScatterLoweringTest.desc(DataType.FLOAT32, normalized);
        TensorDescriptor outputDescriptor = descriptor(DataType.FLOAT32, inputShape,
                new long[] {outputStride, 1}, 3);
        var base = CpuScatterLoweringTest.context(new Operation(RmsNormKind.RMS_NORM,
                new RmsNormAttrs(normalized, ScalarValue.float32(EPSILON32))), List.of(0, 1),
                List.of(inputDescriptor, scaleDescriptor), outputDescriptor);
        Prepared prepared = mixedPrepared(base, List.of(CarrierAccess.MEMORY_SEGMENT,
                CarrierAccess.FLOAT_ARRAY, CarrierAccess.MEMORY_SEGMENT), arena,
                "mixed-rms-scaled-general");
        MemorySegment input = arena.allocate((1L + ROWS * (long) inputStride) * Short.BYTES, 2);
        for (int row = 0; row < ROWS; row++) for (int i = 0; i < WIDTH; i++)
            input.set(SHORT, (1L + row * (long) inputStride + i) * Short.BYTES,
                    bfloat(value(row * WIDTH + i)));
        float[] scale = (float[]) parameter(DataType.FLOAT32, false);
        MemorySegment generated = arena.allocate((3L + ROWS * (long) outputStride) * Float.BYTES, 4);
        MemorySegment direct = arena.allocate(generated.byteSize(), 4);
        Action generatedAction = () -> { validateMixedRms(input, scale, generated);
            prepared.handle.invokeExact(input, scale, generated, prepared.geometry, 0L, (long) ROWS);
            return checksumMixed32(generated, outputStride, 3); };
        Action directAction = () -> { validateMixedRms(input, scale, direct);
            for (int row = 0; row < ROWS; row++) { long inputBase = 1L + row * inputStride;
                long outputBase = 3L + row * outputStride; double stateScale = 0, squares = 0;
                boolean nan = false, infinity = false; for (int i = 0; i < WIDTH; i++) {
                    double v = decode(input.get(SHORT, (inputBase + i) * Short.BYTES));
                    if (Double.isNaN(v)) nan = true; else if (v == Double.POSITIVE_INFINITY
                            || v == Double.NEGATIVE_INFINITY) infinity = true; double a = Math.abs(v);
                    if (a != 0 && stateScale < a) { double r = stateScale / a;
                        squares = 1 + squares * r * r; stateScale = a; }
                    else if (a != 0) { double r = a / stateScale; squares += r * r; } }
                double root = (float) Math.hypot(stateScale * Math.sqrt(squares / WIDTH),
                        Math.sqrt(EPSILON32)); if (nan) root = Double.NaN;
                else if (infinity) root = Double.POSITIVE_INFINITY;
                for (int i = 0; i < WIDTH; i++) { float v = (float) (decode(input.get(SHORT,
                        (inputBase + i) * Short.BYTES)) / (float) root);
                    direct.set(FLOAT, (outputBase + i) * Float.BYTES, v * scale[i]); } }
            return checksumMixed32(direct, outputStride, 3); };
        cases.add(new Case("MIXED_RMS_SCALED_GENERAL", generatedAction, directAction,
                () -> equalMixed32(generated, direct, outputStride, 3), new long[9], new long[9]));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, long[] strides, long offset) {
        return new TensorDescriptor(type, shape,
                java.util.Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false);
    }

    private static Prepared mixedPrepared(PrepareContext<CpuPartitionAnalysisInputs> base,
            List<CarrierAccess> carriers, Arena arena, String name) throws Exception {
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, carriers));
        var plan = new CpuPartitionPreparer().analyze(context).plan(); var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator(); byte[] bytes = generator.generateClassBytes(
                route.specialization(), route.kernelIr()); var artifact = generator.defineClassBytes(route.specialization(), bytes);
        retain(name, bytes, route.specialization().compatibilityBytes(), route.specialization().toString(),
                artifact.entryPoint().type().toString()); long scratch = plan.trailingNormalizationGeometry().orElseThrow().scratchSliceBytes();
        return new Prepared(artifact.entryPoint(), scratch == 0 ? null : arena.allocate(scratch, 8),
                plan.trailingNormalizationGeometry().orElseThrow().pack(new long[carriers.size()]),
                bytes, route.specialization().compatibilityBytes(), route.specialization().toString());
    }

    private static void validateMixed(MemorySegment input, float[] scale, MemorySegment bias,
            double[] output) { if (input.byteSize() < ELEMENTS * 2L || scale.length != WIDTH
            || bias.byteSize() < WIDTH * 8L || output.length < ELEMENTS || !input.scope().isAlive()
            || !bias.scope().isAlive()) throw new AssertionError("mixed layer validation"); }
    private static void validateMixedRms(MemorySegment input, float[] scale, MemorySegment output) {
        if (input.byteSize() < ELEMENTS * 2L || scale.length != WIDTH || output.byteSize()
                < ELEMENTS * 4L || !input.scope().isAlive() || !output.scope().isAlive())
            throw new AssertionError("mixed RMS validation"); }
    private static long checksumMixed64(double[] values, int stride, int offset) { long result = 0;
        for (int row = 0; row < ROWS; row++) for (int i = 0; i < WIDTH; i++) result =
                Long.rotateLeft(result, 1) ^ Double.doubleToRawLongBits(values[offset + row * stride + i]);
        return result; }
    private static long checksumMixed32(MemorySegment values, int stride, int offset) { long result = 0;
        for (int row = 0; row < ROWS; row++) for (int i = 0; i < WIDTH; i++) result = Long.rotateLeft(result, 1)
                ^ Integer.toUnsignedLong(Float.floatToRawIntBits(values.get(FLOAT,
                        (offset + row * (long) stride + i) * Float.BYTES))); return result; }
    private static void equalMixed64(double[] left, double[] right, int stride, int offset) {
        for (int row = 0; row < ROWS; row++) for (int i = 0; i < WIDTH; i++) { int index = offset + row * stride + i;
            if (Double.doubleToRawLongBits(left[index]) != Double.doubleToRawLongBits(right[index]))
                throw new AssertionError("mixed layer/" + row + "/" + i); } }
    private static void equalMixed32(MemorySegment left, MemorySegment right, int stride, int offset) {
        for (int row = 0; row < ROWS; row++) for (int i = 0; i < WIDTH; i++) { long address =
                (offset + row * (long) stride + i) * Float.BYTES; if (Float.floatToRawIntBits(left.get(FLOAT, address))
                != Float.floatToRawIntBits(right.get(FLOAT, address))) throw new AssertionError("mixed RMS/" + row + "/" + i); } }

    private static void addVarianceControl(List<Case> cases, Arena arena) throws Throwable {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.VARIANCE,
                DataType.FLOAT32, Shape.of(ROWS, WIDTH),
                new StatisticalReductionAttrs(List.of(1), false, 1), Shape.of(ROWS));
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan(); var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator(); byte[] bytes = generator.generateClassBytes(
                route.specialization(), route.kernelIr()); var artifact = generator.defineClassBytes(
                        route.specialization(), bytes); MethodHandle handle = artifact.entryPoint();
        retain("control-f32-variance", bytes, route.specialization().compatibilityBytes(),
                route.specialization().toString(), handle.type().toString());
        var geometry = plan.advancedReductionGeometry().orElseThrow();
        MemorySegment generatedScratch = arena.allocate(geometry.scratchSliceBytes(), 8);
        MemorySegment directScratch = arena.allocate(exactBytes(DataType.FLOAT32), 8);
        float[] input = (float[]) input(DataType.FLOAT32), generated = new float[ROWS], direct = new float[ROWS];
        Action generatedAction = () -> { validateControl(input, generated); handle.invokeExact(input,
                generated, generatedScratch, geometry.pack(new long[2]), 0L, (long) ROWS);
            return checksum(DataType.FLOAT32, generated); };
        Action directAction = () -> { validateControl(input, direct); for (int row = 0; row < ROWS; row++) {
            int baseIndex = row * WIDTH; exactReset(directScratch, ExactType.F32);
            for (int i = 0; i < WIDTH; i++) exactAdd32(directScratch, input[baseIndex + i]);
            float mean = Float.intBitsToFloat((int) exactMean(directScratch, ExactType.F32, WIDTH));
            double ds = 0, dc = 0, ss = 0, sc = 0; for (int i = 0; i < WIDTH; i++) {
                double d = (double) input[baseIndex + i] - mean, y = d - dc, n = ds + y;
                dc = (n - ds) - y; ds = n; double square = d * d, q = square - sc, z = ss + q;
                sc = (z - ss) - q; ss = z; } double numerator = ss - ds * ds / WIDTH;
            if (numerator < 0) numerator = 0; direct[row] = (float) (numerator / (WIDTH - 1)); }
            return checksum(DataType.FLOAT32, direct); };
        cases.add(new Case("CONTROL_F32_VARIANCE", generatedAction, directAction,
                () -> equal(DataType.FLOAT32, generated, direct), new long[9], new long[9]));
    }

    private static void addSoftmaxControl(List<Case> cases) throws Throwable {
        var base = CpuSoftmaxLoweringTest.context(
                io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind.SOFTMAX,
                DataType.FLOAT32, Shape.of(ROWS, WIDTH), 1);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY)));
        var plan = new CpuPartitionPreparer().analyze(context).plan(); var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator(); byte[] bytes = generator.generateClassBytes(
                route.specialization(), route.kernelIr()); MethodHandle handle = generator.defineClassBytes(
                        route.specialization(), bytes).entryPoint();
        retain("control-f32-softmax", bytes, route.specialization().compatibilityBytes(),
                route.specialization().toString(), handle.type().toString());
        float[] input = (float[]) input(DataType.FLOAT32), generated = new float[ELEMENTS], direct = new float[ELEMENTS];
        Action generatedAction = () -> { validateControl(input, generated); handle.invokeExact(input,
                generated, plan.softmaxGeometry().orElseThrow().pack(new long[2]), 0L, (long) ROWS);
            return checksum(DataType.FLOAT32, generated); };
        Action directAction = () -> { validateControl(input, direct); for (int row = 0; row < ROWS; row++) {
            int baseIndex = row * WIDTH; double maximum = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < WIDTH; i++) maximum = Math.max(maximum, input[baseIndex + i]);
            double sum = 0, compensation = 0; for (int i = 0; i < WIDTH; i++) {
                double y = Math.exp((double) input[baseIndex + i] - maximum) - compensation;
                double next = sum + y; compensation = (next - sum) - y; sum = next; }
            for (int i = 0; i < WIDTH; i++) direct[baseIndex + i] = (float)
                    (Math.exp((double) input[baseIndex + i] - maximum) / sum); }
            return checksum(DataType.FLOAT32, direct); };
        cases.add(new Case("CONTROL_F32_SOFTMAX", generatedAction, directAction,
                () -> equal(DataType.FLOAT32, generated, direct), new long[9], new long[9]));
    }

    private static void addPointwiseControl(List<Case> cases) throws Throwable {
        var denseRead = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var denseWrite = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var ir = new CpuKernelIr(List.of(new CpuKernelIr.Value(0, DataType.FLOAT32,
                CpuKernelIr.Value.Kind.INPUT, denseRead), new CpuKernelIr.Value(1, DataType.FLOAT32,
                CpuKernelIr.Value.Kind.INPUT, denseRead), new CpuKernelIr.Value(2, DataType.FLOAT32,
                CpuKernelIr.Value.Kind.OUTPUT, denseWrite)), List.of(new CpuKernelIr.Instruction(
                        CpuPointwiseOpcode.ADD, List.of(0, 1), 2)), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(2, 0)));
        var specialization = new CpuKernelSpecialization(
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(
                        ir.structuralKey()), CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32),
                List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY),
                0, -1); var generator = new CpuClassFileKernelGenerator(); byte[] bytes =
                generator.generateClassBytes(specialization, ir); MethodHandle handle =
                generator.defineClassBytes(specialization, bytes).entryPoint();
        retain("control-f32-add", bytes, specialization.compatibilityBytes(),
                specialization.toString(), handle.type().toString());
        float[] left = (float[]) input(DataType.FLOAT32), right = (float[]) parameterBody(),
                generated = new float[ELEMENTS], direct = new float[ELEMENTS];
        long[] geometry = {ELEMENTS, 0, 0, 0, 0, 1, 1, 1, ELEMENTS, ELEMENTS, ELEMENTS};
        Action generatedAction = () -> { if (left.length != right.length || generated == left || generated == right)
                throw new AssertionError("control validation"); handle.invokeExact(left, right, generated,
                        geometry, 0L, (long) ELEMENTS); return checksum(DataType.FLOAT32, generated); };
        Action directAction = () -> { if (left.length != right.length || direct == left || direct == right)
                throw new AssertionError("control validation"); for (int i = 0; i < ELEMENTS; i++)
                    direct[i] = left[i] + right[i]; return checksum(DataType.FLOAT32, direct); };
        cases.add(new Case("CONTROL_F32_ADD", generatedAction, directAction,
                () -> equal(DataType.FLOAT32, generated, direct), new long[9], new long[9]));
    }

    private static Object input(DataType type) {
        if (type == DataType.FLOAT64) { double[] result = new double[ELEMENTS];
            for (int i = 0; i < result.length; i++) result[i] = value(i); return result; }
        if (type == DataType.FLOAT32) { float[] result = new float[ELEMENTS];
            for (int i = 0; i < result.length; i++) result[i] = value(i); return result; }
        short[] result = new short[ELEMENTS]; for (int i = 0; i < result.length; i++)
            result[i] = bfloat(value(i)); return result;
    }
    private static Object parameter(DataType type, boolean bias) { double[] source = new double[WIDTH];
        for (int i = 0; i < WIDTH; i++) source[i] = bias ? (i % 11 - 5) * .015625
                : .75 + (i % 13) * .03125; if (type == DataType.FLOAT64) return source;
        if (type == DataType.FLOAT32) { float[] result = new float[WIDTH]; for (int i = 0; i < WIDTH; i++)
            result[i] = (float) source[i]; return result; } short[] result = new short[WIDTH];
        for (int i = 0; i < WIDTH; i++) result[i] = bfloat((float) source[i]); return result; }
    private static Object output(DataType type) { return type == DataType.FLOAT64
            ? new double[ELEMENTS] : type == DataType.FLOAT32 ? new float[ELEMENTS] : new short[ELEMENTS]; }
    private static Object parameterBody() { float[] result = new float[ELEMENTS];
        for (int i = 0; i < result.length; i++) result[i] = (i % 29 - 14) * .0625f; return result; }
    private static float value(int index) { return (index % 37 - 18) * .03125f
            + (index / WIDTH % 7 - 3) * .0078125f; }
    private static CarrierAccess arrayCarrier(DataType type) { return type == DataType.FLOAT64
            ? CarrierAccess.DOUBLE_ARRAY : type == DataType.FLOAT32
                    ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY; }
    private static short bfloat(float value) { int bits = Float.floatToRawIntBits(value);
        if ((bits & 0x7fffffff) > 0x7f800000) return (short) 0x7fc0;
        return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16); }
    private static float decode(short value) { return Float.intBitsToFloat(Short.toUnsignedInt(value) << 16); }
    private static float bfloatEpsilon() { return Float.intBitsToFloat(0x3728 << 16); }

    private static void validateControl(float[] input, float[] output) {
        if (input.length != ELEMENTS || output.length != ELEMENTS && output.length != ROWS
                || input == output) throw new AssertionError("control validation");
    }
    private static long checksum(DataType type, Object values) { long result = 0;
        if (type == DataType.FLOAT64) for (double value : (double[]) values)
            result = Long.rotateLeft(result, 1) ^ Double.doubleToRawLongBits(value);
        else if (type == DataType.FLOAT32) for (float value : (float[]) values)
            result = Long.rotateLeft(result, 1) ^ Integer.toUnsignedLong(Float.floatToRawIntBits(value));
        else for (short value : (short[]) values) result = Long.rotateLeft(result, 1)
                ^ Short.toUnsignedLong(value); return result; }
    private static void equal(DataType type, Object generated, Object direct) {
        if (type == DataType.FLOAT64) { double[] left = (double[]) generated, right = (double[]) direct;
            for (int i = 0; i < left.length; i++) if (Double.doubleToRawLongBits(left[i])
                    != Double.doubleToRawLongBits(right[i])) throw new AssertionError(type + "/" + i); }
        else if (type == DataType.FLOAT32) { float[] left = (float[]) generated, right = (float[]) direct;
            for (int i = 0; i < left.length; i++) if (Float.floatToRawIntBits(left[i])
                    != Float.floatToRawIntBits(right[i])) throw new AssertionError(type + "/" + i
                            + "/" + left[i] + "/" + right[i]); }
        else if (!Arrays.equals((short[]) generated, (short[]) direct)) throw new AssertionError(type); }

    private static long time(Action action, int repetitions) throws Throwable { long start = System.nanoTime(), value = 0;
        for (int i = 0; i < repetitions; i++) value ^= action.run(); sink ^= value; return System.nanoTime() - start; }
    private static int repetitions(Action action) throws Throwable { int result = 1;
        while (time(action, result) < MIN_BATCH_NANOS) result = Math.multiplyExact(result, 2); return result; }
    private static long median(long[] values) { long[] copy = values.clone(); Arrays.sort(copy); return copy[4]; }

    private static long exactBytes(DataType type) { return 8L + 8L * exactLimbs(type == DataType.FLOAT64
            ? ExactType.F64 : ExactType.F32); }
    private static int exactLimbs(ExactType type) { int countBits = 64 - Long.numberOfLeadingZeros(WIDTH - 1);
        return Math.toIntExact(((long) type.maximumExponent + 1 - type.minimumUnitExponent
                + countBits + 1 + 63) / 64); }
    private static long get(MemorySegment state, long offset) { return state.get(ValueLayout.JAVA_LONG, offset); }
    private static void set(MemorySegment state, long offset, long value) { state.set(ValueLayout.JAVA_LONG, offset, value); }
    private static void exactReset(MemorySegment state, ExactType type) { set(state, 0, 0);
        for (int i = 0; i < exactLimbs(type); i++) set(state, 8L + 8L * i, 0); }
    private static void exactAdd64(MemorySegment state, double value) { long bits = Double.doubleToRawLongBits(value);
        long fraction = bits & 0x000f_ffff_ffff_ffffL, exponent = bits >>> 52 & 0x7ff;
        if (exponent == 0x7ff || exponent == 0 && fraction == 0) return; long significand; int shift;
        if (exponent == 0) { significand = fraction; shift = 0; }
        else { significand = (1L << 52) | fraction; shift = (int) exponent - 1023 - 52 + 1074; }
        exactAdd(state, exactLimbs(ExactType.F64), significand, shift, bits < 0); }
    private static void exactAdd32(MemorySegment state, float value) { int bits = Float.floatToRawIntBits(value);
        long fraction = bits & 0x007f_ffffL, exponent = bits >>> 23 & 0xff;
        if (exponent == 0xff || exponent == 0 && fraction == 0) return; long significand; int shift;
        if (exponent == 0) { significand = fraction; shift = 0; }
        else { significand = (1L << 23) | fraction; shift = (int) exponent - 127 - 23 + 149; }
        exactAdd(state, exactLimbs(ExactType.F32), significand, shift, bits < 0); }
    private static void exactAdd(MemorySegment state, int limbs, long significand, int shift,
            boolean negative) { int word = shift >>> 6, bit = shift & 63; long carry = negative ? 1 : 0;
        for (int i = 0; i < limbs; i++) { long part = i == word ? significand << bit
                : bit != 0 && i == word + 1 ? significand >>> (64 - bit) : 0;
            if (negative) part = ~part; long withCarry = part + carry;
            long first = Long.compareUnsigned(withCarry, part) < 0 ? 1 : 0;
            long old = get(state, 8L + 8L * i), sum = old + withCarry;
            long second = Long.compareUnsigned(sum, old) < 0 ? 1 : 0; carry = first | second;
            set(state, 8L + 8L * i, sum); } }
    private static long exactMean(MemorySegment state, ExactType type, long divisor) {
        int limbs = exactLimbs(type); boolean negative = get(state, 8L * limbs) < 0;
        if (negative) { long carry = 1; for (int i = 0; i < limbs; i++) { long old = ~get(state, 8L + 8L * i);
            long next = old + carry; set(state, 8L + 8L * i, next);
            carry = Long.compareUnsigned(next, old) < 0 ? 1 : 0; } }
        long remainder = 0; for (int word = limbs - 1; word >= 0; word--) { long old = get(state, 8L + 8L * word), quotient = 0;
            for (int bit = 63; bit >= 0; bit--) { long next = (remainder << 1) | (old >>> bit & 1);
                if (Long.compareUnsigned(next, divisor) >= 0) { next -= divisor; quotient |= 1L << bit; }
                remainder = next; } set(state, 8L + 8L * word, quotient); }
        int top = limbs - 1; while (top > 0 && get(state, 8L + 8L * top) == 0) top--;
        long topWord = get(state, 8L + 8L * top); if (topWord == 0 && remainder == 0) return negative ? signMask(type) : 0;
        int bitLength = topWord == 0 ? 0 : top * 64 + 64 - Long.numberOfLeadingZeros(topWord);
        long unbiased = (long) type.minimumUnitExponent + bitLength - 1;
        int shift = unbiased >= 1 - type.bias ? bitLength - type.precision : 0;
        long rounded = extract(state, limbs, Math.max(0, shift)); int guard; boolean sticky;
        if (shift > 0) { guard = bit(state, shift - 1); sticky = shift > 1 && anyBelow(state, shift - 1) || remainder != 0; }
        else { int comparison = Long.compareUnsigned(remainder << 1, divisor);
            guard = comparison >= 0 ? 1 : 0; sticky = comparison > 0; }
        if (guard != 0 && (sticky || (rounded & 1) != 0)) rounded++;
        if (unbiased >= 1 - type.bias && rounded == (1L << type.precision)) { rounded >>>= 1; unbiased++; }
        if (unbiased > type.maximumExponent) return signMask(type) | exponentMask(type) << type.fractionBits;
        long result = negative ? signMask(type) : 0; if (unbiased >= 1 - type.bias)
            result |= (unbiased + type.bias) << type.fractionBits | rounded & fractionMask(type);
        else if (rounded >= (1L << type.fractionBits)) result |= 1L << type.fractionBits;
        else result |= rounded; return result; }
    private static long extract(MemorySegment state, int limbs, int shift) { int word = shift >>> 6, bit = shift & 63;
        if (word >= limbs) return 0; long result = get(state, 8L + 8L * word) >>> bit;
        if (bit != 0 && word + 1 < limbs) result |= get(state, 8L + 8L * (word + 1)) << (64 - bit); return result; }
    private static int bit(MemorySegment state, int position) { return (int) (get(state,
            8L + 8L * (position >>> 6)) >>> (position & 63) & 1); }
    private static boolean anyBelow(MemorySegment state, int count) { int full = count >>> 6, bits = count & 63;
        for (int i = 0; i < full; i++) if (get(state, 8L + 8L * i) != 0) return true;
        return bits != 0 && (get(state, 8L + 8L * full) & ((1L << bits) - 1)) != 0; }
    private static long signMask(ExactType type) { return type == ExactType.F64 ? Long.MIN_VALUE : 1L << 31; }
    private static long exponentMask(ExactType type) { return type == ExactType.F64 ? 0x7ffL : 0xffL; }
    private static long fractionMask(ExactType type) { return (1L << type.fractionBits) - 1; }

    private static void retain(String name, byte[] bytes, byte[] compatibility,
            String specialization, String descriptor) throws Exception {
        var model = ClassFile.of().parse(bytes); StringBuilder members = new StringBuilder();
        java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast)
                .forEach(member -> members.append(member.owner().asInternalName()).append('.')
                        .append(member.name().stringValue()).append(member.type().stringValue()).append('\n'));
        if (members.indexOf("java/nio/ByteOrder.nativeOrder") >= 0
                || members.indexOf("java/lang/foreign/ValueLayout.withOrder") >= 0)
            throw new AssertionError(name + " rebuilds native layouts at invocation time\n" + members);
        String root = System.getProperty("synaptik.cpu.normalization.evidence");
        if (root == null) root = System.getenv("SYNAPTIK_CPU_NORMALIZATION_EVIDENCE");
        if (root == null) return; Path generated = Path.of(root, "generated"); Files.createDirectories(generated);
        Files.write(generated.resolve(name + ".class"), bytes);
        Files.write(generated.resolve(name + ".compatibility"), compatibility);
        Files.writeString(generated.resolve(name + ".specialization"), specialization + "\n");
        Files.writeString(generated.resolve(name + ".descriptor"), descriptor + "\n");
        Files.writeString(generated.resolve(name + ".members"), members.toString());
    }
}
