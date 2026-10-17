package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Opt-in, single-process diagnostic, not a qualification gate or historical campaign.
 * Eighteen bounded cases compare generated entries with independently authored specialized Java.
 * Preparation/allocation is excluded; 3000 warmup pairs precede seven alternating-order samples,
 * each containing 64 full-array invocations of 4097 elements. No timing threshold changes policy.
 */
final class CpuActivationDiagnosticTest {
    private static final int COUNT = 4097, REPEATS = 64;
    private static volatile long sink;
    @FunctionalInterface private interface Action { void run() throws Throwable; }

    @Test void boundedGeneratedVersusDirect() throws Throwable {
        assumeTrue("true".equals(System.getenv("SYNAPTIK_CPU_0022_DIAGNOSTIC")));
        Path root = Path.of("build/cpu-0022-diagnostic");
        Files.createDirectories(root);
        StringBuilder report = new StringBuilder("case,generated_ns_per_element,direct_ns_per_element,median_ratio,min_ratio,max_ratio\n");
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32))
            for (boolean vector : List.of(false, true))
                for (boolean gelu : List.of(false, true))
                    numeric(type, vector, gelu, report);
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            for (CpuPointwiseOpcode kind : CpuActivationNumericsTest.COMPOSITES) low(type, kind, report);
        Files.writeString(root.resolve("ratios.csv"), report);
        Files.writeString(root.resolve("environment.txt"), "java=" + System.getProperty("java.runtime.version")
                + "\nos=" + System.getProperty("os.name") + " " + System.getProperty("os.arch")
                + "\nfloatLanes=" + FloatVector.SPECIES_PREFERRED.length()
                + "\ndoubleLanes=" + DoubleVector.SPECIES_PREFERRED.length()
                + "\nprocesses=1 warmupPairs=3000 samples=7 repeats=64 count=4097\n"
                + "diagnostic only; no route decision, universal bound, or five-fork qualification\n");
        System.out.print(report);
    }

    private static void numeric(DataType type, boolean vector, boolean gelu, StringBuilder report) throws Throwable {
        var kind = gelu ? CpuPointwiseOpcode.GELU_EXACT : CpuPointwiseOpcode.ERF;
        var fixture = CpuActivationNumericsTest.fixture(type, kind,
                vector ? ExecutionStrategy.VECTOR : ExecutionStrategy.SCALAR, false, false, false);
        MethodHandle entry = fixture.kernel().entryPoint();
        long[] geometry = CpuActivationNumericsTest.geometry(COUNT, 0, 0, 0, 1, COUNT);
        Action generated, direct;
        if (type == DataType.FLOAT64) {
            double[] input = new double[COUNT], actual = new double[COUNT], expected = new double[COUNT];
            for (int i = 0; i < COUNT; i++) input[i] = i % 32 == 0 ? (i % 64 == 0 ? -1e100 : 1e100)
                    : (i % 225 - 112) / 16d;
            generated = () -> { entry.invokeExact(input, actual, geometry, 0L, (long) COUNT); sink = Double.doubleToRawLongBits(actual[COUNT / 2]); };
            direct = () -> { directDouble(input, expected, vector, gelu); sink = Double.doubleToRawLongBits(expected[COUNT / 2]); };
            generated.run(); direct.run();
            assertArrayEquals(expected, actual, 3e-16);
        } else {
            float[] input = new float[COUNT], actual = new float[COUNT], expected = new float[COUNT];
            for (int i = 0; i < COUNT; i++) input[i] = i % 32 == 0 ? (i % 64 == 0 ? -1e10f : 1e10f)
                    : (i % 225 - 112) / 16f;
            generated = () -> { entry.invokeExact(input, actual, geometry, 0L, (long) COUNT); sink = Float.floatToRawIntBits(actual[COUNT / 2]); };
            direct = () -> { directFloat(input, expected, vector, gelu); sink = Float.floatToRawIntBits(expected[COUNT / 2]); };
            generated.run(); direct.run();
            assertArrayEquals(expected, actual, 2e-7f);
        }
        measure(type + "-" + (vector ? "vector-" : "scalar-") + kind, generated, direct, report);
    }

    private static void low(DataType type, CpuPointwiseOpcode kind, StringBuilder report) throws Throwable {
        MethodHandle entry = CpuActivationNumericsTest.fixture(type, kind, ExecutionStrategy.SCALAR,
                false, false, false).kernel().entryPoint();
        long[] geometry = CpuActivationNumericsTest.geometry(COUNT, 0, 0, 0, 1, COUNT);
        short[] input = new short[COUNT], actual = new short[COUNT], expected = new short[COUNT];
        for (int i = 0; i < COUNT; i++) input[i] = CpuActivationDirectOracle.encode(type,
                kind == CpuPointwiseOpcode.RSQRT ? (i % 224 + 1) / 16f : (i % 225 - 112) / 16f);
        Action generated = () -> { entry.invokeExact(input, actual, geometry, 0L, (long) COUNT); sink = actual[COUNT / 2]; };
        Action direct = switch (type) {
            case BFLOAT16 -> switch (kind) {
                case RSQRT -> () -> { directBFRSQRT(input, expected); sink = expected[COUNT / 2]; };
                case SIGMOID -> () -> { directBFSIGMOID(input, expected); sink = expected[COUNT / 2]; };
                case GELU_EXACT -> () -> { directBFGELU_EXACT(input, expected); sink = expected[COUNT / 2]; };
                case GELU_TANH_APPROXIMATION -> () -> { directBFGELU_TANH_APPROXIMATION(input, expected); sink = expected[COUNT / 2]; };
                case SILU -> () -> { directBFSILU(input, expected); sink = expected[COUNT / 2]; };
                default -> throw new AssertionError(kind);
            };
            case FLOAT16 -> switch (kind) {
                case RSQRT -> () -> { directFRSQRT(input, expected); sink = expected[COUNT / 2]; };
                case SIGMOID -> () -> { directFSIGMOID(input, expected); sink = expected[COUNT / 2]; };
                case GELU_EXACT -> () -> { directFGELU_EXACT(input, expected); sink = expected[COUNT / 2]; };
                case GELU_TANH_APPROXIMATION -> () -> { directFGELU_TANH_APPROXIMATION(input, expected); sink = expected[COUNT / 2]; };
                case SILU -> () -> { directFSILU(input, expected); sink = expected[COUNT / 2]; };
                default -> throw new AssertionError(kind);
            };
            default -> throw new AssertionError(type);
        };
        generated.run(); direct.run(); assertArrayEquals(expected, actual);
        measure(type + "-" + kind, generated, direct, report);
    }

    private static void measure(String name, Action generated, Action direct, StringBuilder report) throws Throwable {
        for (int i = 0; i < 3000; i++) { generated.run(); direct.run(); }
        double[] ratios = new double[7], generatedTimes = new double[7], directTimes = new double[7];
        for (int sample = 0; sample < 7; sample++) {
            if ((sample & 1) == 0) { generatedTimes[sample] = time(generated); directTimes[sample] = time(direct); }
            else { directTimes[sample] = time(direct); generatedTimes[sample] = time(generated); }
            ratios[sample] = generatedTimes[sample] / directTimes[sample];
        }
        Arrays.sort(ratios); Arrays.sort(generatedTimes); Arrays.sort(directTimes);
        report.append(String.format(Locale.ROOT, "%s,%.6f,%.6f,%.6f,%.6f,%.6f%n", name,
                generatedTimes[3] / (REPEATS * (double) COUNT), directTimes[3] / (REPEATS * (double) COUNT),
                ratios[3], ratios[0], ratios[6]));
    }

    private static long time(Action action) throws Throwable {
        long start = System.nanoTime();
        for (int i = 0; i < REPEATS; i++) action.run();
        return System.nanoTime() - start;
    }

    private static void directDouble(double[] input, double[] output, boolean vector, boolean gelu) {
        int i = 0;
        if (vector) {
            var species = DoubleVector.SPECIES_PREFERRED;
            int bound = species.loopBound(input.length);
            if (gelu) for (; i < bound; i += species.length()) {
                DoubleVector x = DoubleVector.fromArray(species, input, i);
                x.mul(0.5).mul(CpuActivationDirectOracle.erf(x.div(Math.sqrt(2))).add(1)).intoArray(output, i);
            } else for (; i < bound; i += species.length())
                CpuActivationDirectOracle.erf(DoubleVector.fromArray(species, input, i)).intoArray(output, i);
        }
        if (gelu) for (; i < input.length; i++) {
            double x = input[i]; output[i] = 0.5 * x * (1 + CpuActivationDirectOracle.erf(x / Math.sqrt(2)));
        } else for (; i < input.length; i++) output[i] = CpuActivationDirectOracle.erf(input[i]);
    }

    private static void directFloat(float[] input, float[] output, boolean vector, boolean gelu) {
        int i = 0;
        if (vector) {
            var species = FloatVector.SPECIES_PREFERRED;
            int bound = species.loopBound(input.length);
            if (gelu) for (; i < bound; i += species.length()) {
                FloatVector x = FloatVector.fromArray(species, input, i);
                x.mul(0.5f).mul(CpuActivationDirectOracle.erf(x.div((float) Math.sqrt(2))).add(1f)).intoArray(output, i);
            } else for (; i < bound; i += species.length())
                CpuActivationDirectOracle.erf(FloatVector.fromArray(species, input, i)).intoArray(output, i);
        }
        if (gelu) for (; i < input.length; i++) {
            double x = input[i]; output[i] = (float) (0.5 * x * (1 + CpuActivationDirectOracle.erf(x / Math.sqrt(2))));
        } else for (; i < input.length; i++) output[i] = (float) CpuActivationDirectOracle.erf(input[i]);
    }

    private static void directBFRSQRT(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.intBitsToFloat(input[i] << 16);
            float result = 1f / (float) StrictMath.sqrt(x);
            output[i] = CpuActivationDirectOracle.encode(DataType.BFLOAT16, result);
        }
    }

    private static void directBFSIGMOID(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.intBitsToFloat(input[i] << 16);
            float result = CpuActivationDirectOracle.sigmoid(x);
            output[i] = CpuActivationDirectOracle.encode(DataType.BFLOAT16, result);
        }
    }

    private static void directBFGELU_EXACT(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.intBitsToFloat(input[i] << 16);
            float result = CpuActivationDirectOracle.gelu(x);
            output[i] = CpuActivationDirectOracle.encode(DataType.BFLOAT16, result);
        }
    }

    private static void directBFGELU_TANH_APPROXIMATION(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.intBitsToFloat(input[i] << 16);
            float result = CpuActivationDirectOracle.geluTanh(x);
            output[i] = CpuActivationDirectOracle.encode(DataType.BFLOAT16, result);
        }
    }

    private static void directBFSILU(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.intBitsToFloat(input[i] << 16);
            float result = CpuActivationDirectOracle.silu(x);
            output[i] = CpuActivationDirectOracle.encode(DataType.BFLOAT16, result);
        }
    }

    private static void directFRSQRT(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.float16ToFloat(input[i]);
            float result = 1f / (float) StrictMath.sqrt(x);
            output[i] = Float.isNaN(result) ? (short) 0x7e00 : Float.floatToFloat16(result);
        }
    }

    private static void directFSIGMOID(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.float16ToFloat(input[i]);
            float result = CpuActivationDirectOracle.sigmoid(x);
            output[i] = Float.isNaN(result) ? (short) 0x7e00 : Float.floatToFloat16(result);
        }
    }

    private static void directFGELU_EXACT(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.float16ToFloat(input[i]);
            float result = CpuActivationDirectOracle.gelu(x);
            output[i] = Float.isNaN(result) ? (short) 0x7e00 : Float.floatToFloat16(result);
        }
    }

    private static void directFGELU_TANH_APPROXIMATION(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.float16ToFloat(input[i]);
            float result = CpuActivationDirectOracle.geluTanh(x);
            output[i] = Float.isNaN(result) ? (short) 0x7e00 : Float.floatToFloat16(result);
        }
    }

    private static void directFSILU(short[] input, short[] output) {
        for (int i = 0; i < input.length; i++) {
            float x = Float.float16ToFloat(input[i]);
            float result = CpuActivationDirectOracle.silu(x);
            output[i] = Float.isNaN(result) ? (short) 0x7e00 : Float.floatToFloat16(result);
        }
    }
}
