package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Opt-in, one-process v2 diagnostic for generated and directly written softmax bodies.
 *
 * <p>Each side uses a contiguous heap carrier, three passes, binary32 working sites, Kahan
 * accumulation, and (for FLOAT16) one final narrowing. Preparation and equality checking are
 * outside timing. This is a bounded observation, not a route gate or a replacement for the
 * historical multi-fork evidence.</p>
 */
final class CpuSoftmaxV2PerformanceDiagnosticTest {
    private static final int ROWS = 128;
    private static final int WIDTH = 2048;
    private static final int REPEATS = 4;
    private static final int WARMUP_PAIRS = 50;
    private static final int SAMPLES = 7;
    private static volatile long sink;

    @FunctionalInterface
    private interface Body { void run() throws Throwable; }

    private record Pair(String name, Body generated, Body direct, Runnable verify) { }

    @Test
    void boundedGeneratedVersusCleanJava() throws Throwable {
        assumeTrue("true".equals(System.getenv("SYNAPTIK_CPU_0033_DIAGNOSTIC")));
        List<Pair> pairs = new ArrayList<>();
        for (SoftmaxKind kind : SoftmaxKind.values()) {
            pairs.add(floatPair(kind));
            pairs.add(halfPair(kind));
        }
        for (Pair pair : pairs) {
            pair.generated.run();
            pair.direct.run();
            pair.verify.run(); // raw equality is a precondition, not a timed tolerance
        }

        Random order = new Random(0x0033_2026L);
        for (int warmup = 0; warmup < WARMUP_PAIRS; warmup++) {
            Collections.shuffle(pairs, order);
            for (Pair pair : pairs) runPair(pair, order.nextBoolean());
        }
        Map<String, long[]> generated = new java.util.HashMap<>();
        Map<String, long[]> direct = new java.util.HashMap<>();
        Map<String, double[]> ratios = new java.util.HashMap<>();
        for (Pair pair : pairs) {
            generated.put(pair.name, new long[SAMPLES]);
            direct.put(pair.name, new long[SAMPLES]);
            ratios.put(pair.name, new double[SAMPLES]);
        }
        for (int sample = 0; sample < SAMPLES; sample++) {
            Collections.shuffle(pairs, order);
            for (Pair pair : pairs) {
                long[] measurement = timedPair(pair, order.nextBoolean());
                generated.get(pair.name)[sample] = measurement[0] / REPEATS;
                direct.get(pair.name)[sample] = measurement[1] / REPEATS;
                ratios.get(pair.name)[sample] = (double) measurement[0] / measurement[1];
            }
        }
        System.out.printf(Locale.ROOT,
                "softmax-v2 diagnostic: java=%s os=%s/%s rows=%d width=%d warmupPairs=%d "
                + "samples=%d repeats=%d one JVM; ns per full invocation; ratio=generated/direct%n",
                System.getProperty("java.runtime.version"), System.getProperty("os.name"),
                System.getProperty("os.arch"), ROWS, WIDTH, WARMUP_PAIRS, SAMPLES, REPEATS);
        System.out.println("case,generated_median_ns,direct_median_ns,paired_median_ratio,min_ratio,max_ratio");
        pairs.sort(java.util.Comparator.comparing(Pair::name));
        for (Pair pair : pairs) {
            pair.verify.run();
            long[] g = generated.get(pair.name), d = direct.get(pair.name);
            double[] r = ratios.get(pair.name);
            Arrays.sort(g); Arrays.sort(d); Arrays.sort(r);
            System.out.printf(Locale.ROOT, "%s,%d,%d,%.6f,%.6f,%.6f%n",
                    pair.name, g[SAMPLES / 2], d[SAMPLES / 2], r[SAMPLES / 2], r[0], r[SAMPLES - 1]);
        }
    }

    private static Pair floatPair(SoftmaxKind kind) {
        MethodHandle entry = entry(kind, DataType.FLOAT32, CarrierAccess.FLOAT_ARRAY);
        long[] geometry = geometry(kind, DataType.FLOAT32, CarrierAccess.FLOAT_ARRAY);
        float[] input = new float[ROWS * WIDTH], actual = new float[input.length], expected = new float[input.length];
        for (int i = 0; i < input.length; i++) input[i] = value(i);
        Body generated = () -> {
            entry.invokeExact(input, actual, geometry, 0L, (long) ROWS);
            sink = Float.floatToRawIntBits(actual[actual.length - 1]);
        };
        Body direct = kind == SoftmaxKind.SOFTMAX
                ? () -> {
                    directFloatSoftmax(input, expected);
                    sink = Float.floatToRawIntBits(expected[expected.length - 1]);
                } : () -> {
                    directFloatLogSoftmax(input, expected);
                    sink = Float.floatToRawIntBits(expected[expected.length - 1]);
                };
        Runnable verify = () -> {
            for (int i = 0; i < actual.length; i++) assertEquals(
                    Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(actual[i]),
                    kind + "/FLOAT32/" + i);
        };
        return new Pair(kind + "/FLOAT32", generated, direct, verify);
    }

    private static Pair halfPair(SoftmaxKind kind) {
        MethodHandle entry = entry(kind, DataType.FLOAT16, CarrierAccess.SHORT_ARRAY);
        long[] geometry = geometry(kind, DataType.FLOAT16, CarrierAccess.SHORT_ARRAY);
        short[] input = new short[ROWS * WIDTH], actual = new short[input.length], expected = new short[input.length];
        for (int i = 0; i < input.length; i++) input[i] = Float.floatToFloat16(value(i));
        Body generated = () -> {
            entry.invokeExact(input, actual, geometry, 0L, (long) ROWS);
            sink = Short.toUnsignedInt(actual[actual.length - 1]);
        };
        Body direct = kind == SoftmaxKind.SOFTMAX
                ? () -> {
                    directHalfSoftmax(input, expected);
                    sink = Short.toUnsignedInt(expected[expected.length - 1]);
                } : () -> {
                    directHalfLogSoftmax(input, expected);
                    sink = Short.toUnsignedInt(expected[expected.length - 1]);
                };
        Runnable verify = () -> {
            for (int i = 0; i < actual.length; i++) assertEquals(expected[i], actual[i],
                    kind + "/FLOAT16/" + i);
        };
        return new Pair(kind + "/FLOAT16", generated, direct, verify);
    }

    private static MethodHandle entry(SoftmaxKind kind, DataType type, CarrierAccess carrier) {
        var route = plan(kind, type, carrier).units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(route.specialization(),
                generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
    }

    private static long[] geometry(SoftmaxKind kind, DataType type, CarrierAccess carrier) {
        return plan(kind, type, carrier).softmaxGeometry().orElseThrow().pack(new long[2]);
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            plan(SoftmaxKind kind, DataType type, CarrierAccess carrier) {
        var base = CpuSoftmaxLoweringTest.context(kind, type, Shape.of(ROWS, WIDTH), 1);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                        List.of(carrier, carrier)));
        return new CpuPartitionPreparer().analyze(context).plan();
    }

    private static float value(int index) {
        return ((index * 37 % 257) - 128) / 32.0f + (index / WIDTH % 5) * 0.25f;
    }

    private static void runPair(Pair pair, boolean generatedFirst) throws Throwable {
        if (generatedFirst) { pair.generated.run(); pair.direct.run(); }
        else { pair.direct.run(); pair.generated.run(); }
    }

    private static long[] timedPair(Pair pair, boolean generatedFirst) throws Throwable {
        long generated, direct;
        if (generatedFirst) {
            generated = time(pair.generated);
            direct = time(pair.direct);
        } else {
            direct = time(pair.direct);
            generated = time(pair.generated);
        }
        return new long[] {generated, direct};
    }

    private static long time(Body body) throws Throwable {
        long start = System.nanoTime();
        for (int repeat = 0; repeat < REPEATS; repeat++) body.run();
        return System.nanoTime() - start;
    }

    private static void directFloatSoftmax(float[] input, float[] output) {
        for (int row = 0; row < ROWS; row++) {
            int base = row * WIDTH;
            float maximum = Float.NEGATIVE_INFINITY;
            for (int i = base; i < base + WIDTH; i++) if (input[i] > maximum) maximum = input[i];
            float sum = 0, compensation = 0;
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = input[i] - maximum;
                float addend = (float) Math.exp(shifted) - compensation;
                float temporary = sum + addend;
                compensation = (temporary - sum) - addend;
                sum = temporary;
            }
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = input[i] - maximum;
                output[i] = (float) Math.exp(shifted) / sum;
            }
        }
    }

    private static void directFloatLogSoftmax(float[] input, float[] output) {
        for (int row = 0; row < ROWS; row++) {
            int base = row * WIDTH;
            float maximum = Float.NEGATIVE_INFINITY;
            for (int i = base; i < base + WIDTH; i++) if (input[i] > maximum) maximum = input[i];
            float sum = 0, compensation = 0;
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = input[i] - maximum;
                float addend = (float) Math.exp(shifted) - compensation;
                float temporary = sum + addend;
                compensation = (temporary - sum) - addend;
                sum = temporary;
            }
            float logarithm = (float) Math.log(sum);
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = input[i] - maximum;
                output[i] = shifted - logarithm;
            }
        }
    }

    private static void directHalfSoftmax(short[] input, short[] output) {
        for (int row = 0; row < ROWS; row++) {
            int base = row * WIDTH;
            float maximum = Float.NEGATIVE_INFINITY;
            for (int i = base; i < base + WIDTH; i++) {
                float value = Float.float16ToFloat(input[i]);
                if (value > maximum) maximum = value;
            }
            float sum = 0, compensation = 0;
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = Float.float16ToFloat(input[i]) - maximum;
                float addend = (float) Math.exp(shifted) - compensation;
                float temporary = sum + addend;
                compensation = (temporary - sum) - addend;
                sum = temporary;
            }
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = Float.float16ToFloat(input[i]) - maximum;
                output[i] = Float.floatToFloat16((float) Math.exp(shifted) / sum);
            }
        }
    }

    private static void directHalfLogSoftmax(short[] input, short[] output) {
        for (int row = 0; row < ROWS; row++) {
            int base = row * WIDTH;
            float maximum = Float.NEGATIVE_INFINITY;
            for (int i = base; i < base + WIDTH; i++) {
                float value = Float.float16ToFloat(input[i]);
                if (value > maximum) maximum = value;
            }
            float sum = 0, compensation = 0;
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = Float.float16ToFloat(input[i]) - maximum;
                float addend = (float) Math.exp(shifted) - compensation;
                float temporary = sum + addend;
                compensation = (temporary - sum) - addend;
                sum = temporary;
            }
            float logarithm = (float) Math.log(sum);
            for (int i = base; i < base + WIDTH; i++) {
                float shifted = Float.float16ToFloat(input[i]) - maximum;
                output[i] = Float.floatToFloat16(shifted - logarithm);
            }
        }
    }
}
