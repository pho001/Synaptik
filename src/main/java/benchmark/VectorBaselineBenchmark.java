package benchmark;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/**
 * Pure-Java baseline benchmark (no Tensor/Graph/Optimizer integration).
 *
 * Measures scalar vs Vector API over the same expression:
 *   y = sigmoid((a + b) * c + 0.25 * a)
 *
 * Also includes arithmetic-only path:
 *   y = (a + b) * c + 0.25 * a
 *
 * Run:
 *   java --add-modules jdk.incubator.vector benchmark.VectorBaselineBenchmark
 */
public final class VectorBaselineBenchmark {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final int SIZE = 1_000_000;
    private static final int WARMUP = 80;
    private static final int ITERS = 220;
    private static final int REPEATS = 7;

    private VectorBaselineBenchmark() {}

    public static void main(String[] args) {
        float[] a = randomArray(SIZE, 0.05f, 42);
        float[] b = randomArray(SIZE, -0.02f, 43);
        float[] c = randomArray(SIZE, 0.01f, 44);
        float[] out = new float[SIZE];

        System.out.println("=== Vector Baseline Benchmark (pure Java) ===");
        System.out.println("size=" + SIZE + ", warmup=" + WARMUP + ", iters=" + ITERS + ", repeats=" + REPEATS);
        System.out.println("vector lanes (float)=" + SPECIES.length());
        System.out.println();

        double[] arithScalar = runRepeated(() -> arithScalar(a, b, c, out));
        double[] arithVector = runRepeated(() -> arithVector(a, b, c, out));
        printStats("ARITH scalar", arithScalar);
        printStats("ARITH vector", arithVector);
        printSpeedup(arithScalar, arithVector);
        System.out.println();

        double[] sigScalar = runRepeated(() -> sigmoidScalar(a, b, c, out));
        double[] sigVector = runRepeated(() -> sigmoidVector(a, b, c, out));
        printStats("SIGMOID scalar", sigScalar);
        printStats("SIGMOID vector", sigVector);
        printSpeedup(sigScalar, sigVector);
        System.out.println();

        // Correctness sanity
        float[] ref = new float[SIZE];
        float[] test = new float[SIZE];
        sigmoidScalar(a, b, c, ref);
        sigmoidVector(a, b, c, test);
        System.out.println("maxAbsDiff(sigmoid scalar vs vector) = " + maxAbsDiff(ref, test));
    }

    private static void arithScalar(float[] a, float[] b, float[] c, float[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = (a[i] + b[i]) * c[i] + (0.25f * a[i]);
        }
    }

    private static void arithVector(float[] a, float[] b, float[] c, float[] out) {
        int width = SPECIES.length();
        int i = 0;
        int upper = out.length - (out.length % width);
        FloatVector k = FloatVector.broadcast(SPECIES, 0.25f);
        for (; i < upper; i += width) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            FloatVector vc = FloatVector.fromArray(SPECIES, c, i);
            FloatVector vy = va.add(vb).mul(vc).add(va.mul(k));
            vy.intoArray(out, i);
        }
        for (; i < out.length; i++) {
            out[i] = (a[i] + b[i]) * c[i] + (0.25f * a[i]);
        }
    }

    private static void sigmoidScalar(float[] a, float[] b, float[] c, float[] out) {
        for (int i = 0; i < out.length; i++) {
            float x = (a[i] + b[i]) * c[i] + (0.25f * a[i]);
            out[i] = (float) (1.0 / (1.0 + Math.exp(-x)));
        }
    }

    private static void sigmoidVector(float[] a, float[] b, float[] c, float[] out) {
        int width = SPECIES.length();
        int i = 0;
        int upper = out.length - (out.length % width);
        float[] tmp = new float[width];
        FloatVector k = FloatVector.broadcast(SPECIES, 0.25f);

        for (; i < upper; i += width) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            FloatVector vc = FloatVector.fromArray(SPECIES, c, i);
            FloatVector vx = va.add(vb).mul(vc).add(va.mul(k));
            vx.intoArray(tmp, 0);
            for (int lane = 0; lane < width; lane++) {
                float x = tmp[lane];
                tmp[lane] = (float) (1.0 / (1.0 + Math.exp(-x)));
            }
            FloatVector.fromArray(SPECIES, tmp, 0).intoArray(out, i);
        }

        for (; i < out.length; i++) {
            float x = (a[i] + b[i]) * c[i] + (0.25f * a[i]);
            out[i] = (float) (1.0 / (1.0 + Math.exp(-x)));
        }
    }

    private static double[] runRepeated(Runnable kernel) {
        double[] ms = new double[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            for (int i = 0; i < WARMUP; i++) kernel.run();
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) kernel.run();
            long t1 = System.nanoTime();
            ms[r] = (t1 - t0) / 1_000_000.0 / ITERS;
        }
        return ms;
    }

    private static void printStats(String label, double[] samples) {
        double median = percentile(samples, 50);
        double mean = Arrays.stream(samples).average().orElse(0.0);
        double p90 = percentile(samples, 90);
        System.out.printf(Locale.ROOT, "%-16s median=%8.4f ms  mean=%8.4f ms  p90=%8.4f ms%n",
                label, median, mean, p90);
    }

    private static void printSpeedup(double[] scalar, double[] vector) {
        double s = percentile(scalar, 50);
        double v = percentile(vector, 50);
        System.out.printf(Locale.ROOT, "speedup (median): %.2fx%n", s / v);
    }

    private static double percentile(double[] values, int p) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        double rank = (p / 100.0) * (sorted.length - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) return sorted[lo];
        double w = rank - lo;
        return sorted[lo] * (1.0 - w) + sorted[hi] * w;
    }

    private static float[] randomArray(int n, float scale, int seed) {
        Random rnd = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) (Math.sin(i * 0.1) + (rnd.nextDouble() - 0.5) * scale * 10.0);
        }
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float max = 0.0f;
        for (int i = 0; i < a.length; i++) {
            float d = Math.abs(a[i] - b[i]);
            if (d > max) max = d;
        }
        return max;
    }
}
