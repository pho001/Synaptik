package benchmark;

import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import config.backend.CpuKernelConfig;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.rules.FuseElementWiseRule;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.Locale;

public final class FusedBroadcastBenchmark {
    private static final int WARMUP = 80;
    private static final int ITERS = 220;
    private static final int REPEATS = 7;

    private FusedBroadcastBenchmark() {}

    public static void main(String[] args) {
        int b0 = 256;
        int b1 = 8;
        int f = 256;

        double[] aVals = new double[b0 * 1 * f];
        double[] bVals = new double[1 * b1 * f];
        double[] cVals = new double[b0 * b1 * f];
        fillInputs(aVals, bVals, cVals);

        GraphOptimizer noOpt = new GraphOptimizer();
        GraphOptimizer fuseOnly = new GraphOptimizer();
        fuseOnly.addRule(new FuseElementWiseRule());

        double[] expected = runOnce(noOpt, CpuKernelConfig.defaultsTraining(), aVals, bVals, cVals);

        CpuKernelConfig scalarCfg = new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, Integer.MAX_VALUE);
        CpuKernelConfig vectorCfg = new CpuKernelConfig(4, 32, 32, 32, 1, Integer.MAX_VALUE);
        CpuKernelConfig parallelCfg = new CpuKernelConfig(4, 32, 32, 32, Integer.MAX_VALUE, 1);
        CpuKernelConfig parallelVectorCfg = new CpuKernelConfig(4, 32, 32, 32, 1, 1);

        double[] s = runRepeated(fuseOnly, scalarCfg, aVals, bVals, cVals, expected);
        double[] v = runRepeated(fuseOnly, vectorCfg, aVals, bVals, cVals, expected);
        double[] p = runRepeated(fuseOnly, parallelCfg, aVals, bVals, cVals, expected);
        double[] pv = runRepeated(fuseOnly, parallelVectorCfg, aVals, bVals, cVals, expected);

        System.out.println("=== Fused Broadcast Benchmark ===");
        System.out.println("shape(a)=[256,1,256], shape(b)=[1,8,256], shape(c)=[256,8,256], out=[256,8,256]");
        System.out.println("expr: sigmoid((a + b) * c + a)");
        System.out.println("dtype=" + DataType.FLOAT64 + ", warmup=" + WARMUP + ", iters=" + ITERS + ", repeats=" + REPEATS);
        System.out.println();

        printStats("SCALAR", s);
        printStats("VECTOR", v);
        printStats("PARALLEL", p);
        printStats("PAR_VECTOR", pv);
        System.out.println();
        printSpeedup("VECTOR", s, v);
        printSpeedup("PARALLEL", s, p);
        printSpeedup("PAR_VECTOR", s, pv);
    }

    private static double[] runRepeated(
            GraphOptimizer optimizer,
            CpuKernelConfig config,
            double[] aVals,
            double[] bVals,
            double[] cVals,
            double[] expected
    ) {
        double[] ms = new double[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            for (int i = 0; i < WARMUP; i++) {
                runOnce(optimizer, config, aVals, bVals, cVals);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                runOnce(optimizer, config, aVals, bVals, cVals);
            }
            long t1 = System.nanoTime();
            ms[r] = (t1 - t0) / 1_000_000.0 / ITERS;
        }

        double[] out = runOnce(optimizer, config, aVals, bVals, cVals);
        double maxAbs = maxAbsDiff(expected, out);
        if (maxAbs > 1e-9) {
            throw new IllegalStateException("Correctness mismatch, maxAbsDiff=" + maxAbs);
        }
        return ms;
    }

    private static double[] runOnce(
            GraphOptimizer optimizer,
            CpuKernelConfig config,
            double[] aVals,
            double[] bVals,
            double[] cVals
    ) {
        RuntimeConfig runtimeConfig = new RuntimeConfig(config, ApproximationConfig.defaults(), BlasConfig.disabled());
        Tensor a = new Tensor(aVals.clone(), new int[]{256, 1, 256}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(bVals.clone(), new int[]{1, 8, 256}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(cVals.clone(), new int[]{256, 8, 256}, null, "c", DataType.FLOAT64);
        Tensor out = a.add(b).mul(c).add(a).sigmoid();
        out.compute(optimizer, runtimeConfig, ExecutionMode.FORWARD);
        return out.toDoubleArrayCopy();
    }

    private static void fillInputs(double[] a, double[] b, double[] c) {
        for (int i = 0; i < a.length; i++) {
            a[i] = Math.sin(i * 0.01) + (i % 7) * 0.03;
        }
        for (int i = 0; i < b.length; i++) {
            b[i] = Math.cos(i * 0.02) + (i % 11) * 0.02;
        }
        for (int i = 0; i < c.length; i++) {
            c[i] = 0.2 + Math.sin(i * 0.005) * 0.4;
        }
    }

    private static void printStats(String label, double[] samples) {
        double median = percentile(samples, 50);
        double mean = Arrays.stream(samples).average().orElse(0.0);
        double p90 = percentile(samples, 90);
        System.out.printf(Locale.ROOT, "%-12s median=%8.4f ms  mean=%8.4f ms  p90=%8.4f ms%n",
                label, median, mean, p90);
    }

    private static void printSpeedup(String mode, double[] scalar, double[] modeSamples) {
        double s = percentile(scalar, 50);
        double m = percentile(modeSamples, 50);
        System.out.printf(Locale.ROOT, "speedup vs SCALAR (%s): %.3fx%n", mode, s / m);
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

    private static double maxAbsDiff(double[] a, double[] b) {
        double max = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.abs(a[i] - b[i]);
            if (d > max) max = d;
        }
        return max;
    }
}
