package benchmark;

import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import config.backend.CpuKernelConfig;
import graph.CompiledGraph;
import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public final class MatMulBenchmark {
    private static final int WARMUP = 12;
    private static final int ITERS = 30;
    private static final int REPEATS = 5;
    private static final double EPS64 = 1e-8;
    private static final double EPS32 = 5e-4;

    private record ShapeCase(String name, int m, int n, int k) {}

    private MatMulBenchmark() {}

    public static void main(String[] args) {
        DataType dtype = resolveDType();
        List<ShapeCase> cases = List.of(
                new ShapeCase("SQUARE_256", 256, 256, 256),
                new ShapeCase("SQUARE_512", 512, 512, 512),
                new ShapeCase("TALL_SKINNY", 1024, 128, 256),
                new ShapeCase("SKINNY_WIDE", 128, 1024, 256)
        );

        int[] tileM = new int[]{16, 32, 64};
        int[] tileN = new int[]{16, 32, 64};
        int[] tileK = new int[]{16, 32, 64};

        System.out.println("=== MatMul Benchmark (CPU Java) ===");
        System.out.println("dtype=" + dtype + ", warmup=" + WARMUP + ", iters=" + ITERS + ", repeats=" + REPEATS);
        System.out.println("grid: tileM=" + Arrays.toString(tileM)
                + ", tileN=" + Arrays.toString(tileN)
                + ", tileK=" + Arrays.toString(tileK));
        System.out.println();

        for (ShapeCase sc : cases) {
            runCase(dtype, sc, tileM, tileN, tileK);
            System.out.println();
        }
    }

    private static void runCase(DataType dtype, ShapeCase sc, int[] tileM, int[] tileN, int[] tileK) {
        int m = sc.m();
        int n = sc.n();
        int k = sc.k();
        long seed = 17L + m * 31L + n * 131L + k * 331L;
        double[] aData = randomData(m * k, seed);
        double[] bData = randomData(k * n, seed + 1);
        double[] expected = referenceMatMul(aData, bData, m, n, k);

        List<Result> results = new ArrayList<>();
        for (int tm : tileM) {
            for (int tn : tileN) {
                for (int tk : tileK) {
                    CpuKernelConfig cfg = new CpuKernelConfig(
                            4, tm, tn, tk,
                            Integer.MAX_VALUE, Integer.MAX_VALUE,
                            0, 4, 4096
                    );
                    double[] samples = runRepeated(dtype, cfg, aData, bData, m, n, k, expected);
                    results.add(new Result(tm, tn, tk, percentile(samples, 50), Arrays.stream(samples).average().orElse(0.0), percentile(samples, 90)));
                }
            }
        }

        results.sort((x, y) -> Double.compare(x.medianMs, y.medianMs));

        System.out.println("[CASE] " + sc.name() + "  (" + m + "x" + k + ") x (" + k + "x" + n + ")");
        System.out.printf(Locale.ROOT, "%-9s %-9s %-9s %-12s %-12s %-12s%n",
                "tileM", "tileN", "tileK", "median_ms", "mean_ms", "p90_ms");
        for (int i = 0; i < Math.min(8, results.size()); i++) {
            Result r = results.get(i);
            System.out.printf(Locale.ROOT, "%-9d %-9d %-9d %-12.4f %-12.4f %-12.4f%n",
                    r.tileM, r.tileN, r.tileK, r.medianMs, r.meanMs, r.p90Ms);
        }
    }

    private static double[] runRepeated(
            DataType dtype,
            CpuKernelConfig config,
            double[] aData,
            double[] bData,
            int m,
            int n,
            int k,
            double[] expected
    ) {
        double[] samples = new double[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            for (int i = 0; i < WARMUP; i++) {
                runOnce(dtype, config, aData, bData, m, n, k);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                runOnce(dtype, config, aData, bData, m, n, k);
            }
            long t1 = System.nanoTime();
            samples[r] = (t1 - t0) / 1_000_000.0 / ITERS;
        }

        double[] out = runOnce(dtype, config, aData, bData, m, n, k);
        double maxAbs = maxAbsDiff(expected, out);
        double eps = dtype == DataType.FLOAT64 ? EPS64 : EPS32;
        if (maxAbs > eps) {
            throw new IllegalStateException("matmul mismatch for tiles ["
                    + config.matMulTileM() + "," + config.matMulTileN() + "," + config.matMulTileK()
                    + "], maxAbs=" + maxAbs + ", eps=" + eps);
        }

        return samples;
    }

    private static double[] runOnce(
            DataType dtype,
            CpuKernelConfig config,
            double[] aData,
            double[] bData,
            int m,
            int n,
            int k
    ) {
        RuntimeConfig runtimeConfig = new RuntimeConfig(config, ApproximationConfig.defaults(), BlasConfig.disabled());
        Tensor a = tensorFrom(aData, new int[]{m, k}, "A", dtype);
        Tensor b = tensorFrom(bData, new int[]{k, n}, "B", dtype);
        Tensor out = a.matmul(b);
        CompiledGraph.compile(out, new GraphOptimizer()).execute(runtimeConfig, ExecutionMode.FORWARD);
        return out.toDoubleArrayCopy();
    }

    private static Tensor tensorFrom(double[] src, int[] shape, String label, DataType dtype) {
        return switch (dtype) {
            case FLOAT64 -> new Tensor(src.clone(), shape, null, label, DataType.FLOAT64);
            case FLOAT32 -> {
                float[] f = new float[src.length];
                for (int i = 0; i < src.length; i++) f[i] = (float) src[i];
                yield new Tensor(f, shape, null, label, DataType.FLOAT32);
            }
            case FLOAT16 -> throw new IllegalArgumentException("MatMulBenchmark supports FLOAT64/FLOAT32 only");
        };
    }

    private static double[] referenceMatMul(double[] a, double[] b, int m, int n, int k) {
        double[] out = new double[m * n];
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int oRow = i * n;
            for (int p = 0; p < k; p++) {
                double av = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    out[oRow + j] += av * b[bRow + j];
                }
            }
        }
        return out;
    }

    private static double[] randomData(int len, long seed) {
        Random rnd = new Random(seed);
        double[] out = new double[len];
        for (int i = 0; i < len; i++) {
            out[i] = (rnd.nextDouble() - 0.5) * 2.0;
        }
        return out;
    }

    private static double maxAbsDiff(double[] a, double[] b) {
        double max = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.abs(a[i] - b[i]);
            if (d > max) max = d;
        }
        return max;
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

    private static DataType resolveDType() {
        String raw = System.getProperty("benchmark.dtype", DataType.FLOAT64.name()).trim().toUpperCase(Locale.ROOT);
        try {
            DataType dtype = DataType.valueOf(raw);
            if (dtype == DataType.FLOAT16) {
                throw new IllegalArgumentException("FLOAT16 is not supported by this benchmark yet.");
            }
            return dtype;
        } catch (Exception ex) {
            throw new IllegalStateException("Invalid benchmark.dtype: " + raw + " (use FLOAT64 or FLOAT32)", ex);
        }
    }

    private static final class Result {
        final int tileM;
        final int tileN;
        final int tileK;
        final double medianMs;
        final double meanMs;
        final double p90Ms;

        Result(int tileM, int tileN, int tileK, double medianMs, double meanMs, double p90Ms) {
            this.tileM = tileM;
            this.tileN = tileN;
            this.tileK = tileK;
            this.medianMs = medianMs;
            this.meanMs = meanMs;
            this.p90Ms = p90Ms;
        }
    }
}
