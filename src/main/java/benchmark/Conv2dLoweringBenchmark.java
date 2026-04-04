package benchmark;

import backend.blas.BlasProvider;
import backend.blas.BlasThreadPolicy;
import backend.blas.BlasThreadPolicy;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.optimizer.CseConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.RewriteConfig;
import config.optimizer.FuseConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.Conv2dOptions;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public final class Conv2dLoweringBenchmark {
    private static final int WARMUP = Math.max(1, Integer.getInteger("benchmark.conv2dWarmup", 8));
    private static final int ITERS = Math.max(1, Integer.getInteger("benchmark.conv2dIters", 20));
    private static final int REPEATS = Math.max(1, Integer.getInteger("benchmark.conv2dRepeats", 3));
    private static volatile double SINK;

    private record ShapeCase(
            String name,
            int batch,
            int inChannels,
            int outChannels,
            int height,
            int width,
            int kernelH,
            int kernelW,
            int strideH,
            int strideW,
            int padH,
            int padW,
            int groups
    ) {}

    private Conv2dLoweringBenchmark() {
    }

    public static void main(String[] args) {
        DataType dtype = resolveDType();
        List<ShapeCase> cases = List.of(
                new ShapeCase("CNN_3x3_S1", 2, 64, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1),
                new ShapeCase("CNN_3x3_S2", 2, 64, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1),
                new ShapeCase("CNN_1x1_28", 2, 256, 256, 28, 28, 1, 1, 1, 1, 0, 0, 1),
                new ShapeCase("BOTTLENECK_EXPAND_1x1", 2, 256, 1024, 14, 14, 1, 1, 1, 1, 0, 0, 1),
                new ShapeCase("BOTTLENECK_PROJECT_1x1", 2, 1024, 256, 14, 14, 1, 1, 1, 1, 0, 0, 1),
                new ShapeCase("GROUPED_3x3_G2", 2, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 2),
                new ShapeCase("GROUPED_3x3_G4", 2, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 4),
                new ShapeCase("DEPTHWISE_3x3", 2, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 64),
                new ShapeCase("DEPTHWISE_3x3_S2", 2, 128, 128, 28, 28, 3, 3, 2, 2, 1, 1, 128)
        );

        System.out.println("=== Conv2d Lowering Benchmark ===");
        System.out.println("dtype=" + dtype + ", warmup=" + WARMUP + ", iters=" + ITERS + ", repeats=" + REPEATS);
        String explicitOpenBlas = System.getProperty("openblas.lib", "").trim();
        if (!explicitOpenBlas.isEmpty()) {
            System.out.println("openblas.lib=" + explicitOpenBlas);
        }
        System.out.println();

        for (ShapeCase shape : cases) {
            runCase(dtype, shape);
            System.out.println();
        }
    }

    private static void runCase(DataType dtype, ShapeCase shape) {
        long seed = 97L + shape.batch * 13L + shape.inChannels * 17L + shape.outChannels * 19L;
        int inputSize = shape.batch * shape.inChannels * shape.height * shape.width;
        int weightSize = shape.outChannels * (shape.inChannels / shape.groups) * shape.kernelH * shape.kernelW;
        int biasSize = shape.outChannels;
        double[] input = randomData(inputSize, seed);
        double[] weight = randomData(weightSize, seed + 1);
        double[] bias = randomData(biasSize, seed + 2);
        Conv2dOptions options = new Conv2dOptions(
                shape.strideH, shape.strideW,
                shape.padH, shape.padW,
                1, 1,
                shape.groups
        );

        double[] reference = runOnce(dtype, shape, input, weight, bias, options, Mode.DIRECT);
        Result direct = benchmarkMode(dtype, shape, input, weight, bias, options, reference, Mode.DIRECT);
        Result gemmJava = benchmarkMode(dtype, shape, input, weight, bias, options, reference, Mode.GEMM_JAVA);
        Result gemmBlas = benchmarkMode(dtype, shape, input, weight, bias, options, reference, Mode.GEMM_BLAS);

        System.out.println("[CASE] " + shape.name
                + "  input=[" + shape.batch + "," + shape.inChannels + "," + shape.height + "," + shape.width + "]"
                + "  weight=[" + shape.outChannels + "," + (shape.inChannels / shape.groups) + "," + shape.kernelH + "," + shape.kernelW + "]"
                + "  groups=" + shape.groups);
        System.out.printf(Locale.ROOT, "%-14s %-12s %-12s %-12s%n", "mode", "median_ms", "mean_ms", "p90_ms");
        printResult("direct", direct);
        printResult("gemm_java", gemmJava);
        if (gemmBlas != null) {
            printResult("gemm_openblas", gemmBlas);
        } else {
            System.out.printf(Locale.ROOT, "%-14s %-12s %-12s %-12s%n", "gemm_openblas", "n/a", "n/a", "n/a");
        }
        Result best = direct;
        String bestName = "direct";
        if (gemmJava != null && gemmJava.medianMs < best.medianMs) {
            best = gemmJava;
            bestName = "gemm_java";
        }
        if (gemmBlas != null && gemmBlas.medianMs < best.medianMs) {
            best = gemmBlas;
            bestName = "gemm_openblas";
        }
        System.out.printf(Locale.ROOT,
                "best=%s  speedup_java=%.3fx  speedup_openblas=%s%n",
                bestName,
                direct.medianMs / gemmJava.medianMs,
                gemmBlas == null ? "n/a" : String.format(Locale.ROOT, "%.3fx", direct.medianMs / gemmBlas.medianMs)
        );
    }

    private static void printResult(String name, Result result) {
        System.out.printf(Locale.ROOT, "%-14s %-12.4f %-12.4f %-12.4f%n", name, result.medianMs, result.meanMs, result.p90Ms);
    }

    private static Result benchmarkMode(
            DataType dtype,
            ShapeCase shape,
            double[] input,
            double[] weight,
            double[] bias,
            Conv2dOptions options,
            double[] reference,
            Mode mode
    ) {
        if (mode == Mode.GEMM_BLAS && !backend.blas.OpenBlasFfmBridge.isAvailable()) {
            return null;
        }

        double[] samples = new double[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            for (int i = 0; i < WARMUP; i++) {
                runOnce(dtype, shape, input, weight, bias, options, mode);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                runOnce(dtype, shape, input, weight, bias, options, mode);
            }
            long t1 = System.nanoTime();
            samples[r] = (t1 - t0) / 1_000_000.0 / ITERS;
        }

        double[] actual = runOnce(dtype, shape, input, weight, bias, options, mode);
        double maxAbs = maxAbsDiff(reference, actual);
        double eps = dtype == DataType.FLOAT64 ? 1e-8 : 5e-4;
        if (maxAbs > eps) {
            throw new IllegalStateException("Conv2d benchmark mismatch for mode " + mode + ", maxAbs=" + maxAbs + ", eps=" + eps);
        }
        SINK += actual[0];
        return new Result(percentile(samples, 50), Arrays.stream(samples).average().orElse(0.0), percentile(samples, 90));
    }

    private static double[] runOnce(
            DataType dtype,
            ShapeCase shape,
            double[] inputData,
            double[] weightData,
            double[] biasData,
            Conv2dOptions options,
            Mode mode
    ) {
        Tensor input = tensorFrom(inputData, new int[]{shape.batch, shape.inChannels, shape.height, shape.width}, "input", dtype);
        Tensor weight = tensorFrom(weightData, new int[]{shape.outChannels, shape.inChannels / shape.groups, shape.kernelH, shape.kernelW}, "weight", dtype);
        Tensor bias = tensorFrom(biasData, new int[]{shape.outChannels}, "bias", dtype);
        Tensor out = input.conv2d(weight, bias, options);

        CompiledGraph compiled = CompiledGraph.compile(out, optimizerFor(mode));
        compiled.execute(runtimeConfigFor(mode), ExecutionMode.FORWARD);
        return out.toDoubleArrayCopy();
    }

    private static OptimizerConfig optimizerFor(Mode mode) {
        return switch (mode) {
            case DIRECT -> OptimizerConfig.noOptimization();
            case GEMM_JAVA, GEMM_BLAS -> new OptimizerConfig(
                    List.of(OptimizerStage.AR),
                    new RewriteConfig(Conv2dLoweringConfig.always()),
                    CseConfig.strictDefaults(),
                    FuseConfig.trainingDefaults(),
                    config.optimizer.MemoryConfig.defaults()
            );
        };
    }

    private static RuntimeConfig runtimeConfigFor(Mode mode) {
        CpuKernelConfig cpu = CpuKernelConfig.defaultsInference();
        BlasThreadPolicy threadPolicy = resolveThreadPolicy();
        int threads = resolveThreadCount();
        BlasConfig blas = switch (mode) {
            case DIRECT, GEMM_JAVA -> BlasConfig.disabled();
            case GEMM_BLAS -> new BlasConfig(BlasProvider.OPENBLAS_FFM, 1L, false, 1000.0d, false, threadPolicy, threads);
        };
        return new RuntimeConfig(cpu, ApproximationConfig.defaults(), blas);
    }

    private static BlasThreadPolicy resolveThreadPolicy() {
        String raw = System.getProperty("benchmark.blasThreadPolicy", BlasThreadPolicy.AUTO.name()).trim().toUpperCase(Locale.ROOT);
        try {
            return BlasThreadPolicy.valueOf(raw);
        } catch (Exception ex) {
            throw new IllegalStateException("Invalid benchmark.blasThreadPolicy: " + raw + ". Allowed: AUTO, FIXED", ex);
        }
    }

    private static int resolveThreadCount() {
        return Math.max(1, Integer.getInteger("benchmark.blasThreads", 1));
    }

    private static Tensor tensorFrom(double[] src, int[] shape, String label, DataType dtype) {
        return switch (dtype) {
            case FLOAT64 -> new Tensor(src.clone(), shape, null, label, DataType.FLOAT64);
            case FLOAT32 -> {
                float[] f = new float[src.length];
                for (int i = 0; i < src.length; i++) f[i] = (float) src[i];
                yield new Tensor(f, shape, null, label, DataType.FLOAT32);
            }
            case FLOAT16 -> {
                short[] h = new short[src.length];
                for (int i = 0; i < src.length; i++) h[i] = backend.kernels.cpu.CpuDTypeOps.toHalfBits((float) src[i]);
                yield new Tensor(h, shape, null, label, DataType.FLOAT16);
            }
            case INT32, BOOL -> throw new IllegalArgumentException("Conv2dLoweringBenchmark supports floating dtypes only.");
        };
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
        String raw = System.getProperty("benchmark.dtype", DataType.FLOAT32.name()).trim().toUpperCase(Locale.ROOT);
        try {
            DataType dtype = DataType.valueOf(raw);
            if (dtype == DataType.BOOL || dtype == DataType.INT32) {
                throw new IllegalArgumentException("Unsupported benchmark dtype: " + raw);
            }
            return dtype;
        } catch (Exception ex) {
            throw new IllegalStateException("Invalid benchmark.dtype: " + raw + ". Allowed: FLOAT16, FLOAT32, FLOAT64", ex);
        }
    }

    private enum Mode {
        DIRECT,
        GEMM_JAVA,
        GEMM_BLAS
    }

    private record Result(double medianMs, double meanMs, double p90Ms) {
    }
}
