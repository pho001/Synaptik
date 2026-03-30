package benchmark;

import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;

import java.util.Locale;
import java.util.Random;

/**
 * Microbenchmark for tensor layout transforms.
 *
 * Runs:
 * - reshape view path
 * - permute/transpose view path
 * - permute + contiguous materialization
 * - expandDims + squeeze chain
 *
 * Usage:
 *   java --add-modules jdk.incubator.vector benchmark.LayoutOpsBenchmark
 */
public final class LayoutOpsBenchmark {
    private static final int SIZE = 1_000_000;
    private static final int WARMUP = 100;
    private static final int ITERS = 500;
    private static volatile double SINK = 0.0;

    private LayoutOpsBenchmark() {}

    public static void main(String[] args) {
        System.out.println("=== Layout Ops Benchmark ===");
        System.out.println("size=" + SIZE + ", warmup=" + WARMUP + ", iters=" + ITERS);
        System.out.println();

        runForDType(DataType.FLOAT32);
        runForDType(DataType.FLOAT64);
    }

    private static void runForDType(DataType dataType) {
        GraphOptimizer optimizer = new GraphOptimizer();
        double[] values = randomData(SIZE, 42);
        Tensor base = new Tensor(values.clone(), new int[]{1000, 1000}, null, "base", dataType);

        Tensor reshapeGraph = base.reshape(2000, 500);
        Tensor permuteGraph = base.permute(1, 0);
        Tensor permuteContiguousGraph = base.permute(1, 0).contiguous();
        Tensor transposeGraph = base.transpose();
        Tensor expandSqueezeGraph = base.expandDims(0).squeeze(0);

        // JIT + graph compilation priming
        reshapeGraph.compute(optimizer);
        permuteGraph.compute(optimizer);
        permuteContiguousGraph.compute(optimizer);
        transposeGraph.compute(optimizer);
        expandSqueezeGraph.compute(optimizer);

        double reshapeMs = bench(() -> runAndTouch(reshapeGraph, optimizer));
        double permuteViewMs = bench(() -> runAndTouch(permuteGraph, optimizer));
        double permuteContigMs = bench(() -> runAndTouch(permuteContiguousGraph, optimizer));
        double transposeViewMs = bench(() -> runAndTouch(transposeGraph, optimizer));
        double expandSqueezeMs = bench(() -> runAndTouch(expandSqueezeGraph, optimizer));

        System.out.println("[" + dataType + "]");
        System.out.printf(Locale.ROOT, "reshape(view):            %.6f ms%n", reshapeMs);
        System.out.printf(Locale.ROOT, "permute(view):            %.6f ms%n", permuteViewMs);
        System.out.printf(Locale.ROOT, "permute+contiguous(copy): %.6f ms%n", permuteContigMs);
        System.out.printf(Locale.ROOT, "transpose(view):          %.6f ms%n", transposeViewMs);
        System.out.printf(Locale.ROOT, "expand+squeeze(view):     %.6f ms%n", expandSqueezeMs);
        System.out.println();
    }

    private static void runAndTouch(Tensor graph, GraphOptimizer optimizer) {
        graph.compute(optimizer);
        SINK += graph.getByFlatIndex(0);
    }

    private static double bench(Runnable task) {
        for (int i = 0; i < WARMUP; i++) {
            task.run();
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < ITERS; i++) {
            task.run();
        }
        long t1 = System.nanoTime();
        return (t1 - t0) / 1_000_000.0 / ITERS;
    }

    private static double[] randomData(int n, int seed) {
        Random rnd = new Random(seed);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = Math.sin(i * 0.01) + (rnd.nextDouble() - 0.5) * 0.1;
        }
        return out;
    }
}
