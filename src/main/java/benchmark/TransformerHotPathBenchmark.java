package benchmark;

import benchmark.scenario.PreparedHotPathScenario;
import benchmark.scenario.TransformerHotPathScenarioFactory;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import config.optimizer.OptimizerConfig;
import tensor.DataType;
import backend.runtime.ExecutionMode;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public final class TransformerHotPathBenchmark {
    private static final int WARMUP = Math.max(1, Integer.getInteger("benchmark.hotpathWarmup", 20));
    private static final int ITERS = Math.max(1, Integer.getInteger("benchmark.hotpathIters", 40));
    private static final int REPEATS = Math.max(1, Integer.getInteger("benchmark.hotpathRepeats", 5));
    private static volatile double SINK;

    private TransformerHotPathBenchmark() {
    }

    public static void main(String[] args) {
        ExecutionProfile profile = loadProfile();
        System.out.println("=== Transformer Hot Path Benchmark ===");
        System.out.println("profile=" + profile.profileName()
                + ", candidate=" + profile.candidateName()
                + ", dtype=" + profile.dataType()
                + ", mode=" + profile.mode()
                + ", workload=" + profile.workload());
        System.out.println("warmup=" + WARMUP + ", iters=" + ITERS + ", repeats=" + REPEATS);
        System.out.println();

        List<PreparedHotPathScenario> scenarios = TransformerHotPathScenarioFactory.create(profile);
        System.out.printf(Locale.ROOT, "%-24s %-12s %-12s %-12s%n", "scenario", "median_ms", "mean_ms", "p90_ms");
        for (PreparedHotPathScenario scenario : scenarios) {
            double[] samples = bench(scenario);
            System.out.printf(Locale.ROOT, "%-24s %-12.4f %-12.4f %-12.4f%n",
                    scenario.name(),
                    percentile(samples, 50),
                    Arrays.stream(samples).average().orElse(0.0),
                    percentile(samples, 90));
        }
        System.out.println();
        System.out.println("sink=" + SINK);
    }

    private static ExecutionProfile loadProfile() {
        Path path = Path.of(System.getProperty("benchmark.profilePath", "config/optimizer-profile.json"));
        ExecutionProfile defaults = new ExecutionProfile(
                "transformer-hotpath",
                "transformer-hotpath",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                OptimizerConfig.inferenceDefaults(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.transformerHotPathDefaults()
        );
        ExecutionProfile loaded = OptimizerProfileIO.loadExecutionProfileOrDefault(path, defaults);
        if (loaded.workload().kind() == config.profile.WorkloadKind.NONE) {
            return new ExecutionProfile(
                    loaded.profileName(),
                    loaded.candidateName(),
                    loaded.dataType(),
                    loaded.mode(),
                    loaded.optimizer(),
                    loaded.runtime(),
                    WorkloadProfile.transformerHotPathDefaults()
            );
        }
        return loaded;
    }

    private static double[] bench(PreparedHotPathScenario scenario) {
        double[] samples = new double[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            for (int i = 0; i < WARMUP; i++) {
                scenario.run();
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS; i++) {
                scenario.run();
            }
            long t1 = System.nanoTime();
            samples[r] = (t1 - t0) / 1_000_000.0 / ITERS;
        }
        scenario.run();
        SINK += scenario.sink();
        return samples;
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
}
