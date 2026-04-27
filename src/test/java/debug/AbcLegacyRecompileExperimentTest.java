package debug;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.OptimizerFactory;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class AbcLegacyRecompileExperimentTest {
    @Test
    void experimentF64ForwardBackward() {
        run(DataType.FLOAT64, "f64");
    }

    @Test
    void experimentF32ForwardBackward() {
        run(DataType.FLOAT32, "f32");
    }

    private static void run(DataType dataType, String dtypeId) {
        ExecutionProfile profile = loadBestProfile(dtypeId);
        Reference reference = reference(profile);

        System.out.println();
        System.out.println("ABC_LEGACY_RECOMPILE_EXPERIMENT :: " + dtypeId);
        System.out.println("stageOrder=" + profile.optimizer().stageOrder());

        report("baseline", instantiateRoot(profile), profile, reference);
        report("preopt-x1", preoptimizedRoot(profile, 1), profile, reference);
        report("preopt-x2", preoptimizedRoot(profile, 2), profile, reference);
    }

    private static void report(String name, Tensor root, ExecutionProfile profile, Reference reference) {
        long compileStart = System.nanoTime();
        CompiledGraph compiled = CompiledGraph.compile(root, profile.optimizer());
        long compileEnd = System.nanoTime();

        long prepareStart = System.nanoTime();
        PreparedExecution prepared = compiled.prepare(profile.runtime());
        long prepareEnd = System.nanoTime();

        Stats stats = benchmark(prepared, profile.mode(), 30, 100, 3);

        List<Tensor> graph = compiled.getCompiledGraphAsList();
        long backwardNodes = graph.stream().filter(Tensor::isBackward).count();
        long fusedNodes = graph.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == Operation.OpType.FUSED)
                .count();
        prepared.execute(profile.mode());
        double outputDiff = Math.abs(root.scalarAsDouble() - reference.output());
        String gradState = gradientState(root, reference.gradientsByLabel());

        System.out.printf(
                java.util.Locale.US,
                "%-12s forwardNodes=%4d compiledNodes=%4d backwardNodes=%4d fusedNodes=%3d compileMs=%8.3f prepareMs=%8.3f medianMs=%8.6f meanMs=%8.6f p90Ms=%8.6f outDiff=%10.3e grad=%s%n",
                name,
                root.forwardOutput().topologicalSort().size(),
                graph.size(),
                backwardNodes,
                fusedNodes,
                (compileEnd - compileStart) / 1_000_000.0d,
                (prepareEnd - prepareStart) / 1_000_000.0d,
                stats.medianMs(),
                stats.meanMs(),
                stats.p90Ms(),
                outputDiff,
                gradState
        );
        if ("preopt-x1".equals(name)) {
            dumpGraph("PREOPT_FORWARD_ROOT", root.topologicalSort());
            dumpGraph("PREOPT_COMPILED_GRAPH", graph);
        }
    }

    private static Reference reference(ExecutionProfile profile) {
        Tensor root = instantiateRoot(profile);
        CompiledGraph.compile(root, profile.optimizer()).prepare(profile.runtime()).execute(profile.mode());
        return new Reference(root.scalarAsDouble(), captureGradients(root, List.of("A", "B", "C")));
    }

    private static Tensor instantiateRoot(ExecutionProfile profile) {
        WorkloadInstance workload = StandardWorkloads
                .abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + profile.dataType().name().toLowerCase())
                .instantiate(new WorkloadEnvironment(profile));
        return workload.root();
    }

    private static Tensor preoptimizedRoot(ExecutionProfile profile, int rounds) {
        Tensor currentRoot = instantiateRoot(profile);
        for (int round = 0; round < rounds; round++) {
            GraphOptimizer optimizer = OptimizerFactory.create(profile.optimizer());
            Tensor forwardAnchor = currentRoot.forwardOutput();
            List<Tensor> optimized = optimizer.optimize(forwardAnchor.topologicalSort());
            currentRoot = extractForwardRoot(optimized);
        }
        return currentRoot;
    }

    private static Tensor extractForwardRoot(List<Tensor> optimizedGraph) {
        for (Tensor tensor : optimizedGraph) {
            if (Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(tensor.getLabel())) {
                List<Tensor> prev = tensor.getPrevTensors();
                if (prev == null || prev.isEmpty()) {
                    throw new IllegalStateException("Optimized forward anchor has no input.");
                }
                return prev.getFirst();
            }
        }
        throw new IllegalStateException("Optimized graph is missing forward anchor.");
    }

    private static Stats benchmark(PreparedExecution prepared, ExecutionMode mode, int warmupIters, int measureIters, int repeats) {
        for (int i = 0; i < warmupIters; i++) {
            prepared.execute(mode);
        }
        double[] samples = new double[repeats];
        for (int r = 0; r < repeats; r++) {
            long start = System.nanoTime();
            for (int i = 0; i < measureIters; i++) {
                prepared.execute(mode);
            }
            long end = System.nanoTime();
            samples[r] = (end - start) / 1_000_000.0d / measureIters;
        }
        Arrays.sort(samples);
        double mean = Arrays.stream(samples).average().orElse(0.0d);
        double median = percentile(samples, 50);
        double p90 = percentile(samples, 90);
        return new Stats(mean, median, p90);
    }

    private static double percentile(double[] sortedValues, int p) {
        if (sortedValues.length == 1) {
            return sortedValues[0];
        }
        double rank = (p / 100.0d) * (sortedValues.length - 1);
        int low = (int) Math.floor(rank);
        int high = (int) Math.ceil(rank);
        if (low == high) {
            return sortedValues[low];
        }
        double w = rank - low;
        return sortedValues[low] * (1.0d - w) + sortedValues[high] * w;
    }

    private static ExecutionProfile loadBestProfile(String dtypeId) {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", dtypeId + "-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-" + dtypeId + "-best-profile.json")
        );
        return new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for " + dtypeId))
                .profile();
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }

    private record Stats(double meanMs, double medianMs, double p90Ms) {
    }

    private record Reference(double output, Map<String, double[]> gradientsByLabel) {
    }

    private static String gradientState(Tensor root, Map<String, double[]> referenceGradients) {
        Map<String, Tensor> tensorsByLabel = tensorsByLabel(root);
        double maxDiff = 0.0d;
        for (Map.Entry<String, double[]> entry : referenceGradients.entrySet()) {
            Tensor tensor = tensorsByLabel.get(entry.getKey());
            if (tensor == null || tensor.getGradient() == null) {
                return "missing:" + entry.getKey();
            }
            double[] actual = tensor.getGradient().toDoubleArrayCopy();
            double[] expected = entry.getValue();
            if (actual.length != expected.length) {
                return "shape:" + entry.getKey();
            }
            for (int i = 0; i < actual.length; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(actual[i] - expected[i]));
            }
        }
        return String.format(java.util.Locale.US, "maxDiff=%.3e", maxDiff);
    }

    private static Map<String, double[]> captureGradients(Tensor root, List<String> labels) {
        Map<String, Tensor> tensorsByLabel = tensorsByLabel(root);
        LinkedHashMap<String, double[]> out = new LinkedHashMap<>();
        for (String label : labels) {
            Tensor tensor = tensorsByLabel.get(label);
            if (tensor == null || tensor.getGradient() == null) {
                throw new IllegalStateException("Missing gradient for reference label " + label);
            }
            out.put(label, tensor.getGradient().toDoubleArrayCopy());
        }
        return Map.copyOf(out);
    }

    private static Map<String, Tensor> tensorsByLabel(Tensor root) {
        LinkedHashMap<String, Tensor> out = new LinkedHashMap<>();
        for (Tensor tensor : root.topologicalSort()) {
            out.putIfAbsent(tensor.getLabel(), tensor);
        }
        return out;
    }

    private static void dumpGraph(String title, List<Tensor> graph) {
        System.out.println(title);
        for (int i = 0; i < graph.size(); i++) {
            Tensor tensor = graph.get(i);
            String op = tensor.getOperation() == null ? "LEAF" : tensor.getOperation().opType().name();
            String expr = tensor.getOperation() == null ? "" : tensor.getOperation().getExpression();
            List<Tensor> prev = tensor.getPrevTensors();
            String inputs = prev == null ? "[]" : prev.stream().map(Tensor::getLabel).toList().toString();
            System.out.printf(
                    java.util.Locale.US,
                    "  [%02d] label=%s op=%s expr=%s backward=%s inputs=%s%n",
                    i,
                    tensor.getLabel(),
                    op,
                    expr,
                    tensor.isBackward(),
                    inputs
            );
        }
    }
}
