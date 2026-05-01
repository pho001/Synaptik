package tuning.benchmark.report;

import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.workload.StandardWorkloads;

import java.util.List;

/**
 * Checked v1.3 hot-path target registry for coverage triage and benchmark-suite construction.
 */
public final class GpuHotPathCoverageTargets {
    private GpuHotPathCoverageTargets() {
    }

    public static List<GpuHotPathCoverageTarget> defaults() {
        return List.of(
                new GpuHotPathCoverageTarget(
                        "transformer_block_hot_path",
                        "transformer",
                        List.of("GPUDAG", "GPUNORM", "GPUFUSEX", "GPUMULTI", "GPUHARDEN"),
                        19,
                        "Exercises multi-op transformer block regions with normalization, epilogues, and handoffs."
                ),
                new GpuHotPathCoverageTarget(
                        "mlp_classifier_small",
                        "mlp",
                        List.of("GPUSTORAGE", "GPUFUSEX", "GPUMULTI", "GPUHARDEN"),
                        18,
                        "Exercises linear, bias, activation, dtype/storage, and fused epilogue coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "conv2d_resnet_3x3",
                        "conv",
                        List.of("GPUNORM", "GPUMULTI", "GPUHARDEN"),
                        17,
                        "Exercises conv-adjacent lowering and longer device-owned region coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "layer_norm_small",
                        "normalization",
                        List.of("GPUSTORAGE", "GPUNORM", "GPUMULTI", "GPUHARDEN"),
                        17,
                        "Exercises normalization, reduction-style, and storage-residency coverage."
                )
        );
    }

    public static List<String> defaultWorkloadNames() {
        return defaults().stream().map(GpuHotPathCoverageTarget::workloadName).toList();
    }

    public static BenchmarkSuiteRequest benchmarkSuite(List<BenchmarkEntry> entries) {
        return StandardWorkloads.benchmarkSuite(defaultWorkloadNames(), entries);
    }
}
