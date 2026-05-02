package tuning.benchmark.report;

import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.workload.StandardWorkloads;

import java.util.List;

/**
 * Checked hot-path target registry for coverage triage and benchmark-suite construction.
 */
public final class GpuHotPathCoverageTargets {
    private GpuHotPathCoverageTargets() {
    }

    public static List<GpuHotPathCoverageTarget> defaults() {
        return List.of(
                new GpuHotPathCoverageTarget(
                        "reduction_chain_small",
                        "reduction",
                        List.of("GPUNATIVE", "GPURED", "GPUCLOSE"),
                        23,
                        "Exercises SUM, MEAN, REDUCE_MIN, and REDUCE_MAX fallback closure."
                ),
                new GpuHotPathCoverageTarget(
                        "transformer_block_hot_path",
                        "transformer",
                        List.of("GPUNATIVE", "GPUSDPA", "GPUNORMX", "GPUCLOSE"),
                        25,
                        "Exercises forward SDPA, normalization-adjacent reductions, epilogues, and handoffs."
                ),
                new GpuHotPathCoverageTarget(
                        "mlp_classifier_small",
                        "mlp",
                        List.of("GPUNATIVE", "GPUFUSEX", "GPUCLOSE"),
                        28,
                        "Exercises linear, bias, activation, dtype/storage, and fused epilogue coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "conv2d_resnet_3x3",
                        "conv",
                        List.of("GPUNATIVE", "GPUCONVBOOL", "GPUCLOSE"),
                        27,
                        "Exercises conv lowering and longer device-owned region coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "max_pool2d_small",
                        "pool",
                        List.of("GPUNATIVE", "GPUCONVBOOL", "GPUCLOSE"),
                        27,
                        "Exercises pool lowering and explicit fallback evidence."
                ),
                new GpuHotPathCoverageTarget(
                        "layer_norm_small",
                        "normalization",
                        List.of("GPUNATIVE", "GPUNORMX", "GPUCLOSE"),
                        24,
                        "Exercises normalization, reduction-style, and storage-residency coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "rms_norm_small",
                        "normalization",
                        List.of("GPUNATIVE", "GPUNORMX", "GPUCLOSE"),
                        24,
                        "Exercises RMS normalization and reduction-adjacent coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "cross_entropy_small",
                        "loss_index",
                        List.of("GPUNATIVE", "GPULOSSIDX", "GPUCLOSE"),
                        26,
                        "Exercises index-target loss and INT32 target residency/fallback evidence."
                ),
                new GpuHotPathCoverageTarget(
                        "bool_compare_where_small",
                        "bool_compare",
                        List.of("GPUNATIVE", "GPUCONVBOOL", "GPUCLOSE"),
                        27,
                        "Exercises BOOL compare output feeding WHERE without hiding materialization."
                )
        );
    }

    public static List<String> defaultWorkloadNames() {
        return defaults().stream().map(GpuHotPathCoverageTarget::workloadName).toList();
    }

    public static List<GpuCoverageHotPathExpectation> defaultExpectations() {
        return expectationsForBackend("GPU_METAL");
    }

    public static List<GpuCoverageHotPathExpectation> expectationsForBackend(String backend) {
        String resolvedBackend = backend == null || backend.isBlank() ? "GPU_METAL" : backend;
        return List.of(
                new GpuCoverageHotPathExpectation(
                        "reduction_chain_small",
                        resolvedBackend,
                        reductionSupportedPolicy(resolvedBackend),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "transformer_block_hot_path",
                        resolvedBackend,
                        transformerSdpaPolicy(resolvedBackend),
                        transformerSdpaVisibleReasons(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "mlp_classifier_small",
                        resolvedBackend,
                        GpuCoverageGatePolicy.hotPathTarget(resolvedBackend, 0.5d, 2, 1, 1, 1),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "conv2d_resnet_3x3",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("unsupported-layout", "CONV", "DAG_CANDIDATE_SHORTENED"),
                        false
                ),
                new GpuCoverageHotPathExpectation(
                        "max_pool2d_small",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("POOL", "MAX_POOL2D", "CONV_POOL"),
                        false
                ),
                new GpuCoverageHotPathExpectation(
                        "layer_norm_small",
                        resolvedBackend,
                        normalizationSupportedPolicy(resolvedBackend),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "rms_norm_small",
                        resolvedBackend,
                        normalizationSupportedPolicy(resolvedBackend),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "cross_entropy_small",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("LOSS", "INT32", "CROSS_ENTROPY_LOSS_INDICES"),
                        false
                ),
                new GpuCoverageHotPathExpectation(
                        "bool_compare_where_small",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("BOOL", "COMPARE_BOOL", "GT"),
                        false
                )
        );
    }

    public static BenchmarkSuiteRequest benchmarkSuite(List<BenchmarkEntry> entries) {
        return StandardWorkloads.benchmarkSuite(defaultWorkloadNames(), entries);
    }

    private static GpuCoverageGatePolicy partialBlockerPolicy(String backend) {
        return new GpuCoverageGatePolicy(
                backend,
                0.0d,
                0,
                0,
                0,
                0,
                Integer.MAX_VALUE,
                Integer.MAX_VALUE,
                Integer.MAX_VALUE,
                Integer.MAX_VALUE,
                false
        );
    }

    private static GpuCoverageGatePolicy transformerSdpaPolicy(String backend) {
        if ("GPU_METAL".equals(backend)) {
            return new GpuCoverageGatePolicy(
                    backend,
                    0.1d,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    true
            );
        }
        return partialBlockerPolicy(backend);
    }

    private static GpuCoverageGatePolicy reductionSupportedPolicy(String backend) {
        return new GpuCoverageGatePolicy(
                backend,
                0.1d,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                true
        );
    }

    private static List<String> transformerSdpaVisibleReasons(String backend) {
        if ("GPU_CUDA".equals(backend)) {
            return List.of("CAPABILITY_MISSING", "SCALED_DOT_PRODUCT_ATTENTION", "transformer_block_hot_path");
        }
        if ("GPU_METAL".equals(backend)) {
            return List.of();
        }
        return List.of("ATTENTION", "SDPA", "SCALED_DOT_PRODUCT_ATTENTION");
    }

    private static GpuCoverageGatePolicy normalizationSupportedPolicy(String backend) {
        return new GpuCoverageGatePolicy(
                backend,
                1.0d,
                1,
                0,
                5,
                1,
                0,
                0,
                0,
                1,
                true
        );
    }
}
