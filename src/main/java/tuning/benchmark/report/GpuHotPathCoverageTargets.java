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
                        "mlp_classifier_small_bf16",
                        "dtype_bf16",
                        List.of("GPUNATIVE", "GPUFUSEX", "GPUCLOSE", "METALBF16", "CUDADTYPE"),
                        30,
                        "Exercises BF16 linear, bias, activation, dtype residency, and fused epilogue coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "conv2d_resnet_3x3",
                        "conv",
                        List.of("GPUNATIVE", "GPUCONVBOOL", "GPUCLOSE", "METALCONVPOOL"),
                        35,
                        "Exercises Conv2D/Conv2D_GEMM lowering and longer device-owned region coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "max_pool2d_small",
                        "pool",
                        List.of("GPUNATIVE", "GPUCONVBOOL", "GPUCLOSE", "METALCONVPOOL"),
                        35,
                        "Exercises max-pool lowering and native Metal evidence."
                ),
                new GpuHotPathCoverageTarget(
                        "avg_pool2d_small",
                        "pool",
                        List.of("GPUNATIVE", "GPUCONVBOOL", "GPUCLOSE", "METALCONVPOOL"),
                        35,
                        "Exercises avg-pool lowering and native Metal evidence."
                ),
                new GpuHotPathCoverageTarget(
                        "layer_norm_small",
                        "normalization",
                        List.of("GPUNATIVE", "GPUNORMX", "GPUCLOSE"),
                        24,
                        "Exercises normalization, reduction-style, and storage-residency coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "layer_norm_small_bf16",
                        "dtype_bf16",
                        List.of("GPUNATIVE", "GPUNORMX", "GPUCLOSE", "METALBF16", "CUDADTYPE"),
                        30,
                        "Exercises BF16 LayerNorm, dtype residency, and normalization tolerance coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "rms_norm_small",
                        "normalization",
                        List.of("GPUNATIVE", "GPUNORMX", "GPUCLOSE"),
                        24,
                        "Exercises RMS normalization and reduction-adjacent coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "rms_norm_small_bf16",
                        "dtype_bf16",
                        List.of("GPUNATIVE", "GPUNORMX", "GPUCLOSE", "METALBF16", "CUDADTYPE"),
                        30,
                        "Exercises BF16 RMSNorm, dtype residency, and normalization tolerance coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "reduction_chain_small_bf16",
                        "dtype_bf16",
                        List.of("GPUNATIVE", "GPURED", "GPUCLOSE", "METALBF16", "CUDADTYPE"),
                        30,
                        "Exercises BF16 SUM, MEAN, REDUCE_MIN, and REDUCE_MAX native coverage."
                ),
                new GpuHotPathCoverageTarget(
                        "dense_loss_small",
                        "loss_dense",
                        List.of("GPUNATIVE", "METALLOSS", "GPUCLOSE"),
                        37,
                        "Exercises dense NLL and dense cross-entropy Metal lowering with native buffer evidence."
                ),
                new GpuHotPathCoverageTarget(
                        "cross_entropy_small",
                        "loss_index",
                        List.of("GPUNATIVE", "GPULOSSIDX", "GPUCLOSE"),
                        26,
                        "Exercises index-target loss and INT32 target residency/fallback evidence."
                ),
                new GpuHotPathCoverageTarget(
                        "training_transformer_block_hot_path",
                        "training_attention",
                        List.of("GPUNATIVE", "GPUSDPA", "GPUCLOSE", "METALTRAIN"),
                        38,
                        "Exercises FORWARD_BACKWARD transformer coverage and keeps unsupported SDPA backward report-visible."
                ),
                new GpuHotPathCoverageTarget(
                        "training_dense_loss_small",
                        "training_loss_dense",
                        List.of("GPUNATIVE", "METALLOSS", "GPUCLOSE", "METALTRAIN"),
                        38,
                        "Exercises dense loss training coverage with gradient publication separated from internal CPU exits."
                ),
                new GpuHotPathCoverageTarget(
                        "training_reduction_chain_small",
                        "training_reduction",
                        List.of("GPUNATIVE", "GPURED", "GPUCLOSE", "METALTRAIN"),
                        38,
                        "Exercises reduction backward coverage and gradient publication gates."
                ),
                new GpuHotPathCoverageTarget(
                        "training_layer_norm_small",
                        "training_normalization",
                        List.of("GPUNATIVE", "GPUNORMX", "GPUCLOSE", "METALTRAIN"),
                        38,
                        "Exercises normalization/reduction training coverage and gradient publication gates."
                ),
                new GpuHotPathCoverageTarget(
                        "training_cross_entropy_small",
                        "training_loss_index",
                        List.of("GPUNATIVE", "GPULOSSIDX", "GPUCLOSE", "METALTRAIN"),
                        38,
                        "Exercises index-target loss gradient blocker visibility in FORWARD_BACKWARD runs."
                ),
                new GpuHotPathCoverageTarget(
                        "bool_compare_where_small",
                        "bool_compare",
                        List.of("GPUNATIVE", "GPUCONVBOOL", "GPUCLOSE", "METALBOOL"),
                        31,
                        "Exercises BOOL compare output feeding WHERE without hiding materialization."
                ),
                new GpuHotPathCoverageTarget(
                        "gather_take_small",
                        "index_gather",
                        List.of("GPUNATIVE", "GPULOSSIDX", "GPUCLOSE", "METALINTIDX", "CUDADTYPE", "CUDAINDEX"),
                        32,
                        "Exercises forward GATHER and TAKE_ALONG_AXIS with INT32 index residency and native Metal execution."
                ),
                new GpuHotPathCoverageTarget(
                        "scatter_index_gradient_small",
                        "index_scatter_gradient",
                        List.of("GPUNATIVE", "GPULOSSIDX", "GPUCLOSE", "METALSCATTER"),
                        36,
                        "Exercises SCATTER_ADD, GATHER_GRAD, and TAKE_ALONG_AXIS_GRAD as separately reported duplicate-index blockers."
                ),
                new GpuHotPathCoverageTarget(
                        "layout_broadcast_repair_small",
                        "layout_repair",
                        List.of("GPUNATIVE", "GPUCLOSE", "METALLAYOUT", "CUDADTYPE"),
                        33,
                        "Exercises zero-stride broadcast view repair through GPU layout materialization."
                ),
                new GpuHotPathCoverageTarget(
                        "masked_sdpa_small",
                        "attention",
                        List.of("GPUNATIVE", "GPUSDPA", "GPUCLOSE", "METALSDPAMASK"),
                        34,
                        "Exercises direct external BOOL masked SDPA staying on Metal."
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
                        "mlp_classifier_small_bf16",
                        resolvedBackend,
                        bf16MlpPolicy(resolvedBackend),
                        List.of(),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "conv2d_resnet_3x3",
                        resolvedBackend,
                        conv2dPolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend) ? List.of() : List.of("unsupported-layout", "CONV", "DAG_CANDIDATE_SHORTENED"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "max_pool2d_small",
                        resolvedBackend,
                        pool2dPolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend) ? List.of() : List.of("POOL", "MAX_POOL2D", "CONV_POOL"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "avg_pool2d_small",
                        resolvedBackend,
                        pool2dPolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend) ? List.of() : List.of("POOL", "AVG_POOL2D", "CONV_POOL"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "layer_norm_small",
                        resolvedBackend,
                        normalizationSupportedPolicy(resolvedBackend),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "layer_norm_small_bf16",
                        resolvedBackend,
                        bf16NormalizationPolicy(resolvedBackend),
                        List.of(),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "rms_norm_small",
                        resolvedBackend,
                        normalizationSupportedPolicy(resolvedBackend),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "rms_norm_small_bf16",
                        resolvedBackend,
                        bf16NormalizationPolicy(resolvedBackend),
                        List.of(),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "reduction_chain_small_bf16",
                        resolvedBackend,
                        bf16ReductionPolicy(resolvedBackend),
                        List.of(),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "dense_loss_small",
                        resolvedBackend,
                        denseLossPolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend)
                                ? List.of()
                                : List.of("DAG_PRIMITIVE_UNSUPPORTED", "NLL_LOSS", "CROSS_ENTROPY_LOSS"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "cross_entropy_small",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("UNSUPPORTED_INDEX_SEMANTICS", "INT32", "CROSS_ENTROPY_LOSS_INDICES"),
                        false
                ),
                new GpuCoverageHotPathExpectation(
                        "training_transformer_block_hot_path",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("SCALED_DOT_PRODUCT_ATTENTION_BACKWARD", "BRIDGE_UNAVAILABLE", "forward SDPA DAG unsupported"),
                        false
                ),
                new GpuCoverageHotPathExpectation(
                        "training_dense_loss_small",
                        resolvedBackend,
                        trainingDenseLossPolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend)
                                ? List.of()
                                : List.of("DAG_PRIMITIVE_UNSUPPORTED", "NLL_LOSS", "CROSS_ENTROPY_LOSS"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "training_reduction_chain_small",
                        resolvedBackend,
                        trainingReductionPolicy(resolvedBackend),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "training_layer_norm_small",
                        resolvedBackend,
                        trainingNormalizationPolicy(resolvedBackend),
                        List.of(),
                        true
                ),
                new GpuCoverageHotPathExpectation(
                        "training_cross_entropy_small",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("UNSUPPORTED_INDEX_SEMANTICS", "CROSS_ENTROPY_LOSS_INDICES_GRAD", "INT32"),
                        false
                ),
                new GpuCoverageHotPathExpectation(
                        "bool_compare_where_small",
                        resolvedBackend,
                        boolCompareWherePolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend) ? List.of() : List.of("BOOL", "COMPARE_BOOL", "GT"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "gather_take_small",
                        resolvedBackend,
                        gatherTakePolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend) ? List.of() : List.of("CAPABILITY_MISSING", "GATHER", "TAKE_ALONG_AXIS"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "scatter_index_gradient_small",
                        resolvedBackend,
                        partialBlockerPolicy(resolvedBackend),
                        List.of("UNSUPPORTED_DUPLICATE_INDEX", "SCATTER_ADD", "GATHER_GRAD", "TAKE_ALONG_AXIS_GRAD"),
                        false
                ),
                new GpuCoverageHotPathExpectation(
                        "layout_broadcast_repair_small",
                        resolvedBackend,
                        layoutRepairPolicy(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend) ? List.of() : List.of("GPU_LAYOUT", "BROADCAST_GPU_MATERIALIZATION"),
                        "GPU_METAL".equals(resolvedBackend)
                ),
                new GpuCoverageHotPathExpectation(
                        "masked_sdpa_small",
                        resolvedBackend,
                        transformerSdpaPolicy(resolvedBackend),
                        transformerSdpaVisibleReasons(resolvedBackend),
                        "GPU_METAL".equals(resolvedBackend)
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

    private static GpuCoverageGatePolicy boolCompareWherePolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return new GpuCoverageGatePolicy(
                backend,
                0.5d,
                4,
                1,
                4,
                0,
                0,
                0,
                0,
                1,
                true
        );
    }

    private static GpuCoverageGatePolicy gatherTakePolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return new GpuCoverageGatePolicy(
                backend,
                0.5d,
                4,
                1,
                4,
                0,
                0,
                0,
                0,
                1,
                true
        );
    }

    private static GpuCoverageGatePolicy conv2dPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return new GpuCoverageGatePolicy(
                backend,
                0.5d,
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

    private static GpuCoverageGatePolicy pool2dPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return new GpuCoverageGatePolicy(
                backend,
                0.5d,
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

    private static GpuCoverageGatePolicy layoutRepairPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return new GpuCoverageGatePolicy(
                backend,
                0.2d,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                true
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

    private static GpuCoverageGatePolicy bf16MlpPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return GpuCoverageGatePolicy.hotPathTarget(backend, 0.5d, 2, 1, 1, 1);
    }

    private static GpuCoverageGatePolicy bf16NormalizationPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return normalizationSupportedPolicy(backend);
    }

    private static GpuCoverageGatePolicy bf16ReductionPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return reductionSupportedPolicy(backend);
    }

    private static GpuCoverageGatePolicy denseLossPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return new GpuCoverageGatePolicy(
                backend,
                0.1d,
                1,
                0,
                3,
                0,
                0,
                0,
                0,
                1,
                true
        );
    }

    private static GpuCoverageGatePolicy trainingDenseLossPolicy(String backend) {
        if (!"GPU_METAL".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return GpuCoverageGatePolicy.trainingHotPathTarget(backend, 0.1d, 1, 0, 3, 0, 8);
    }

    private static GpuCoverageGatePolicy trainingReductionPolicy(String backend) {
        if (!"GPU_METAL".equals(backend) && !"GPU_CUDA".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return GpuCoverageGatePolicy.trainingHotPathTarget(backend, 0.1d, 1, 0, 1, 0, 8);
    }

    private static GpuCoverageGatePolicy trainingNormalizationPolicy(String backend) {
        if (!"GPU_METAL".equals(backend) && !"GPU_CUDA".equals(backend)) {
            return partialBlockerPolicy(backend);
        }
        return GpuCoverageGatePolicy.trainingHotPathTarget(backend, 0.1d, 1, 0, 5, 0, 8);
    }
}
