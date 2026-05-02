package tuning.workload;

import config.profile.WorkloadProfile;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateSpace;
import tuning.autotune.AutotuneRequest;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.autotune.TuningDefaults;
import tuning.preset.TuningPreset;
import tensor.loss.LossReduction;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.List;

public final class StandardWorkloads {
    private static final int ABC_BLAS_BENCHMARK_BATCH = 256;
    private static final int ABC_BLAS_BENCHMARK_FEATURES = 2_048;

    private StandardWorkloads() {
    }

    public static MatMulWorkloadSpec matmul(String name, int batch, int m, int k, int n) {
        return new MatMulWorkloadSpec(name, batch, m, k, n);
    }

    public static Conv2dWorkloadSpec conv2d(
            String name,
            int batch,
            int inChannels,
            int outChannels,
            int height,
            int width,
            int kernelH,
            int kernelW,
            Conv2dOptions options,
            boolean withBias
    ) {
        return new Conv2dWorkloadSpec(name, batch, inChannels, outChannels, height, width, kernelH, kernelW, options, withBias);
    }

    public static TransformerHotPathWorkloadSpec transformerHotPath(String name) {
        return new TransformerHotPathWorkloadSpec(name);
    }

    /**
     * Creates a transformer hot-path workload with a default shape profile.
     *
     * @param name workload name used in reports and persistence
     * @param profile default transformer shape used when candidate profiles do not carry workload metadata
     * @return workload specification
     */
    public static TransformerHotPathWorkloadSpec transformerHotPath(String name, WorkloadProfile profile) {
        return new TransformerHotPathWorkloadSpec(name, profile);
    }

    public static TransformerBlockHotPathWorkloadSpec transformerBlockHotPath(String name) {
        return new TransformerBlockHotPathWorkloadSpec(name);
    }

    /**
     * Creates a transformer-block workload with a default shape profile.
     *
     * @param name workload name used in reports and persistence
     * @param profile default transformer shape used when candidate profiles do not carry workload metadata
     * @return workload specification
     */
    public static TransformerBlockHotPathWorkloadSpec transformerBlockHotPath(String name, WorkloadProfile profile) {
        return new TransformerBlockHotPathWorkloadSpec(name, profile);
    }

    public static AbcSequenceMatmulWorkloadSpec abcSequenceMatmul(String name, int batch, int features) {
        return new AbcSequenceMatmulWorkloadSpec(name, batch, features);
    }

    public static AbcSequenceMatmulWorkloadSpec abcSequenceMatmulBlasBenchmark(String name) {
        return abcSequenceMatmul(name, ABC_BLAS_BENCHMARK_BATCH, ABC_BLAS_BENCHMARK_FEATURES);
    }

    public static MlpClassificationWorkloadSpec mlpClassification(
            String name,
            int batch,
            int inputFeatures,
            int hidden1,
            int hidden2,
            int classes,
            LossReduction reduction
    ) {
        return new MlpClassificationWorkloadSpec(name, batch, inputFeatures, hidden1, hidden2, classes, reduction);
    }

    public static NormalizationWorkloadSpec normalization(
            String name,
            NormalizationWorkloadSpec.NormalizationKind kind,
            int batch,
            int channels,
            int height,
            int width,
            double epsilon
    ) {
        return new NormalizationWorkloadSpec(name, kind, batch, channels, height, width, epsilon);
    }

    public static ReductionWorkloadSpec reductionChain(String name, int batch, int features) {
        return new ReductionWorkloadSpec(name, batch, features);
    }

    public static BoolCompareMaskWorkloadSpec boolCompareWhere(String name, int batch, int features) {
        return new BoolCompareMaskWorkloadSpec(name, batch, features);
    }

    public static GatherTakeWorkloadSpec gatherTake(String name, int batch, int features, int picks) {
        return new GatherTakeWorkloadSpec(name, batch, features, picks);
    }

    public static ScatterIndexGradientWorkloadSpec scatterIndexGradient(String name, int batch, int features, int picks) {
        return new ScatterIndexGradientWorkloadSpec(name, batch, features, picks);
    }

    public static LayoutRepairWorkloadSpec layoutBroadcastRepair(String name, int batch, int features) {
        return new LayoutRepairWorkloadSpec(name, batch, features);
    }

    public static MaskedSdpaWorkloadSpec maskedSdpa(String name, int batch, int heads, int tokens, int headDim, int valueDim) {
        return new MaskedSdpaWorkloadSpec(name, batch, heads, tokens, headDim, valueDim);
    }

    public static Pool2dWorkloadSpec pool2d(
            String name,
            Pool2dWorkloadSpec.PoolKind kind,
            int batch,
            int channels,
            int height,
            int width,
            Pool2dOptions options
    ) {
        return new Pool2dWorkloadSpec(name, kind, batch, channels, height, width, options);
    }

    public static LossWorkloadSpec indexedLoss(
            String name,
            LossWorkloadSpec.LossKind kind,
            int batch,
            int classes,
            LossReduction reduction
    ) {
        return new LossWorkloadSpec(name, kind, batch, classes, reduction);
    }

    public static LossWorkloadSpec denseLoss(
            String name,
            int batch,
            int classes,
            LossReduction reduction
    ) {
        return new LossWorkloadSpec(
                name,
                LossWorkloadSpec.LossKind.DENSE_CROSS_ENTROPY_AND_NLL,
                batch,
                classes,
                reduction
        );
    }

    public static WorkloadCatalog defaultCatalog() {
        return new WorkloadCatalog()
                .register(matmul("matmul_small", 1, 64, 64, 64))
                .register(matmul("matmul_batched_attention_like", 8, 128, 64, 64))
                .register(abcSequenceMatmul(
                        "abc_sequence_matmul_small",
                        64, 256
                ))
                .register(conv2d(
                        "conv2d_resnet_3x3",
                        2, 64, 128, 56, 56, 3, 3,
                        new Conv2dOptions(1, 1, 1, 1, 1, 1, 1),
                        true
                ))
                .register(mlpClassification(
                        "mlp_classifier_small",
                        16, 32, 48, 24, 6,
                        LossReduction.MEAN
                ))
                .register(mlpClassification(
                        "mlp_classifier_small_bf16",
                        16, 32, 48, 24, 6,
                        LossReduction.MEAN
                ))
                .register(mlpClassification(
                        "mlp_classifier_blas_heavy",
                        64, 256, 512, 256, 32,
                        LossReduction.MEAN
                ))
                .register(reductionChain("reduction_chain_small", 8, 32))
                .register(reductionChain("reduction_chain_small_bf16", 8, 32))
                .register(normalization("layer_norm_small", NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM, 4, 64, 8, 1, 1e-5))
                .register(normalization("layer_norm_small_bf16", NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM, 4, 64, 8, 1, 1e-5))
                .register(normalization("rms_norm_small", NormalizationWorkloadSpec.NormalizationKind.RMS_NORM, 4, 64, 8, 1, 1e-5))
                .register(normalization("rms_norm_small_bf16", NormalizationWorkloadSpec.NormalizationKind.RMS_NORM, 4, 64, 8, 1, 1e-5))
                .register(pool2d("max_pool2d_small", Pool2dWorkloadSpec.PoolKind.MAX, 2, 8, 16, 16, Pool2dOptions.square(2)))
                .register(pool2d("avg_pool2d_small", Pool2dWorkloadSpec.PoolKind.AVG, 2, 8, 16, 16, Pool2dOptions.square(2)))
                .register(denseLoss("dense_loss_small", 8, 16, LossReduction.MEAN))
                .register(indexedLoss("cross_entropy_small", LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES, 8, 16, LossReduction.MEAN))
                .register(boolCompareWhere("bool_compare_where_small", 8, 32))
                .register(gatherTake("gather_take_small", 8, 16, 4))
                .register(scatterIndexGradient("scatter_index_gradient_small", 8, 16, 4))
                .register(layoutBroadcastRepair("layout_broadcast_repair_small", 8, 32))
                .register(maskedSdpa("masked_sdpa_small", 2, 2, 8, 8, 8))
                .register(transformerHotPath("transformer_hot_path"))
                .register(transformerHotPath("transformer_hot_path_large", WorkloadProfile.transformerHotPathLarge()))
                .register(transformerHotPath("transformer_hot_path_long_seq", WorkloadProfile.transformerHotPathLongSeq()))
                .register(transformerHotPath("transformer_hot_path_ffn_heavy", WorkloadProfile.transformerHotPathFfnHeavy()))
                .register(transformerHotPath("transformer_hot_path_attention_heavy", WorkloadProfile.transformerHotPathAttentionHeavy()))
                .register(transformerBlockHotPath("transformer_block_hot_path"))
                .register(transformerBlockHotPath("transformer_block_hot_path_large", WorkloadProfile.transformerHotPathLarge()))
                .register(transformerBlockHotPath("transformer_block_hot_path_long_seq", WorkloadProfile.transformerHotPathLongSeq()))
                .register(transformerBlockHotPath("transformer_block_hot_path_ffn_heavy", WorkloadProfile.transformerHotPathFfnHeavy()))
                .register(transformerBlockHotPath("transformer_block_hot_path_attention_heavy", WorkloadProfile.transformerHotPathAttentionHeavy()));
    }

    public static BenchmarkRequest benchmark(String workloadName, List<BenchmarkEntry> entries, TuningPreset preset) {
        return defaultCatalog().benchmarkRequest(workloadName, entries, preset);
    }

    public static BenchmarkRequest benchmark(String workloadName, List<BenchmarkEntry> entries) {
        return defaultCatalog().benchmarkRequest(workloadName, entries);
    }

    public static BenchmarkSuiteRequest benchmarkSuite(List<String> workloadNames, List<BenchmarkEntry> entries, TuningPreset preset) {
        return defaultCatalog().benchmarkSuiteRequest(workloadNames, entries, preset);
    }

    public static BenchmarkSuiteRequest benchmarkSuite(List<String> workloadNames, List<BenchmarkEntry> entries) {
        return defaultCatalog().benchmarkSuiteRequest(workloadNames, entries);
    }

    public static AutotuneRequest autotune(
            String workloadName,
            config.profile.ExecutionProfile seedProfile,
            CandidateSpace candidateSpace,
            TuningPreset preset,
            tuning.store.PersistencePolicy persistence
    ) {
        return TuningDefaults.autotune(
                preset,
                defaultCatalog().require(workloadName),
                seedProfile,
                candidateSpace,
                persistence
        );
    }

    public static AutotuneRequest autotune(
            String workloadName,
            config.profile.ExecutionProfile seedProfile,
            CandidateSpace candidateSpace,
            tuning.store.PersistencePolicy persistence
    ) {
        return TuningDefaults.recommendedAutotune(
                defaultCatalog().require(workloadName),
                seedProfile,
                candidateSpace,
                persistence
        );
    }

    public static WorkloadProfile transformerHotPathDefaults() {
        return WorkloadProfile.transformerHotPathDefaults();
    }
}
