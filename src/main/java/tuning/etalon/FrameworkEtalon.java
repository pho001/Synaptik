package tuning.etalon;

import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import tensor.DataType;
import tuning.measure.MeasurementPolicy;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkSuiteRequest;
import tuning.session.TuningDefaults;
import tuning.session.TuningPreset;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import java.util.List;

public final class FrameworkEtalon {
    private FrameworkEtalon() {
    }

    public static BenchmarkSuiteRequest inferenceSuite(TuningPreset preset) {
        return TuningDefaults.benchmarkSuite(
                preset == null ? TuningPreset.BALANCED : preset,
                inferenceWorkloads(),
                inferenceCandidates()
        );
    }

    public static BenchmarkSuiteRequest inferenceRegressionSuite() {
        return new BenchmarkSuiteRequest(
                inferenceWorkloads(),
                inferenceCandidates(),
                regressionMeasurement(),
                TuningPreset.BALANCED.benchmarkValidation(),
                TuningPreset.BALANCED.reportPolicy()
        );
    }

    public static BenchmarkSuiteRequest trainingSuite(TuningPreset preset) {
        return TuningDefaults.benchmarkSuite(
                preset == null ? TuningPreset.BALANCED : preset,
                trainingWorkloads(),
                trainingCandidates()
        );
    }

    public static List<WorkloadSpec> inferenceWorkloads() {
        return List.of(
                StandardWorkloads.matmul("etalon_matmul_small", 1, 64, 64, 64),
                StandardWorkloads.abcSequenceMatmul("etalon_abc_sequence_small", 64, 256),
                StandardWorkloads.conv2d(
                        "etalon_conv2d_resnet_3x3",
                        2, 64, 128, 56, 56, 3, 3,
                        tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                        true
                ),
                StandardWorkloads.normalization(
                        "etalon_layer_norm_small",
                        tuning.workload.NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM,
                        4, 64, 8, 1, 1e-5
                ),
                StandardWorkloads.pool2d(
                        "etalon_max_pool2d_small",
                        tuning.workload.Pool2dWorkloadSpec.PoolKind.MAX,
                        2, 8, 16, 16,
                        tensor.options.Pool2dOptions.square(2)
                ),
                StandardWorkloads.transformerHotPath("etalon_transformer_hot_path")
        );
    }

    public static List<WorkloadSpec> trainingWorkloads() {
        return List.of(
                StandardWorkloads.matmul("etalon_train_matmul_small", 1, 64, 64, 64),
                StandardWorkloads.abcSequenceMatmul("etalon_train_abc_sequence_small", 64, 256),
                StandardWorkloads.mlpClassification(
                        "etalon_train_mlp_small",
                        16, 32, 48, 24, 6,
                        tensor.loss.LossReduction.MEAN
                ),
                StandardWorkloads.mlpClassification(
                        "etalon_train_mlp_heavy",
                        64, 256, 512, 256, 32,
                        tensor.loss.LossReduction.MEAN
                ),
                StandardWorkloads.indexedLoss(
                        "etalon_train_cross_entropy",
                        tuning.workload.LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES,
                        8, 16,
                        tensor.loss.LossReduction.MEAN
                )
        );
    }

    public static List<BenchmarkEntry> inferenceCandidates() {
        return List.of(
                entry("f64_infer_default", DataType.FLOAT64, ExecutionMode.FORWARD, OptimizerConfig.inferenceDefaults(), RuntimeConfig.inferenceDefaults()),
                entry("f32_infer_default", DataType.FLOAT32, ExecutionMode.FORWARD, OptimizerConfig.inferenceDefaults(), RuntimeConfig.inferenceDefaults()),
                entry("bf16_infer_default", DataType.BFLOAT16, ExecutionMode.FORWARD, OptimizerConfig.inferenceDefaults(), RuntimeConfig.inferenceDefaults()),
                entry("f64_infer_no_fuse", DataType.FLOAT64, ExecutionMode.FORWARD, OptimizerConfig.inferenceDefaults().withStageOrder(List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.MEM)), RuntimeConfig.inferenceDefaults()),
                entry("f32_infer_no_fuse", DataType.FLOAT32, ExecutionMode.FORWARD, OptimizerConfig.inferenceDefaults().withStageOrder(List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.MEM)), RuntimeConfig.inferenceDefaults()),
                entry("f64_infer_blas", DataType.FLOAT64, ExecutionMode.FORWARD, OptimizerConfig.inferenceDefaults(), withRuntime(RuntimeConfig.inferenceDefaults(), 100000, BlasProvider.OPENBLAS_FFM, 1_000_000L, 0)),
                entry("f32_infer_blas", DataType.FLOAT32, ExecutionMode.FORWARD, OptimizerConfig.inferenceDefaults(), withRuntime(RuntimeConfig.inferenceDefaults(), 100000, BlasProvider.OPENBLAS_FFM, 1_000_000L, 0))
        );
    }

    public static List<BenchmarkEntry> trainingCandidates() {
        return List.of(
                entry("f64_train_default", DataType.FLOAT64, ExecutionMode.FORWARD_BACKWARD, OptimizerConfig.trainingDefaults(), RuntimeConfig.trainingDefaults()),
                entry("f32_train_default", DataType.FLOAT32, ExecutionMode.FORWARD_BACKWARD, OptimizerConfig.trainingDefaults(), RuntimeConfig.trainingDefaults()),
                entry("bf16_train_default", DataType.BFLOAT16, ExecutionMode.FORWARD_BACKWARD, OptimizerConfig.trainingDefaults(), RuntimeConfig.trainingDefaults()),
                entry("f64_train_fuse_mem", DataType.FLOAT64, ExecutionMode.FORWARD_BACKWARD, OptimizerConfig.trainingDefaults().withStageOrder(List.of(OptimizerStage.FUSE, OptimizerStage.MEM)), RuntimeConfig.trainingDefaults()),
                entry("f32_train_fuse_mem", DataType.FLOAT32, ExecutionMode.FORWARD_BACKWARD, OptimizerConfig.trainingDefaults().withStageOrder(List.of(OptimizerStage.FUSE, OptimizerStage.MEM)), RuntimeConfig.trainingDefaults()),
                entry("f64_train_blas", DataType.FLOAT64, ExecutionMode.FORWARD_BACKWARD, OptimizerConfig.trainingDefaults(), withRuntime(RuntimeConfig.trainingDefaults(), 100000, BlasProvider.OPENBLAS_FFM, 1_000_000L, 0)),
                entry("f32_train_blas", DataType.FLOAT32, ExecutionMode.FORWARD_BACKWARD, OptimizerConfig.trainingDefaults(), withRuntime(RuntimeConfig.trainingDefaults(), 100000, BlasProvider.OPENBLAS_FFM, 1_000_000L, 0))
        );
    }

    private static BenchmarkEntry entry(
            String name,
            DataType dataType,
            ExecutionMode mode,
            OptimizerConfig optimizer,
            RuntimeConfig runtime
    ) {
        return BenchmarkEntry.candidate(
                name,
                new ExecutionProfile(name, name, dataType, mode, optimizer, runtime, WorkloadProfile.none())
        );
    }

    private static RuntimeConfig withRuntime(
            RuntimeConfig base,
            int matmulParallelMin,
            BlasProvider provider,
            long blasMinWork,
            int threads
    ) {
        CpuKernelConfig cpu = base.kernel().cpu();
        CpuKernelConfig tunedCpu = new CpuKernelConfig(
                cpu.loopUnrollFactor(),
                cpu.matMulTileM(),
                cpu.matMulTileN(),
                cpu.matMulTileK(),
                cpu.cheapVectorMinSize(),
                cpu.transcendentalVectorMinSize(),
                cpu.fusedCheapVectorMinSize(),
                cpu.fusedTranscendentalVectorMinSize(),
                cpu.reductionVectorMinSize(),
                cpu.attentionVectorMinSize(),
                cpu.cheapParallelMinSize(),
                cpu.transcendentalParallelMinSize(),
                cpu.fusedCheapParallelMinSize(),
                cpu.fusedTranscendentalParallelMinSize(),
                cpu.reductionParallelMinSize(),
                cpu.attentionParallelMinSize(),
                cpu.contiguousMaterializeThreshold(),
                cpu.cheapF64MaterializeThreshold(),
                cpu.cheapF32MaterializeThreshold(),
                cpu.cheapBF16MaterializeThreshold(),
                cpu.whereMaterializeThreshold(),
                cpu.lowCostTargetChunksPerWorker(),
                cpu.mediumCostTargetChunksPerWorker(),
                cpu.highCostTargetChunksPerWorker(),
                cpu.minScalarChunkSize(),
                cpu.minVectorChunkSize(),
                cpu.minReductionChunkSize(),
                cpu.commonPoolLowCostMaxWorkPerWorker(),
                cpu.fusedCheapContiguousAsmVectorWidth(),
                cpu.fusedCheapStridedAsmVectorWidth(),
                cpu.fusedNonCheapContiguousAsmVectorWidth(),
                cpu.fusedNonCheapStridedAsmVectorWidth(),
                cpu.sumAccuracyMode(),
                matmulParallelMin,
                cpu.attentionMatMulPolicy(),
                cpu.matMulMicroKernel(),
                cpu.attentionMatMulMicroKernel(),
                cpu.attentionMatMulTileM(),
                cpu.attentionMatMulTileN(),
                cpu.attentionMatMulTileK()
        );
        return new RuntimeConfig(
                new KernelTuningConfig(tunedCpu, base.kernel().cuda(), base.kernel().opencl()),
                base.approximation(),
                new BlasConfig(
                        provider,
                        blasMinWork,
                        base.blas().f32RequireMgeK(),
                        base.blas().f32MaxNOverK(),
                        base.blas().debug(),
                        threads
                )
        );
    }

    private static MeasurementPolicy regressionMeasurement() {
        return new MeasurementPolicy(8, 8, 3, true, true, true, true, false);
    }
}
