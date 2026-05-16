package tuning.benchmark;

import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.ApproximationConfig;
import config.runtime.BFloat16TrainingPolicy;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import tensor.DataType;
import tensor.loss.LossReduction;
import tuning.measure.MeasurementPolicy;
import tuning.reporting.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;

import java.util.List;

/**
 * Factory for the Wave 8 BF16 performance truth benchmark.
 */
public final class Bf16PerformanceBenchmark {
    public static final String MLP_WORKLOAD_NAME = "bf16_performance_mlp";
    public static final String MATMUL_WORKLOAD_NAME = "bf16_performance_matmul";
    public static final String TRAINING_WORKLOAD_NAME = "bf16_performance_training";

    public static final String F32_MLP_BASELINE = "f32-mlp-baseline";
    public static final String BF16_PROMOTED_MLP = "bf16-promoted-mlp";
    public static final String BF16_SBGEMM_CONTINUATION = "bf16-sbgemm-continuation";
    public static final String BF16_BGEMM_OUTPUT = "bf16-bgemm-output";
    public static final String BF16_TRAINING_POLICY = "bf16-training-policy";

    private Bf16PerformanceBenchmark() {
    }

    /**
     * Creates a compact suite that exercises BF16 MLP, matmul, and training policy evidence.
     *
     * @return benchmark suite request
     */
    public static BenchmarkSuiteRequest suite() {
        return suite(measurementPolicy());
    }

    /**
     * Creates a compact suite with caller-selected measurement policy.
     *
     * @param measurement measurement policy; {@code null} uses the BF16 benchmark default
     * @return benchmark suite request
     */
    public static BenchmarkSuiteRequest suite(MeasurementPolicy measurement) {
        return new BenchmarkSuiteRequest(
                List.of(
                        StandardWorkloads.mlpClassification(
                                MLP_WORKLOAD_NAME,
                                32,
                                128,
                                256,
                                128,
                                16,
                                LossReduction.MEAN
                        ),
                        StandardWorkloads.matmul(MATMUL_WORKLOAD_NAME, 1, 128, 128, 128),
                        StandardWorkloads.mlpClassification(
                                TRAINING_WORKLOAD_NAME,
                                16,
                                64,
                                128,
                                64,
                                8,
                                LossReduction.MEAN
                        )
                ),
                entries(),
                measurement == null ? measurementPolicy() : measurement,
                ValidationPolicy.disabled(),
                ReportPolicy.defaults()
        );
    }

    /**
     * Creates the default MLP benchmark request for quick local evidence checks.
     *
     * @return benchmark request
     */
    public static BenchmarkRequest request() {
        return request(measurementPolicy());
    }

    /**
     * Creates the default MLP benchmark request with caller-selected measurement policy.
     *
     * @param measurement measurement policy; {@code null} uses the BF16 benchmark default
     * @return benchmark request
     */
    public static BenchmarkRequest request(MeasurementPolicy measurement) {
        return new BenchmarkRequest(
                StandardWorkloads.mlpClassification(
                        MLP_WORKLOAD_NAME,
                        32,
                        128,
                        256,
                        128,
                        16,
                        LossReduction.MEAN
                ),
                entries(),
                measurement == null ? measurementPolicy() : measurement,
                ValidationPolicy.disabled(),
                ReportPolicy.defaults()
        );
    }

    /**
     * Returns the named profiles compared by the BF16 evidence benchmark.
     *
     * @return benchmark entries
     */
    public static List<BenchmarkEntry> entries() {
        return List.of(
                BenchmarkEntry.baseline(F32_MLP_BASELINE, profile(
                        F32_MLP_BASELINE,
                        DataType.FLOAT32,
                        ExecutionMode.FORWARD,
                        CompileConfig.inference(),
                        openBlasRuntime(DataType.FLOAT32)
                )),
                BenchmarkEntry.candidate(BF16_PROMOTED_MLP, profile(
                        BF16_PROMOTED_MLP,
                        DataType.BFLOAT16,
                        ExecutionMode.FORWARD,
                        CompileConfig.inference(),
                        RuntimeConfig.inferenceDefaults()
                )),
                BenchmarkEntry.candidate(BF16_SBGEMM_CONTINUATION, profile(
                        BF16_SBGEMM_CONTINUATION,
                        DataType.BFLOAT16,
                        ExecutionMode.FORWARD,
                        CompileConfig.inference(),
                        openBlasRuntime(DataType.BFLOAT16)
                )),
                BenchmarkEntry.candidate(BF16_BGEMM_OUTPUT, profile(
                        BF16_BGEMM_OUTPUT,
                        DataType.BFLOAT16,
                        ExecutionMode.FORWARD,
                        CompileConfig.inference(),
                        openBlasRuntime(DataType.BFLOAT16)
                )),
                BenchmarkEntry.candidate(BF16_TRAINING_POLICY, profile(
                        BF16_TRAINING_POLICY,
                        DataType.BFLOAT16,
                        ExecutionMode.FORWARD_BACKWARD,
                        CompileConfig.training(),
                        RuntimeConfig.trainingDefaults()
                                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
                                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY)
                                .withBFloat16TrainingPolicy(BFloat16TrainingPolicy.ACTIVATIONS_ONLY)
                ))
        );
    }

    /**
     * Uses a short steady-state loop while preserving cold-run trace evidence.
     *
     * @return measurement policy
     */
    public static MeasurementPolicy measurementPolicy() {
        return new MeasurementPolicy(2, 5, 3, true, true, true, true, true);
    }

    private static ExecutionProfile profile(
            String name,
            DataType dataType,
            ExecutionMode mode,
            CompileConfig compile,
            RuntimeConfig runtime
    ) {
        return new ExecutionProfile(name, name, dataType, mode, compile, runtime, WorkloadProfile.none());
    }

    private static RuntimeConfig openBlasRuntime(DataType dataType) {
        RuntimeConfig base = dataType == DataType.BFLOAT16
                ? RuntimeConfig.inferenceDefaults(DataType.BFLOAT16)
                : RuntimeConfig.inferenceDefaults(DataType.FLOAT32);
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        100.0d,
                        BlasStorageMode.AUTO,
                        false,
                        1
                ),
                base.conv2d(),
                base.fused(),
                base.accelerator(),
                dataType == DataType.BFLOAT16 ? CpuStorageProfile.AUTO : base.cpuStorageProfile(),
                NativeCpuFailurePolicy.FALLBACK_TO_ARRAY,
                base.deviceTransferPolicy(),
                base.nativeCpuMemory()
        );
    }
}
