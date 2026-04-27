package debug;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadKind;

import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

final class MetalBackwardBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = new tuning.measure.MeasurementPolicy(
            4,
            8,
            3,
            true,
            true,
            true,
            true,
            false
    );

    @Test
    void benchmarkMatmulSumBackwardF32() {
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_BACKWARD_MATMUL_SUM_F32",
                matmulSumWorkload("metal_gpu_backward_matmul_sum_f32", 512, 1024, 1024),
                baselineProfile("metal-cpu-backward-matmul-sum-f32"),
                gpuProfile("metal-backward-matmul-sum-f32-gpu")
        );
    }

    @Test
    void benchmarkLinearBiasTanhSumBackwardF32() {
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_BACKWARD_LINEAR_BIAS_TANH_SUM_F32",
                linearBiasTanhSumWorkload("metal_gpu_backward_linear_bias_tanh_sum_f32", 512, 1024, 1024),
                baselineProfile("metal-cpu-backward-linear-bias-tanh-sum-f32"),
                gpuProfile("metal-backward-linear-bias-tanh-sum-f32-gpu")
        );
    }

    private static void run(
            String label,
            TensorRootWorkloadSpec workload,
            ExecutionProfile baseline,
            ExecutionProfile gpu
    ) {
        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(
                        BenchmarkEntry.baseline("cpu-baseline", baseline),
                        BenchmarkEntry.candidate("gpu-metal", gpu)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(label);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static TensorRootWorkloadSpec matmulSumWorkload(String name, int m, int k, int n) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor a = new Tensor(sequence(m * k), new int[]{m, k}, null, "a", DataType.FLOAT32);
                    Tensor b = new Tensor(sequence(k * n), new int[]{k, n}, null, "b", DataType.FLOAT32);
                    a.setRequiresGrad(true);
                    b.setRequiresGrad(true);
                    Tensor matmul = a.matmul(b);
                    if (isGpuProfile(environment)) {
                        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
                    }
                    return matmul.sum();
                }
        );
    }

    private static TensorRootWorkloadSpec linearBiasTanhSumWorkload(String name, int batch, int in, int outDim) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor input = new Tensor(sequence(batch * in), new int[]{batch, in}, null, "input", DataType.FLOAT32);
                    Tensor weight = new Tensor(sequence(in * outDim), new int[]{in, outDim}, null, "weight", DataType.FLOAT32);
                    Tensor bias = new Tensor(sequence(outDim), new int[]{outDim}, null, "bias", DataType.FLOAT32);
                    input.setRequiresGrad(true);
                    weight.setRequiresGrad(true);
                    bias.setRequiresGrad(true);
                    Tensor linear = input.linear(weight, bias);
                    Tensor out = linear.tanh();
                    if (isGpuProfile(environment)) {
                        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
                        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
                    }
                    return out.sum();
                }
        );
    }

    private static boolean isGpuProfile(WorkloadEnvironment environment) {
        return environment.profile().candidateName().contains("-gpu");
    }

    private static ExecutionProfile baselineProfile(String name) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile gpuProfile(String name) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
    }

    private static float[] sequence(int size) {
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            out[i] = (i % 17) * 0.0625f + 0.125f;
        }
        return out;
    }

    private static void assumeMetalMpsConfigured() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        String envLib = System.getenv("SYNAPTIK_METAL_MPS_LIB");
        assumeTrue((explicitLib != null && !explicitLib.isBlank()) || (envLib != null && !envLib.isBlank()));
    }
}
