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

final class AppleGpuMatMulBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = new tuning.measure.MeasurementPolicy(
            8,
            16,
            3,
            true,
            true,
            true,
            true,
            false
    );

    @Test
    void benchmarkMatmulF32() {
        assumeAppleShimConfigured();
        run(
                "APPLE_GPU_MATMUL_F32",
                matmulWorkload("apple_gpu_matmul_f32", 256, 512, 512),
                baselineProfile("apple-cpu-matmul-f32"),
                gpuProfile("apple-metal-matmul-f32-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddReluF32() {
        assumeAppleShimConfigured();
        run(
                "APPLE_GPU_MATMUL_ADD_RELU_F32",
                matmulAddReluWorkload("apple_gpu_matmul_add_relu_f32", 256, 512, 512),
                baselineProfile("apple-cpu-matmul-add-relu-f32"),
                gpuProfile("apple-metal-matmul-add-relu-f32-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddTanhF32() {
        assumeAppleShimConfigured();
        run(
                "APPLE_GPU_MATMUL_ADD_TANH_F32",
                matmulAddTanhWorkload("apple_gpu_matmul_add_tanh_f32", 256, 512, 512),
                baselineProfile("apple-cpu-matmul-add-tanh-f32"),
                gpuProfile("apple-metal-matmul-add-tanh-f32-gpu")
        );
    }

    @Test
    void benchmarkMatmulLargeF32() {
        assumeAppleShimConfigured();
        run(
                "APPLE_GPU_MATMUL_F32_LARGE",
                matmulWorkload("apple_gpu_matmul_f32_large", 512, 1024, 1024),
                baselineProfile("apple-cpu-matmul-f32-large"),
                gpuProfile("apple-metal-matmul-f32-large-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddReluLargeF32() {
        assumeAppleShimConfigured();
        run(
                "APPLE_GPU_MATMUL_ADD_RELU_F32_LARGE",
                matmulAddReluWorkload("apple_gpu_matmul_add_relu_f32_large", 512, 1024, 1024),
                baselineProfile("apple-cpu-matmul-add-relu-f32-large"),
                gpuProfile("apple-metal-matmul-add-relu-f32-large-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddTanhLargeF32() {
        assumeAppleShimConfigured();
        run(
                "APPLE_GPU_MATMUL_ADD_TANH_F32_LARGE",
                matmulAddTanhWorkload("apple_gpu_matmul_add_tanh_f32_large", 512, 1024, 1024),
                baselineProfile("apple-cpu-matmul-add-tanh-f32-large"),
                gpuProfile("apple-metal-matmul-add-tanh-f32-large-gpu")
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

    private static TensorRootWorkloadSpec matmulWorkload(String name, int m, int k, int n) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor a = new Tensor(sequence(m * k), new int[]{m, k}, null, "a", DataType.FLOAT32);
                    Tensor b = new Tensor(sequence(k * n), new int[]{k, n}, null, "b", DataType.FLOAT32);
                    Tensor out = a.matmul(b);
                    if (isGpuProfile(environment)) {
                        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
                    }
                    return out;
                }
        );
    }

    private static TensorRootWorkloadSpec matmulAddReluWorkload(String name, int m, int k, int n) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor a = new Tensor(sequence(m * k), new int[]{m, k}, null, "a", DataType.FLOAT32);
                    Tensor b = new Tensor(sequence(k * n), new int[]{k, n}, null, "b", DataType.FLOAT32);
                    Tensor bias = new Tensor(sequence(n), new int[]{n}, null, "bias", DataType.FLOAT32);
                    Tensor matmul = a.matmul(b);
                    Tensor add = matmul.add(bias);
                    Tensor out = add.relu();
                    if (isGpuProfile(environment)) {
                        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
                        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
                        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
                    }
                    return out;
                }
        );
    }

    private static TensorRootWorkloadSpec matmulAddTanhWorkload(String name, int m, int k, int n) {
        return new TensorRootWorkloadSpec(
                name,
                WorkloadKind.GENERIC,
                environment -> {
                    Tensor a = new Tensor(sequence(m * k), new int[]{m, k}, null, "a", DataType.FLOAT32);
                    Tensor b = new Tensor(sequence(k * n), new int[]{k, n}, null, "b", DataType.FLOAT32);
                    Tensor bias = new Tensor(sequence(n), new int[]{n}, null, "bias", DataType.FLOAT32);
                    Tensor matmul = a.matmul(b);
                    Tensor add = matmul.add(bias);
                    Tensor out = add.tanh();
                    if (isGpuProfile(environment)) {
                        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
                        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
                        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
                    }
                    return out;
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
                ExecutionMode.FORWARD,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile gpuProfile(String name) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }

    private static float[] sequence(int size) {
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            out[i] = (i % 13) * 0.125f + 0.25f;
        }
        return out;
    }

    private static void assumeAppleShimConfigured() {
        String explicitLib = System.getProperty("synaptik.apple.mps.lib");
        String envLib = System.getenv("SYNAPTIK_APPLE_MPS_LIB");
        assumeTrue((explicitLib != null && !explicitLib.isBlank()) || (envLib != null && !envLib.isBlank()));
    }
}
