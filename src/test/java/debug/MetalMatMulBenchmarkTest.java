package debug;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.validate.ValidationPolicy;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadGraph;
import tuning.workload.WorkloadGraphFactory;
import tuning.workload.WorkloadKind;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

final class MetalMatMulBenchmarkTest {
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
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_MATMUL_F32",
                matmulWorkload("metal_gpu_matmul_f32", 256, 512, 512),
                baselineProfile("metal-cpu-matmul-f32"),
                gpuProfile("metal-matmul-f32-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddReluF32() {
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_MATMUL_ADD_RELU_F32",
                matmulAddReluWorkload("metal_gpu_matmul_add_relu_f32", 256, 512, 512),
                baselineProfile("metal-cpu-matmul-add-relu-f32"),
                gpuProfile("metal-matmul-add-relu-f32-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddTanhF32() {
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_MATMUL_ADD_TANH_F32",
                matmulAddTanhWorkload("metal_gpu_matmul_add_tanh_f32", 256, 512, 512),
                baselineProfile("metal-cpu-matmul-add-tanh-f32"),
                gpuProfile("metal-matmul-add-tanh-f32-gpu")
        );
    }

    @Test
    void benchmarkMatmulLargeF32() {
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_MATMUL_F32_LARGE",
                matmulWorkload("metal_gpu_matmul_f32_large", 512, 1024, 1024),
                baselineProfile("metal-cpu-matmul-f32-large"),
                gpuProfile("metal-matmul-f32-large-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddReluLargeF32() {
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_MATMUL_ADD_RELU_F32_LARGE",
                matmulAddReluWorkload("metal_gpu_matmul_add_relu_f32_large", 512, 1024, 1024),
                baselineProfile("metal-cpu-matmul-add-relu-f32-large"),
                gpuProfile("metal-matmul-add-relu-f32-large-gpu")
        );
    }

    @Test
    void benchmarkMatmulAddTanhLargeF32() {
        assumeMetalMpsConfigured();
        run(
                "METAL_GPU_MATMUL_ADD_TANH_F32_LARGE",
                matmulAddTanhWorkload("metal_gpu_matmul_add_tanh_f32_large", 512, 1024, 1024),
                baselineProfile("metal-cpu-matmul-add-tanh-f32-large"),
                gpuProfile("metal-matmul-add-tanh-f32-large-gpu")
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
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println(label);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static TensorRootWorkloadSpec matmulWorkload(String name, int m, int k, int n) {
        return TensorRootWorkloadSpec.fromGraphFactory(
                name,
                WorkloadKind.GENERIC,
                (WorkloadGraphFactory) environment -> {
                    Tensor a = new Tensor(sequence(m * k), new int[]{m, k}, null, "a", DataType.FLOAT32);
                    Tensor b = new Tensor(sequence(k * n), new int[]{k, n}, null, "b", DataType.FLOAT32);
                    Tensor out = a.matmul(b);
                    BackendIntentPlan backendIntentPlan = isGpuProfile(environment)
                            ? BackendIntentPlan.of(out, ComputeBackend.GPU_METAL)
                            : BackendIntentPlan.empty();
                    return new WorkloadGraph(out, backendIntentPlan);
                }
        );
    }

    private static TensorRootWorkloadSpec matmulAddReluWorkload(String name, int m, int k, int n) {
        return TensorRootWorkloadSpec.fromGraphFactory(
                name,
                WorkloadKind.GENERIC,
                (WorkloadGraphFactory) environment -> {
                    Tensor a = new Tensor(sequence(m * k), new int[]{m, k}, null, "a", DataType.FLOAT32);
                    Tensor b = new Tensor(sequence(k * n), new int[]{k, n}, null, "b", DataType.FLOAT32);
                    Tensor bias = new Tensor(sequence(n), new int[]{n}, null, "bias", DataType.FLOAT32);
                    Tensor matmul = a.matmul(b);
                    Tensor add = matmul.add(bias);
                    Tensor out = add.relu();
                    BackendIntentPlan backendIntentPlan = isGpuProfile(environment)
                            ? BackendIntentPlan.of(ComputeBackend.GPU_METAL, matmul, add, out)
                            : BackendIntentPlan.empty();
                    return new WorkloadGraph(out, backendIntentPlan);
                }
        );
    }

    private static TensorRootWorkloadSpec matmulAddTanhWorkload(String name, int m, int k, int n) {
        return TensorRootWorkloadSpec.fromGraphFactory(
                name,
                WorkloadKind.GENERIC,
                (WorkloadGraphFactory) environment -> {
                    Tensor a = new Tensor(sequence(m * k), new int[]{m, k}, null, "a", DataType.FLOAT32);
                    Tensor b = new Tensor(sequence(k * n), new int[]{k, n}, null, "b", DataType.FLOAT32);
                    Tensor bias = new Tensor(sequence(n), new int[]{n}, null, "bias", DataType.FLOAT32);
                    Tensor matmul = a.matmul(b);
                    Tensor add = matmul.add(bias);
                    Tensor out = add.tanh();
                    BackendIntentPlan backendIntentPlan = isGpuProfile(environment)
                            ? BackendIntentPlan.of(ComputeBackend.GPU_METAL, matmul, add, out)
                            : BackendIntentPlan.empty();
                    return new WorkloadGraph(out, backendIntentPlan);
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
                CompileConfig.noGraphOptimizationBaseline(),
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
                CompileConfig.noGraphOptimizationBaseline(),
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

    private static void assumeMetalMpsConfigured() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        String envLib = System.getenv("SYNAPTIK_METAL_MPS_LIB");
        assumeTrue((explicitLib != null && !explicitLib.isBlank()) || (envLib != null && !envLib.isBlank()));
    }
}
