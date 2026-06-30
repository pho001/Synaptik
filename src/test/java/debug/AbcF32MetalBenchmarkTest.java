package debug;

import backend.contract.ComputeBackend;
import runtime.contract.ExecutionMode;
import config.profile.ExecutionProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.validate.ValidationPolicy;
import tuning.validate.ValidationReference;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadGraph;
import tuning.workload.WorkloadGraphFactory;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;
import graph.compile.intent.BackendIntentPlan;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

final class AbcF32MetalBenchmarkTest {
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
    void benchmarkOptimizedCpuVsMetal() {
        assumeMetalMpsConfigured();

        ExecutionProfile cpuProfile = loadBestProfile();
        ExecutionProfile gpuProfile = new ExecutionProfile(
                cpuProfile.profileName() + "-gpu-metal",
                cpuProfile.candidateName() + "-gpu-metal",
                cpuProfile.dataType(),
                cpuProfile.mode(),
                cpuProfile.compile(),
                cpuProfile.runtime(),
                cpuProfile.workload()
        );

        BenchmarkRequest request = new BenchmarkRequest(
                abcF32MetalWorkload("abc_sequence_matmul_f32_metal_gpu"),
                List.of(
                        BenchmarkEntry.baseline("optimized-cpu", cpuProfile),
                        BenchmarkEntry.candidate("optimized-gpu-metal", gpuProfile)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("ABC_F32_METAL_GPU_BENCHMARK");
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static TensorRootWorkloadSpec abcF32MetalWorkload(String name) {
        return TensorRootWorkloadSpec.fromGraphFactory(
                name,
                WorkloadKind.ABC_SEQUENCE_MATMUL,
                (WorkloadGraphFactory) AbcF32MetalBenchmarkTest::buildGraph,
                environment -> ValidationReference.none(),
                environment -> new WorkloadMetadata(
                        name,
                        WorkloadKind.ABC_SEQUENCE_MATMUL,
                        Map.of("batch", 256, "features", 2048)
                )
        );
    }

    private static WorkloadGraph buildGraph(WorkloadEnvironment environment) {
        ExecutionProfile profile = environment.profile();
        boolean requiresGrad = profile.mode() == ExecutionMode.FORWARD_BACKWARD;
        DataType dataType = profile.dataType();

        Tensor A = tensor("A", 801, dataType, requiresGrad, 256, 2048, 1.5, 0.9);
        Tensor B = tensor("B", 802, dataType, requiresGrad, 256, 2048, 1.1, 0.7);
        Tensor C = tensor("C", 803, dataType, requiresGrad, 256, 2048, 0.2, 0.15);

        Tensor T1 = A.div(B);
        Tensor T2 = A.sub(C);
        Tensor T3 = B.add(C);
        Tensor T4 = T1.div(T2);
        Tensor T5 = T3.mul(T4);
        Tensor T6 = T4.add(T5);
        Tensor T7 = T6.pow(2.0);

        Tensor matmul = A.matmul(B.transpose());
        Tensor matmulProjected = matmul.mean(1, true).expand(T7.getShapeUnsafe());
        Tensor root = T7.add(matmulProjected).mean();
        BackendIntentPlan backendIntentPlan = profile.candidateName().contains("-gpu-metal")
                ? BackendIntentPlan.of(matmul, ComputeBackend.GPU_METAL)
                : BackendIntentPlan.empty();
        return new WorkloadGraph(root, backendIntentPlan);
    }

    private static Tensor tensor(
            String label,
            int seed,
            DataType dataType,
            boolean requiresGrad,
            int rows,
            int cols,
            double base,
            double amplitude
    ) {
        double[] data = new double[rows * cols];
        java.util.Random random = new java.util.Random(seed);
        for (int i = 0; i < data.length; i++) {
            data[i] = base + Math.abs(Math.sin(seed * 0.01 + i * 0.031)) * amplitude + random.nextDouble() * 0.03;
        }
        return tensor.factory.TensorDataFactory.shapedTensor(label, data, requiresGrad, dataType, rows, cols);
    }

    private static ExecutionProfile loadBestProfile() {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", "f32-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-f32-best-profile.json")
        );
        return new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for f32 at " + profilePath))
                .profile();
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }

    private static void assumeMetalMpsConfigured() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        String envLib = System.getenv("SYNAPTIK_METAL_MPS_LIB");
        assumeTrue((explicitLib != null && !explicitLib.isBlank()) || (envLib != null && !envLib.isBlank()));
    }
}
