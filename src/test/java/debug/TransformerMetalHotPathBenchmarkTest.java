package debug;

import runtime.contract.ExecutionMode;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuPartitionConfig;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import config.runtime.AcceleratorConfig;
import config.runtime.RuntimeConfig;
import runtime.execution.PublicationPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.calibration.store.CalibrationArtifactLayout;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.measure.MeasurementPolicy;
import tuning.store.HardwareFingerprint;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

final class TransformerMetalHotPathBenchmarkTest {
    private static final MeasurementPolicy MEASUREMENT = new MeasurementPolicy(
            4,
            8,
            3,
            true,
            true,
            true,
            true,
            true,
            PublicationPolicy.NONE
    );

    @Test
    void benchmarkMediumTransformerBlockTrainingCpuVsMetal() {
        runMediumTransformerBlockTrainingCpuVsMetal(DataType.FLOAT32, "f32");
    }

    @Test
    void benchmarkMediumTransformerBlockTrainingBf16CpuVsMetal() {
        runMediumTransformerBlockTrainingCpuVsMetal(DataType.BFLOAT16, "bf16");
    }

    private static void runMediumTransformerBlockTrainingCpuVsMetal(DataType dataType, String dtypeId) {
        assumeMetalMpsConfigured();
        WorkloadProfile workload = WorkloadProfile.transformerHotPathMedium();
        RuntimeConfig calibratedRuntime = loadCalibratedRuntime(dataType, dtypeId).toRuntimeConfig();
        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.transformerBlockHotPath("transformer_block_hot_path_medium_" + dtypeId, workload),
                List.of(
                        BenchmarkEntry.baseline("cpu-no-opt", noOptCpuProfile(dataType, dtypeId, workload)),
                        BenchmarkEntry.candidate("cpu-calibrated", calibratedCpuProfile(dataType, dtypeId, workload, calibratedRuntime)),
                        BenchmarkEntry.candidate("metal-calibrated-greedy-buffer", metalProfile(dataType, dtypeId, workload, calibratedRuntime))
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("TRANSFORMER_METAL_HOT_PATH_MEDIUM_" + dtypeId.toUpperCase(java.util.Locale.ROOT));
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile noOptCpuProfile(DataType dataType, String dtypeId, WorkloadProfile workload) {
        return new ExecutionProfile(
                "transformer-medium-" + dtypeId + "-cpu-no-opt",
                "cpu-no-opt",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.noOptNoVecNoPar(),
                workload
        );
    }

    private static ExecutionProfile calibratedCpuProfile(
            DataType dataType,
            String dtypeId,
            WorkloadProfile workload,
            RuntimeConfig calibratedRuntime
    ) {
        return new ExecutionProfile(
                "transformer-medium-" + dtypeId + "-cpu-calibrated",
                "cpu-calibrated",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training(),
                calibratedRuntime.withAccelerator(AcceleratorConfig.disabled()),
                workload
        );
    }

    private static ExecutionProfile metalProfile(
            DataType dataType,
            String dtypeId,
            WorkloadProfile workload,
            RuntimeConfig calibratedRuntime
    ) {
        return new ExecutionProfile(
                "transformer-medium-" + dtypeId + "-metal-calibrated-greedy-buffer",
                "metal-calibrated-greedy-buffer",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training()
                        .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withCpuPartitions(CpuPartitionConfig.defaults()))
                        .withPartitionExecution(CompileConfig.training().partitionExecution().withCpuFusion(CpuFusionConfig.defaults())),
                calibratedRuntime,
                workload
        );
    }

    private static PlatformRuntimeProfile loadCalibratedRuntime(DataType dataType, String dtypeId) {
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        Path path = resolveExisting(
                CalibrationArtifactLayout.of(Path.of("profiles"), platformId)
                        .latestProfilePath(dtypeId, ExecutionMode.FORWARD_BACKWARD.name()),
                Path.of("profiles", "platform", platformId, "calibration", dtypeId + "-forward-backward.json")
        );
        assumeTrue(
                Files.exists(path),
                "Run calibration first: missing platform runtime profile at " + path
        );
        return PlatformRuntimeProfileIO.loadStrict(path, calibratedRuntimeFallback(platformId, hardware, dataType));
    }

    private static PlatformRuntimeProfile calibratedRuntimeFallback(
            String platformId,
            HardwareFingerprint hardware,
            DataType dataType
    ) {
        return PlatformRuntimeProfile.fromExecutionProfile(
                platformId,
                hardware.key(),
                "fallback",
                new ExecutionProfile(
                        "transformer-medium-" + dtypeId(dataType) + "-calibration-fallback",
                        "transformer-medium-" + dtypeId(dataType) + "-calibration-fallback",
                        dataType,
                        ExecutionMode.FORWARD_BACKWARD,
                        CompileConfig.training(),
                        RuntimeConfig.trainingDefaults(),
                        WorkloadProfile.none()
                )
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }

    private static String dtypeId(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> "f64";
            case FLOAT32 -> "f32";
            case BFLOAT16 -> "bf16";
            default -> dataType.name().toLowerCase(java.util.Locale.ROOT);
        };
    }

    private static void assumeMetalMpsConfigured() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank(), "Set -Dsynaptik.metal.mps.lib to run Metal benchmark.");
    }
}
