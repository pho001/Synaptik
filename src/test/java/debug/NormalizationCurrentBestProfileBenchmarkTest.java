package debug;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.validate.ValidationPolicy;
import tuning.workload.NormalizationWorkloadSpec;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class NormalizationCurrentBestProfileBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;
    private static final int BATCH = 8;
    private static final int CHANNELS = 512;
    private static final int HEIGHT = 128;
    private static final int WIDTH = 1;
    private static final double EPSILON = 1e-5;

    @Test
    void benchmarkLayerNormF64ForwardBackward() {
        run(NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM, DataType.FLOAT64, "f64");
    }

    @Test
    void benchmarkLayerNormF32ForwardBackward() {
        run(NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM, DataType.FLOAT32, "f32");
    }

    @Test
    void benchmarkLayerNormBF16ForwardBackward() {
        run(NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM, DataType.BFLOAT16, "bf16");
    }

    @Test
    void benchmarkRmsNormF64ForwardBackward() {
        run(NormalizationWorkloadSpec.NormalizationKind.RMS_NORM, DataType.FLOAT64, "f64");
    }

    @Test
    void benchmarkRmsNormF32ForwardBackward() {
        run(NormalizationWorkloadSpec.NormalizationKind.RMS_NORM, DataType.FLOAT32, "f32");
    }

    @Test
    void benchmarkRmsNormBF16ForwardBackward() {
        run(NormalizationWorkloadSpec.NormalizationKind.RMS_NORM, DataType.BFLOAT16, "bf16");
    }

    private static void run(
            NormalizationWorkloadSpec.NormalizationKind kind,
            DataType dataType,
            String dtypeId
    ) {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", dtypeId + "-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-" + dtypeId + "-best-profile.json")
        );
        ExecutionProfile bestProfile = new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for " + dtypeId))
                .profile();

        String workloadName = kind.name().toLowerCase() + "_" + dtypeId + "_transformer_like";
        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.normalization(workloadName, kind, BATCH, CHANNELS, HEIGHT, WIDTH, EPSILON),
                List.of(
                        BenchmarkEntry.baseline("baseline-no-opt", baselineProfile(kind, dataType, dtypeId)),
                        BenchmarkEntry.candidate("best-profile", bestProfile)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("NORMALIZATION_CURRENT_BEST_PROFILE_BENCHMARK :: " + kind.name() + " :: " + dtypeId);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile baselineProfile(
            NormalizationWorkloadSpec.NormalizationKind kind,
            DataType dataType,
            String dtypeId
    ) {
        return new ExecutionProfile(
                kind.name().toLowerCase() + "-baseline-" + dtypeId,
                kind.name().toLowerCase() + "-baseline-" + dtypeId,
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }
}
