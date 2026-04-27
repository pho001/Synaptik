package debug;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
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
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class AbcCurrentBestProfileBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkF64ForwardBackward() {
        run(DataType.FLOAT64, "f64");
    }

    @Test
    void benchmarkF32ForwardBackward() {
        run(DataType.FLOAT32, "f32");
    }

    @Test
    void benchmarkBF16ForwardBackward() {
        run(DataType.BFLOAT16, "bf16");
    }

    private static void run(DataType dataType, String dtypeId) {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", dtypeId + "-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-" + dtypeId + "-best-profile.json")
        );
        ExecutionProfile bestProfile = new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for " + dtypeId))
                .profile();

        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + dtypeId),
                List.of(
                        BenchmarkEntry.baseline("baseline-no-opt", baselineProfile(dataType, dtypeId)),
                        BenchmarkEntry.candidate("best-profile", bestProfile)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("ABC_CURRENT_BEST_PROFILE_BENCHMARK :: " + dtypeId);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile baselineProfile(DataType dataType, String dtypeId) {
        return new ExecutionProfile(
                "abc-baseline-" + dtypeId,
                "abc-baseline-" + dtypeId,
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.noOptimization(),
                RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }
}
