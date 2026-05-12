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
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

final class AbcForwardProfileBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkF64Forward() {
        run(DataType.FLOAT64, "f64");
    }

    @Test
    void benchmarkF32Forward() {
        run(DataType.FLOAT32, "f32");
    }

    @Test
    void benchmarkBF16Forward() {
        run(DataType.BFLOAT16, "bf16");
    }

    private static void run(DataType dataType, String dtypeId) {
        ExecutionProfile currentBest = loadBestProfile(dtypeId);
        ExecutionProfile optimizedForward = new ExecutionProfile(
                currentBest.profileName() + "-forward",
                currentBest.candidateName() + "-forward",
                dataType,
                ExecutionMode.FORWARD,
                currentBest.compile(),
                currentBest.runtime(),
                WorkloadProfile.none()
        );

        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + dtypeId + "_forward"),
                List.of(
                        BenchmarkEntry.baseline("baseline-no-opt-forward", baselineProfile(dataType, dtypeId)),
                        BenchmarkEntry.candidate("best-profile-forward", optimizedForward)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("ABC_FORWARD_PROFILE_BENCHMARK :: " + dtypeId);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile loadBestProfile(String dtypeId) {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", dtypeId + "-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-" + dtypeId + "-best-profile.json")
        );
        return new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for " + dtypeId + " at " + profilePath))
                .profile();
    }

    private static ExecutionProfile baselineProfile(DataType dataType, String dtypeId) {
        return new ExecutionProfile(
                "abc-baseline-" + dtypeId + "-forward",
                "abc-baseline-" + dtypeId + "-forward",
                dataType,
                ExecutionMode.FORWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }
}
