package debug;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.PlatformCalibrationPaths;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

final class AbcProfileOverrideBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD;

    @Test
    void benchmarkBF16ForwardBackward() {
        run(DataType.BFLOAT16, "bf16");
    }

    private static void run(DataType dataType, String dtypeId) {
        ExecutionProfile current = loadBestProfile(resolveDefaultBestProfile(dtypeId), dtypeId);
        List<BenchmarkEntry> entries = new ArrayList<>();
        entries.add(BenchmarkEntry.baseline("baseline-no-opt", baselineProfile(dataType, dtypeId)));
        entries.add(BenchmarkEntry.candidate("current-best", current));

        String overridePathValue = propertyOrEnv("synaptik.overrideProfilePath", "SYNAPTIK_OVERRIDE_PROFILE_PATH");
        if (overridePathValue != null && !overridePathValue.isBlank()) {
            Path overridePath = Path.of(overridePathValue);
            String overrideLabel = propertyOrEnv("synaptik.overrideLabel", "SYNAPTIK_OVERRIDE_LABEL");
            if (overrideLabel == null || overrideLabel.isBlank()) {
                overrideLabel = "override";
            }
            entries.add(BenchmarkEntry.candidate(overrideLabel, loadBestProfile(overridePath, dtypeId)));
        }

        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_" + dtypeId),
                entries,
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("ABC_PROFILE_OVERRIDE_BENCHMARK :: " + dtypeId);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile loadBestProfile(Path path, String dtypeId) {
        return new JsonFileBestProfileStore()
                .load(path)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for " + dtypeId + " at " + path))
                .profile();
    }

    private static Path resolveDefaultBestProfile(String dtypeId) {
        return resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", dtypeId + "-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-" + dtypeId + "-best-profile.json")
        );
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

    private static String propertyOrEnv(String propertyName, String envName) {
        String property = System.getProperty(propertyName);
        if (property != null && !property.isBlank()) {
            return property;
        }
        String env = System.getenv(envName);
        return (env == null || env.isBlank()) ? null : env;
    }
}
