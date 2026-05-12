package debug;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.compile.MemoryPlanningConfig;
import config.profile.ExecutionProfile;
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

final class AbcF32StageOrderHotspotBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void compareCommonStageOrderVariants() {
        ExecutionProfile best = loadBestProfile();
        ExecutionProfile arFuse = withCompile(best, "ar-fuse", best.compile()
                .withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false))
                .withMemoryPlanning(MemoryPlanningConfig.disabledUnlessRequired()));
        ExecutionProfile arPartFuseMem = withCompile(best, "ar-part-fuse-mem", best.compile()
                .withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false)));
        ExecutionProfile cleanupFuseMem = withCompile(best, "cleanup-fuse-mem", best.compile());
        ExecutionProfile cleanupNoMem = withCompile(best, "cleanup-fuse", best.compile()
                .withMemoryPlanning(MemoryPlanningConfig.disabledUnlessRequired()));

        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_f32_stage_order_probe"),
                List.of(
                        BenchmarkEntry.candidate("best-current", best),
                        BenchmarkEntry.candidate("ar-fuse", arFuse),
                        BenchmarkEntry.candidate("ar-part-fuse-mem", arPartFuseMem),
                        BenchmarkEntry.candidate("cleanup-fuse-mem", cleanupFuseMem),
                        BenchmarkEntry.candidate("cleanup-fuse", cleanupNoMem)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println();
        System.out.println("ABC_F32_STAGE_ORDER_HOTSPOT_BENCHMARK");
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile loadBestProfile() {
        Path profilePath = resolveExisting(
                Path.of("profiles", "platform", PlatformCalibrationPaths.platformId(HardwareFingerprint.capture()), "tuning", "abc", "f32-best-profile.json"),
                Path.of("build", "tuning", "best-profiles", "abc-f32-best-profile.json")
        );
        return new JsonFileBestProfileStore()
                .load(profilePath)
                .orElseThrow(() -> new IllegalStateException("Missing best profile for f32"))
                .profile();
    }

    private static ExecutionProfile withCompile(
            ExecutionProfile base,
            String suffix,
            CompileConfig compile
    ) {
        return new ExecutionProfile(
                base.profileName() + "-" + suffix,
                base.candidateName() + "-" + suffix,
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                compile,
                base.runtime(),
                base.workload()
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }
}
