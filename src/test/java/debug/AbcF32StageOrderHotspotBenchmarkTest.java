package debug;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerStage;
import config.profile.ExecutionProfile;
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
import java.util.List;

final class AbcF32StageOrderHotspotBenchmarkTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void compareCommonStageOrderVariants() {
        ExecutionProfile best = loadBestProfile();
        ExecutionProfile arFuse = withStageOrder(best, "ar-fuse", List.of(
                OptimizerStage.AR,
                OptimizerStage.PART,
                OptimizerStage.FUSE
        ));
        ExecutionProfile arPartFuseMem = withStageOrder(best, "ar-part-fuse-mem", List.of(
                OptimizerStage.AR,
                OptimizerStage.PART,
                OptimizerStage.FUSE,
                OptimizerStage.MEM
        ));
        ExecutionProfile arFuseCseMem = withStageOrder(best, "ar-fuse-cse-mem", List.of(
                OptimizerStage.AR,
                OptimizerStage.CSE,
                OptimizerStage.PART,
                OptimizerStage.FUSE,
                OptimizerStage.MEM
        ));
        ExecutionProfile cseArFuseMem = withStageOrder(best, "cse-ar-fuse-mem", List.of(
                OptimizerStage.CSE,
                OptimizerStage.AR,
                OptimizerStage.PART,
                OptimizerStage.FUSE,
                OptimizerStage.MEM
        ));

        BenchmarkRequest request = new BenchmarkRequest(
                StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_f32_stage_order_probe"),
                List.of(
                        BenchmarkEntry.candidate("best-current", best),
                        BenchmarkEntry.candidate("ar-fuse", arFuse),
                        BenchmarkEntry.candidate("ar-part-fuse-mem", arPartFuseMem),
                        BenchmarkEntry.candidate("ar-fuse-cse-mem", arFuseCseMem),
                        BenchmarkEntry.candidate("cse-ar-fuse-mem", cseArFuseMem)
                ),
                MEASUREMENT,
                ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
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

    private static ExecutionProfile withStageOrder(
            ExecutionProfile base,
            String suffix,
            List<OptimizerStage> stageOrder
    ) {
        return new ExecutionProfile(
                base.profileName() + "-" + suffix,
                base.candidateName() + "-" + suffix,
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                base.optimizer().withStageOrder(stageOrder),
                base.runtime(),
                base.workload()
        );
    }

    private static Path resolveExisting(Path preferred, Path fallback) {
        return Files.exists(preferred) ? preferred : fallback;
    }
}
