package synaptik.app;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.WorkloadProfile;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
import tuning.measure.DefaultMeasurementEngine;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.report.TextPlatformCalibrationResultRenderer;
import tuning.report.TextTuningResultRenderer;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.session.AutotuneRequest;
import tuning.session.AutotuneSession;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkSession;
import tuning.session.LoggingPlatformCalibrationProgressListener;
import tuning.session.PlatformCalibrationDefaults;
import tuning.session.PlatformCalibrationRequest;
import tuning.session.PlatformCalibrationSession;
import tuning.session.TuningDefaults;
import tuning.session.TuningPreset;
import tuning.store.PersistencePolicy;
import tuning.store.PlatformCalibrationLayout;
import tuning.store.PlatformCalibrationPaths;
import tuning.store.PlatformCalibrationSaveHelper;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.validate.DefaultValidationEngine;
import tuning.workload.StandardWorkloads;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        runAbcFlow("f64", tensor.DataType.FLOAT64);
        runAbcFlow("f32", tensor.DataType.FLOAT32);
        runAbcFlow("bf16", tensor.DataType.BFLOAT16);
    }

    private static void runAbcFlow(String dtypeId, tensor.DataType dataType) {
        System.out.println("\n==============================");
        System.out.println("Synaptik " + dtypeId.toUpperCase() + " calibration + ABC tuning example");
        System.out.println("==============================");

        ExecutionProfile calibrationSeed = new ExecutionProfile(
                "platform-seed-" + dtypeId + "-inference",
                "platform-seed-" + dtypeId + "-inference",
                dataType,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        Path rootDir = Path.of("build", "platform-calibration", dtypeId);
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(rootDir, calibrationSeed);
        PlatformCalibrationRequest baseCalibration = PlatformCalibrationDefaults.balancedInferenceFull(
                layout.platformId(),
                calibrationSeed,
                layout.profilePath()
        );
        PlatformCalibrationRequest calibrationRequest = new PlatformCalibrationRequest(
                baseCalibration.platformId(),
                baseCalibration.profileName(),
                baseCalibration.dataType(),
                baseCalibration.executionMode(),
                baseCalibration.graphPolicy(),
                baseCalibration.seedRuntimeProfile(),
                baseCalibration.steps(),
                baseCalibration.outputProfilePath(),
                LoggingPlatformCalibrationProgressListener.defaults()
        );

        var calibrationResult = PlatformCalibrationSession.create(calibrationRequest).run();
        PlatformCalibrationSaveHelper.saveAll(
                calibrationResult,
                layout.profilePath(),
                layout.jsonReportPath(),
                layout.textReportPath()
        );

        System.out.println("\n=== Platform Calibration (" + dtypeId.toUpperCase() + ") ===");
        System.out.println(TextPlatformCalibrationResultRenderer.render(calibrationResult));
        System.out.println("profilePath=" + calibrationResult.outputProfilePath());

        var abcWorkload = StandardWorkloads.abcSequenceMatmul("abc_sequence_matmul_" + dtypeId, 64, 10_000);
        ExecutionProfile calibratedSeed = ExecutionProfileAssembler.assemble(
                "abc-" + dtypeId + "-calibrated",
                "abc-" + dtypeId + "-calibrated",
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                calibrationResult.finalRuntimeProfile(),
                GraphExecutionPolicy.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var stageCandidateSpace = new ProfileGridCandidateSpace(
                calibratedSeed,
                List.of(ProfileMutators.constrainedStageOrderSpace())
        );

        PersistencePolicy autotunePersistence = new PersistencePolicy(
                true,
                true,
                Path.of("build", "tuning", "best-profiles", "abc-" + dtypeId + "-best-profile.json"),
                Path.of("build", "tuning", "history", "abc-" + dtypeId + "-history.jsonl")
        );

        AutotuneRequest autotuneRequest = TuningDefaults.balancedAutotune(
                abcWorkload,
                calibratedSeed,
                stageCandidateSpace,
                autotunePersistence
        );
        var tuningResult = AutotuneSession.create(
                autotuneRequest,
                new ExhaustiveSearchStrategy(),
                new DefaultMeasurementEngine(),
                new DefaultValidationEngine(),
                new JsonFileBestProfileStore(),
                new JsonFileTuningHistoryStore()
        ).run();

        System.out.println("\n=== ABC Autotune (" + dtypeId.toUpperCase() + ") ===");
        System.out.println(TextTuningResultRenderer.render(tuningResult));
        System.out.println("autotuneBestProfilePath=" + autotunePersistence.bestProfilePath());
        System.out.println("autotuneHistoryPath=" + autotunePersistence.historyPath());

        ExecutionProfile baselineProfile = new ExecutionProfile(
                "abc-baseline-no-opt-" + dtypeId,
                "abc-baseline-no-opt-" + dtypeId,
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );

        List<BenchmarkEntry> benchmarkEntries = new ArrayList<>();
        benchmarkEntries.add(BenchmarkEntry.baseline("baseline-no-opt", baselineProfile));
        stageCandidateSpace.generate(abcWorkload).forEach(candidate ->
                benchmarkEntries.add(BenchmarkEntry.candidate(candidate.name(), candidate.profile()))
        );

        var abcBenchmark = TuningDefaults.benchmark(
                TuningPreset.BALANCED,
                abcWorkload,
                benchmarkEntries
        );
        var abcReport = BenchmarkSession.create(abcBenchmark).run();

        System.out.println("\n=== ABC Benchmark: All Permissible Stage Orders (" + dtypeId.toUpperCase() + ") ===");
        System.out.println(TextBenchmarkReportRenderer.render(abcReport));
    }
}
