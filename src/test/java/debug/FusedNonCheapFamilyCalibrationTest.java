package debug;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.report.TextBenchmarkSuiteReportRenderer;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSession;
import tuning.session.BenchmarkSuiteRequest;
import tuning.session.BenchmarkSuiteSession;
import tuning.session.PlatformCalibrationDefaults;
import tuning.session.PlatformCalibrationScore;
import tuning.session.PlatformCalibrationStep;
import tuning.session.RuntimeProfileCandidate;
import tuning.store.PlatformCalibrationLayout;
import tuning.store.PlatformCalibrationPaths;
import tuning.workload.CalibrationWorkloads;
import tuning.workload.WorkloadSpec;

import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;

final class FusedNonCheapFamilyCalibrationTest {
    private static final tuning.measure.MeasurementPolicy MEASUREMENT = DebugMeasurementPolicies.STANDARD_WITH_TRACE;

    @Test
    void calibrateNonCheapFusedFamiliesF32ForwardAndBenchmark() {
        ExecutionProfile seed = inferenceSeedProfile();
        GraphExecutionPolicy loweredPolicy = GraphExecutionPolicy.of(
                OptimizerConfig.inferenceDefaults().withStageOrder(List.of(OptimizerStage.FUSE))
        );
        PlatformCalibrationLayout layout = PlatformCalibrationPaths.defaultLayout(
                Path.of("build", "platform-calibration", "f32"),
                seed
        );
        PlatformRuntimeProfile fallback = PlatformRuntimeProfile.fromExecutionProfile(
                layout.platformId(),
                layout.hardware().key(),
                "fallback",
                seed
        );
        PlatformRuntimeProfile current = PlatformRuntimeProfileIO.loadOrDefault(layout.profilePath(), fallback);
        PlatformRuntimeProfile original = current;

        current = calibrateStep(
                "FUSED_NON_CHEAP_CONTIGUOUS_CALIBRATION",
                PlatformCalibrationDefaults.fusedNonCheapContiguousStep(
                        "fused_noncheap_contiguous_family",
                        tuning.session.TuningPreset.BALANCED,
                        DataType.FLOAT32
                ),
                current,
                layout,
                loweredPolicy
        );
        current = calibrateStep(
                "FUSED_NON_CHEAP_STRIDED_CALIBRATION",
                PlatformCalibrationDefaults.fusedNonCheapStridedStep(
                        "fused_noncheap_strided_family",
                        tuning.session.TuningPreset.BALANCED,
                        DataType.FLOAT32
                ),
                current,
                layout,
                loweredPolicy
        );

        benchmarkWorkload(
                "FUSED_NON_CHEAP_CONTIGUOUS_PROFILE_BENCHMARK",
                CalibrationWorkloads.fusedTranscendental("fused_noncheap_contiguous_current_profile", 65_536),
                original,
                current,
                loweredPolicy
        );
        benchmarkWorkload(
                "FUSED_NON_CHEAP_STRIDED_PROFILE_BENCHMARK",
                CalibrationWorkloads.fusedTranscendentalStrided("fused_noncheap_strided_current_profile", 256, 256),
                original,
                current,
                loweredPolicy
        );
    }

    private static PlatformRuntimeProfile calibrateStep(
            String header,
            PlatformCalibrationStep step,
            PlatformRuntimeProfile baseProfile,
            PlatformCalibrationLayout layout,
            GraphExecutionPolicy loweredPolicy
    ) {
        List<RuntimeProfileCandidate> generated = step.candidateSpaceFactory()
                .create(baseProfile)
                .generate(step.workloads().getFirst());
        List<BenchmarkEntry> calibrationEntries = generated.stream()
                .map(candidate -> BenchmarkEntry.candidate(
                        candidate.name(),
                        ExecutionProfileAssembler.assemble(
                                header.toLowerCase(),
                                candidate.name(),
                                DataType.FLOAT32,
                                ExecutionMode.FORWARD,
                                candidate.runtimeProfile(),
                                loweredPolicy
                        )
                ))
                .toList();

        var suiteRequest = new BenchmarkSuiteRequest(
                step.workloads(),
                calibrationEntries,
                MEASUREMENT,
                step.preset().benchmarkValidation(),
                step.preset().reportPolicy()
        );
        var suiteReport = BenchmarkSuiteSession.create(suiteRequest).run();
        RuntimeProfileCandidate winner = generated.stream()
                .min(Comparator.comparingDouble(candidate -> candidateScore(candidate.name(), step, suiteReport).score()))
                .orElseThrow();
        PlatformRuntimeProfileIO.save(layout.profilePath(), winner.runtimeProfile());

        System.out.println(header);
        System.out.println("profilePath=" + layout.profilePath());
        System.out.println("winner=" + winner.name());
        System.out.println(TextBenchmarkSuiteReportRenderer.render(suiteReport));
        return winner.runtimeProfile();
    }

    private static void benchmarkWorkload(
            String header,
            WorkloadSpec workload,
            PlatformRuntimeProfile original,
            PlatformRuntimeProfile calibrated,
            GraphExecutionPolicy loweredPolicy
    ) {
        BenchmarkRequest request = new BenchmarkRequest(
                workload,
                List.of(
                        BenchmarkEntry.baseline("manual-no-opt", baselineProfile()),
                        BenchmarkEntry.candidate(
                                "manual-fuse-current-profile",
                                ExecutionProfileAssembler.assemble(
                                        header.toLowerCase() + "-current",
                                        "manual-fuse-current-profile",
                                        DataType.FLOAT32,
                                        ExecutionMode.FORWARD,
                                        original,
                                        loweredPolicy
                                )
                        ),
                        BenchmarkEntry.candidate(
                                "manual-fuse-calibrated-profile",
                                ExecutionProfileAssembler.assemble(
                                        header.toLowerCase() + "-calibrated",
                                        "manual-fuse-calibrated-profile",
                                        DataType.FLOAT32,
                                        ExecutionMode.FORWARD,
                                        calibrated,
                                        loweredPolicy
                                )
                        )
                ),
                MEASUREMENT,
                tuning.validate.ValidationPolicy.disabled(),
                tuning.report.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println(header);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static PlatformCalibrationScore candidateScore(
            String candidateName,
            PlatformCalibrationStep step,
            tuning.report.BenchmarkSuiteReport suiteReport
    ) {
        PlatformCalibrationScore score = step.scorePolicy().score(candidateName, suiteReport);
        if (!score.valid()) {
            return PlatformCalibrationScore.invalid(score.explanation());
        }
        return score;
    }

    private static ExecutionProfile inferenceSeedProfile() {
        return new ExecutionProfile(
                "platform-seed-f32-inference",
                "platform-seed-f32-inference",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile baselineProfile() {
        return new ExecutionProfile(
                "manual-no-opt",
                "manual-no-opt",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }
}
