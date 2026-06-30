package debug;

import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.benchmark.report.TextBenchmarkReportRenderer;
import tuning.benchmark.report.TextBenchmarkSuiteReportRenderer;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.benchmark.BenchmarkSuiteSession;
import tuning.calibration.PlatformCalibrationDefaults;
import tuning.calibration.PlatformCalibrationScore;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.runtime.RuntimeProfileCandidate;
import tuning.calibration.store.PlatformCalibrationLayout;
import tuning.calibration.store.PlatformCalibrationPaths;
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
                CompileConfig.inference().withGraphOptimization(GraphOptimizationConfig.noGraphOptimization()).withMemoryPlanning(config.compile.MemoryPlanningConfig.disabledUnlessRequired())
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
                "FUSED_ASM_WIDTH_CALIBRATION",
                PlatformCalibrationDefaults.fusedAsmWidthStep(
                        "fused_asm_width_family",
                        tuning.preset.TuningPreset.BALANCED,
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
                tuning.reporting.ReportPolicy.defaults()
        );

        var report = BenchmarkSession.create(request).run();
        System.out.println(header);
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static PlatformCalibrationScore candidateScore(
            String candidateName,
            PlatformCalibrationStep step,
            tuning.benchmark.report.BenchmarkSuiteReport suiteReport
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
                CompileConfig.inference(),
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
                CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.noOptNoVecNoPar(),
                WorkloadProfile.none()
        );
    }
}
