package tuning.calibration;

import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import tuning.benchmark.report.BenchmarkSuiteReport;
import tuning.calibration.family.CalibrationFamilyId;

import java.util.List;
import java.util.Objects;

public record PlatformCalibrationStepResult(
        String name,
        CalibrationFamilyId family,
        PlatformRuntimeProfile seedRuntimeProfile,
        BenchmarkSuiteReport benchmarkReport,
        List<PlatformCalibrationCandidateSummary> candidateSummaries,
        PlatformCalibrationCandidateSummary winner,
        PlatformRuntimeProfile selectedRuntimeProfile,
        ExecutionProfile selectedExecutionProfile,
        PlatformCalibrationScore selectedScore,
        String scoreMetric
) {
    public PlatformCalibrationStepResult {
        name = (name == null || name.isBlank()) ? "calibration-step" : name;
        family = family == null ? CalibrationFamilyId.SCHEDULER : family;
        Objects.requireNonNull(seedRuntimeProfile, "seedRuntimeProfile cannot be null");
        Objects.requireNonNull(benchmarkReport, "benchmarkReport cannot be null");
        candidateSummaries = candidateSummaries == null ? List.of() : List.copyOf(candidateSummaries);
        Objects.requireNonNull(winner, "winner cannot be null");
        Objects.requireNonNull(selectedRuntimeProfile, "selectedRuntimeProfile cannot be null");
        Objects.requireNonNull(selectedExecutionProfile, "selectedExecutionProfile cannot be null");
        Objects.requireNonNull(selectedScore, "selectedScore cannot be null");
        scoreMetric = (scoreMetric == null || scoreMetric.isBlank()) ? "averageMedianMs" : scoreMetric;
    }
}
