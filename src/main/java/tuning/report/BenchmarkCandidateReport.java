package tuning.report;

import tuning.candidate.Candidate;
import tuning.measure.MeasurementResult;
import tuning.validate.ValidationResult;

public record BenchmarkCandidateReport(
        Candidate candidate,
        ValidationResult validation,
        MeasurementResult measurement,
        boolean success,
        String failureReason,
        BenchmarkBaselineKind baselineKind
) {
    public BenchmarkCandidateReport {
        failureReason = failureReason == null ? "" : failureReason;
        baselineKind = baselineKind == null ? BenchmarkBaselineKind.NONE : baselineKind;
    }

    public static BenchmarkCandidateReport success(
            Candidate candidate,
            ValidationResult validation,
            MeasurementResult measurement
    ) {
        return new BenchmarkCandidateReport(candidate, validation, measurement, true, "", BenchmarkBaselineKind.NONE);
    }

    public static BenchmarkCandidateReport success(
            Candidate candidate,
            ValidationResult validation,
            MeasurementResult measurement,
            BenchmarkBaselineKind baselineKind
    ) {
        return new BenchmarkCandidateReport(candidate, validation, measurement, true, "", baselineKind);
    }

    public static BenchmarkCandidateReport failure(
            Candidate candidate,
            ValidationResult validation,
            String failureReason
    ) {
        return new BenchmarkCandidateReport(candidate, validation, null, false, failureReason, BenchmarkBaselineKind.NONE);
    }

    public static BenchmarkCandidateReport failure(
            Candidate candidate,
            ValidationResult validation,
            String failureReason,
            BenchmarkBaselineKind baselineKind
    ) {
        return new BenchmarkCandidateReport(candidate, validation, null, false, failureReason, baselineKind);
    }
}
