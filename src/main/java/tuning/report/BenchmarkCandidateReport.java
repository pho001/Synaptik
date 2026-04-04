package tuning.report;

import tuning.candidate.Candidate;
import tuning.measure.MeasurementResult;
import tuning.validate.ValidationResult;

public record BenchmarkCandidateReport(
        Candidate candidate,
        ValidationResult validation,
        MeasurementResult measurement,
        boolean success,
        String failureReason
) {
    public BenchmarkCandidateReport {
        failureReason = failureReason == null ? "" : failureReason;
    }

    public static BenchmarkCandidateReport success(
            Candidate candidate,
            ValidationResult validation,
            MeasurementResult measurement
    ) {
        return new BenchmarkCandidateReport(candidate, validation, measurement, true, "");
    }

    public static BenchmarkCandidateReport failure(
            Candidate candidate,
            ValidationResult validation,
            String failureReason
    ) {
        return new BenchmarkCandidateReport(candidate, validation, null, false, failureReason);
    }
}
