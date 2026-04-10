package tuning.report;

import tuning.candidate.Candidate;
import tuning.measure.MeasurementResult;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkEntryRole;
import tuning.validate.ValidationResult;

public record BenchmarkCandidateReport(
        BenchmarkEntry entry,
        ValidationResult validation,
        MeasurementResult measurement,
        boolean success,
        String failureReason
) {
    public BenchmarkCandidateReport {
        failureReason = failureReason == null ? "" : failureReason;
    }

    public static BenchmarkCandidateReport success(
            BenchmarkEntry entry,
            ValidationResult validation,
            MeasurementResult measurement
    ) {
        return new BenchmarkCandidateReport(entry, validation, measurement, true, "");
    }

    public static BenchmarkCandidateReport success(
            Candidate candidate,
            ValidationResult validation,
            MeasurementResult measurement
    ) {
        return success(BenchmarkEntry.candidate(candidate.name(), candidate.profile()), validation, measurement);
    }

    public static BenchmarkCandidateReport failure(
            BenchmarkEntry entry,
            ValidationResult validation,
            String failureReason
    ) {
        return new BenchmarkCandidateReport(entry, validation, null, false, failureReason);
    }

    public static BenchmarkCandidateReport failure(
            Candidate candidate,
            ValidationResult validation,
            String failureReason
    ) {
        return failure(BenchmarkEntry.candidate(candidate.name(), candidate.profile()), validation, failureReason);
    }

    public boolean baseline() {
        return entry != null && entry.role() == BenchmarkEntryRole.BASELINE;
    }

    public Candidate candidate() {
        return entry == null ? null : entry.toCandidate();
    }
}
