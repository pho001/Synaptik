package tuning.session;

import tuning.candidate.Candidate;
import tuning.measure.MeasurementPolicy;
import tuning.report.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Objects;

public record BenchmarkRequest(
        WorkloadSpec workload,
        List<Candidate> candidates,
        MeasurementPolicy measurement,
        ValidationPolicy validation,
        ReportPolicy report
) {
    public BenchmarkRequest {
        Objects.requireNonNull(workload, "workload cannot be null");
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
        measurement = measurement == null ? MeasurementPolicy.defaults() : measurement;
        validation = validation == null ? ValidationPolicy.disabled() : validation;
        report = report == null ? ReportPolicy.defaults() : report;
    }
}
