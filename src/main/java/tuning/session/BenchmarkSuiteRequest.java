package tuning.session;

import tuning.candidate.Candidate;
import tuning.measure.MeasurementPolicy;
import tuning.report.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;

public record BenchmarkSuiteRequest(
        List<WorkloadSpec> workloads,
        List<Candidate> candidates,
        MeasurementPolicy measurement,
        ValidationPolicy validation,
        ReportPolicy report,
        BaselinePolicy baselines
) {
    public BenchmarkSuiteRequest {
        workloads = workloads == null ? List.of() : List.copyOf(workloads);
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
        measurement = measurement == null ? MeasurementPolicy.defaults() : measurement;
        validation = validation == null ? ValidationPolicy.disabled() : validation;
        report = report == null ? ReportPolicy.defaults() : report;
        baselines = baselines == null ? BaselinePolicy.defaults() : baselines;
    }

    public BenchmarkSuiteRequest(
            List<WorkloadSpec> workloads,
            List<Candidate> candidates,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            ReportPolicy report
    ) {
        this(workloads, candidates, measurement, validation, report, BaselinePolicy.defaults());
    }
}
