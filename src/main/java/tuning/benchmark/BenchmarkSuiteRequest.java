package tuning.benchmark;

import tuning.measure.MeasurementPolicy;
import tuning.reporting.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;

public record BenchmarkSuiteRequest(
        List<WorkloadSpec> workloads,
        List<BenchmarkEntry> entries,
        MeasurementPolicy measurement,
        ValidationPolicy validation,
        ReportPolicy report
) {
    public BenchmarkSuiteRequest {
        workloads = workloads == null ? List.of() : List.copyOf(workloads);
        entries = entries == null ? List.of() : List.copyOf(entries);
        measurement = measurement == null ? MeasurementPolicy.defaults() : measurement;
        validation = validation == null ? ValidationPolicy.disabled() : validation;
        report = report == null ? ReportPolicy.defaults() : report;

        long baselineCount = entries.stream().filter(entry -> entry.role() == BenchmarkEntryRole.BASELINE).count();
        if (baselineCount > 1) {
            throw new IllegalArgumentException("BenchmarkSuiteRequest can contain at most one baseline entry.");
        }
    }
}
