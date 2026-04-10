package tuning.session;

import tuning.measure.MeasurementPolicy;
import tuning.report.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Objects;

public record BenchmarkRequest(
        WorkloadSpec workload,
        List<BenchmarkEntry> entries,
        MeasurementPolicy measurement,
        ValidationPolicy validation,
        ReportPolicy report
) {
    public BenchmarkRequest {
        Objects.requireNonNull(workload, "workload cannot be null");
        entries = entries == null ? List.of() : List.copyOf(entries);
        measurement = measurement == null ? MeasurementPolicy.defaults() : measurement;
        validation = validation == null ? ValidationPolicy.disabled() : validation;
        report = report == null ? ReportPolicy.defaults() : report;

        long baselineCount = entries.stream().filter(entry -> entry.role() == BenchmarkEntryRole.BASELINE).count();
        if (baselineCount > 1) {
            throw new IllegalArgumentException("BenchmarkRequest can contain at most one baseline entry.");
        }
    }
}
