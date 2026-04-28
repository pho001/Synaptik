package tuning.benchmark;

import tuning.measure.MeasurementPolicy;
import tuning.reporting.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;

/**
 * Immutable request for benchmarking the same entries across multiple workloads.
 *
 * <p>The suite session runs a regular {@link BenchmarkRequest} for each workload
 * and aggregates the reports. It does not search candidate spaces, update
 * runtime profiles, or write artifacts by itself.</p>
 *
 * @param workloads workloads to run; {@code null} becomes empty
 * @param entries profiles to measure against each workload; {@code null} becomes empty
 * @param measurement measurement controls; {@code null} means defaults
 * @param validation validation controls; {@code null} disables validation
 * @param report report rendering policy; {@code null} means defaults
 */
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
