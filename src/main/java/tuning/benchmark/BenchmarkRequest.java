package tuning.benchmark;

import tuning.measure.MeasurementPolicy;
import tuning.reporting.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Objects;

/**
 * Immutable request for benchmarking supplied execution profiles against one
 * workload.
 *
 * <p>Benchmarking is observational: it validates and measures the entries the
 * caller provides and never creates, searches, or persists new profiles, best
 * records, history entries, calibration artifacts, or benchmark reports.
 * Explicit report persistence, when desired by a tool, must be a separate
 * caller action outside the benchmark session. Use {@code tuning.autotune.AutotuneSession}
 * when the system should select among a candidate space, and use
 * {@code tuning.calibration.PlatformCalibrationSession} when runtime-platform
 * knobs should be learned across calibration workloads.</p>
 *
 * <p>The request normalizes optional policies to defaults and defensively copies
 * {@link #entries()}. At most one entry may be marked as
 * {@link BenchmarkEntryRole#BASELINE}; reports use it only for relative speedup
 * calculations.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * BenchmarkRequest request = new BenchmarkRequest(
 *         workload,
 *         List.of(BenchmarkEntry.baseline("default", defaultProfile),
 *                 BenchmarkEntry.candidate("candidate-a", tunedProfile)),
 *         MeasurementPolicy.defaults(),
 *         ValidationPolicy.defaults(),
 *         ReportPolicy.defaults());
 * BenchmarkReport report = BenchmarkSession.create(request).run();
 * }</pre>
 *
 * @param workload workload to instantiate once per entry; required
 * @param entries profiles to validate and measure; {@code null} becomes empty
 * @param measurement measurement controls; {@code null} means defaults
 * @param validation validation controls; {@code null} disables validation
 * @param report report rendering policy; {@code null} means defaults
 */
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
