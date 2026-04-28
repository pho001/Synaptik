package tuning.api;

import config.profile.ExecutionProfile;
import tuning.autotune.TuningDefaults;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSession;
import tuning.benchmark.report.BenchmarkReport;
import tuning.measure.MeasurementPolicy;
import tuning.preset.TuningPreset;
import tuning.reporting.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Fluent builder for benchmarking explicit execution profiles.
 *
 * <p>The builder is a convenience layer over {@link BenchmarkRequest} and {@link BenchmarkSession}.
 * It never searches or mutates profiles; it only measures the entries provided by the caller.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * BenchmarkReport report = Synaptik.tuning()
 *         .benchmark()
 *         .workload(StandardWorkloads.matmul("m64", 1, 64, 64, 64))
 *         .quick()
 *         .report().hotStepLimit(5).includeTrace().done()
 *         .baseline("no-opt", baselineProfile)
 *         .candidate("calibrated", calibratedProfile)
 *         .run();
 * }</pre>
 *
 * <p>This builder is mutable and not thread-safe. Build a new instance for each benchmark run.</p>
 */
public final class BenchmarkDsl {
    private WorkloadSpec workload;
    private final List<BenchmarkEntry> entries = new ArrayList<>();
    private TuningPreset preset = TuningPreset.BALANCED;
    private MeasurementPolicy measurement;
    private ValidationPolicy validation;
    private ReportPolicy report;

    /**
     * Selects the workload to benchmark.
     *
     * @param workload workload spec; required before {@link #toRequest()} or {@link #run()}
     * @return this builder
     */
    public BenchmarkDsl workload(WorkloadSpec workload) {
        this.workload = Objects.requireNonNull(workload, "workload cannot be null");
        return this;
    }

    /**
     * Selects a workload from {@link StandardWorkloads#defaultCatalog()}.
     *
     * @param workloadName catalog workload name
     * @return this builder
     */
    public BenchmarkDsl workload(String workloadName) {
        return workload(StandardWorkloads.defaultCatalog().require(workloadName));
    }

    /**
     * Selects the standard ABC sequence-matmul BLAS benchmark workload.
     *
     * @param name workload name to report
     * @return this builder
     */
    public BenchmarkDsl abcBlasBenchmark(String name) {
        return workload(StandardWorkloads.abcSequenceMatmulBlasBenchmark(name));
    }

    /**
     * Marker for readability in chains that compare entries. It has no side effects.
     *
     * @return this builder
     */
    public BenchmarkDsl compare() {
        return this;
    }

    /**
     * Adds the baseline entry used for relative speedup calculations.
     *
     * @param name display name
     * @param profile execution profile to measure
     * @return this builder
     */
    public BenchmarkDsl baseline(String name, ExecutionProfile profile) {
        entries.add(BenchmarkEntry.baseline(name, profile));
        return this;
    }

    /**
     * Adds a candidate entry.
     *
     * @param name display name
     * @param profile execution profile to measure
     * @return this builder
     */
    public BenchmarkDsl candidate(String name, ExecutionProfile profile) {
        entries.add(BenchmarkEntry.candidate(name, profile));
        return this;
    }

    /**
     * Sets the preset used for default benchmark measurement, validation, and report policy.
     *
     * @param preset preset; {@code null} falls back to {@code BALANCED}
     * @return this builder
     */
    public BenchmarkDsl preset(TuningPreset preset) {
        this.preset = preset == null ? TuningPreset.BALANCED : preset;
        return this;
    }

    /**
     * Uses the quick preset.
     *
     * @return this builder
     */
    public BenchmarkDsl quick() {
        return preset(TuningPreset.QUICK);
    }

    /**
     * Uses the balanced preset.
     *
     * @return this builder
     */
    public BenchmarkDsl balanced() {
        return preset(TuningPreset.BALANCED);
    }

    /**
     * Uses the thorough preset.
     *
     * @return this builder
     */
    public BenchmarkDsl thorough() {
        return preset(TuningPreset.THOROUGH);
    }

    /**
     * Uses an explicit measurement policy instead of the preset's benchmark measurement.
     *
     * @param measurement measurement policy
     * @return this builder
     */
    public BenchmarkDsl measurement(MeasurementPolicy measurement) {
        this.measurement = measurement;
        return this;
    }

    /**
     * Uses an explicit validation policy instead of the preset's benchmark validation.
     *
     * @param validation validation policy
     * @return this builder
     */
    public BenchmarkDsl validation(ValidationPolicy validation) {
        this.validation = validation;
        return this;
    }

    /**
     * Disables benchmark validation.
     *
     * @return this builder
     */
    public BenchmarkDsl validationDisabled() {
        validation = ValidationPolicy.disabled();
        return this;
    }

    /**
     * Opens the grouped benchmark-report selector.
     *
     * <p>Use this for dot-style report configuration, for example
     * {@code benchmark().report().hotStepLimit(5).includeTrace().done()}.</p>
     *
     * @return report selector
     */
    public Reports report() {
        return new Reports(this);
    }

    /**
     * Uses an explicit report policy.
     *
     * @param report report policy
     * @return this builder
     */
    public BenchmarkDsl report(ReportPolicy report) {
        this.report = report;
        return this;
    }

    /**
     * Builds the low-level benchmark request represented by this fluent configuration.
     *
     * @return benchmark request
     */
    public BenchmarkRequest toRequest() {
        if (workload == null) {
            throw new IllegalStateException("Benchmark workload must be selected before building a request.");
        }
        BenchmarkRequest defaults = TuningDefaults.benchmark(preset, workload, List.copyOf(entries));
        return new BenchmarkRequest(
                workload,
                List.copyOf(entries),
                measurement == null ? defaults.measurement() : measurement,
                validation == null ? defaults.validation() : validation,
                report == null ? defaults.report() : report
        );
    }

    /**
     * Runs the benchmark through the existing {@link BenchmarkSession}.
     *
     * @return benchmark report
     */
    public BenchmarkReport run() {
        return BenchmarkSession.create(toRequest()).run();
    }

    private ReportPolicy resolvedReport() {
        return report == null ? preset.reportPolicy() : report;
    }

    private BenchmarkDsl reportHotStepLimit(int hotStepLimit) {
        ReportPolicy current = resolvedReport();
        report = new ReportPolicy(hotStepLimit, current.includeTrace(), current.includeFailedCandidates());
        return this;
    }

    private BenchmarkDsl reportIncludeTrace(boolean includeTrace) {
        ReportPolicy current = resolvedReport();
        report = new ReportPolicy(current.hotStepLimit(), includeTrace, current.includeFailedCandidates());
        return this;
    }

    private BenchmarkDsl reportIncludeFailedCandidates(boolean includeFailedCandidates) {
        ReportPolicy current = resolvedReport();
        report = new ReportPolicy(current.hotStepLimit(), current.includeTrace(), includeFailedCandidates);
        return this;
    }

    /**
     * Grouped report selector used by {@link BenchmarkDsl#report()}.
     */
    public static final class Reports {
        private final BenchmarkDsl parent;

        private Reports(BenchmarkDsl parent) {
            this.parent = parent;
        }

        /**
         * Uses the default report policy.
         *
         * @return parent builder
         */
        public BenchmarkDsl defaults() {
            return parent.report(ReportPolicy.defaults());
        }

        /**
         * Uses a compact report: no hot-step details and no trace detail.
         *
         * @return parent builder
         */
        public BenchmarkDsl compact() {
            return parent.report(new ReportPolicy(0, false, true));
        }

        /**
         * Uses a detailed report with trace detail, failed candidates, and the default hot-step limit.
         *
         * @return parent builder
         */
        public BenchmarkDsl detailed() {
            return parent.report(ReportPolicy.defaults());
        }

        /**
         * Uses an explicit report policy.
         *
         * @param policy report policy
         * @return parent builder
         */
        public BenchmarkDsl policy(ReportPolicy policy) {
            return parent.report(policy);
        }

        /**
         * Sets how many hot execution steps should be highlighted in rendered reports.
         *
         * @param hotStepLimit maximum hot-step rows; must be non-negative
         * @return this selector
         */
        public Reports hotStepLimit(int hotStepLimit) {
            parent.reportHotStepLimit(hotStepLimit);
            return this;
        }

        /**
         * Includes trace detail in reports.
         *
         * @return this selector
         */
        public Reports includeTrace() {
            parent.reportIncludeTrace(true);
            return this;
        }

        /**
         * Excludes trace detail from reports.
         *
         * @return this selector
         */
        public Reports excludeTrace() {
            parent.reportIncludeTrace(false);
            return this;
        }

        /**
         * Includes failed candidates in reports.
         *
         * @return this selector
         */
        public Reports includeFailedCandidates() {
            parent.reportIncludeFailedCandidates(true);
            return this;
        }

        /**
         * Excludes failed candidates from reports.
         *
         * @return this selector
         */
        public Reports excludeFailedCandidates() {
            parent.reportIncludeFailedCandidates(false);
            return this;
        }

        /**
         * Returns to the benchmark builder after changing report fields.
         *
         * @return parent builder
         */
        public BenchmarkDsl done() {
            return parent;
        }
    }
}
