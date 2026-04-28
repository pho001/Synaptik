package tuning.workload;

import tuning.measure.MeasurementPolicy;
import tuning.reporting.ReportPolicy;
import tuning.benchmark.BenchmarkEntry;
import tuning.benchmark.BenchmarkRequest;
import tuning.benchmark.BenchmarkSuiteRequest;
import tuning.autotune.TuningDefaults;
import tuning.preset.TuningPreset;
import tuning.validate.ValidationPolicy;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Registry of named workload specifications plus convenience request builders.
 *
 * <p>The catalog is mutable during setup and not synchronized. Register
 * workloads on one thread before sharing it for read-only request construction.</p>
 */
public final class WorkloadCatalog {
    private final Map<String, WorkloadSpec> workloads = new LinkedHashMap<>();

    /**
     * Registers a workload by {@link WorkloadSpec#name()}.
     *
     * @param workload workload to register
     * @return this catalog for fluent setup
     * @throws IllegalArgumentException if the name is blank or already registered
     */
    public WorkloadCatalog register(WorkloadSpec workload) {
        Objects.requireNonNull(workload, "workload cannot be null");
        String name = workload.name();
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("workload name cannot be blank");
        }
        if (workloads.containsKey(name)) {
            throw new IllegalArgumentException("Duplicate workload registration: " + name);
        }
        workloads.put(name, workload);
        return this;
    }

    /**
     * Resolves a workload by name.
     *
     * @param name registered workload name
     * @return workload specification
     * @throws IllegalArgumentException if no workload is registered with that name
     */
    public WorkloadSpec require(String name) {
        WorkloadSpec workload = workloads.get(name);
        if (workload == null) {
            throw new IllegalArgumentException("Unknown workload: " + name);
        }
        return workload;
    }

    /**
     * Resolves multiple workloads by name.
     *
     * @param names workload names; {@code null} or empty returns an empty list
     * @return immutable list of workload specs
     */
    public List<WorkloadSpec> requireAll(List<String> names) {
        if (names == null || names.isEmpty()) {
            return List.of();
        }
        List<WorkloadSpec> out = new ArrayList<>(names.size());
        for (String name : names) {
            out.add(require(name));
        }
        return List.copyOf(out);
    }

    /**
     * @return registered workload names in registration order
     */
    public List<String> names() {
        return List.copyOf(workloads.keySet().stream().toList());
    }

    /**
     * Builds a benchmark request for a registered workload with explicit policies.
     *
     * @param workloadName registered workload name
     * @param entries benchmark entries
     * @param measurement measurement policy
     * @param validation validation policy
     * @param report report policy
     * @return benchmark request
     */
    public BenchmarkRequest benchmarkRequest(
            String workloadName,
            List<BenchmarkEntry> entries,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            ReportPolicy report
    ) {
        return new BenchmarkRequest(require(workloadName), entries, measurement, validation, report);
    }

    /**
     * Builds a benchmark request using a tuning preset.
     *
     * @param workloadName registered workload name
     * @param entries benchmark entries
     * @param preset preset supplying measurement, validation, and reporting
     * @return benchmark request
     */
    public BenchmarkRequest benchmarkRequest(
            String workloadName,
            List<BenchmarkEntry> entries,
            TuningPreset preset
    ) {
        return TuningDefaults.benchmark(preset, require(workloadName), entries);
    }

    /**
     * Builds a benchmark request using the recommended preset for the workload.
     *
     * @param workloadName registered workload name
     * @param entries benchmark entries
     * @return benchmark request
     */
    public BenchmarkRequest benchmarkRequest(
            String workloadName,
            List<BenchmarkEntry> entries
    ) {
        return TuningDefaults.recommendedBenchmark(require(workloadName), entries);
    }

    /**
     * Builds a suite request for registered workloads with explicit policies.
     *
     * @param workloadNames registered workload names
     * @param entries benchmark entries reused for each workload
     * @param measurement measurement policy
     * @param validation validation policy
     * @param report report policy
     * @return benchmark suite request
     */
    public BenchmarkSuiteRequest benchmarkSuiteRequest(
            List<String> workloadNames,
            List<BenchmarkEntry> entries,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            ReportPolicy report
    ) {
        return new BenchmarkSuiteRequest(requireAll(workloadNames), entries, measurement, validation, report);
    }

    /**
     * Builds a suite request using one tuning preset.
     *
     * @param workloadNames registered workload names
     * @param entries benchmark entries reused for each workload
     * @param preset preset supplying measurement, validation, and reporting
     * @return benchmark suite request
     */
    public BenchmarkSuiteRequest benchmarkSuiteRequest(
            List<String> workloadNames,
            List<BenchmarkEntry> entries,
            TuningPreset preset
    ) {
        return TuningDefaults.benchmarkSuite(preset, requireAll(workloadNames), entries);
    }

    /**
     * Builds a suite request using the recommended suite preset.
     *
     * @param workloadNames registered workload names
     * @param entries benchmark entries reused for each workload
     * @return benchmark suite request
     */
    public BenchmarkSuiteRequest benchmarkSuiteRequest(
            List<String> workloadNames,
            List<BenchmarkEntry> entries
    ) {
        return TuningDefaults.recommendedBenchmarkSuite(requireAll(workloadNames), entries);
    }
}
