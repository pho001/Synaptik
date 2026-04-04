package tuning.workload;

import tuning.candidate.Candidate;
import tuning.measure.MeasurementPolicy;
import tuning.report.ReportPolicy;
import tuning.session.BenchmarkRequest;
import tuning.session.BenchmarkSuiteRequest;
import tuning.session.TuningDefaults;
import tuning.session.TuningPreset;
import tuning.validate.ValidationPolicy;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class WorkloadCatalog {
    private final Map<String, WorkloadSpec> workloads = new LinkedHashMap<>();

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

    public WorkloadSpec require(String name) {
        WorkloadSpec workload = workloads.get(name);
        if (workload == null) {
            throw new IllegalArgumentException("Unknown workload: " + name);
        }
        return workload;
    }

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

    public List<String> names() {
        return List.copyOf(workloads.keySet().stream().toList());
    }

    public BenchmarkRequest benchmarkRequest(
            String workloadName,
            List<Candidate> candidates,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            ReportPolicy report
    ) {
        return new BenchmarkRequest(require(workloadName), candidates, measurement, validation, report);
    }

    public BenchmarkRequest benchmarkRequest(
            String workloadName,
            List<Candidate> candidates,
            TuningPreset preset
    ) {
        return TuningDefaults.benchmark(preset, require(workloadName), candidates);
    }

    public BenchmarkSuiteRequest benchmarkSuiteRequest(
            List<String> workloadNames,
            List<Candidate> candidates,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            ReportPolicy report
    ) {
        return new BenchmarkSuiteRequest(requireAll(workloadNames), candidates, measurement, validation, report);
    }

    public BenchmarkSuiteRequest benchmarkSuiteRequest(
            List<String> workloadNames,
            List<Candidate> candidates,
            TuningPreset preset
    ) {
        return TuningDefaults.benchmarkSuite(preset, requireAll(workloadNames), candidates);
    }
}
