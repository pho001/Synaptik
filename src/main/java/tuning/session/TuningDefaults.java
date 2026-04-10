package tuning.session;

import config.profile.ExecutionProfile;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateSpace;
import tuning.candidate.ListCandidateSpace;
import tuning.measure.MeasurementPolicy;
import tuning.report.ReportPolicy;
import tuning.search.SearchPolicy;
import tuning.store.PersistencePolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;

public final class TuningDefaults {
    private TuningDefaults() {
    }


    public static MeasurementPolicy quickMeasurement() {
        return new MeasurementPolicy(0, 3, 1, true, true, true, true, false);
    }

    public static MeasurementPolicy balancedMeasurement() {
        return new MeasurementPolicy(2, 8, 3, true, true, true, true, false);
    }

    public static MeasurementPolicy thoroughMeasurement() {
        return new MeasurementPolicy(4, 16, 5, true, true, true, true, true);
    }

    public static ValidationPolicy quickValidation() {
        return ValidationPolicy.quickDTypeAware(false);
    }

    public static ValidationPolicy balancedValidation() {
        return ValidationPolicy.balancedDTypeAware(false);
    }

    public static ValidationPolicy thoroughValidation() {
        return ValidationPolicy.thoroughDTypeAware(true);
    }

    public static ReportPolicy defaultReportPolicy() {
        return ReportPolicy.defaults();
    }

    public static SearchPolicy quickSearchPolicy() {
        return new SearchPolicy(16, 2, 2, true);
    }

    public static SearchPolicy balancedSearchPolicy() {
        return new SearchPolicy(32, 4, 4, true);
    }

    public static SearchPolicy thoroughSearchPolicy() {
        return new SearchPolicy(96, 8, 6, true);
    }

    public static BenchmarkRequest benchmark(TuningPreset preset, WorkloadSpec workload, List<BenchmarkEntry> entries) {
        TuningPreset resolved = preset == null ? TuningPreset.QUICK : preset;
        return new BenchmarkRequest(
                workload,
                entries,
                resolved.benchmarkMeasurement(),
                resolved.benchmarkValidation(),
                resolved.reportPolicy()
        );
    }

    public static BenchmarkRequest recommendedBenchmark(WorkloadSpec workload, List<BenchmarkEntry> entries) {
        return benchmark(WorkloadPresetFamily.benchmarkPresetFor(workload), workload, entries);
    }

    public static BenchmarkRequest quickBenchmark(WorkloadSpec workload, List<BenchmarkEntry> entries) {
        return benchmark(TuningPreset.QUICK, workload, entries);
    }

    public static BenchmarkRequest balancedBenchmark(WorkloadSpec workload, List<BenchmarkEntry> entries) {
        return benchmark(TuningPreset.BALANCED, workload, entries);
    }

    public static BenchmarkRequest thoroughBenchmark(WorkloadSpec workload, List<BenchmarkEntry> entries) {
        return benchmark(TuningPreset.THOROUGH, workload, entries);
    }

    public static BenchmarkSuiteRequest benchmarkSuite(TuningPreset preset, List<WorkloadSpec> workloads, List<BenchmarkEntry> entries) {
        TuningPreset resolved = preset == null ? TuningPreset.QUICK : preset;
        return new BenchmarkSuiteRequest(
                workloads,
                entries,
                resolved.benchmarkMeasurement(),
                resolved.benchmarkValidation(),
                resolved.reportPolicy()
        );
    }

    public static BenchmarkSuiteRequest recommendedBenchmarkSuite(List<WorkloadSpec> workloads, List<BenchmarkEntry> entries) {
        return benchmarkSuite(WorkloadPresetFamily.benchmarkPresetForSuite(workloads), workloads, entries);
    }

    public static BenchmarkSuiteRequest quickBenchmarkSuite(List<WorkloadSpec> workloads, List<BenchmarkEntry> entries) {
        return benchmarkSuite(TuningPreset.QUICK, workloads, entries);
    }

    public static BenchmarkSuiteRequest balancedBenchmarkSuite(List<WorkloadSpec> workloads, List<BenchmarkEntry> entries) {
        return benchmarkSuite(TuningPreset.BALANCED, workloads, entries);
    }

    public static BenchmarkSuiteRequest thoroughBenchmarkSuite(List<WorkloadSpec> workloads, List<BenchmarkEntry> entries) {
        return benchmarkSuite(TuningPreset.THOROUGH, workloads, entries);
    }

    public static AutotuneRequest autotune(
            TuningPreset preset,
            WorkloadSpec workload,
            ExecutionProfile seedProfile,
            CandidateSpace candidateSpace,
            PersistencePolicy persistence
    ) {
        TuningPreset resolved = preset == null ? TuningPreset.QUICK : preset;
        return AutotuneRequest.fromSeedExecutionProfile(
                workload,
                seedProfile,
                candidateSpace,
                resolved.autotuneMeasurement(),
                resolved.autotuneValidation(),
                resolved.autotuneSearch(),
                persistence == null ? PersistencePolicy.disabled() : persistence,
                null
        );
    }

    public static AutotuneRequest recommendedAutotune(
            WorkloadSpec workload,
            ExecutionProfile seedProfile,
            CandidateSpace candidateSpace,
            PersistencePolicy persistence
    ) {
        return autotune(WorkloadPresetFamily.autotunePresetFor(workload), workload, seedProfile, candidateSpace, persistence);
    }

    public static AutotuneRequest quickAutotune(WorkloadSpec workload, ExecutionProfile seedProfile, CandidateSpace candidateSpace) {
        return autotune(TuningPreset.QUICK, workload, seedProfile, candidateSpace, PersistencePolicy.disabled());
    }

    public static AutotuneRequest balancedAutotune(WorkloadSpec workload, ExecutionProfile seedProfile, CandidateSpace candidateSpace, PersistencePolicy persistence) {
        return autotune(TuningPreset.BALANCED, workload, seedProfile, candidateSpace, persistence);
    }

    public static AutotuneRequest thoroughAutotune(WorkloadSpec workload, ExecutionProfile seedProfile, CandidateSpace candidateSpace, PersistencePolicy persistence) {
        return autotune(TuningPreset.THOROUGH, workload, seedProfile, candidateSpace, persistence);
    }

    public static CandidateSpace singleCandidate(ExecutionProfile profile) {
        return new ListCandidateSpace(List.of(new Candidate(profile.candidateName(), profile)));
    }
}
