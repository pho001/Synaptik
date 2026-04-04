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
        return new ValidationPolicy(1e-8, 1e-8, false, true);
    }

    public static ValidationPolicy thoroughValidation() {
        return new ValidationPolicy(1e-9, 1e-9, true, true);
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

    public static BenchmarkRequest quickBenchmark(WorkloadSpec workload, List<Candidate> candidates) {
        return new BenchmarkRequest(workload, candidates, quickMeasurement(), quickValidation(), defaultReportPolicy());
    }

    public static BenchmarkSuiteRequest quickBenchmarkSuite(List<WorkloadSpec> workloads, List<Candidate> candidates) {
        return new BenchmarkSuiteRequest(workloads, candidates, quickMeasurement(), quickValidation(), defaultReportPolicy());
    }

    public static AutotuneRequest quickAutotune(WorkloadSpec workload, CandidateSpace candidateSpace) {
        return new AutotuneRequest(workload, candidateSpace, quickMeasurement(), quickValidation(), quickSearchPolicy(), PersistencePolicy.disabled());
    }

    public static AutotuneRequest balancedAutotune(WorkloadSpec workload, CandidateSpace candidateSpace, PersistencePolicy persistence) {
        return new AutotuneRequest(workload, candidateSpace, balancedMeasurement(), quickValidation(), balancedSearchPolicy(), persistence);
    }

    public static AutotuneRequest thoroughAutotune(WorkloadSpec workload, CandidateSpace candidateSpace, PersistencePolicy persistence) {
        return new AutotuneRequest(workload, candidateSpace, thoroughMeasurement(), thoroughValidation(), thoroughSearchPolicy(), persistence);
    }

    public static CandidateSpace singleCandidate(ExecutionProfile profile) {
        return new ListCandidateSpace(List.of(new Candidate(profile.candidateName(), profile)));
    }
}
