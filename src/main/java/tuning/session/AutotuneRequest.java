package tuning.session;

import tuning.candidate.CandidateSpace;
import tuning.measure.MeasurementPolicy;
import tuning.search.SearchPolicy;
import tuning.store.PersistencePolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.Objects;

public record AutotuneRequest(
        WorkloadSpec workload,
        CandidateSpace candidateSpace,
        MeasurementPolicy measurement,
        ValidationPolicy validation,
        SearchPolicy search,
        PersistencePolicy persistence
) {
    public AutotuneRequest {
        Objects.requireNonNull(workload, "workload cannot be null");
        Objects.requireNonNull(candidateSpace, "candidateSpace cannot be null");
        measurement = measurement == null ? MeasurementPolicy.defaults() : measurement;
        validation = validation == null ? ValidationPolicy.disabled() : validation;
        search = search == null ? new SearchPolicy(64, 8, 4, true) : search;
        persistence = persistence == null ? PersistencePolicy.disabled() : persistence;
    }
}
