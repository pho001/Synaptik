package tuning.candidate;

import tuning.workload.WorkloadSpec;

import java.util.List;

public record ListCandidateSpace(
        List<Candidate> candidates
) implements CandidateSpace {
    public ListCandidateSpace {
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
    }

    @Override
    public List<Candidate> generate(WorkloadSpec workload) {
        return candidates;
    }
}
