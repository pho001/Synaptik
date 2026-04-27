package tuning.search;

import tuning.candidate.CandidateSpace;
import tuning.autotune.AutotuneRequest;

public record SearchContext(
        AutotuneRequest request,
        CandidateSpace candidateSpace
) {
}
