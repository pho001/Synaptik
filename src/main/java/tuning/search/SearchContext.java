package tuning.search;

import tuning.candidate.CandidateSpace;
import tuning.session.AutotuneRequest;

public record SearchContext(
        AutotuneRequest request,
        CandidateSpace candidateSpace
) {
}
