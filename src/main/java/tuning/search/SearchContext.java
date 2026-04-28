package tuning.search;

import tuning.candidate.CandidateSpace;
import tuning.autotune.AutotuneRequest;

/**
 * Immutable context supplied to search strategies.
 *
 * @param request autotune request being searched
 * @param candidateSpace candidate space selected by the request
 */
public record SearchContext(
        AutotuneRequest request,
        CandidateSpace candidateSpace
) {
}
