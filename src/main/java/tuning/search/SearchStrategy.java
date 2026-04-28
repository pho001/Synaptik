package tuning.search;

import tuning.benchmark.report.BenchmarkCandidateReport;

import java.util.List;
import java.util.Set;

/**
 * Selects candidates from a {@link SearchContext} for autotune evaluation.
 *
 * <p>Search strategies should be deterministic for the same context and history.
 * They select candidates only; validation, measurement, and persistence remain
 * responsibilities of the autotune session.</p>
 */
public interface SearchStrategy {
    /**
     * Selects the initial batch of candidates.
     *
     * @param context request and candidate-space context
     * @return selected candidates and optional preferred candidate
     */
    SearchResult search(SearchContext context);

    /**
     * @return whether this strategy supports refinement after measurements
     */
    default boolean supportsRefinement() {
        return false;
    }

    /**
     * Selects an additional batch after prior candidates have been evaluated.
     *
     * @param context request and candidate-space context
     * @param evaluatedSoFar reports collected so far
     * @param round current refinement round, starting at one
     * @param seenFingerprints executable fingerprints already evaluated
     * @return additional candidates to evaluate; empty by default
     */
    default SearchResult refine(
            SearchContext context,
            List<BenchmarkCandidateReport> evaluatedSoFar,
            int round,
            Set<String> seenFingerprints
    ) {
        return new SearchResult(List.of(), null);
    }
}
