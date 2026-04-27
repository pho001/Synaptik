package tuning.search;

import tuning.benchmark.report.BenchmarkCandidateReport;

import java.util.List;
import java.util.Set;

public interface SearchStrategy {
    SearchResult search(SearchContext context);

    default boolean supportsRefinement() {
        return false;
    }

    default SearchResult refine(
            SearchContext context,
            List<BenchmarkCandidateReport> evaluatedSoFar,
            int round,
            Set<String> seenFingerprints
    ) {
        return new SearchResult(List.of(), null);
    }
}
