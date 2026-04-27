package tuning.search;

import tuning.candidate.Candidate;

import java.util.List;

public final class SingleCandidateSearchStrategy implements SearchStrategy {
    @Override
    public SearchResult search(SearchContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        List<Candidate> generated = context.candidateSpace().generate(context.request().workload());
        if (generated.isEmpty()) {
            return new SearchResult(List.of(), null);
        }
        Candidate selected = generated.getFirst();
        return new SearchResult(List.of(selected), selected);
    }
}
