package tuning.search;

import tuning.candidate.Candidate;

import java.util.List;

public final class ExhaustiveSearchStrategy implements SearchStrategy {
    @Override
    public SearchResult search(SearchContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        List<Candidate> generated = context.candidateSpace().generate(context.request().workload());
        int limit = Math.min(context.request().search().maxCandidates(), generated.size());
        List<Candidate> selected = generated.subList(0, limit);
        Candidate preferred = selected.isEmpty() ? null : selected.getFirst();
        return new SearchResult(selected, preferred);
    }
}
