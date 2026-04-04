package tuning.search;

import tuning.candidate.Candidate;

import java.util.List;

public final class FirstKSearchStrategy implements SearchStrategy {
    private final int k;

    public FirstKSearchStrategy(int k) {
        this.k = Math.max(1, k);
    }

    @Override
    public SearchResult search(SearchContext context) {
        List<Candidate> generated = context.candidateSpace().generate(context.request().workload());
        int limit = Math.min(Math.min(k, context.request().search().maxCandidates()), generated.size());
        List<Candidate> selected = generated.subList(0, limit);
        return new SearchResult(selected, selected.isEmpty() ? null : selected.getFirst());
    }
}
