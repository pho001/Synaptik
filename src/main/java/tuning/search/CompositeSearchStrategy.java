package tuning.search;

import tuning.candidate.Candidate;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;

public final class CompositeSearchStrategy implements SearchStrategy {
    private final List<SearchStrategy> delegates;

    public CompositeSearchStrategy(List<SearchStrategy> delegates) {
        this.delegates = delegates == null ? List.of() : List.copyOf(delegates);
    }

    @Override
    public SearchResult search(SearchContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        LinkedHashMap<String, Candidate> merged = new LinkedHashMap<>();
        Candidate preferred = null;

        for (SearchStrategy delegate : delegates) {
            SearchResult result = delegate.search(context);
            if (preferred == null && result.preferredCandidate() != null) {
                preferred = result.preferredCandidate();
            }
            for (Candidate candidate : result.selectedCandidates()) {
                merged.putIfAbsent(candidate.name(), candidate);
                if (merged.size() >= context.request().search().maxCandidates()) {
                    break;
                }
            }
            if (merged.size() >= context.request().search().maxCandidates()) {
                break;
            }
        }

        List<Candidate> selected = new ArrayList<>(merged.values());
        if (preferred == null && !selected.isEmpty()) {
            preferred = selected.getFirst();
        }
        return new SearchResult(List.copyOf(selected), preferred);
    }
}
