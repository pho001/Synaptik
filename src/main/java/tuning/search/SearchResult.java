package tuning.search;

import tuning.candidate.Candidate;

import java.util.List;

public record SearchResult(
        List<Candidate> selectedCandidates,
        Candidate preferredCandidate
) {
    public SearchResult {
        selectedCandidates = selectedCandidates == null ? List.of() : List.copyOf(selectedCandidates);
    }
}
