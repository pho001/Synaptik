package tuning.search;

import tuning.candidate.Candidate;

import java.util.List;

/**
 * Candidate selection returned by a search strategy.
 *
 * @param selectedCandidates candidates to evaluate in this batch; {@code null}
 * becomes empty
 * @param preferredCandidate optional strategy-selected leader before measurement
 */
public record SearchResult(
        List<Candidate> selectedCandidates,
        Candidate preferredCandidate
) {
    public SearchResult {
        selectedCandidates = selectedCandidates == null ? List.of() : List.copyOf(selectedCandidates);
    }
}
