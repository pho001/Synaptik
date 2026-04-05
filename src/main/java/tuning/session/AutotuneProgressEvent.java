package tuning.session;

public record AutotuneProgressEvent(
        AutotuneProgressPhase phase,
        int round,
        int totalCandidateCount,
        int selectedCount,
        int evaluatedCount,
        int validCount,
        String candidateName,
        String candidateFingerprint,
        String bestCandidateName,
        double bestMedianMs,
        String message
) {
    public AutotuneProgressEvent {
        phase = phase == null ? AutotuneProgressPhase.STARTED : phase;
        candidateName = candidateName == null ? "" : candidateName;
        candidateFingerprint = candidateFingerprint == null ? "" : candidateFingerprint;
        bestCandidateName = bestCandidateName == null ? "" : bestCandidateName;
        message = message == null ? "" : message;
    }
}
