package tuning.session;

import java.util.Map;

public record PlatformCalibrationCandidateSummary(
        String candidateId,
        Map<String, String> knobAssignments,
        PlatformCalibrationScore score
) {
    public PlatformCalibrationCandidateSummary {
        candidateId = candidateId == null ? "" : candidateId;
        knobAssignments = knobAssignments == null ? Map.of() : Map.copyOf(knobAssignments);
        if (score == null) {
            score = PlatformCalibrationScore.invalid("missing score");
        }
    }
}
