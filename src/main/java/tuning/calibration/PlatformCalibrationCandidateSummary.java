package tuning.calibration;

import java.util.Map;

/**
 * Scored summary for one runtime-profile candidate in a calibration step.
 *
 * @param candidateId candidate identifier
 * @param knobAssignments human-readable knob values changed by the candidate
 * @param score score assigned by the step's score policy
 */
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
