package tuning.calibration.runtime;

import config.profile.PlatformRuntimeProfile;

import java.util.Map;
import java.util.Objects;

/**
 * Candidate runtime profile generated during platform calibration.
 *
 * <p>The candidate contains only platform runtime policy. Calibration sessions
 * assemble a full execution profile by combining it with the request's fixed
 * graph policy, dtype, and execution mode.</p>
 *
 * @param name candidate id used in reports and selection
 * @param runtimeProfile runtime profile represented by this candidate; required
 * @param knobAssignments changed knobs for progress, reports, and family validation
 */
public record RuntimeProfileCandidate(
        String name,
        PlatformRuntimeProfile runtimeProfile,
        Map<String, String> knobAssignments
) {
    public RuntimeProfileCandidate {
        name = name == null || name.isBlank() ? "candidate" : name;
        Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        knobAssignments = knobAssignments == null ? Map.of() : Map.copyOf(knobAssignments);
    }
}
