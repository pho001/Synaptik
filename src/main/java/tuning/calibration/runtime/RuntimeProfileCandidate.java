package tuning.calibration.runtime;

import config.profile.PlatformRuntimeProfile;

import java.util.Map;
import java.util.Objects;

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
