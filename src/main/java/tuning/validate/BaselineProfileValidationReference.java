package tuning.validate;

import config.profile.ExecutionProfile;

import java.util.List;
import java.util.Objects;

public record BaselineProfileValidationReference(
        ExecutionProfile baselineProfile,
        List<String> gradientTargetLabels
) {
    public BaselineProfileValidationReference {
        Objects.requireNonNull(baselineProfile, "baselineProfile cannot be null");
        gradientTargetLabels = gradientTargetLabels == null ? List.of() : List.copyOf(gradientTargetLabels);
    }
}
