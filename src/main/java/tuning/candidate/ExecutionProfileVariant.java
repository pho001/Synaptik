package tuning.candidate;

import config.profile.ExecutionProfile;

import java.util.Objects;

public record ExecutionProfileVariant(
        String suffix,
        ExecutionProfile profile
) {
    public ExecutionProfileVariant {
        suffix = (suffix == null || suffix.isBlank()) ? "variant" : suffix;
        Objects.requireNonNull(profile, "profile cannot be null");
    }
}
