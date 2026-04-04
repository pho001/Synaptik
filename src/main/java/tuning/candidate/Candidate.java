package tuning.candidate;

import config.profile.ExecutionProfile;

import java.util.Objects;

public record Candidate(
        String name,
        ExecutionProfile profile
) {
    public Candidate {
        name = (name == null || name.isBlank()) ? "candidate" : name;
        Objects.requireNonNull(profile, "profile cannot be null");
    }
}
