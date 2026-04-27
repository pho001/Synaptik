package tuning.candidate;

import config.profile.ExecutionProfile;

import java.util.Objects;

public record Candidate(
        String name,
        ExecutionProfile profile,
        CandidateKind kind,
        CandidateMetadata metadata
) {
    public Candidate {
        name = (name == null || name.isBlank()) ? "candidate" : name;
        Objects.requireNonNull(profile, "profile cannot be null");
        kind = kind == null ? CandidateKind.GENERIC : kind;
        metadata = metadata == null ? CandidateMetadata.generic() : metadata;
    }

    public Candidate(String name, ExecutionProfile profile) {
        this(name, profile, CandidateKind.GENERIC, CandidateMetadata.generic());
    }
}
