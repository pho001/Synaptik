package tuning.candidate;

import config.profile.ExecutionProfile;

import java.util.Objects;

/**
 * Candidate execution profile considered by benchmark, autotune, and validation
 * flows.
 *
 * <p>Candidate metadata describes where the profile came from and whether it is
 * production eligible. The profile itself remains the executable contract used
 * by measurement engines.</p>
 *
 * @param name candidate id used in progress, reports, and persistence
 * @param profile executable profile; required
 * @param kind high-level candidate category
 * @param metadata candidate provenance and eligibility metadata
 */
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

    /**
     * Creates a generic production-eligible candidate.
     *
     * @param name candidate name
     * @param profile executable profile
     */
    public Candidate(String name, ExecutionProfile profile) {
        this(name, profile, CandidateKind.GENERIC, CandidateMetadata.generic());
    }
}
