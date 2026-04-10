package tuning.session;

import config.profile.ExecutionProfile;
import tuning.candidate.Candidate;

import java.util.Objects;

public record BenchmarkEntry(
        String name,
        BenchmarkEntryRole role,
        ExecutionProfile profile
) {
    public BenchmarkEntry {
        name = (name == null || name.isBlank()) ? "entry" : name;
        role = role == null ? BenchmarkEntryRole.CANDIDATE : role;
        Objects.requireNonNull(profile, "profile cannot be null");
    }

    public static BenchmarkEntry candidate(String name, ExecutionProfile profile) {
        return new BenchmarkEntry(name, BenchmarkEntryRole.CANDIDATE, profile);
    }

    public static BenchmarkEntry baseline(String name, ExecutionProfile profile) {
        return new BenchmarkEntry(name, BenchmarkEntryRole.BASELINE, profile);
    }

    public Candidate toCandidate() {
        return new Candidate(name, profile);
    }
}
