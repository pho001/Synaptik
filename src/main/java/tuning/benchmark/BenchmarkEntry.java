package tuning.benchmark;

import config.profile.ExecutionProfile;
import tuning.candidate.Candidate;

import java.util.Objects;

/**
 * Profile entry supplied to benchmark and calibration measurement flows.
 *
 * <p>A baseline entry is measured like any other entry but excluded from
 * best-candidate selection and used for speedup calculations. A candidate entry
 * participates in best-candidate selection.</p>
 *
 * @param name display and report name; blank values become {@code "entry"}
 * @param role candidate or baseline role; {@code null} becomes candidate
 * @param profile execution profile to validate and measure; required
 */
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

    /**
     * Creates a candidate benchmark entry.
     *
     * @param name report name
     * @param profile profile to measure
     * @return candidate entry
     */
    public static BenchmarkEntry candidate(String name, ExecutionProfile profile) {
        return new BenchmarkEntry(name, BenchmarkEntryRole.CANDIDATE, profile);
    }

    /**
     * Creates a baseline benchmark entry.
     *
     * @param name report name
     * @param profile baseline profile to measure
     * @return baseline entry
     */
    public static BenchmarkEntry baseline(String name, ExecutionProfile profile) {
        return new BenchmarkEntry(name, BenchmarkEntryRole.BASELINE, profile);
    }

    /**
     * Converts this benchmark entry into the candidate representation consumed by
     * measurement and validation engines.
     *
     * @return candidate with this entry's name and profile
     */
    public Candidate toCandidate() {
        return new Candidate(name, profile);
    }
}
