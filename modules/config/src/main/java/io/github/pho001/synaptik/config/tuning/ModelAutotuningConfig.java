package io.github.pho001.synaptik.config.tuning;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * Describes one declarative request for later model-autotuning composition.
 *
 * <p>Possessing this value means that model autotuning was requested. The record retains the
 * caller's immutable policy values and the exact explicit workload-cache path without accessing
 * the filesystem or selecting a default. It does not run tuning, enumerate candidates, interpret
 * backend configuration, read or write a cache, prepare an executable, or perform Engine or
 * Runtime lifecycle work.</p>
 *
 * <p>A later composition owner may translate these Config-owned values into the independent
 * tuning-tool request contract. That owner also supplies model identity, representative inputs,
 * tunable occurrences, and execution behavior; none of those values is part of this facade.</p>
 *
 * @param objective selection objective; must not be {@code null}; retained by reference
 * @param budget complete-work and sampling bounds; must not be {@code null}; retained by
 *     reference
 * @param representativeProfile opaque representative-profile identity; must not be {@code null};
 *     retained by reference
 * @param fallbackPolicy policy for a tuning transaction that cannot produce a complete result;
 *     must not be {@code null}; retained by reference
 * @param workloadCache explicit workload-cache file path; must not be {@code null}; the exact
 *     reference is retained without normalization or I/O
 */
public record ModelAutotuningConfig(
        Objective objective,
        Budget budget,
        RepresentativeProfileIdentity representativeProfile,
        FallbackPolicy fallbackPolicy,
        Path workloadCache) {
    /**
     * Creates an immutable declarative model-autotuning request.
     *
     * @param objective selection objective; must not be {@code null}; retained by reference
     * @param budget complete-work and sampling bounds; must not be {@code null}; retained by
     *     reference
     * @param representativeProfile opaque representative-profile identity; must not be
     *     {@code null}; retained by reference
     * @param fallbackPolicy policy for a tuning transaction that cannot produce a complete
     *     result; must not be {@code null}; retained by reference
     * @param workloadCache explicit workload-cache file path; must not be {@code null}; the exact
     *     reference is retained without normalization or I/O
     * @throws NullPointerException if any component is {@code null}; the exception message names
     *     that component
     */
    public ModelAutotuningConfig {
        if (objective == null) {
            throw new NullPointerException("objective");
        }
        if (budget == null) {
            throw new NullPointerException("budget");
        }
        if (representativeProfile == null) {
            throw new NullPointerException("representativeProfile");
        }
        if (fallbackPolicy == null) {
            throw new NullPointerException("fallbackPolicy");
        }
        if (workloadCache == null) {
            throw new NullPointerException("workloadCache");
        }
    }

    /**
     * Returns the requested selection objective.
     *
     * @return the exact non-null objective retained at construction
     */
    @Override
    public Objective objective() {
        return objective;
    }

    /**
     * Returns the complete-work and sampling bounds.
     *
     * @return the exact non-null immutable budget retained at construction
     */
    @Override
    public Budget budget() {
        return budget;
    }

    /**
     * Returns the identity of the representative profile for this request.
     *
     * @return the exact non-null immutable identity retained at construction
     */
    @Override
    public RepresentativeProfileIdentity representativeProfile() {
        return representativeProfile;
    }

    /**
     * Returns the policy for a tuning transaction that cannot produce a complete result.
     *
     * @return the exact non-null fallback policy retained at construction
     */
    @Override
    public FallbackPolicy fallbackPolicy() {
        return fallbackPolicy;
    }

    /**
     * Returns the explicit workload-cache file path.
     *
     * @return the exact non-null path reference retained at construction; no filesystem state is
     *     inspected or changed
     */
    @Override
    public Path workloadCache() {
        return workloadCache;
    }

    /** Identifies how later tuning selects among already eligible complete candidates. */
    public enum Objective {
        /** Selects the first encountered candidate with the lowest median elapsed nanoseconds. */
        MIN_MEDIAN_ELAPSED_NANOS
    }

    /**
     * Declares fixed bounds for complete cache-miss work and candidate sampling.
     *
     * <p>The maxima bound complete candidate sets; the warmup and timed counts count complete
     * candidate executions. This value has no wall-clock cutoff and performs no measurement.</p>
     *
     * @param maximumDistinctCacheMisses positive maximum number of distinct cache misses
     * @param maximumCandidatesPerMiss positive maximum complete candidate count for each miss
     * @param warmupCount non-negative number of untimed complete executions per candidate
     * @param timedSampleCount positive odd number of timed complete executions per candidate
     */
    public record Budget(
            int maximumDistinctCacheMisses,
            int maximumCandidatesPerMiss,
            int warmupCount,
            int timedSampleCount) {
        /**
         * Creates a validated fixed sampling budget.
         *
         * @param maximumDistinctCacheMisses positive maximum number of distinct cache misses
         * @param maximumCandidatesPerMiss positive maximum complete candidate count for each miss
         * @param warmupCount non-negative number of untimed complete executions per candidate
         * @param timedSampleCount positive odd number of timed complete executions per candidate
         * @throws IllegalArgumentException if either maximum is not positive, the warmup count is
         *     negative, or the timed sample count is not positive and odd
         */
        public Budget {
            if (maximumDistinctCacheMisses <= 0) {
                throw new IllegalArgumentException(
                        "maximumDistinctCacheMisses must be positive");
            }
            if (maximumCandidatesPerMiss <= 0) {
                throw new IllegalArgumentException("maximumCandidatesPerMiss must be positive");
            }
            if (warmupCount < 0) {
                throw new IllegalArgumentException("warmupCount must be non-negative");
            }
            if (timedSampleCount <= 0 || (timedSampleCount & 1) == 0) {
                throw new IllegalArgumentException(
                        "timedSampleCount must be positive and odd");
            }
        }

        /**
         * Returns the maximum number of distinct cache misses.
         *
         * @return the positive maximum
         */
        @Override
        public int maximumDistinctCacheMisses() {
            return maximumDistinctCacheMisses;
        }

        /**
         * Returns the maximum complete candidate count for one cache miss.
         *
         * @return the positive maximum
         */
        @Override
        public int maximumCandidatesPerMiss() {
            return maximumCandidatesPerMiss;
        }

        /**
         * Returns the untimed complete-execution count per candidate.
         *
         * @return the non-negative warmup count
         */
        @Override
        public int warmupCount() {
            return warmupCount;
        }

        /**
         * Returns the timed complete-execution count per candidate.
         *
         * @return the positive odd sample count
         */
        @Override
        public int timedSampleCount() {
            return timedSampleCount;
        }
    }

    /**
     * Immutable caller-owned identity for the representative profile attached to tuning evidence.
     *
     * <p>The schema version tells the caller how to interpret the otherwise opaque byte sequence.
     * Construction and access both copy the sequence. Equality and hashing use the schema version
     * and byte content; diagnostic text reports only the version and byte length.</p>
     */
    public static final class RepresentativeProfileIdentity {
        private final int schemaVersion;
        private final byte[] bytes;

        /**
         * Creates an immutable snapshot of a representative-profile identity.
         *
         * @param schemaVersion positive caller-owned schema version
         * @param bytes non-null, non-empty opaque identity bytes; the array is snapshotted
         * @throws NullPointerException if {@code bytes} is {@code null}; the exception message is
         *     {@code bytes}
         * @throws IllegalArgumentException if {@code schemaVersion} is not positive or
         *     {@code bytes} is empty
         */
        public RepresentativeProfileIdentity(int schemaVersion, byte[] bytes) {
            if (schemaVersion <= 0) {
                throw new IllegalArgumentException("schemaVersion must be positive");
            }
            if (bytes == null) {
                throw new NullPointerException("bytes");
            }
            if (bytes.length == 0) {
                throw new IllegalArgumentException("bytes must not be empty");
            }
            this.schemaVersion = schemaVersion;
            this.bytes = bytes.clone();
        }

        /**
         * Returns the caller-owned schema version for the opaque identity bytes.
         *
         * @return the positive schema version
         */
        public int schemaVersion() {
            return schemaVersion;
        }

        /**
         * Returns the opaque representative-profile identity bytes.
         *
         * @return a fresh caller-owned copy of the non-empty identity sequence
         */
        public byte[] bytes() {
            return bytes.clone();
        }

        /**
         * Compares this identity with another value by schema version and byte content.
         *
         * @param other value to compare; may be {@code null}
         * @return {@code true} only when {@code other} is a representative-profile identity with
         *     the same schema version and byte content
         */
        @Override
        public boolean equals(Object other) {
            return this == other
                    || other instanceof RepresentativeProfileIdentity that
                            && schemaVersion == that.schemaVersion
                            && Arrays.equals(bytes, that.bytes);
        }

        /**
         * Returns a content-based hash consistent with {@link #equals(Object)}.
         *
         * @return the hash derived from the schema version and identity-byte content
         */
        @Override
        public int hashCode() {
            return 31 * schemaVersion + Arrays.hashCode(bytes);
        }

        /**
         * Returns diagnostic text that does not expose the opaque identity bytes.
         *
         * @return text containing only the schema version and byte length; never {@code null}
         */
        @Override
        public String toString() {
            return "RepresentativeProfileIdentity[schemaVersion="
                    + schemaVersion
                    + ", byteLength="
                    + bytes.length
                    + "]";
        }
    }

    /**
     * Declares how later composition proceeds when tuning cannot produce a complete result.
     *
     * <p>This policy grants no candidate eligibility, numerical relaxation, cache compatibility,
     * partial selection, or permission to suppress an unrelated preparation failure.</p>
     */
    public enum FallbackPolicy {
        /**
         * Requires later composition to report failure when the requested tuning transaction
         * cannot produce a complete selected result.
         */
        REQUIRE_TUNED_RESULT,

        /**
         * Permits later composition to abandon an unavailable or failed tuning transaction and
         * continue through ordinary safe heuristic preparation.
         *
         * <p>This does not turn corrupt or incompatible cache data into a hit, grant candidate
         * eligibility, accept a partial selection, suppress unrelated preparation failures, or
         * require Config to catch an exception.</p>
         */
        ALLOW_SAFE_HEURISTIC
    }
}
