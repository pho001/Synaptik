package io.github.pho001.synaptik.config.tuning;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * Describes one declarative request for later model-autotuning composition.
 *
 * <p>Possessing this value means that model autotuning was requested. The record retains the
 * caller's immutable policy values and the exact explicit cache paths without accessing the
 * filesystem or selecting a default. It does not run tuning, enumerate candidates, interpret
 * backend configuration, read or write a cache, prepare an executable, or perform Engine or
 * Runtime lifecycle work.</p>
 *
 * <p>A later composition owner may translate these Config-owned values into the independent
 * tuning-tool request contract. That owner also supplies model identity, representative inputs,
 * tunable occurrences, and execution behavior; none of those values is part of this facade.</p>
 *
 * @param objective selection objective; must not be {@code null}; retained by reference
 * @param budget Phase-1 cache-miss and sampling bounds; must not be {@code null}; retained by
 *     reference; none of its values controls complete-plan Phase 2
 * @param representativeProfile opaque representative-profile identity; must not be {@code null};
 *     retained by reference
 * @param fallbackPolicy policy for a tuning transaction that cannot produce a complete result;
 *     must not be {@code null}; retained by reference
 * @param workloadCache explicit workload-cache file path; must not be {@code null}; the exact
 *     reference is retained without normalization or I/O
 * @param completePlanBudget independent complete-plan timing and resource bounds; must not be
 *     {@code null}; retained by reference
 * @param modelPlanCache explicit model-plan-cache path; must not be {@code null} or have an empty
 *     string form; the exact reference is retained without normalization or I/O
 */
public record ModelAutotuningConfig(
        Objective objective,
        Budget budget,
        RepresentativeProfileIdentity representativeProfile,
        FallbackPolicy fallbackPolicy,
        Path workloadCache,
        CompletePlanBudget completePlanBudget,
        Path modelPlanCache) {
    /**
     * Creates an immutable declarative model-autotuning request.
     *
     * @param objective selection objective; must not be {@code null}; retained by reference
     * @param budget Phase-1 cache-miss and sampling bounds; must not be {@code null}; retained by
     *     reference; none of its values controls complete-plan Phase 2
     * @param representativeProfile opaque representative-profile identity; must not be
     *     {@code null}; retained by reference
     * @param fallbackPolicy policy for a tuning transaction that cannot produce a complete
     *     result; must not be {@code null}; retained by reference
     * @param workloadCache explicit workload-cache file path; must not be {@code null}; the exact
     *     reference is retained without normalization or I/O
     * @param completePlanBudget independent complete-plan timing and resource bounds; must not be
     *     {@code null}; retained by reference
     * @param modelPlanCache explicit model-plan-cache path; must not be {@code null} or have an
     *     empty string form; the exact reference is retained without normalization or I/O
     * @throws NullPointerException if any component is {@code null}; the exception message names
     *     that component
     * @throws IllegalArgumentException if {@code modelPlanCache} has an empty string form
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
        if (completePlanBudget == null) {
            throw new NullPointerException("completePlanBudget");
        }
        if (modelPlanCache == null) {
            throw new NullPointerException("modelPlanCache");
        }
        if (modelPlanCache.toString().isEmpty()) {
            throw new IllegalArgumentException("modelPlanCache must not be empty");
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
     * Returns the Phase-1 cache-miss and sampling bounds.
     *
     * @return the exact non-null immutable Phase-1 budget retained at construction; none of its
     *     values controls complete-plan Phase 2
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

    /**
     * Returns the independent complete-plan timing and resource bounds.
     *
     * @return the exact non-null immutable complete-plan budget retained at construction
     */
    @Override
    public CompletePlanBudget completePlanBudget() {
        return completePlanBudget;
    }

    /**
     * Returns the explicit model-plan-cache path.
     *
     * <p>Later composition supplies this path to the complete-plan transaction. The current CPU
     * producer declares session-scoped reuse, so that transaction does not access the path. A
     * future persistent producer may cause the tuning owner to use it under that owner's
     * validation and publication rules.</p>
     *
     * @return the exact non-null path reference retained at construction; no filesystem state is
     *     inspected or changed
     */
    @Override
    public Path modelPlanCache() {
        return modelPlanCache;
    }

    /** Identifies how later tuning selects among already eligible complete candidates. */
    public enum Objective {
        /** Selects the first encountered candidate with the lowest median elapsed nanoseconds. */
        MIN_MEDIAN_ELAPSED_NANOS
    }

    /**
     * Declares fixed bounds for Phase-1 workload-cache misses and candidate sampling.
     *
     * <p>The maxima bound Phase-1 cache misses and their complete local-candidate sets; the warmup
     * and timed counts count complete local-candidate executions. None of these values controls
     * complete-plan Phase 2. This value has no wall-clock cutoff and performs no measurement.</p>
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
     * Declares independent fixed bounds for the later complete-plan tuning transaction.
     *
     * <p>The timing counts belong only to complete-plan Phase 2 and are independent from the
     * Phase-1 {@link Budget} counts. Later preflight checks the actual complete-plan candidate
     * count {@code N} against the total-execution ceiling with the checked formula
     * {@code N * (1 + W + S)}, where {@code W} and {@code S} are this value's warmup and timed
     * sample counts. This value performs no enumeration, correctness work, or measurement.</p>
     *
     * @param maximumPlanCandidates positive maximum complete-plan candidate count
     * @param warmupCount non-negative Phase-2 untimed execution count per candidate
     * @param timedSampleCount positive odd Phase-2 timed execution count per candidate
     * @param maximumTotalPlanExecutions positive maximum execution count for the whole Phase-2
     *     transaction
     * @param maximumAggregateCorrectnessBytes non-negative aggregate byte ceiling for canonical
     *     publication data captured and compared by later correctness collaboration; zero is valid
     */
    public record CompletePlanBudget(
            int maximumPlanCandidates,
            int warmupCount,
            int timedSampleCount,
            long maximumTotalPlanExecutions,
            long maximumAggregateCorrectnessBytes) {
        /**
         * Creates validated independent complete-plan timing and resource bounds.
         *
         * @param maximumPlanCandidates positive maximum complete-plan candidate count
         * @param warmupCount non-negative Phase-2 untimed execution count per candidate
         * @param timedSampleCount positive odd Phase-2 timed execution count per candidate
         * @param maximumTotalPlanExecutions positive maximum execution count for the whole
         *     Phase-2 transaction
         * @param maximumAggregateCorrectnessBytes non-negative aggregate byte ceiling for
         *     canonical publication data captured and compared by later correctness collaboration;
         *     zero is valid and is neither a cache-file limit nor a per-result allocation promise
         * @throws IllegalArgumentException if the candidate or total-execution maximum is not
         *     positive, the warmup count or correctness-byte ceiling is negative, or the timed
         *     sample count is not positive and odd
         */
        public CompletePlanBudget {
            if (maximumPlanCandidates <= 0) {
                throw new IllegalArgumentException("maximumPlanCandidates must be positive");
            }
            if (warmupCount < 0) {
                throw new IllegalArgumentException("warmupCount must be non-negative");
            }
            if (timedSampleCount <= 0 || (timedSampleCount & 1) == 0) {
                throw new IllegalArgumentException(
                        "timedSampleCount must be positive and odd");
            }
            if (maximumTotalPlanExecutions <= 0) {
                throw new IllegalArgumentException(
                        "maximumTotalPlanExecutions must be positive");
            }
            if (maximumAggregateCorrectnessBytes < 0) {
                throw new IllegalArgumentException(
                        "maximumAggregateCorrectnessBytes must be non-negative");
            }
        }

        /**
         * Returns the maximum actual candidate count accepted by later Phase-2 preflight.
         *
         * @return the positive maximum complete-plan candidate count
         */
        @Override
        public int maximumPlanCandidates() {
            return maximumPlanCandidates;
        }

        /**
         * Returns the Phase-2 untimed execution count for each complete-plan candidate.
         *
         * @return the non-negative Phase-2 warmup count per candidate
         */
        @Override
        public int warmupCount() {
            return warmupCount;
        }

        /**
         * Returns the Phase-2 timed execution count for each complete-plan candidate.
         *
         * @return the positive odd Phase-2 timed sample count per candidate
         */
        @Override
        public int timedSampleCount() {
            return timedSampleCount;
        }

        /**
         * Returns the whole-transaction Phase-2 execution ceiling.
         *
         * @return the positive ceiling later checked against {@code N * (1 + W + S)}
         */
        @Override
        public long maximumTotalPlanExecutions() {
            return maximumTotalPlanExecutions;
        }

        /**
         * Returns the aggregate canonical-publication correctness ceiling for later collaboration.
         *
         * @return the non-negative ceiling in bytes; zero is valid and is neither a cache-file
         *     limit nor a per-result allocation promise
         */
        @Override
        public long maximumAggregateCorrectnessBytes() {
            return maximumAggregateCorrectnessBytes;
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
