package io.github.pho001.synaptik.tools.tuning;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Read-only, bounded inspection of tuning artifacts and already-created tuning evidence.
 *
 * <p>Artifact inspection snapshots at most 16 MiB and exposes opaque fields only as byte lengths
 * and lowercase SHA-256 digests. A matching key still requires the current backend-owned decoder;
 * this class never decodes a decision, executes work, prepares a plan, or changes a cache file.
 * Path reads use one read-only channel. A same-length concurrent replacement is assessed only
 * through the bytes read and their checksum, and bytes appended after the final growth probe are
 * outside the completed snapshot.
 */
public final class TuningInspection {
    /**
     * Supplies a deterministic test seam for the bounded read-only Path snapshot algorithm.
     * Implementations report the size observed before reading and transfer bytes without taking
     * ownership of the destination buffer.
     */
    interface ReadChannel {
        /**
         * Returns the size observed before the snapshot allocation.
         *
         * @return the observed byte count, which must be non-negative
         * @throws IOException if the channel cannot report its size
         */
        long size() throws IOException;

        /**
         * Transfers bytes into the caller-owned destination at its current position.
         *
         * @param target non-null caller-owned destination; the implementation advances its
         *     position by the number of transferred bytes and does not retain it
         * @return the number of bytes transferred, or a negative value at end of input
         * @throws IOException if bytes cannot be read
         */
        int read(ByteBuffer target) throws IOException;
    }

    private TuningInspection() {
        throw new AssertionError("no instances");
    }

    /**
     * Structural state of an inspected artifact: absent Path, valid complete artifact, or
     * malformed bounded snapshot. Byte-array inspection cannot produce {@link #MISSING}.
     */
    public enum ArtifactStatus {
        /** The requested Path did not exist at read-only open. */
        MISSING,
        /** The complete snapshot satisfies the supported format. */
        VALID,
        /** The snapshot or its content violates a bounded format rule. */
        INVALID
    }

    /** Origin of the inspected snapshot, either one read-only Path channel or supplied bytes. */
    public enum ArtifactSource {
        /** Bytes were acquired through one read-only Path channel. */
        PATH,
        /** Bytes were defensively snapshotted from a caller-owned array. */
        BYTE_ARRAY
    }

    /**
     * Stable category for malformed artifact bytes. These values classify content and bounded
     * snapshot races; ordinary Path open and read failures remain {@link IOException}s.
     */
    public enum InvalidReason {
        /** The snapshot cannot contain the fixed header and trailing checksum. */
        SIZE_TOO_SMALL,
        /** The observed input exceeds the 16 MiB artifact bound. */
        OVERSIZE,
        /** The Path channel ended before its initially observed size. */
        TRUNCATED,
        /** The Path channel supplied a byte after the initially observed size. */
        GROWTH_DURING_READ,
        /** The trailing SHA-256 value does not match the payload. */
        CHECKSUM_MISMATCH,
        /** The format-identifying magic bytes do not match. */
        INVALID_MAGIC,
        /** The declared artifact schema is not supported. */
        UNSUPPORTED_SCHEMA,
        /** The entry count is negative or exceeds 65,536. */
        INVALID_ENTRY_COUNT,
        /** A length-prefixed opaque value is empty or exceeds 1 MiB. */
        INVALID_LENGTH,
        /** A key, value, numeric code, or timing summary violates its structure. */
        INVALID_STRUCTURE_OR_SUMMARY,
        /** Entries are duplicated or not in strict canonical order. */
        DUPLICATE_OR_NONCANONICAL_ORDER,
        /** Bytes remain between the final declared entry and checksum. */
        TRAILING_PAYLOAD
    }

    /**
     * Result of comparing stored key fields with an optional current expectation. An exact key
     * match still requires the current backend-owned decoder and is not a compatibility verdict.
     */
    public enum CompatibilityStatus {
        /** No current expectation was supplied. */
        NOT_EVALUATED,
        /** One or more stored fields differ from the expectation. */
        MISMATCH,
        /** Stored fields match, but the backend-owned decoder must authenticate the decision. */
        KEY_MATCH_REQUIRES_BACKEND_DECODER,
        /** The expectation declares session-only reuse, so a file entry cannot be reused. */
        SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE
    }

    /**
     * Directly provable stored-key mismatch. Reports order reasons by the corresponding cache-key
     * field order, with the session-scope reason last.
     */
    public enum MismatchReason {
        /** Stored producer schema differs. */
        PRODUCER_SCHEMA_MISMATCH,
        /** Stored producer bytes differ. */
        PRODUCER_VALUE_MISMATCH,
        /** Stored decision-codec schema differs. */
        DECISION_CODEC_SCHEMA_MISMATCH,
        /** Stored decision-codec bytes differ. */
        DECISION_CODEC_VALUE_MISMATCH,
        /** Stored model schema differs. */
        MODEL_SCHEMA_MISMATCH,
        /** Stored model bytes differ. */
        MODEL_VALUE_MISMATCH,
        /** Stored representative-profile schema differs. */
        PROFILE_SCHEMA_MISMATCH,
        /** Stored representative-profile bytes differ. */
        PROFILE_VALUE_MISMATCH,
        /** Stored target schema differs. */
        TARGET_SCHEMA_MISMATCH,
        /** Stored target bytes differ. */
        TARGET_VALUE_MISMATCH,
        /** Stored compatibility schema differs. */
        COMPATIBILITY_SCHEMA_MISMATCH,
        /** Stored compatibility bytes differ. */
        COMPATIBILITY_VALUE_MISMATCH,
        /** Stored selection objective differs. */
        OBJECTIVE_MISMATCH,
        /** Stored correctness-policy code differs. */
        CORRECTNESS_POLICY_MISMATCH,
        /** Stored policy-identity schema differs. */
        POLICY_SCHEMA_MISMATCH,
        /** Stored policy-identity bytes differ. */
        POLICY_VALUE_MISMATCH,
        /** Stored untimed warmup count differs. */
        WARMUP_COUNT_MISMATCH,
        /** Stored timed-sample count differs. */
        TIMED_SAMPLE_COUNT_MISMATCH,
        /** The current compatibility declares session-only reuse. */
        SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE
    }

    /**
     * Redacted description of a non-empty unversioned opaque value. The original bytes are not
     * retained or exposed.
     *
     * @param byteLength positive encoded length in bytes
     * @param sha256 non-null 64-character lowercase hexadecimal SHA-256 digest
     */
    public record OpaqueSummary(int byteLength, String sha256) {
        /**
         * Validates the redacted description.
         *
         * @param byteLength positive encoded length in bytes
         * @param sha256 non-null 64-character lowercase hexadecimal SHA-256 digest
         * @throws NullPointerException if {@code sha256} is null
         * @throws IllegalArgumentException if the length is not positive or the digest is not
         *     lowercase 64-character hexadecimal text
         */
        public OpaqueSummary {
            if (byteLength <= 0) throw new IllegalArgumentException("byteLength must be positive");
            requireDigest(sha256);
        }
    }

    /**
     * Redacted description of a non-empty schema-versioned opaque value. The original bytes are
     * not retained or exposed.
     *
     * @param schemaVersion positive producer-defined schema version
     * @param byteLength positive encoded length in bytes
     * @param sha256 non-null 64-character lowercase hexadecimal SHA-256 digest
     */
    public record VersionedOpaqueSummary(int schemaVersion, int byteLength, String sha256) {
        /**
         * Validates the redacted versioned description.
         *
         * @param schemaVersion positive producer-defined schema version
         * @param byteLength positive encoded length in bytes
         * @param sha256 non-null 64-character lowercase hexadecimal SHA-256 digest
         * @throws NullPointerException if {@code sha256} is null
         * @throws IllegalArgumentException if the schema or length is not positive or the digest
         *     is not lowercase 64-character hexadecimal text
         */
        public VersionedOpaqueSummary {
            if (schemaVersion <= 0) {
                throw new IllegalArgumentException("schemaVersion must be positive");
            }
            if (byteLength <= 0) throw new IllegalArgumentException("byteLength must be positive");
            requireDigest(sha256);
        }
    }

    /**
     * Immutable expected key for workload-cache comparison. The compatibility retains its
     * producer-declared reuse scope even though that scope is absent from the compact file.
     *
     * @param compatibility non-null current backend-owned compatibility identity
     * @param objective non-null current selection objective
     * @param warmupCount non-negative number of untimed warmups
     * @param timedSampleCount positive odd number of timed samples
     */
    public record WorkloadExpectation(
            WorkloadTuningRequest.WorkloadCompatibility compatibility,
            WorkloadTuningRequest.Objective objective,
            int warmupCount,
            int timedSampleCount) {
        /**
         * Validates all expected key fields without performing cache I/O.
         *
         * @param compatibility non-null current backend-owned compatibility identity
         * @param objective non-null current selection objective
         * @param warmupCount non-negative number of untimed warmups
         * @param timedSampleCount positive odd number of timed samples
         * @throws NullPointerException if {@code compatibility} or {@code objective} is null
         * @throws IllegalArgumentException if a sampling count violates its constraint
         */
        public WorkloadExpectation {
            Objects.requireNonNull(compatibility, "compatibility");
            Objects.requireNonNull(objective, "objective");
            validateSampling(warmupCount, timedSampleCount);
        }
    }

    /**
     * Immutable expected key for model-plan-cache comparison. Its compatibility retains the
     * producer-declared reuse scope, which is not persisted in the compact file.
     *
     * @param compatibility non-null current producer and decision-codec compatibility
     * @param modelFingerprint non-null current model evidence identity
     * @param profileFingerprint non-null current representative-profile identity
     * @param targetFingerprint non-null current target identity
     * @param objective non-null current selection objective
     * @param correctnessPolicy non-null current correctness policy
     * @param policyIdentity non-null current policy identity
     * @param warmupCount non-negative number of untimed warmups
     * @param timedSampleCount positive odd number of timed samples
     */
    public record ModelPlanExpectation(
            CompletePlanTuningRequest.PlanCompatibility compatibility,
            CompletePlanTuningRequest.ModelFingerprint modelFingerprint,
            CompletePlanTuningRequest.ProfileFingerprint profileFingerprint,
            CompletePlanTuningRequest.TargetFingerprint targetFingerprint,
            CompletePlanTuningRequest.Objective objective,
            CompletePlanTuningRequest.CorrectnessPolicy correctnessPolicy,
            CompletePlanTuningRequest.PolicyIdentity policyIdentity,
            int warmupCount,
            int timedSampleCount) {
        /**
         * Validates all expected key fields without performing cache I/O.
         *
         * @param compatibility non-null current producer and decision-codec compatibility
         * @param modelFingerprint non-null current model evidence identity
         * @param profileFingerprint non-null current representative-profile identity
         * @param targetFingerprint non-null current target identity
         * @param objective non-null current selection objective
         * @param correctnessPolicy non-null current correctness policy
         * @param policyIdentity non-null current policy identity
         * @param warmupCount non-negative number of untimed warmups
         * @param timedSampleCount positive odd number of timed samples
         * @throws NullPointerException if any reference component is null
         * @throws IllegalArgumentException if a sampling count violates its constraint
         */
        public ModelPlanExpectation {
            Objects.requireNonNull(compatibility, "compatibility");
            Objects.requireNonNull(modelFingerprint, "modelFingerprint");
            Objects.requireNonNull(profileFingerprint, "profileFingerprint");
            Objects.requireNonNull(targetFingerprint, "targetFingerprint");
            Objects.requireNonNull(objective, "objective");
            Objects.requireNonNull(correctnessPolicy, "correctnessPolicy");
            Objects.requireNonNull(policyIdentity, "policyIdentity");
            validateSampling(warmupCount, timedSampleCount);
        }
    }

    /**
     * One redacted compact workload-cache entry in canonical file order. The file carries no
     * model/profile identity, occurrence evidence, raw samples, correctness evidence, or
     * executable state.
     *
     * @param compatibility non-null redacted stored compatibility identity
     * @param objective stored positive wire value for the objective
     * @param warmupCount stored non-negative untimed warmup count
     * @param timedSampleCount stored positive odd timed-sample count
     * @param encodedDecision non-null redacted selected decision
     * @param selectedCandidate non-null redacted winner identity
     * @param minimumNanos minimum stored elapsed time in nanoseconds
     * @param medianNanos integer-middle stored median in nanoseconds
     * @param maximumNanos maximum stored elapsed time in nanoseconds
     * @param sampleCount stored positive sample count
     * @param compatibilityStatus non-null result of optional exact-key comparison
     * @param mismatches non-null deterministic mismatch sequence; defensively copied
     */
    public record WorkloadEntryInspection(
            VersionedOpaqueSummary compatibility,
            int objective,
            int warmupCount,
            int timedSampleCount,
            OpaqueSummary encodedDecision,
            OpaqueSummary selectedCandidate,
            long minimumNanos,
            long medianNanos,
            long maximumNanos,
            int sampleCount,
            CompatibilityStatus compatibilityStatus,
            List<MismatchReason> mismatches) {
        /**
         * Snapshots the deterministic mismatch list.
         *
         * @param compatibility non-null redacted stored compatibility identity
         * @param objective stored positive wire value for the objective
         * @param warmupCount stored non-negative untimed warmup count
         * @param timedSampleCount stored positive odd timed-sample count
         * @param encodedDecision non-null redacted selected decision
         * @param selectedCandidate non-null redacted winner identity
         * @param minimumNanos minimum stored elapsed time in nanoseconds
         * @param medianNanos integer-middle stored median in nanoseconds
         * @param maximumNanos maximum stored elapsed time in nanoseconds
         * @param sampleCount stored positive sample count
         * @param compatibilityStatus non-null result of optional exact-key comparison
         * @param mismatches non-null deterministic mismatch sequence; defensively copied
         * @throws NullPointerException if a reference component is null
         */
        public WorkloadEntryInspection {
            Objects.requireNonNull(compatibility, "compatibility");
            Objects.requireNonNull(encodedDecision, "encodedDecision");
            Objects.requireNonNull(selectedCandidate, "selectedCandidate");
            Objects.requireNonNull(compatibilityStatus, "compatibilityStatus");
            mismatches = List.copyOf(Objects.requireNonNull(mismatches, "mismatches"));
        }
    }

    /**
     * Complete read-only workload-cache report. A valid report has a positive schema, whole-file
     * digest, and canonical entries. Missing and invalid reports have no entries; nullable digest,
     * schema, and reason components follow the status invariant documented by the constructor.
     *
     * @param status non-null structural status
     * @param source non-null snapshot origin
     * @param observedByteCount non-negative observed file or array length in bytes
     * @param artifactSchema positive for valid artifacts, a safely read positive value for some
     *     invalid artifacts, or zero when unavailable or when the raw schema is non-positive
     * @param sha256 lowercase whole-file SHA-256 for valid artifacts and some invalid snapshots,
     *     otherwise null
     * @param invalidReason non-null only when {@code status} is {@code INVALID}
     * @param entries non-null canonical entries only when valid; defensively copied
     */
    public record WorkloadCacheInspection(
            ArtifactStatus status,
            ArtifactSource source,
            long observedByteCount,
            int artifactSchema,
            String sha256,
            InvalidReason invalidReason,
            List<WorkloadEntryInspection> entries) {
        /**
         * Snapshots entries and validates status-dependent report state.
         *
         * @param status non-null structural status
         * @param source non-null snapshot origin
         * @param observedByteCount non-negative observed file or array length in bytes
         * @param artifactSchema positive for valid artifacts, a safely read positive value for
         *     some invalid artifacts, or zero when unavailable or when the raw schema is
         *     non-positive
         * @param sha256 lowercase whole-file SHA-256 for valid artifacts and some invalid
         *     snapshots, otherwise null
         * @param invalidReason non-null only for invalid artifacts
         * @param entries non-null canonical entries only for valid artifacts; defensively copied
         * @throws NullPointerException if {@code status}, {@code source}, or {@code entries} is
         *     null, or if a required digest is null
         * @throws IllegalArgumentException if numeric metadata or status-dependent fields are
         *     inconsistent, or a present digest is malformed
         */
        public WorkloadCacheInspection {
            Objects.requireNonNull(status, "status");
            Objects.requireNonNull(source, "source");
            entries = List.copyOf(Objects.requireNonNull(entries, "entries"));
            validateArtifactState(status, observedByteCount, artifactSchema, sha256,
                    invalidReason, entries);
        }
    }

    /**
     * One redacted compact model-plan-cache entry in canonical file order. The file carries no
     * prepared recipe, decision meaning, correctness reference/action, raw sample, or executable.
     *
     * @param producer non-null redacted producer identity
     * @param decisionCodec non-null redacted decision-codec identity
     * @param model non-null redacted model identity
     * @param profile non-null redacted representative-profile identity
     * @param target non-null redacted target identity
     * @param compatibility non-null redacted plan compatibility
     * @param objective stored positive wire value for the objective
     * @param correctnessPolicy stored positive wire value for the correctness policy
     * @param policy non-null redacted policy identity
     * @param warmupCount stored non-negative untimed warmup count
     * @param timedSampleCount stored positive odd timed-sample count
     * @param encodedDecision non-null redacted selected decision
     * @param selectedCandidate non-null redacted winner identity
     * @param minimumNanos minimum stored elapsed time in nanoseconds
     * @param medianNanos integer-middle stored median in nanoseconds
     * @param maximumNanos maximum stored elapsed time in nanoseconds
     * @param sampleCount stored positive sample count
     * @param compatibilityStatus non-null result of optional exact-key comparison
     * @param mismatches non-null deterministic mismatch sequence; defensively copied
     */
    public record ModelPlanEntryInspection(
            VersionedOpaqueSummary producer,
            VersionedOpaqueSummary decisionCodec,
            VersionedOpaqueSummary model,
            VersionedOpaqueSummary profile,
            VersionedOpaqueSummary target,
            VersionedOpaqueSummary compatibility,
            int objective,
            int correctnessPolicy,
            VersionedOpaqueSummary policy,
            int warmupCount,
            int timedSampleCount,
            OpaqueSummary encodedDecision,
            OpaqueSummary selectedCandidate,
            long minimumNanos,
            long medianNanos,
            long maximumNanos,
            int sampleCount,
            CompatibilityStatus compatibilityStatus,
            List<MismatchReason> mismatches) {
        /**
         * Snapshots the deterministic mismatch list.
         *
         * @param producer non-null redacted producer identity
         * @param decisionCodec non-null redacted decision-codec identity
         * @param model non-null redacted model identity
         * @param profile non-null redacted representative-profile identity
         * @param target non-null redacted target identity
         * @param compatibility non-null redacted plan compatibility
         * @param objective stored positive wire value for the objective
         * @param correctnessPolicy stored positive wire value for the correctness policy
         * @param policy non-null redacted policy identity
         * @param warmupCount stored non-negative untimed warmup count
         * @param timedSampleCount stored positive odd timed-sample count
         * @param encodedDecision non-null redacted selected decision
         * @param selectedCandidate non-null redacted winner identity
         * @param minimumNanos minimum stored elapsed time in nanoseconds
         * @param medianNanos integer-middle stored median in nanoseconds
         * @param maximumNanos maximum stored elapsed time in nanoseconds
         * @param sampleCount stored positive sample count
         * @param compatibilityStatus non-null result of optional exact-key comparison
         * @param mismatches non-null deterministic mismatch sequence; defensively copied
         * @throws NullPointerException if a reference component is null
         */
        public ModelPlanEntryInspection {
            Objects.requireNonNull(producer, "producer");
            Objects.requireNonNull(decisionCodec, "decisionCodec");
            Objects.requireNonNull(model, "model");
            Objects.requireNonNull(profile, "profile");
            Objects.requireNonNull(target, "target");
            Objects.requireNonNull(compatibility, "compatibility");
            Objects.requireNonNull(policy, "policy");
            Objects.requireNonNull(encodedDecision, "encodedDecision");
            Objects.requireNonNull(selectedCandidate, "selectedCandidate");
            Objects.requireNonNull(compatibilityStatus, "compatibilityStatus");
            mismatches = List.copyOf(Objects.requireNonNull(mismatches, "mismatches"));
        }
    }

    /**
     * Complete read-only model-plan-cache report with the same status-dependent metadata
     * invariants as {@link WorkloadCacheInspection}.
     *
     * @param status non-null structural status
     * @param source non-null snapshot origin
     * @param observedByteCount non-negative observed file or array length in bytes
     * @param artifactSchema positive for valid artifacts, a safely read positive value for some
     *     invalid artifacts, or zero when unavailable or when the raw schema is non-positive
     * @param sha256 lowercase whole-file SHA-256 for valid artifacts and some invalid snapshots,
     *     otherwise null
     * @param invalidReason non-null only when {@code status} is {@code INVALID}
     * @param entries non-null canonical entries only when valid; defensively copied
     */
    public record ModelPlanCacheInspection(
            ArtifactStatus status,
            ArtifactSource source,
            long observedByteCount,
            int artifactSchema,
            String sha256,
            InvalidReason invalidReason,
            List<ModelPlanEntryInspection> entries) {
        /**
         * Snapshots entries and validates status-dependent report state.
         *
         * @param status non-null structural status
         * @param source non-null snapshot origin
         * @param observedByteCount non-negative observed file or array length in bytes
         * @param artifactSchema positive for valid artifacts, a safely read positive value for
         *     some invalid artifacts, or zero when unavailable or when the raw schema is
         *     non-positive
         * @param sha256 lowercase whole-file SHA-256 for valid artifacts and some invalid
         *     snapshots, otherwise null
         * @param invalidReason non-null only for invalid artifacts
         * @param entries non-null canonical entries only for valid artifacts; defensively copied
         * @throws NullPointerException if {@code status}, {@code source}, or {@code entries} is
         *     null, or if a required digest is null
         * @throws IllegalArgumentException if numeric metadata or status-dependent fields are
         *     inconsistent, or a present digest is malformed
         */
        public ModelPlanCacheInspection {
            Objects.requireNonNull(status, "status");
            Objects.requireNonNull(source, "source");
            entries = List.copyOf(Objects.requireNonNull(entries, "entries"));
            validateArtifactState(status, observedByteCount, artifactSchema, sha256,
                    invalidReason, entries);
        }
    }

    /**
     * Redacted workload occurrence context and its positive caller-defined weight.
     *
     * @param context non-null redacted occurrence context
     * @param weight positive caller-defined aggregation weight
     */
    public record OccurrenceEvidenceSummary(OpaqueSummary context, long weight) {
        /**
         * Validates immutable occurrence evidence.
         *
         * @param context non-null redacted occurrence context
         * @param weight positive caller-defined aggregation weight
         * @throws NullPointerException if {@code context} is null
         * @throws IllegalArgumentException if {@code weight} is not positive
         */
        public OccurrenceEvidenceSummary {
            Objects.requireNonNull(context, "context");
            if (weight <= 0) throw new IllegalArgumentException("weight must be positive");
        }
    }

    /**
     * Detached raw timing evidence for one measured workload candidate. All elapsed values and
     * summary values are nanoseconds.
     *
     * @param candidate non-null redacted candidate identity
     * @param elapsedSamplesNanos non-null ordered raw timed samples; defensively copied
     * @param minimumNanos minimum elapsed nanoseconds
     * @param medianNanos integer-middle median elapsed nanoseconds
     * @param maximumNanos maximum elapsed nanoseconds
     * @param sampleCount positive number of timed samples
     */
    public record CandidateMeasurementSummary(
            OpaqueSummary candidate,
            List<Long> elapsedSamplesNanos,
            long minimumNanos,
            long medianNanos,
            long maximumNanos,
            int sampleCount) {
        /**
         * Snapshots raw nanosecond samples.
         *
         * @param candidate non-null redacted candidate identity
         * @param elapsedSamplesNanos non-null ordered raw timed samples; defensively copied
         * @param minimumNanos minimum elapsed nanoseconds
         * @param medianNanos integer-middle median elapsed nanoseconds
         * @param maximumNanos maximum elapsed nanoseconds
         * @param sampleCount positive number of timed samples
         * @throws NullPointerException if {@code candidate}, the sample list, or a sample is null
         */
        public CandidateMeasurementSummary {
            Objects.requireNonNull(candidate, "candidate");
            elapsedSamplesNanos = List.copyOf(
                    Objects.requireNonNull(elapsedSamplesNanos, "elapsedSamplesNanos"));
        }
    }

    /**
     * Detached rich evidence for one deduplicated workload. Cache-hit source values have no
     * fabricated candidate measurement rows.
     *
     * @param compatibility non-null redacted compatibility identity
     * @param reuseScope non-null producer-declared reuse scope
     * @param totalWeight positive sum of occurrence weights
     * @param occurrences non-null ordered detached occurrence summaries; defensively copied
     * @param source non-null measured or cache-hit source
     * @param candidates non-null ordered measured candidates, empty for cache hits; defensively
     *     copied
     * @param selectedCandidate non-null redacted winner identity
     * @param minimumNanos selected minimum elapsed nanoseconds
     * @param medianNanos selected integer-middle median elapsed nanoseconds
     * @param maximumNanos selected maximum elapsed nanoseconds
     * @param sampleCount selected positive sample count
     */
    public record WorkloadEvidenceSummary(
            VersionedOpaqueSummary compatibility,
            WorkloadTuningRequest.ReuseScope reuseScope,
            long totalWeight,
            List<OccurrenceEvidenceSummary> occurrences,
            WorkloadTuningResult.Source source,
            List<CandidateMeasurementSummary> candidates,
            OpaqueSummary selectedCandidate,
            long minimumNanos,
            long medianNanos,
            long maximumNanos,
            int sampleCount) {
        /**
         * Snapshots occurrence and candidate evidence.
         *
         * @param compatibility non-null redacted compatibility identity
         * @param reuseScope non-null producer-declared reuse scope
         * @param totalWeight positive sum of occurrence weights
         * @param occurrences non-null ordered detached occurrence summaries; defensively copied
         * @param source non-null measured or cache-hit source
         * @param candidates non-null ordered measured candidates, empty for cache hits;
         *     defensively copied
         * @param selectedCandidate non-null redacted winner identity
         * @param minimumNanos selected minimum elapsed nanoseconds
         * @param medianNanos selected integer-middle median elapsed nanoseconds
         * @param maximumNanos selected maximum elapsed nanoseconds
         * @param sampleCount selected positive sample count
         * @throws NullPointerException if a reference component or list element is null
         */
        public WorkloadEvidenceSummary {
            Objects.requireNonNull(compatibility, "compatibility");
            Objects.requireNonNull(reuseScope, "reuseScope");
            occurrences = List.copyOf(Objects.requireNonNull(occurrences, "occurrences"));
            Objects.requireNonNull(source, "source");
            candidates = List.copyOf(Objects.requireNonNull(candidates, "candidates"));
            Objects.requireNonNull(selectedCandidate, "selectedCandidate");
        }
    }

    /**
     * Detached rich workload-evidence report created from an existing result; construction does
     * no cache lookup, candidate enumeration, measurement, decoding, or execution.
     *
     * @param model non-null redacted model evidence identity
     * @param profile non-null redacted representative-profile identity
     * @param objective non-null selection objective
     * @param maximumDistinctCacheMisses non-negative miss budget
     * @param maximumCandidatesPerMiss positive per-miss candidate budget
     * @param warmupCount non-negative untimed warmup count
     * @param timedSampleCount positive odd timed-sample count
     * @param workloads non-null ordered detached workload summaries; defensively copied
     */
    public record WorkloadEvidenceInspection(
            VersionedOpaqueSummary model,
            VersionedOpaqueSummary profile,
            WorkloadTuningRequest.Objective objective,
            int maximumDistinctCacheMisses,
            int maximumCandidatesPerMiss,
            int warmupCount,
            int timedSampleCount,
            List<WorkloadEvidenceSummary> workloads) {
        /**
         * Snapshots the ordered workload summaries.
         *
         * @param model non-null redacted model evidence identity
         * @param profile non-null redacted representative-profile identity
         * @param objective non-null selection objective
         * @param maximumDistinctCacheMisses non-negative miss budget
         * @param maximumCandidatesPerMiss positive per-miss candidate budget
         * @param warmupCount non-negative untimed warmup count
         * @param timedSampleCount positive odd timed-sample count
         * @param workloads non-null ordered detached workload summaries; defensively copied
         * @throws NullPointerException if a reference component or workload is null
         */
        public WorkloadEvidenceInspection {
            Objects.requireNonNull(model, "model");
            Objects.requireNonNull(profile, "profile");
            Objects.requireNonNull(objective, "objective");
            workloads = List.copyOf(Objects.requireNonNull(workloads, "workloads"));
        }
    }

    /**
     * Detached correctness and raw timing evidence for one measured complete-plan candidate. All
     * elapsed values and summary values are nanoseconds.
     *
     * @param candidate non-null redacted versioned candidate identity
     * @param correctnessAction non-null recorded correctness outcome
     * @param elapsedSamplesNanos non-null ordered raw timed samples; defensively copied
     * @param minimumNanos minimum elapsed nanoseconds
     * @param medianNanos integer-middle median elapsed nanoseconds
     * @param maximumNanos maximum elapsed nanoseconds
     * @param sampleCount positive number of timed samples
     */
    public record CompleteCandidateMeasurementSummary(
            VersionedOpaqueSummary candidate,
            CompletePlanTuningResult.CorrectnessAction correctnessAction,
            List<Long> elapsedSamplesNanos,
            long minimumNanos,
            long medianNanos,
            long maximumNanos,
            int sampleCount) {
        /**
         * Snapshots raw nanosecond samples.
         *
         * @param candidate non-null redacted versioned candidate identity
         * @param correctnessAction non-null recorded correctness outcome
         * @param elapsedSamplesNanos non-null ordered raw timed samples; defensively copied
         * @param minimumNanos minimum elapsed nanoseconds
         * @param medianNanos integer-middle median elapsed nanoseconds
         * @param maximumNanos maximum elapsed nanoseconds
         * @param sampleCount positive number of timed samples
         * @throws NullPointerException if a reference component, sample list, or sample is null
         */
        public CompleteCandidateMeasurementSummary {
            Objects.requireNonNull(candidate, "candidate");
            Objects.requireNonNull(correctnessAction, "correctnessAction");
            elapsedSamplesNanos = List.copyOf(
                    Objects.requireNonNull(elapsedSamplesNanos, "elapsedSamplesNanos"));
        }
    }

    /**
     * Detached rich complete-plan evidence report created from an existing result. A cache-hit
     * source has no fabricated candidate correctness or measurement rows.
     *
     * @param model non-null redacted model identity
     * @param profile non-null redacted representative-profile identity
     * @param target non-null redacted target identity
     * @param objective non-null selection objective
     * @param correctnessPolicy non-null correctness policy
     * @param policy non-null redacted policy identity
     * @param source non-null measured or cache-hit source
     * @param candidates non-null ordered measured candidates, empty for cache hits; defensively
     *     copied
     * @param selectedCandidate non-null redacted selected candidate identity
     * @param minimumNanos selected minimum elapsed nanoseconds
     * @param medianNanos selected integer-middle median elapsed nanoseconds
     * @param maximumNanos selected maximum elapsed nanoseconds
     * @param sampleCount selected positive sample count
     */
    public record CompletePlanEvidenceInspection(
            VersionedOpaqueSummary model,
            VersionedOpaqueSummary profile,
            VersionedOpaqueSummary target,
            CompletePlanTuningRequest.Objective objective,
            CompletePlanTuningRequest.CorrectnessPolicy correctnessPolicy,
            VersionedOpaqueSummary policy,
            CompletePlanTuningResult.Source source,
            List<CompleteCandidateMeasurementSummary> candidates,
            VersionedOpaqueSummary selectedCandidate,
            long minimumNanos,
            long medianNanos,
            long maximumNanos,
            int sampleCount) {
        /**
         * Snapshots ordered complete-candidate evidence.
         *
         * @param model non-null redacted model identity
         * @param profile non-null redacted representative-profile identity
         * @param target non-null redacted target identity
         * @param objective non-null selection objective
         * @param correctnessPolicy non-null correctness policy
         * @param policy non-null redacted policy identity
         * @param source non-null measured or cache-hit source
         * @param candidates non-null ordered measured candidates, empty for cache hits;
         *     defensively copied
         * @param selectedCandidate non-null redacted selected candidate identity
         * @param minimumNanos selected minimum elapsed nanoseconds
         * @param medianNanos selected integer-middle median elapsed nanoseconds
         * @param maximumNanos selected maximum elapsed nanoseconds
         * @param sampleCount selected positive sample count
         * @throws NullPointerException if a reference component or candidate is null
         */
        public CompletePlanEvidenceInspection {
            Objects.requireNonNull(model, "model");
            Objects.requireNonNull(profile, "profile");
            Objects.requireNonNull(target, "target");
            Objects.requireNonNull(objective, "objective");
            Objects.requireNonNull(correctnessPolicy, "correctnessPolicy");
            Objects.requireNonNull(policy, "policy");
            Objects.requireNonNull(source, "source");
            candidates = List.copyOf(Objects.requireNonNull(candidates, "candidates"));
            Objects.requireNonNull(selectedCandidate, "selectedCandidate");
        }
    }

    /**
     * Inspects one workload-cache path through a read-only snapshot bounded to 16 MiB. Missing is
     * a report status, malformed content is {@code INVALID}, and no directory or file is changed.
     *
     * @param path non-null explicit cache path
     * @return immutable structural report; an absent target reports {@link ArtifactStatus#MISSING}
     * @throws NullPointerException if {@code path} is null
     * @throws IOException for ordinary open or read failures other than a missing target
     */
    public static WorkloadCacheInspection inspectWorkloadCache(Path path) throws IOException {
        return inspectWorkloadSnapshot(read(path), null);
    }

    /**
     * Exercises workload inspection through the deterministic snapshot seam.
     *
     * @param channel non-null read channel, used but not closed or retained
     * @return immutable structural report; never null
     * @throws NullPointerException if {@code channel} is null
     * @throws IOException if size or byte acquisition fails
     */
    static WorkloadCacheInspection inspectWorkloadCache(ReadChannel channel) throws IOException {
        return inspectWorkloadSnapshot(read(Objects.requireNonNull(channel, "channel")), null);
    }

    /**
     * Inspects and compares one workload-cache path without decoding decisions. Mismatches follow
     * stored key order; equality reports decoder-required, while session scope forbids reuse.
     *
     * @param path non-null explicit cache path
     * @param expectation non-null current expected key
     * @return immutable structural and exact-key report
     * @throws NullPointerException if an argument is null
     * @throws IOException for ordinary open or read failures other than a missing target
     */
    public static WorkloadCacheInspection inspectWorkloadCache(
            Path path, WorkloadExpectation expectation) throws IOException {
        Objects.requireNonNull(expectation, "expectation");
        return inspectWorkloadSnapshot(read(path), expectation);
    }

    /**
     * Inspects a snapshot of supplied workload-cache bytes, rejecting more than 16 MiB before a
     * second full allocation. Later caller mutation cannot affect the report.
     *
     * @param bytes non-null bytes, bounded before the defensive copy and never retained
     * @return immutable structural report; malformed bytes report {@code INVALID}, never missing
     * @throws NullPointerException if {@code bytes} is null
     */
    public static WorkloadCacheInspection inspectWorkloadCache(byte[] bytes) {
        return inspectWorkloadSnapshot(snapshot(bytes), null);
    }

    /**
     * Inspects and compares a bounded snapshot of supplied workload-cache bytes. Comparison is
     * deterministic and cannot authenticate the opaque decision.
     *
     * @param bytes non-null bytes, bounded before the defensive copy and never retained
     * @param expectation non-null current expected key
     * @return immutable structural and exact-key report; never null
     * @throws NullPointerException if an argument is null
     */
    public static WorkloadCacheInspection inspectWorkloadCache(
            byte[] bytes, WorkloadExpectation expectation) {
        Objects.requireNonNull(expectation, "expectation");
        return inspectWorkloadSnapshot(snapshot(bytes), expectation);
    }

    /**
     * Inspects one model-plan-cache path through a read-only snapshot bounded to 16 MiB. Missing
     * is a report status, malformed content is {@code INVALID}, and no file is changed.
     *
     * @param path non-null explicit cache path
     * @return immutable structural report; an absent target reports {@link ArtifactStatus#MISSING}
     * @throws NullPointerException if {@code path} is null
     * @throws IOException for ordinary open or read failures other than a missing target
     */
    public static ModelPlanCacheInspection inspectModelPlanCache(Path path) throws IOException {
        return inspectModelPlanSnapshot(read(path), null);
    }

    /**
     * Inspects and compares one model-plan-cache path without decoding decisions. Mismatches
     * follow stored key order; equality reports decoder-required, while session scope forbids
     * persistent reuse.
     *
     * @param path non-null explicit cache path
     * @param expectation non-null current expected key
     * @return immutable structural and exact-key report
     * @throws NullPointerException if an argument is null
     * @throws IOException for ordinary open or read failures other than a missing target
     */
    public static ModelPlanCacheInspection inspectModelPlanCache(
            Path path, ModelPlanExpectation expectation) throws IOException {
        Objects.requireNonNull(expectation, "expectation");
        return inspectModelPlanSnapshot(read(path), expectation);
    }

    /**
     * Inspects a snapshot of supplied model-plan-cache bytes, rejecting more than 16 MiB before a
     * second full allocation. Later caller mutation cannot affect the report.
     *
     * @param bytes non-null bytes, bounded before the defensive copy and never retained
     * @return immutable structural report; malformed bytes report {@code INVALID}, never missing
     * @throws NullPointerException if {@code bytes} is null
     */
    public static ModelPlanCacheInspection inspectModelPlanCache(byte[] bytes) {
        return inspectModelPlanSnapshot(snapshot(bytes), null);
    }

    /**
     * Inspects and compares a bounded snapshot of supplied model-plan-cache bytes. Comparison is
     * deterministic and cannot authenticate the opaque decision or prove executability.
     *
     * @param bytes non-null bytes, bounded before the defensive copy and never retained
     * @param expectation non-null current expected key
     * @return immutable structural and exact-key report; never null
     * @throws NullPointerException if an argument is null
     */
    public static ModelPlanCacheInspection inspectModelPlanCache(
            byte[] bytes, ModelPlanExpectation expectation) {
        Objects.requireNonNull(expectation, "expectation");
        return inspectModelPlanSnapshot(snapshot(bytes), expectation);
    }

    /**
     * Creates a detached redacted view of existing rich workload evidence. This performs no file
     * access, candidate enumeration, measurement, decoding, preparation, or execution; cache-hit
     * evidence remains without fabricated candidate rows.
     *
     * @param evidence non-null already-created evidence
     * @return immutable detached summary retaining raw sample values and their nanosecond units
     * @throws NullPointerException if {@code evidence} is null
     */
    public static WorkloadEvidenceInspection summarize(WorkloadTuningResult.Evidence evidence) {
        Objects.requireNonNull(evidence, "evidence");
        List<WorkloadEvidenceSummary> workloads = new ArrayList<>();
        for (WorkloadTuningResult.WorkloadEvidence workload : evidence.workloads()) {
            List<OccurrenceEvidenceSummary> occurrences = new ArrayList<>();
            for (WorkloadTuningResult.OccurrenceEvidence occurrence : workload.occurrences()) {
                occurrences.add(new OccurrenceEvidenceSummary(
                        opaque(occurrence.contextFingerprint()), occurrence.weight()));
            }
            List<CandidateMeasurementSummary> candidates = new ArrayList<>();
            for (WorkloadTuningResult.CandidateEvidence candidate : workload.candidates()) {
                WorkloadTuningResult.SampleSummary summary = candidate.summary();
                candidates.add(new CandidateMeasurementSummary(
                        opaque(candidate.identity().bytes()), candidate.elapsedSamplesNanos(),
                        summary.minimumNanos(), summary.medianNanos(), summary.maximumNanos(),
                        summary.sampleCount()));
            }
            WorkloadTuningResult.SampleSummary winner = workload.winnerSummary();
            workloads.add(new WorkloadEvidenceSummary(
                    versioned(workload.compatibility().schemaVersion(),
                            workload.compatibility().bytes()),
                    workload.compatibility().reuseScope(), workload.totalWeight(), occurrences,
                    workload.source(), candidates, opaque(workload.winnerIdentity().bytes()),
                    winner.minimumNanos(), winner.medianNanos(), winner.maximumNanos(),
                    winner.sampleCount()));
        }
        WorkloadTuningRequest.Budget budget = evidence.budget();
        return new WorkloadEvidenceInspection(
                versioned(evidence.modelFingerprint().schemaVersion(),
                        evidence.modelFingerprint().bytes()),
                versioned(evidence.profileFingerprint().schemaVersion(),
                        evidence.profileFingerprint().bytes()),
                evidence.objective(), budget.maximumDistinctCacheMisses(),
                budget.maximumCandidatesPerMiss(), budget.warmupCount(),
                budget.timedSampleCount(), workloads);
    }

    /**
     * Creates a detached redacted view of existing rich complete-plan evidence. This performs no
     * file access, correctness work, measurement, decoding, preparation, or execution; cache-hit
     * evidence remains without fabricated candidate rows.
     *
     * @param evidence non-null already-created evidence
     * @return immutable detached summary retaining correctness actions and raw nanosecond samples
     * @throws NullPointerException if {@code evidence} is null
     */
    public static CompletePlanEvidenceInspection summarize(
            CompletePlanTuningResult.Evidence evidence) {
        Objects.requireNonNull(evidence, "evidence");
        List<CompleteCandidateMeasurementSummary> candidates = new ArrayList<>();
        for (CompletePlanTuningResult.CandidateEvidence candidate : evidence.candidates()) {
            CompletePlanTuningResult.SampleSummary summary = candidate.summary();
            candidates.add(new CompleteCandidateMeasurementSummary(
                    versioned(candidate.identity().schemaVersion(), candidate.identity().bytes()),
                    candidate.correctnessAction(), candidate.elapsedSamplesNanos(),
                    summary.minimumNanos(), summary.medianNanos(), summary.maximumNanos(),
                    summary.sampleCount()));
        }
        CompletePlanTuningResult.SampleSummary selected = evidence.selectedSummary();
        return new CompletePlanEvidenceInspection(
                versioned(evidence.modelFingerprint().schemaVersion(),
                        evidence.modelFingerprint().bytes()),
                versioned(evidence.profileFingerprint().schemaVersion(),
                        evidence.profileFingerprint().bytes()),
                versioned(evidence.targetFingerprint().schemaVersion(),
                        evidence.targetFingerprint().bytes()),
                evidence.objective(), evidence.correctnessPolicy(),
                versioned(evidence.policyIdentity().schemaVersion(),
                        evidence.policyIdentity().bytes()),
                evidence.source(), candidates,
                versioned(evidence.selectedCandidateIdentity().schemaVersion(),
                        evidence.selectedCandidateIdentity().bytes()),
                selected.minimumNanos(), selected.medianNanos(), selected.maximumNanos(),
                selected.sampleCount());
    }

    private static WorkloadCacheInspection inspectWorkloadSnapshot(
            Snapshot snapshot, WorkloadExpectation expectation) {
        if (snapshot.missing) return missingWorkload();
        if (snapshot.failure != null) {
            return invalidWorkload(snapshot, snapshot.failure);
        }
        try {
            WorkloadCacheFile.Parsed parsed = WorkloadCacheFile.parse(snapshot.bytes);
            List<WorkloadEntryInspection> entries = new ArrayList<>();
            for (WorkloadCacheFile.Entry entry : parsed.entries()) {
                WorkloadCacheFile.Key key = entry.key();
                Comparison comparison = compare(key, expectation);
                WorkloadTuningResult.SampleSummary summary = entry.summary();
                entries.add(new WorkloadEntryInspection(
                        versioned(key.compatibilitySchema(), key.compatibility()), key.objective(),
                        key.warmupCount(), key.timedSampleCount(),
                        opaque(entry.encodedDecision()), opaque(entry.winnerIdentity()),
                        summary.minimumNanos(), summary.medianNanos(), summary.maximumNanos(),
                        summary.sampleCount(), comparison.status, comparison.mismatches));
            }
            return new WorkloadCacheInspection(ArtifactStatus.VALID, snapshot.source,
                    snapshot.observedBytes, parsed.artifactSchema(), digest(snapshot.bytes), null,
                    entries);
        } catch (WorkloadCacheFile.FormatException exception) {
            return invalidWorkload(snapshot, exception.reason());
        }
    }

    private static ModelPlanCacheInspection inspectModelPlanSnapshot(
            Snapshot snapshot, ModelPlanExpectation expectation) {
        if (snapshot.missing) return missingModelPlan();
        if (snapshot.failure != null) {
            return invalidModelPlan(snapshot, snapshot.failure);
        }
        try {
            ModelPlanCacheFile.Parsed parsed = ModelPlanCacheFile.parse(snapshot.bytes);
            List<ModelPlanEntryInspection> entries = new ArrayList<>();
            for (ModelPlanCacheFile.Entry entry : parsed.entries()) {
                ModelPlanCacheFile.Key key = entry.key();
                Comparison comparison = compare(key, expectation);
                CompletePlanTuningResult.SampleSummary summary = entry.summary();
                entries.add(new ModelPlanEntryInspection(
                        versioned(key.producerSchema(), key.producer()),
                        versioned(key.codecSchema(), key.codec()),
                        versioned(key.modelSchema(), key.model()),
                        versioned(key.profileSchema(), key.profile()),
                        versioned(key.targetSchema(), key.target()),
                        versioned(key.compatibilitySchema(), key.compatibility()), key.objective(),
                        key.correctnessPolicy(), versioned(key.policySchema(), key.policy()),
                        key.warmupCount(), key.timedSampleCount(), opaque(entry.decision()),
                        opaque(entry.winner()), summary.minimumNanos(), summary.medianNanos(),
                        summary.maximumNanos(), summary.sampleCount(), comparison.status,
                        comparison.mismatches));
            }
            return new ModelPlanCacheInspection(ArtifactStatus.VALID, snapshot.source,
                    snapshot.observedBytes, parsed.artifactSchema(), digest(snapshot.bytes), null,
                    entries);
        } catch (ModelPlanCacheFile.FormatException exception) {
            return invalidModelPlan(snapshot, exception.reason());
        }
    }

    private static Comparison compare(
            WorkloadCacheFile.Key key, WorkloadExpectation expectation) {
        if (expectation == null) return Comparison.notEvaluated();
        List<MismatchReason> mismatches = new ArrayList<>();
        WorkloadTuningRequest.WorkloadCompatibility compatibility = expectation.compatibility();
        mismatch(key.compatibilitySchema() != compatibility.schemaVersion(), mismatches,
                MismatchReason.COMPATIBILITY_SCHEMA_MISMATCH);
        mismatch(!Arrays.equals(key.compatibility(), compatibility.bytes()), mismatches,
                MismatchReason.COMPATIBILITY_VALUE_MISMATCH);
        mismatch(key.objective() != expectation.objective().ordinal() + 1, mismatches,
                MismatchReason.OBJECTIVE_MISMATCH);
        mismatch(key.warmupCount() != expectation.warmupCount(), mismatches,
                MismatchReason.WARMUP_COUNT_MISMATCH);
        mismatch(key.timedSampleCount() != expectation.timedSampleCount(), mismatches,
                MismatchReason.TIMED_SAMPLE_COUNT_MISMATCH);
        boolean session = compatibility.reuseScope() == WorkloadTuningRequest.ReuseScope.SESSION;
        mismatch(session, mismatches, MismatchReason.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE);
        return finishComparison(mismatches, session);
    }

    private static Comparison compare(
            ModelPlanCacheFile.Key key, ModelPlanExpectation expectation) {
        if (expectation == null) return Comparison.notEvaluated();
        List<MismatchReason> mismatches = new ArrayList<>();
        CompletePlanTuningRequest.PlanCompatibility compatibility = expectation.compatibility();
        compareVersioned(key.producerSchema(), key.producer(),
                compatibility.producerIdentity().schemaVersion(),
                compatibility.producerIdentity().bytes(), MismatchReason.PRODUCER_SCHEMA_MISMATCH,
                MismatchReason.PRODUCER_VALUE_MISMATCH, mismatches);
        compareVersioned(key.codecSchema(), key.codec(),
                compatibility.decisionCodecIdentity().schemaVersion(),
                compatibility.decisionCodecIdentity().bytes(),
                MismatchReason.DECISION_CODEC_SCHEMA_MISMATCH,
                MismatchReason.DECISION_CODEC_VALUE_MISMATCH, mismatches);
        compareVersioned(key.modelSchema(), key.model(),
                expectation.modelFingerprint().schemaVersion(),
                expectation.modelFingerprint().bytes(), MismatchReason.MODEL_SCHEMA_MISMATCH,
                MismatchReason.MODEL_VALUE_MISMATCH, mismatches);
        compareVersioned(key.profileSchema(), key.profile(),
                expectation.profileFingerprint().schemaVersion(),
                expectation.profileFingerprint().bytes(), MismatchReason.PROFILE_SCHEMA_MISMATCH,
                MismatchReason.PROFILE_VALUE_MISMATCH, mismatches);
        compareVersioned(key.targetSchema(), key.target(),
                expectation.targetFingerprint().schemaVersion(),
                expectation.targetFingerprint().bytes(), MismatchReason.TARGET_SCHEMA_MISMATCH,
                MismatchReason.TARGET_VALUE_MISMATCH, mismatches);
        compareVersioned(key.compatibilitySchema(), key.compatibility(),
                compatibility.schemaVersion(), compatibility.bytes(),
                MismatchReason.COMPATIBILITY_SCHEMA_MISMATCH,
                MismatchReason.COMPATIBILITY_VALUE_MISMATCH, mismatches);
        mismatch(key.objective() != expectation.objective().ordinal() + 1, mismatches,
                MismatchReason.OBJECTIVE_MISMATCH);
        mismatch(key.correctnessPolicy() != expectation.correctnessPolicy().ordinal() + 1,
                mismatches, MismatchReason.CORRECTNESS_POLICY_MISMATCH);
        compareVersioned(key.policySchema(), key.policy(),
                expectation.policyIdentity().schemaVersion(), expectation.policyIdentity().bytes(),
                MismatchReason.POLICY_SCHEMA_MISMATCH, MismatchReason.POLICY_VALUE_MISMATCH,
                mismatches);
        mismatch(key.warmupCount() != expectation.warmupCount(), mismatches,
                MismatchReason.WARMUP_COUNT_MISMATCH);
        mismatch(key.timedSampleCount() != expectation.timedSampleCount(), mismatches,
                MismatchReason.TIMED_SAMPLE_COUNT_MISMATCH);
        boolean session = compatibility.reuseScope() == CompletePlanTuningRequest.ReuseScope.SESSION;
        mismatch(session, mismatches, MismatchReason.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE);
        return finishComparison(mismatches, session);
    }

    private static Comparison finishComparison(List<MismatchReason> mismatches, boolean session) {
        if (mismatches.isEmpty()) {
            return new Comparison(CompatibilityStatus.KEY_MATCH_REQUIRES_BACKEND_DECODER, List.of());
        }
        if (session && mismatches.size() == 1) {
            return new Comparison(CompatibilityStatus.SESSION_SCOPE_FORBIDS_PERSISTENT_REUSE,
                    mismatches);
        }
        return new Comparison(CompatibilityStatus.MISMATCH, mismatches);
    }

    private static void compareVersioned(int actualSchema, byte[] actual, int expectedSchema,
            byte[] expected, MismatchReason schemaReason, MismatchReason valueReason,
            List<MismatchReason> mismatches) {
        mismatch(actualSchema != expectedSchema, mismatches, schemaReason);
        mismatch(!Arrays.equals(actual, expected), mismatches, valueReason);
    }

    private static void mismatch(boolean condition, List<MismatchReason> mismatches,
            MismatchReason reason) {
        if (condition) mismatches.add(reason);
    }

    private static Snapshot snapshot(byte[] supplied) {
        Objects.requireNonNull(supplied, "bytes");
        if (supplied.length > WorkloadCacheFile.MAX_FILE_BYTES) {
            return new Snapshot(ArtifactSource.BYTE_ARRAY, supplied.length, null, false,
                    InvalidReason.OVERSIZE);
        }
        return new Snapshot(ArtifactSource.BYTE_ARRAY, supplied.length, supplied.clone(), false,
                supplied.length < 8 + Integer.BYTES * 2 + 32
                        ? InvalidReason.SIZE_TOO_SMALL : null);
    }

    private static Snapshot read(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return read(new ReadChannel() {
                @Override
                public long size() throws IOException {
                    return channel.size();
                }

                @Override
                public int read(ByteBuffer target) throws IOException {
                    return channel.read(target);
                }
            });
        } catch (NoSuchFileException exception) {
            return new Snapshot(ArtifactSource.PATH, 0, null, true, null);
        }
    }

    private static Snapshot read(ReadChannel channel) throws IOException {
        long size = channel.size();
        if (size < 0) {
            throw new IOException("inspection channel returned a negative size");
        }
        if (size > WorkloadCacheFile.MAX_FILE_BYTES) {
            return new Snapshot(ArtifactSource.PATH, size, null, false,
                    InvalidReason.OVERSIZE);
        }
        byte[] bytes = new byte[Math.toIntExact(size)];
        ByteBuffer target = ByteBuffer.wrap(bytes);
        while (target.hasRemaining()) {
            int count = channel.read(target);
            if (count < 0) {
                return new Snapshot(ArtifactSource.PATH, size, null, false,
                        InvalidReason.TRUNCATED);
            }
        }
        if (channel.read(ByteBuffer.allocate(1)) >= 0) {
            return new Snapshot(ArtifactSource.PATH, size, bytes, false,
                    InvalidReason.GROWTH_DURING_READ);
        }
        return new Snapshot(ArtifactSource.PATH, size, bytes, false,
                size < 8 + Integer.BYTES * 2 + 32
                        ? InvalidReason.SIZE_TOO_SMALL : null);
    }

    private static WorkloadCacheInspection missingWorkload() {
        return new WorkloadCacheInspection(ArtifactStatus.MISSING, ArtifactSource.PATH, 0, 0,
                null, null, List.of());
    }

    private static ModelPlanCacheInspection missingModelPlan() {
        return new ModelPlanCacheInspection(ArtifactStatus.MISSING, ArtifactSource.PATH, 0, 0,
                null, null, List.of());
    }

    private static WorkloadCacheInspection invalidWorkload(
            Snapshot snapshot, InvalidReason reason) {
        return new WorkloadCacheInspection(ArtifactStatus.INVALID, snapshot.source,
                snapshot.observedBytes, safeSchema(snapshot.bytes, reason),
                snapshot.bytes == null ? null : digest(snapshot.bytes), reason, List.of());
    }

    private static ModelPlanCacheInspection invalidModelPlan(
            Snapshot snapshot, InvalidReason reason) {
        return new ModelPlanCacheInspection(ArtifactStatus.INVALID, snapshot.source,
                snapshot.observedBytes, safeSchema(snapshot.bytes, reason),
                snapshot.bytes == null ? null : digest(snapshot.bytes), reason, List.of());
    }

    private static int safeSchema(byte[] bytes, InvalidReason reason) {
        if (bytes == null || bytes.length < 8 + Integer.BYTES * 2 + 32
                || reason == InvalidReason.CHECKSUM_MISMATCH
                || reason == InvalidReason.INVALID_MAGIC
                || reason == InvalidReason.GROWTH_DURING_READ
                || reason == InvalidReason.TRUNCATED) return 0;
        int schema = ByteBuffer.wrap(bytes, 8, Integer.BYTES).getInt();
        return Math.max(schema, 0);
    }

    private static OpaqueSummary opaque(byte[] bytes) {
        return new OpaqueSummary(bytes.length, digest(bytes));
    }

    private static VersionedOpaqueSummary versioned(int schema, byte[] bytes) {
        return new VersionedOpaqueSummary(schema, bytes.length, digest(bytes));
    }

    private static String digest(byte[] bytes) {
        try {
            byte[] value = MessageDigest.getInstance("SHA-256").digest(bytes);
            return java.util.HexFormat.of().formatHex(value);
        } catch (NoSuchAlgorithmException exception) {
            throw new AssertionError("SHA-256 is required by the JDK", exception);
        }
    }

    private static void requireDigest(String value) {
        Objects.requireNonNull(value, "sha256");
        if (value.length() != 64) throw new IllegalArgumentException("invalid SHA-256 text");
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
                throw new IllegalArgumentException("SHA-256 must be lowercase hexadecimal");
            }
        }
    }

    private static void validateSampling(int warmupCount, int timedSampleCount) {
        if (warmupCount < 0) throw new IllegalArgumentException("warmupCount must be non-negative");
        if (timedSampleCount <= 0 || (timedSampleCount & 1) == 0) {
            throw new IllegalArgumentException("timedSampleCount must be positive and odd");
        }
    }

    private static void validateArtifactState(ArtifactStatus status, long observedBytes,
            int artifactSchema, String sha256, InvalidReason invalidReason, List<?> entries) {
        if (observedBytes < 0 || artifactSchema < 0) {
            throw new IllegalArgumentException("artifact metadata is negative");
        }
        if (status == ArtifactStatus.MISSING) {
            if (observedBytes != 0 || artifactSchema != 0 || sha256 != null
                    || invalidReason != null || !entries.isEmpty()) {
                throw new IllegalArgumentException("missing artifact report is inconsistent");
            }
        } else if (status == ArtifactStatus.VALID) {
            if (artifactSchema <= 0 || sha256 == null || invalidReason != null) {
                throw new IllegalArgumentException("valid artifact report is inconsistent");
            }
            requireDigest(sha256);
        } else {
            if (invalidReason == null || !entries.isEmpty()) {
                throw new IllegalArgumentException("invalid artifact report is inconsistent");
            }
            if (sha256 != null) requireDigest(sha256);
        }
    }

    private record Snapshot(ArtifactSource source, long observedBytes, byte[] bytes,
            boolean missing, InvalidReason failure) { }

    private record Comparison(CompatibilityStatus status, List<MismatchReason> mismatches) {
        Comparison {
            mismatches = List.copyOf(mismatches);
        }

        static Comparison notEvaluated() {
            return new Comparison(CompatibilityStatus.NOT_EVALUATED, List.of());
        }
    }
}
