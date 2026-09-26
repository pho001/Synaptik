package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable Metal-owned candidates and session compatibility for one supported-operation partition.
 *
 * <p>The stable candidate order begins with the current safe heuristic. The value owns no
 * measurement, cache location, native handle, executable, physical allocation, or Runtime state.
 * Shared Prepare can transport it only through the method-free
 * {@link BackendTuningCandidateBatch} role.</p>
 */
final class MetalNegTuningBatch implements BackendTuningCandidateBatch {
    /** Current candidate and decision meaning. */
    static final int CANDIDATE_SCHEMA_VERSION = 14;
    /** Current canonical workload/target compatibility meaning. */
    static final int COMPATIBILITY_SCHEMA_VERSION = 14;
    /** Current exact profile-qualified Metal operation-composition policy meaning. */
    static final int ROUTE_POLICY_VERSION = 14;

    /** Stable complete private route configurations. */
    enum Candidate {
        /** One-node, one-feed, one-target custom FLOAT32 NEG configuration. */
        CUSTOM_SINGLE_NEG(MetalPreparedRoute.CUSTOM_SINGLE_NEG),
        /** Whole-partition typed MPSGraph profile-qualified operation configuration. */
        MPSGRAPH(MetalPreparedRoute.MPSGRAPH),
        /** Fixed whole-partition Task-0052 custom-program configuration. */
        CUSTOM_TASK0052(MetalPreparedRoute.CUSTOM_TASK0052);

        private final MetalPreparedRoute route;

        Candidate(MetalPreparedRoute route) {
            this.route = Objects.requireNonNull(route, "route");
        }

        /** @return stable positive identity owned by the canonical prepared route */
        int wireIdentity() { return route.wireIdentity(); }

        /** @return exact canonical preparation route represented by this candidate */
        MetalPreparedRoute route() { return route; }

        /**
         * Resolves one schema-local identity through the canonical route owner.
         *
         * @param wireIdentity encoded positive identity
         * @return matching candidate, or empty for an unknown identity
         */
        static Optional<Candidate> fromWireIdentity(int wireIdentity) {
            Optional<MetalPreparedRoute> resolved =
                    MetalPreparedRoute.fromWireIdentity(wireIdentity);
            if (resolved.isEmpty()) return Optional.empty();
            return Optional.of(switch (resolved.orElseThrow()) {
                case CUSTOM_SINGLE_NEG -> CUSTOM_SINGLE_NEG;
                case MPSGRAPH -> MPSGRAPH;
                case CUSTOM_TASK0052 -> CUSTOM_TASK0052;
            });
        }
    }

    /**
     * Canonical bounded workload-fingerprint bytes independent of graph-local object identities.
     *
     * <p>The constructor and accessor defensively copy the byte sequence. Equality and hashing
     * compare byte content.</p>
     */
    static final class WorkloadSignature {
        private final byte[] bytes;

        /**
         * Snapshots canonical schema-thirteen workload-fingerprint bytes.
         *
         * <p>The bytes include the stable explicit numerical-profile wire identity, so otherwise
         * equal workloads under different profiles cannot share workload identity.</p>
         *
         * @param bytes non-null non-empty canonical bytes within the generator bound
         * @throws IllegalArgumentException if {@code bytes} is empty or exceeds the bound
         */
        WorkloadSignature(byte[] bytes) {
            Objects.requireNonNull(bytes, "bytes");
            if (bytes.length == 0 || bytes.length > MetalNegTuningCodec.MAX_WORKLOAD_BYTES) {
                throw new IllegalArgumentException("Metal NEG workload signature size is invalid");
            }
            this.bytes = bytes.clone();
        }

        /** @return a fresh copy of the canonical bytes */
        byte[] bytes() { return bytes.clone(); }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof WorkloadSignature signature
                    && Arrays.equals(bytes, signature.bytes);
        }

        @Override
        public int hashCode() { return Arrays.hashCode(bytes); }

        @Override
        public String toString() { return "MetalNegWorkloadSignature[bytes=" + bytes.length + ']'; }
    }

    /**
     * Exact session target compatibility without a native handle or cross-session device claim.
     *
     * @param abiVersion exact Metal native ABI schema
     * @param sessionNonce exact immutable nonce of the live Metal context
     */
    record TargetCompatibility(
            int abiVersion, MetalDeviceContext.SessionNonce sessionNonce) {
        TargetCompatibility {
            if (abiVersion != MetalNativeApi.ABI_VERSION) {
                throw new IllegalArgumentException("unsupported Metal ABI compatibility version");
            }
            Objects.requireNonNull(sessionNonce, "sessionNonce");
        }
    }

    /**
     * Complete versioned compatibility used by decisions and the session codec.
     *
     * @param schemaVersion exact compatibility schema
     * @param candidateSchemaVersion exact candidate schema
     * @param routePolicyVersion exact route-policy schema
     * @param numericalProfile exact immutable graph-wide numerical-profile identity
     * @param workload exact canonical structural workload
     * @param target exact live-context session compatibility
     */
    record Compatibility(
            int schemaVersion,
            int candidateSchemaVersion,
            int routePolicyVersion,
            NumericalProfile numericalProfile,
            WorkloadSignature workload,
            TargetCompatibility target) {
        Compatibility {
            if (schemaVersion != COMPATIBILITY_SCHEMA_VERSION
                    || candidateSchemaVersion != CANDIDATE_SCHEMA_VERSION
                    || routePolicyVersion != ROUTE_POLICY_VERSION) {
                throw new IllegalArgumentException("unsupported Metal NEG compatibility schema");
            }
            Objects.requireNonNull(numericalProfile, "numericalProfile");
            Objects.requireNonNull(workload, "workload");
            Objects.requireNonNull(target, "target");
        }
    }

    private final Compatibility compatibility;
    private final List<Candidate> candidates;

    /**
     * Creates one complete deterministic positive-budget prefix.
     *
     * @param compatibility non-null exact workload/session compatibility
     * @param candidates non-null non-empty ordered distinct valid candidates
     * @throws NullPointerException if an argument or candidate is {@code null}
     * @throws IllegalArgumentException if candidates are empty, duplicated, or do not start with
     *     a valid safe heuristic
     */
    MetalNegTuningBatch(Compatibility compatibility, List<Candidate> candidates) {
        this.compatibility = Objects.requireNonNull(compatibility, "compatibility");
        this.candidates = List.copyOf(candidates);
        if (this.candidates.isEmpty()
                || this.candidates.stream().distinct().count() != this.candidates.size()
                || this.candidates.size() > Candidate.values().length
                || this.candidates.equals(
                        List.of(Candidate.MPSGRAPH, Candidate.CUSTOM_SINGLE_NEG))) {
            throw new IllegalArgumentException("invalid Metal NEG candidate batch");
        }
    }

    /** @return exact immutable compatibility */
    Compatibility compatibility() { return compatibility; }

    /** @return immutable ordered candidates with the safe heuristic first */
    List<Candidate> candidates() { return candidates; }

    /**
     * Finds one candidate identity in this batch.
     *
     * @param candidate non-null identity to find
     * @return matching retained candidate, or empty when it was pruned by validity or budget
     */
    Optional<Candidate> find(Candidate candidate) {
        Objects.requireNonNull(candidate, "candidate");
        return candidates.contains(candidate) ? Optional.of(candidate) : Optional.empty();
    }

    @Override
    public boolean equals(Object other) {
        return this == other || other instanceof MetalNegTuningBatch batch
                && compatibility.equals(batch.compatibility)
                && candidates.equals(batch.candidates);
    }

    @Override
    public int hashCode() { return Objects.hash(compatibility, candidates); }
}
