package io.github.pho001.synaptik.engine;

import java.util.Arrays;
import java.util.Objects;

/**
 * Defines the opaque, session-scoped result model for representative complete-plan comparison.
 *
 * <p>A reference privately owns only detached canonical publication snapshots and the identity of
 * the representative session that captured them. Occurrence order, aliases, zero-length values,
 * byte lengths, and every represented bit are significant; signed zero and distinct NaN payloads
 * therefore differ. Neither the reference nor the two-valued comparison result exposes
 * publication metadata, payload bytes, Runtime state, or a backend representation. The retained
 * snapshots remain usable only through their associated open, healthy session, which enforces
 * that identity and lifetime before execution and comparison.</p>
 */
final class RepresentativePlanCorrectness {
    /** Prevents construction of the stateless enclosing namespace. */
    private RepresentativePlanCorrectness() {}

    /** Exact canonical represented-byte comparison outcome without a diagnostic payload. */
    enum Comparison {
        /** Every ordered occurrence has identical canonical represented bytes. */
        MATCH,
        /** At least one occurrence count, length, or canonical represented byte differs. */
        MISMATCH
    }

    /**
     * Opaque immutable reference captured by exactly one representative execution session.
     *
     * <p>The constructor is private so only the enclosing correctness model can transfer ownership
     * of detached snapshots into the reference. No accessor returns those arrays or their
     * contents, and the reference retains no result lease or physical representation.</p>
     */
    static final class Reference {
        private final RepresentativeExecutionSession session;
        private final byte[][] publications;

        /**
         * Retains transferred private snapshots for one session association.
         *
         * @param session non-null exact session owning this reference's lifetime
         * @param publications non-null private detached payload array whose ownership transfers;
         *     elements are non-null and the caller does not mutate them after construction
         */
        private Reference(
                RepresentativeExecutionSession session,
                byte[][] publications) {
            this.session = Objects.requireNonNull(session, "session");
            this.publications = Objects.requireNonNull(publications, "publications");
        }

        /**
         * Compares a complete candidate without retaining or exposing its payloads.
         *
         * @param candidatePublications non-null detached payloads in encounter order
         * @return {@link Comparison#MATCH} only for equal occurrence counts, lengths, and bytes
         */
        private Comparison compare(byte[][] candidatePublications) {
            if (candidatePublications.length != publications.length) {
                return Comparison.MISMATCH;
            }
            for (int index = 0; index < publications.length; index++) {
                if (!Arrays.equals(publications[index], candidatePublications[index])) {
                    return Comparison.MISMATCH;
                }
            }
            return Comparison.MATCH;
        }
    }

    /**
     * Transfers already-detached publication snapshots into one opaque reference.
     *
     * @param session non-null live session that owns and exclusively associates the reference
     * @param publications non-null private array of non-null detached canonical payloads in
     *     compiled publication order; ownership transfers and the caller must not mutate it
     * @return a new non-null immutable opaque reference retaining no Runtime or backend resource
     * @throws NullPointerException if an argument is null
     */
    static Reference capture(
            RepresentativeExecutionSession session,
            byte[][] publications) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(publications, "publications");
        for (int index = 0; index < publications.length; index++) {
            Objects.requireNonNull(publications[index], "publications[" + index + "]");
        }
        return new Reference(session, publications);
    }

    /**
     * Compares a complete detached candidate snapshot after authenticating its session owner.
     *
     * @param reference non-null opaque reference previously captured by {@code session}
     * @param session non-null exact live session requesting comparison
     * @param candidatePublications non-null private array of non-null detached canonical payloads
     *     in compiled publication order; retained by neither reference nor result
     * @return {@link Comparison#MATCH} only when every ordered occurrence has identical length and
     *     represented bytes; otherwise {@link Comparison#MISMATCH}; the result retains no payload
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the reference belongs to another session
     */
    static Comparison compare(
            Reference reference,
            RepresentativeExecutionSession session,
            byte[][] candidatePublications) {
        Objects.requireNonNull(reference, "reference");
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(candidatePublications, "candidatePublications");
        if (reference.session != session) {
            throw new IllegalArgumentException(
                    "correctness reference belongs to another representative session");
        }
        return reference.compare(candidatePublications);
    }

    /**
     * Authenticates the identity association without exposing retained publication state.
     *
     * @param reference non-null opaque reference to authenticate
     * @param session non-null exact session expected to own the reference
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the reference belongs to another session
     */
    static void requireAssociation(
            Reference reference,
            RepresentativeExecutionSession session) {
        Objects.requireNonNull(reference, "reference");
        Objects.requireNonNull(session, "session");
        if (reference.session != session) {
            throw new IllegalArgumentException(
                    "correctness reference belongs to another representative session");
        }
    }
}
