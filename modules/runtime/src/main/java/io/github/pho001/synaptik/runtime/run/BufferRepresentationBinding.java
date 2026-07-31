package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.Objects;

/**
 * Associates one exact physical buffer representation with its initial ownership for one run.
 *
 * <p>The record retains both supplied references exactly. {@link RunResourceOwnership#BORROWED}
 * leaves cleanup with the caller, while {@link RunResourceOwnership#RUN_OWNED} transfers cleanup
 * responsibility only after a {@link RunState} has completed all constructor validation. The
 * record describes neither representation validity nor residency, coherence, transfer, slot
 * selection, or backend compatibility.
 *
 * <p>The association is immutable, but the retained physical representation may own mutable
 * backend state. Ordinary record equality and hashing use the retained component values; Runtime
 * duplicate detection instead compares exact representation object identity.
 *
 * @param representation the non-null physical buffer representation retained exactly
 * @param ownership the non-null initial cleanup ownership retained exactly
 */
public record BufferRepresentationBinding(
        BufferRepresentation representation, RunResourceOwnership ownership) {
    /**
     * Creates an exact representation-and-ownership association.
     *
     * <p>The representation is validated before the ownership value.
     *
     * @param representation the physical buffer representation to retain; must be non-null
     * @param ownership the initial cleanup ownership to retain; must be non-null
     * @throws NullPointerException if {@code representation} or {@code ownership} is {@code null}
     */
    public BufferRepresentationBinding(
            BufferRepresentation representation, RunResourceOwnership ownership) {
        this.representation = Objects.requireNonNull(representation, "representation");
        this.ownership = Objects.requireNonNull(ownership, "ownership");
    }

    /**
     * Returns the exact physical representation supplied at construction.
     *
     * @return the retained non-null representation; never a copy or replacement
     */
    public BufferRepresentation representation() {
        return representation;
    }

    /**
     * Returns the exact initial run-ownership value supplied at construction.
     *
     * @return the retained non-null ownership value
     */
    public RunResourceOwnership ownership() {
        return ownership;
    }
}
