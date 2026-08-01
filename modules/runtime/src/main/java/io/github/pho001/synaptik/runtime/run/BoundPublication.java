package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;

/**
 * Records one publication occurrence bound to a direct physical representation for one run.
 *
 * <p>Publication performs no physical work and changes no Runtime validity or ownership. It
 * requires the selected resident copy to be valid at the publication moment, then changes only
 * this occurrence's one-shot flag. Distinct occurrences may intentionally retain the same exact
 * representation for different ordered result positions.
 *
 * <p>This class is not thread-safe. Publication must not race validity mutation, execution,
 * transfer, result construction, or state closure.
 */
public final class BoundPublication {
    private final RunState runState;
    private final PreparedPublication publication;
    private final BufferRepresentation representation;
    private boolean published;

    BoundPublication(
            RunState runState,
            PreparedPublication publication,
            BufferRepresentation representation) {
        this.runState = runState;
        this.publication = publication;
        this.representation = representation;
    }

    /**
     * Completes this occurrence if its exact selected representation is currently valid.
     *
     * @throws IllegalStateException if the retained state is closed, this occurrence is already
     *     complete, or the exact selected representation is invalid
     */
    public void publish() {
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }
        if (published) {
            throw new IllegalStateException("publication is already complete");
        }
        if (!runState.isBufferRepresentationValid(
                publication.bufferIndex(), publication.representationIndex())) {
            throw new IllegalStateException("published buffer representation is invalid");
        }
        published = true;
    }

    /**
     * Reports whether this occurrence published successfully.
     *
     * <p>The local flag remains inspectable after state closure and performs no state access.
     *
     * @return {@code true} after the sole successful publication; otherwise {@code false}
     */
    public boolean isPublished() {
        return published;
    }

    final RunState runState() {
        return runState;
    }

    final PreparedPublication publication() {
        return publication;
    }

    final BufferRepresentation representation() {
        return representation;
    }
}
