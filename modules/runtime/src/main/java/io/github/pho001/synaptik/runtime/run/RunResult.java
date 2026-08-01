package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;
import java.util.Objects;

/**
 * Leases the complete run state after every ordered publication occurrence completes.
 *
 * <p>The result snapshots the direct physical representation references privately in dense result
 * order, including intentional aliases, but exposes no representation, storage, value, or state
 * access. An empty publication list is valid. Successful construction transfers semantic cleanup
 * responsibility for the whole state to this result without changing individual resource
 * ownership. Constructor failure transfers nothing and closes nothing.
 *
 * <p>This class is not thread-safe. After successful construction callers close the leased state
 * only through this result and must not race closure with other state activity. Borrowed
 * representations remain caller-owned for the complete result lifetime.
 */
public final class RunResult implements AutoCloseable {
    private final RunState runState;
    private final BufferRepresentation[] representations;

    /**
     * Validates a complete dense ordered publication list and leases its exact open state.
     *
     * @param runState the exact open state to lease after all validation succeeds; non-null
     * @param publications the dense result-ordered bound occurrences to validate and snapshot;
     *     non-null, with non-null elements all belonging to {@code runState}
     * @throws NullPointerException if an argument or publication element is {@code null}
     * @throws IllegalStateException if the state is closed or an occurrence is not published
     * @throws IllegalArgumentException if an occurrence belongs to another state or its result
     *     index differs from its list position
     */
    public RunResult(RunState runState, List<BoundPublication> publications) {
        Objects.requireNonNull(runState, "runState");
        Objects.requireNonNull(publications, "publications");
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }

        int publicationCount = publications.size();
        for (int index = 0; index < publicationCount; index++) {
            BoundPublication bound =
                    Objects.requireNonNull(publications.get(index), "publications[" + index + "]");
            if (bound.runState() != runState) {
                throw new IllegalArgumentException(
                        "publications[" + index + "] does not belong to supplied run state");
            }
            if (bound.publication().resultIndex() != index) {
                throw new IllegalArgumentException(
                        "publications[" + index + "] result index does not match encounter order");
            }
            if (!bound.isPublished()) {
                throw new IllegalStateException(
                        "publications[" + index + "] is not published");
            }
        }
        var snapshot = new BufferRepresentation[publicationCount];
        for (int index = 0; index < publicationCount; index++) {
            snapshot[index] = publications.get(index).representation();
        }
        this.runState = runState;
        this.representations = snapshot;
    }

    /**
     * Returns the number of ordered published results, including aliases.
     *
     * @return the non-negative private snapshot length; available after closure
     */
    public int resultCount() {
        return representations.length;
    }

    /**
     * Reports the lifecycle status of the complete leased run state.
     *
     * @return {@code true} after result/state closure has begun; otherwise {@code false}
     */
    public boolean isClosed() {
        return runState.isClosed();
    }

    /**
     * Delegates idempotent cleanup of all resources still owned by the leased run state.
     *
     * @throws RuntimeException if state cleanup reports an unchecked failure
     * @throws Error if state cleanup reports an error
     */
    @Override
    public void close() {
        runState.close();
    }
}
