package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.Objects;

/**
 * Names one existing run-state buffer representation as one ordered result position.
 *
 * <p>The immutable recipe uses dense zero-based prepared-plan and run-state positions only. It
 * retains the exact prepared memory plan and performs no graph lookup, physical work, validity
 * change, ownership transfer, or output conversion. One recipe may be reused to bind distinct
 * matching open run states concurrently; each resulting bound occurrence has independent mutable
 * publication state.
 */
public final class PreparedPublication {
    private final PreparedMemoryPlan memoryPlan;
    private final int bufferIndex;
    private final int representationIndex;
    private final int resultIndex;

    /**
     * Creates an immutable publication recipe for one exact prepared buffer coordinate.
     *
     * @param memoryPlan the exact immutable prepared memory plan to retain; must be non-null
     * @param bufferIndex the dense zero-based position in {@code memoryPlan.buffers()}
     * @param representationIndex the dense non-negative per-run representation position
     * @param resultIndex the dense non-negative ordered result position
     * @throws NullPointerException if {@code memoryPlan} is {@code null}
     * @throws IllegalArgumentException if an index is negative or {@code bufferIndex} is outside
     *     the prepared plan
     */
    public PreparedPublication(
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int representationIndex,
            int resultIndex) {
        this.memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan");
        if (bufferIndex < 0) {
            throw new IllegalArgumentException("bufferIndex must be non-negative");
        }
        if (bufferIndex >= memoryPlan.buffers().size()) {
            throw new IllegalArgumentException(
                    "bufferIndex out of prepared-plan range: " + bufferIndex);
        }
        if (representationIndex < 0) {
            throw new IllegalArgumentException("representationIndex must be non-negative");
        }
        if (resultIndex < 0) {
            throw new IllegalArgumentException("resultIndex must be non-negative");
        }
        this.bufferIndex = bufferIndex;
        this.representationIndex = representationIndex;
        this.resultIndex = resultIndex;
    }

    /**
     * Returns the exact prepared memory plan supplied at construction.
     *
     * @return the retained non-null immutable plan reference
     */
    public PreparedMemoryPlan memoryPlan() {
        return memoryPlan;
    }

    /**
     * Returns the dense prepared buffer position.
     *
     * @return the non-negative index in {@code memoryPlan().buffers()}
     */
    public int bufferIndex() {
        return bufferIndex;
    }

    /**
     * Returns the dense representation position within the selected run-state buffer.
     *
     * @return the non-negative representation index
     */
    public int representationIndex() {
        return representationIndex;
    }

    /**
     * Returns the dense ordered result position.
     *
     * @return the non-negative result index
     */
    public int resultIndex() {
        return resultIndex;
    }

    /**
     * Resolves and retains the exact selected physical representation for one run.
     *
     * <p>Binding performs the only representation lookup. It changes no validity or ownership and
     * invokes no backend work. The returned occurrence is not thread-safe.
     *
     * @param runState the exact matching open run state; must be non-null
     * @return a new non-null bound occurrence retaining the exact selected representation
     * @throws NullPointerException if {@code runState} is {@code null}
     * @throws IllegalStateException if {@code runState} is closed
     * @throws IllegalArgumentException if the state retains another plan reference or the
     *     representation position is outside the selected run-state buffer
     */
    public BoundPublication bind(RunState runState) {
        Objects.requireNonNull(runState, "runState");
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }
        if (runState.memoryPlan() != memoryPlan) {
            throw new IllegalArgumentException(
                    "run state memory plan does not match prepared publication memory plan");
        }
        if (representationIndex >= runState.bufferRepresentationCount(bufferIndex)) {
            throw new IllegalArgumentException(
                    "representationIndex out of run-state range: " + representationIndex);
        }
        BufferRepresentation representation =
                runState.bufferRepresentation(bufferIndex, representationIndex).representation();
        return new BoundPublication(runState, this, representation);
    }
}
