package io.github.pho001.synaptik.runtime.execution;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.Objects;

/**
 * Defines one immutable reusable backend-owned recipe for transferring a logical buffer value
 * between two already-created representation positions of the same prepared buffer.
 *
 * <p>The recipe retains one exact {@link PreparedMemoryPlan} and three dense positions. It owns
 * no representation or other resource and performs no allocation, transfer, validity change,
 * route search, backend discovery, or schedule traversal. Transferring to an equivalent
 * destination representation is the materialization operation; there is no separate
 * materialization kind.
 *
 * <p>{@link #bind(RunState)} is the cold checked boundary. It requires the exact open state,
 * resolves the source and destination once, delegates explicit compatibility checks to the
 * concrete backend, and returns a {@link BoundBufferTransfer} associated with the exact state and
 * positions. Concrete subclasses must be immutable and thread-safe so one recipe can bind
 * concurrently to distinct isolated run states.
 */
public abstract class PreparedBufferTransfer {
    private final PreparedMemoryPlan memoryPlan;
    private final int bufferIndex;
    private final int sourceRepresentationIndex;
    private final int destinationRepresentationIndex;

    /**
     * Creates a transfer recipe for two distinct representation positions of one buffer.
     *
     * <p>Construction validates arguments in declaration order. Representation-position bounds
     * are checked only during binding because the prepared memory plan does not describe the
     * number of physical representations. Construction performs no callback or physical work.
     *
     * @param memoryPlan the exact immutable prepared memory plan to retain; must be non-null
     * @param bufferIndex the dense zero-based position in {@code memoryPlan.buffers()}; must be in
     *     range
     * @param sourceRepresentationIndex the non-negative source position in the run-state buffer
     *     bindings
     * @param destinationRepresentationIndex the non-negative, distinct destination position in
     *     the run-state buffer bindings
     * @throws NullPointerException if {@code memoryPlan} is {@code null}
     * @throws IllegalArgumentException if a coordinate is negative, {@code bufferIndex} is
     *     outside the plan, or the representation positions are equal
     */
    protected PreparedBufferTransfer(
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex) {
        this.memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan");
        if (bufferIndex < 0) {
            throw new IllegalArgumentException("bufferIndex must be non-negative");
        }
        if (bufferIndex >= memoryPlan.buffers().size()) {
            throw new IllegalArgumentException(
                    "bufferIndex out of prepared-plan range: " + bufferIndex);
        }
        if (sourceRepresentationIndex < 0) {
            throw new IllegalArgumentException(
                    "sourceRepresentationIndex must be non-negative");
        }
        if (destinationRepresentationIndex < 0) {
            throw new IllegalArgumentException(
                    "destinationRepresentationIndex must be non-negative");
        }
        if (sourceRepresentationIndex == destinationRepresentationIndex) {
            throw new IllegalArgumentException(
                    "sourceRepresentationIndex and destinationRepresentationIndex must be "
                            + "distinct");
        }
        this.bufferIndex = bufferIndex;
        this.sourceRepresentationIndex = sourceRepresentationIndex;
        this.destinationRepresentationIndex = destinationRepresentationIndex;
    }

    /**
     * Returns the exact prepared memory plan associated with this recipe.
     *
     * @return the retained non-null immutable plan reference; never a copy or replacement
     */
    public final PreparedMemoryPlan memoryPlan() {
        return memoryPlan;
    }

    /**
     * Returns the dense prepared buffer position shared by the source and destination.
     *
     * @return the non-negative zero-based position in {@code memoryPlan().buffers()}
     */
    public final int bufferIndex() {
        return bufferIndex;
    }

    /**
     * Returns the source representation position within the selected run-state buffer.
     *
     * @return the non-negative dense source position
     */
    public final int sourceRepresentationIndex() {
        return sourceRepresentationIndex;
    }

    /**
     * Returns the destination representation position within the selected run-state buffer.
     *
     * @return the non-negative dense destination position, distinct from the source position
     */
    public final int destinationRepresentationIndex() {
        return destinationRepresentationIndex;
    }

    /**
     * Cold-binds this recipe to two already-created representations in one exact open run state.
     *
     * <p>Both ranges are validated before representations are resolved. The source is then
     * resolved and checked before the destination. Each compatibility hook and the final bind
     * hook is invoked exactly once when all preceding checks succeed. Binding changes no
     * validity or ownership and performs no transfer.
     *
     * @param runState the exact open per-run state whose representations are selected; must be
     *     non-null and retain {@link #memoryPlan()} by reference identity
     * @return the non-null backend-owned bound action retaining the exact state and coordinates
     * @throws NullPointerException if {@code runState} or the backend-created bound action is
     *     {@code null}
     * @throws IllegalStateException if {@code runState} is closed
     * @throws IllegalArgumentException if the state uses another plan object, either
     *     representation position is absent, either representation is incompatible, or the
     *     returned action has another state or coordinates
     */
    public final BoundBufferTransfer bind(RunState runState) {
        Objects.requireNonNull(runState, "runState");
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }
        if (runState.memoryPlan() != memoryPlan) {
            throw new IllegalArgumentException(
                    "run state memory plan does not match prepared buffer transfer memory plan");
        }

        int representationCount = runState.bufferRepresentationCount(bufferIndex);
        if (sourceRepresentationIndex >= representationCount) {
            throw new IllegalArgumentException(
                    "sourceRepresentationIndex out of run-state range: "
                            + sourceRepresentationIndex);
        }
        if (destinationRepresentationIndex >= representationCount) {
            throw new IllegalArgumentException(
                    "destinationRepresentationIndex out of run-state range: "
                            + destinationRepresentationIndex);
        }

        BufferRepresentation sourceRepresentation =
                runState
                        .bufferRepresentation(bufferIndex, sourceRepresentationIndex)
                        .representation();
        if (!acceptsSourceBufferRepresentation(sourceRepresentation)) {
            throw new IllegalArgumentException(
                    "source buffer representation is incompatible with prepared buffer transfer");
        }

        BufferRepresentation destinationRepresentation =
                runState
                        .bufferRepresentation(bufferIndex, destinationRepresentationIndex)
                        .representation();
        if (!acceptsDestinationBufferRepresentation(destinationRepresentation)) {
            throw new IllegalArgumentException(
                    "destination buffer representation is incompatible with prepared buffer "
                            + "transfer");
        }

        BoundBufferTransfer boundBufferTransfer =
                Objects.requireNonNull(
                        bindCompatible(
                                runState, sourceRepresentation, destinationRepresentation),
                        "boundBufferTransfer");
        if (boundBufferTransfer.runState() != runState) {
            throw new IllegalArgumentException(
                    "bound buffer transfer does not belong to supplied run state");
        }
        if (boundBufferTransfer.bufferIndex() != bufferIndex
                || boundBufferTransfer.sourceRepresentationIndex()
                        != sourceRepresentationIndex
                || boundBufferTransfer.destinationRepresentationIndex()
                        != destinationRepresentationIndex) {
            throw new IllegalArgumentException(
                    "bound buffer transfer does not match prepared buffer transfer positions");
        }
        return boundBufferTransfer;
    }

    /**
     * Reports whether the resolved source has the backend representation required by this recipe.
     *
     * @param representation the exact non-null nominal source resolved from the run state
     * @return {@code true} when the source is compatible; otherwise {@code false}
     */
    protected abstract boolean acceptsSourceBufferRepresentation(
            BufferRepresentation representation);

    /**
     * Reports whether the resolved destination has the backend representation required by this
     * recipe.
     *
     * @param representation the exact non-null nominal destination resolved from the run state
     * @return {@code true} when the destination is compatible; otherwise {@code false}
     */
    protected abstract boolean acceptsDestinationBufferRepresentation(
            BufferRepresentation representation);

    /**
     * Constructs one typed per-run action after both representations pass compatibility checks.
     *
     * <p>The implementation may perform checked casts justified by the preceding hooks. The
     * returned subclass must retain direct concrete typed source and destination fields and must
     * associate itself with exactly the supplied state and recipe coordinates.
     *
     * @param runState the exact non-null open state supplied to {@link #bind(RunState)}
     * @param sourceRepresentation the exact compatible non-null source reference
     * @param destinationRepresentation the exact compatible non-null destination reference
     * @return a non-null bound action with direct concrete references and matching association
     * @throws RuntimeException if backend-specific bound-action construction fails
     * @throws Error if backend-specific bound-action construction reports an error
     */
    protected abstract BoundBufferTransfer bindCompatible(
            RunState runState,
            BufferRepresentation sourceRepresentation,
            BufferRepresentation destinationRepresentation);
}
