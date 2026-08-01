package io.github.pho001.synaptik.runtime.execution;

import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.Objects;

/**
 * Represents one backend-owned buffer transfer bound to direct physical source and destination
 * references for one logical run.
 *
 * <p>The shared base retains only the exact run state and dense coordinates needed for Runtime
 * validity orchestration. A concrete backend subclass retains the compatible concrete source and
 * destination references established during cold binding. The action owns and closes no state or
 * representation.
 *
 * <p>{@link #execute()} first treats an already-valid destination as a no-op. Otherwise it
 * requires a valid source, invokes the backend transfer once, and marks only the destination valid
 * after successful return. A backend {@link RuntimeException} or {@link Error} is propagated
 * unchanged and leaves every Runtime validity bit unchanged. This object is not thread-safe and
 * must not race execution with another validity transition or state closure.
 */
public abstract class BoundBufferTransfer {
    private final RunState runState;
    private final int bufferIndex;
    private final int sourceRepresentationIndex;
    private final int destinationRepresentationIndex;

    /**
     * Associates one bound action with two distinct resident positions in an exact open state.
     *
     * <p>Construction performs no physical work and changes no validity or ownership.
     *
     * @param runState the exact open state to retain; must be non-null
     * @param bufferIndex the dense zero-based buffer position in the state
     * @param sourceRepresentationIndex the dense source position within that buffer
     * @param destinationRepresentationIndex the distinct dense destination position within that
     *     buffer
     * @throws NullPointerException if {@code runState} is {@code null}
     * @throws IllegalStateException if {@code runState} is closed
     * @throws IndexOutOfBoundsException if a buffer or representation position is outside the
     *     state
     * @throws IllegalArgumentException if the representation positions are equal
     */
    protected BoundBufferTransfer(
            RunState runState,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex) {
        this.runState = Objects.requireNonNull(runState, "runState");
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }
        if (bufferIndex < 0 || bufferIndex >= runState.bufferSlotCount()) {
            throw new IndexOutOfBoundsException("bufferIndex out of range: " + bufferIndex);
        }
        int representationCount = runState.bufferRepresentationCount(bufferIndex);
        if (sourceRepresentationIndex < 0
                || sourceRepresentationIndex >= representationCount) {
            throw new IndexOutOfBoundsException(
                    "representationIndex out of range: " + sourceRepresentationIndex);
        }
        if (destinationRepresentationIndex < 0
                || destinationRepresentationIndex >= representationCount) {
            throw new IndexOutOfBoundsException(
                    "representationIndex out of range: " + destinationRepresentationIndex);
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
     * Performs the prepared transfer and its success-only Runtime validity transition.
     *
     * <p>If the destination is valid, this method returns without querying source validity or
     * invoking backend work. Otherwise an invalid source fails before backend invocation. After
     * one successful backend call, only the destination is marked valid; the source and every
     * other copy retain their prior validity.
     *
     * @throws IllegalStateException if the state is closed or an invalid destination has no valid
     *     source
     * @throws RuntimeException if backend transfer work reports an unchecked failure; propagated
     *     unchanged without a validity write, retry, wrapping, or cleanup
     * @throws Error if backend transfer work reports an error; propagated unchanged without a
     *     validity write, retry, wrapping, or cleanup
     */
    public final void execute() {
        if (runState.isBufferRepresentationValid(
                bufferIndex, destinationRepresentationIndex)) {
            return;
        }
        if (!runState.isBufferRepresentationValid(bufferIndex, sourceRepresentationIndex)) {
            throw new IllegalStateException("source buffer representation is invalid");
        }
        executeTransfer();
        runState.setBufferRepresentationValid(
                bufferIndex, destinationRepresentationIndex, true);
    }

    /**
     * Performs the backend physical transfer through direct concrete references retained by the
     * subclass.
     *
     * <p>The implementation must not resolve representations, mutate Runtime validity, allocate,
     * discover a backend or route, retry, invalidate another copy, publish a result, or close a
     * resource.
     *
     * @throws RuntimeException if the concrete backend reports an unchecked transfer failure
     * @throws Error if the concrete backend reports a transfer error
     */
    protected abstract void executeTransfer();

    final RunState runState() {
        return runState;
    }

    final int bufferIndex() {
        return bufferIndex;
    }

    final int sourceRepresentationIndex() {
        return sourceRepresentationIndex;
    }

    final int destinationRepresentationIndex() {
        return destinationRepresentationIndex;
    }
}
