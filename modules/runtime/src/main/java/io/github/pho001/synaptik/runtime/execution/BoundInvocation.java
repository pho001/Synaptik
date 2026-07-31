package io.github.pho001.synaptik.runtime.execution;

import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.Objects;

/**
 * Represents one backend-owned invocation bound to the resources of one logical run.
 *
 * <p>A concrete backend subclass retains direct references to the concrete buffer and workspace
 * representation types accepted during cold binding. This shared base retains the exact
 * {@link RunState} association and provides the final lifecycle guard. It does not own or close
 * the state, any representation, or an immutable prepared resource.
 *
 * <p>An invocation is not thread-safe. Callers may execute it sequentially while its state stays
 * open, but must not execute it concurrently or race execution with state closure. The hot call
 * performs only one state-closed check before dispatching directly to the backend implementation;
 * it does no resource lookup, compatibility check, allocation, transfer, residency decision,
 * publication, backend discovery, or implementation selection.
 */
public abstract class BoundInvocation {
    private final RunState runState;

    /**
     * Associates a backend-owned bound invocation with one exact open run state.
     *
     * @param runState the open run state to retain exactly; must be non-null and remains owned by
     *     its existing lifecycle owner
     * @throws NullPointerException if {@code runState} is {@code null}
     * @throws IllegalStateException if {@code runState} is already closed
     */
    protected BoundInvocation(RunState runState) {
        this.runState = Objects.requireNonNull(runState, "runState");
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }
    }

    /**
     * Executes the backend work already bound to direct concrete representation references.
     *
     * <p>Sequential calls are permitted while the retained run state remains open. A backend
     * {@link RuntimeException} or {@link Error} is propagated unchanged; this method performs no
     * fallback, retry, wrapping, or cleanup.
     *
     * @throws IllegalStateException if the retained run state is closed; the backend is not
     *     called
     * @throws RuntimeException if the backend implementation reports an unchecked execution
     *     failure
     * @throws Error if the backend implementation reports an execution error
     */
    public final void execute() {
        if (runState.isClosed()) {
            throw new IllegalStateException("run state is closed");
        }
        executeBound();
    }

    /**
     * Executes the prepared region through direct concrete references established during bind.
     *
     * <p>The implementation must not perform slot lookup, compatibility casting, backend
     * discovery, route or configuration selection, allocation, transfer, residency decisions,
     * publication, tuning, or tracing. It must not close resources owned elsewhere.
     *
     * @throws RuntimeException if backend execution reports an unchecked failure
     * @throws Error if backend execution reports an error
     */
    protected abstract void executeBound();

    final RunState runState() {
        return runState;
    }
}
