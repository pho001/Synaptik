package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import java.util.Objects;

/**
 * Opaque immutable handle for a prepared recipe owned by one {@link AdvancedEngine}.
 *
 * <p>The handle may be shared safely by concurrent runs through its exact owning Engine. Each run
 * still receives isolated mutable Runtime state. The handle exposes neither its owner nor its
 * inward Runtime recipe publicly. Explicit close or owner-Engine close terminally rejects new
 * work and closes the exact inward preparation through Runtime's lease-aware lifecycle. A result
 * returned by an admitted run owns its independent run-state lifetime and is not closed by
 * closing this handle.</p>
 */
public final class AdvancedPreparedExecution implements AutoCloseable {
    private final AdvancedEngine owner;
    private final PreparedExecution execution;
    private boolean closed;
    private boolean cleanupComplete;
    private Throwable closeFailure;

    /**
     * Retains one exact Engine owner and immutable Runtime recipe.
     *
     * @param owner non-null exact Engine that created and exclusively consumes this handle
     * @param execution non-null immutable prepared recipe retained without copying
     * @throws NullPointerException if either argument is null
     */
    AdvancedPreparedExecution(AdvancedEngine owner, PreparedExecution execution) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.execution = Objects.requireNonNull(execution, "execution");
    }

    /**
     * Returns the exact creating Engine for package-private identity validation.
     *
     * @return the non-null exact owner
     */
    AdvancedEngine owner() {
        return owner;
    }

    /**
     * Returns the retained Runtime recipe only to package-private orchestration.
     *
     * @return the non-null immutable prepared execution
     * @throws IllegalStateException if outward close has begun
     */
    synchronized PreparedExecution execution() {
        if (closed) {
            throw new IllegalStateException("prepared execution is closed");
        }
        return execution;
    }

    /**
     * Reports whether this Engine-facing handle has terminally closed.
     *
     * @return {@code true} only after the inward Runtime owner has transitioned to reject new
     *     leases; an earlier leased run may still be completing deferred physical release
     */
    public synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Closes the exact inward Runtime preparation once and unregisters this handle even when
     * cleanup fails. Concurrent and repeated callers wait for the first attempt, restore
     * interruption, and observe the same retained immediate unchecked failure or error by
     * identity. Delegate retrieval and the inward close transition share this wrapper's
     * synchronization boundary, but a complete run does not hold that boundary. A run that
     * retrieved the delegate first therefore arbitrates with close only at Runtime's unique lease
     * authority. Deferred cleanup failure from such an admitted run is reported to that run's
     * lease-releasing path rather than replayed here.
     *
     * @throws RuntimeException if inward cleanup reports an unchecked failure
     * @throws Error if inward cleanup reports a fatal failure
     */
    @Override
    public void close() {
        boolean interrupted = false;
        synchronized (this) {
            if (closed) {
                while (!cleanupComplete) {
                    try {
                        wait();
                    } catch (InterruptedException ignored) {
                        interrupted = true;
                    }
                }
                if (interrupted) Thread.currentThread().interrupt();
                rethrow(closeFailure);
                return;
            }
            closed = true;
            Throwable failure = null;
            try {
                execution.close();
            } catch (RuntimeException | Error cleanupFailure) {
                failure = cleanupFailure;
            } finally {
                owner.unregister(this);
                closeFailure = failure;
                cleanupComplete = true;
                notifyAll();
            }
            rethrow(failure);
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (failure instanceof Error error) throw error;
    }
}
