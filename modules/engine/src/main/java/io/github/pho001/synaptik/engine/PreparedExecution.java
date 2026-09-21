package io.github.pho001.synaptik.engine;

import java.util.Objects;

/**
 * Immutable reusable ordinary prepared handle owned by one exact {@link Engine}.
 *
 * <p>The handle is the outward owner of one private inward Runtime preparation. Explicit close or
 * owner-Engine close terminally rejects new runs and delegates physical prepared-resource
 * lifetime to Runtime's non-waiting lease protocol. Each admitted run still receives isolated
 * Runtime state. Closing the handle does not close a completed run result or detached host value;
 * those values retain their own documented lifetimes. Originating compile metadata remains
 * readable after closure.</p>
 */
public final class PreparedExecution implements AutoCloseable {
    private final Engine owner;
    private final CompiledGraph compiledGraph;
    private final io.github.pho001.synaptik.runtime.execution.PreparedExecution execution;
    private boolean closed;
    private boolean cleanupComplete;
    private Throwable closeFailure;

    /**
     * Retains the exact ordinary owner, compile handle, and immutable Runtime recipe.
     *
     * @param owner non-null exact Engine owner
     * @param compiledGraph non-null exact originating compile handle
     * @param execution non-null immutable inward prepared recipe
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the compiled handle has another owner
     */
    PreparedExecution(
            Engine owner,
            CompiledGraph compiledGraph,
            io.github.pho001.synaptik.runtime.execution.PreparedExecution execution) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.compiledGraph = Objects.requireNonNull(compiledGraph, "compiledGraph");
        this.execution = Objects.requireNonNull(execution, "execution");
        if (compiledGraph.owner() != owner) {
            throw new IllegalArgumentException("compiled graph belongs to another engine");
        }
    }

    /**
     * Returns the exact ordinary compile handle from which this recipe was prepared.
     *
     * @return the retained non-null originating handle by reference identity
     */
    public CompiledGraph compiledGraph() {
        return compiledGraph;
    }

    Engine owner() {
        return owner;
    }

    /**
     * Returns the inward recipe only to package-private orchestration while this handle is open.
     *
     * @return the non-null immutable inward prepared execution
     * @throws IllegalStateException if outward close has begun
     */
    synchronized io.github.pho001.synaptik.runtime.execution.PreparedExecution execution() {
        if (closed) {
            throw new IllegalStateException("prepared execution is closed");
        }
        return execution;
    }

    /**
     * Reports whether this Engine-facing handle has terminally closed.
     *
     * <p>A {@code true} result means that the inward Runtime owner has already transitioned to
     * reject new leases. An earlier run may still hold a lease, so physical release of persistent
     * prepared resources may complete later when that run releases its lease.</p>
     *
     * @return {@code true} from the terminal outward close transition
     */
    public synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Closes the exact inward Runtime preparation once and unregisters this handle from its
     * Engine owner even when inward cleanup fails.
     *
     * <p>Concurrent and repeated callers wait uninterruptibly for the first cleanup attempt,
     * restore interruption before returning, and rethrow the same retained immediate cleanup
     * failure by identity. Delegate retrieval and the inward close transition share this
     * wrapper's synchronization boundary, but a complete run does not hold that boundary. A run
     * that retrieved the delegate first therefore arbitrates with close only at Runtime's unique
     * lease authority. Runtime close does not wait for an already leased run; a failure raised
     * later by deferred cleanup belongs to the lease-releasing run path and is not retroactively
     * replayed by this wrapper.</p>
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
                owner.lifecycleOwner().unregister(this);
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
