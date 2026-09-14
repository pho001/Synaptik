package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.runtime.run.RunResult;
import java.util.Objects;

/**
 * Owns the lifecycle of one successful advanced Engine run without exposing its values.
 *
 * <p>The immutable publication count remains available after closure, but published values and
 * representations are deliberately inaccessible. Closing is thread-safe and idempotent,
 * delegates to the exact Runtime result at most once, and unregisters this result from its Engine
 * even when Runtime cleanup fails. The owning Engine also closes an open result during Engine
 * shutdown.</p>
 */
public final class AdvancedRunResult implements AutoCloseable {
    private final AdvancedEngine owner;
    private final RunResult delegate;
    private final int resultCount;
    private boolean closed;
    private boolean cleanupComplete;
    private Throwable closeFailure;

    /**
     * Creates one owner-bound result after Runtime execution succeeds.
     *
     * @param owner non-null exact Engine responsible for closing this result if it remains open
     * @param delegate non-null exact Runtime result whose ownership transfers to this wrapper
     * @throws NullPointerException if either argument is null
     */
    AdvancedRunResult(AdvancedEngine owner, RunResult delegate) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        resultCount = delegate.resultCount();
    }

    /**
     * Returns the number of ordered publications produced by the run, including aliases.
     * No publication value or representation is exposed by this result.
     *
     * @return the immutable non-negative count captured at construction
     */
    public int resultCount() {
        return resultCount;
    }

    /**
     * Reports whether closure of this wrapper has begun, either directly or through its Engine.
     *
     * @return {@code true} before or during delegate cleanup after the first close call
     */
    public synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Closes the exact Runtime result once and always unregisters this wrapper from its Engine.
     * Concurrent and repeated calls wait for the first cleanup attempt and then return or rethrow
     * its exact retained failure without invoking Runtime cleanup again. Waiting is
     * uninterruptible; an interrupted waiting thread has its interrupt status restored before
     * this method returns or rethrows the retained failure.
     *
     * @throws RuntimeException if Runtime cleanup reports an unchecked failure
     * @throws Error if Runtime cleanup reports a fatal failure
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
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                rethrow(closeFailure);
                return;
            }
            closed = true;
        }
        Throwable failure = null;
        try {
            delegate.close();
        } catch (RuntimeException | Error cleanupFailure) {
            failure = cleanupFailure;
        } finally {
            owner.unregister(this);
            synchronized (this) {
                closeFailure = failure;
                cleanupComplete = true;
                notifyAll();
            }
        }
        rethrow(failure);
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
    }
}
