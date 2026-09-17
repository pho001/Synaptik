package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.runtime.run.RunResult;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.List;
import java.util.Objects;

/**
 * Owns the lifecycle of one successful advanced Engine run without exposing its values.
 *
 * <p>The immutable publication count remains available after closure, but published values and
 * representations are deliberately inaccessible. Closing is thread-safe and idempotent,
 * delegates to the exact Runtime result at most once, then closes any ordinary-run wrappers in
 * reverse borrow order, and unregisters this result from its Engine even when cleanup fails.
 * Wrapper cleanup never closes caller-owned host storage. The owning Engine also closes an open
 * result during Engine shutdown.</p>
 */
public final class AdvancedRunResult implements AutoCloseable {
    private final AdvancedEngine owner;
    private final RunResult delegate;
    private final List<BufferRepresentation> ownedBorrowedInputs;
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
        this(owner, delegate, List.of());
    }

    /**
     * Creates one owner-bound result and takes ownership of ordinary-created borrow wrappers.
     *
     * @param owner non-null exact Engine responsible for result cleanup
     * @param delegate non-null exact Runtime result whose ownership transfers
     * @param ownedBorrowedInputs non-null borrow-order snapshot of non-null wrappers whose wrapper
     *     ownership transfers; their caller-owned storage does not transfer
     * @throws NullPointerException if an argument or wrapper is {@code null}
     */
    AdvancedRunResult(
            AdvancedEngine owner,
            RunResult delegate,
            List<BufferRepresentation> ownedBorrowedInputs) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        Objects.requireNonNull(ownedBorrowedInputs, "ownedBorrowedInputs");
        for (int index = 0; index < ownedBorrowedInputs.size(); index++) {
            Objects.requireNonNull(
                    ownedBorrowedInputs.get(index), "ownedBorrowedInputs[" + index + "]");
        }
        this.ownedBorrowedInputs = List.copyOf(ownedBorrowedInputs);
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
     * Returns the exact owning lifecycle coordinator to package-private ordinary orchestration.
     *
     * @return the non-null Engine owner retained for this result's complete lifetime
     */
    AdvancedEngine owner() {
        return owner;
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
     * Materializes one ordinary occurrence while holding this result's lifecycle monitor.
     * Result closure is checked before the selector or limit, and the monitor remains held through
     * validation, representation borrowing, physical copy, and detached-value construction.
     *
     * @param result non-null ordinary result backed by this owner
     * @param publication possibly null publication selector validated by the ordinary result only
     *     after Engine and result lifecycle admission
     * @param maximumBytes caller byte limit validated by the ordinary result after selector
     *     authentication
     * @param composition non-null owned composition used for the physical copy
     * @return a fresh detached host value
     * @throws IllegalStateException if result closure has begun
     * @throws RuntimeException if outward validation or Runtime/CPU copying fails
     * @throws Error if copying reports a fatal failure
     */
    synchronized HostTensorValue materializeUnderAdmission(
            io.github.pho001.synaptik.engine.RunResult result,
            io.github.pho001.synaptik.engine.RunResult.Publication publication,
            long maximumBytes,
            EngineBackendComposition composition) {
        if (closed) {
            throw new IllegalStateException("run result is closed");
        }
        return result.materializeOpen(publication, maximumBytes, delegate, composition);
    }

    /**
     * Closes the exact Runtime result, then every owned ordinary borrow wrapper in reverse order,
     * once, and always unregisters this wrapper from its Engine.
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
            failure = closeAndAccumulate(delegate, failure);
            for (int index = ownedBorrowedInputs.size() - 1; index >= 0; index--) {
                failure = closeAndAccumulate(ownedBorrowedInputs.get(index), failure);
            }
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

    private static Throwable closeAndAccumulate(AutoCloseable closeable, Throwable first) {
        try {
            closeable.close();
        } catch (RuntimeException | Error next) {
            if (first == null) {
                return next;
            }
            if (next != first) {
                first.addSuppressed(next);
            }
        } catch (Exception impossible) {
            throw new AssertionError("close contract declared an unexpected checked failure", impossible);
        }
        return first;
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
