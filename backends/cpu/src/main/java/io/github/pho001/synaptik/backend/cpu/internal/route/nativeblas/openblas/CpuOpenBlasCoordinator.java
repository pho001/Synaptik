package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import java.util.Objects;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Explicit CPU-private owner coordinating one OpenBLAS provider lifetime and shared thread budget.
 * Configuration writers and admitted calls are mutually excluded without holding the Java lock
 * across native thread-control or matrix calls. The coordinator controls only work routed through
 * this exact object; it cannot exclude another handle, class loader, coordinator, or arbitrary
 * native caller. Closing quiesces admitted borrowers, restores and verifies the captured positive
 * count through the still-open invocation, and only then closes the transferred provider owner.
 */
public final class CpuOpenBlasCoordinator implements AutoCloseable {
    private enum State { OPEN_UNCONFIGURED, OPEN_CONFIGURED, FAILED, CLOSING, CLOSED }

    private final CpuOpenBlasInvocation invocation;
    private final CpuOpenBlasDiscoverySession.CloseAction closeAction;
    private final CpuConcurrencyBudget budget;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final Condition changed = lock.newCondition();
    private final ThreadLocal<Boolean> admitted = ThreadLocal.withInitial(() -> false);
    private State state = State.OPEN_UNCONFIGURED;
    private boolean writerActive;
    private int activeCalls;
    private Integer originalCount;
    private Integer installedCount;

    /**
     * Creates the sole owner of one transferred provider resource.
     * @param invocation non-null invocation retained by the transferred owner
     * @param closeAction non-null action that closes that exact owner
     * @param budget non-null borrowed shared CPU budget
     * @throws NullPointerException if any argument is {@code null}
     */
    CpuOpenBlasCoordinator(CpuOpenBlasInvocation invocation,
            CpuOpenBlasDiscoverySession.CloseAction closeAction, CpuConcurrencyBudget budget) {
        this.invocation = Objects.requireNonNull(invocation, "invocation");
        this.closeAction = Objects.requireNonNull(closeAction, "closeAction");
        this.budget = Objects.requireNonNull(budget, "budget");
    }

    /**
     * Returns the budget identity used for admission and cold identity validation.
     *
     * @return the exact non-null borrowed budget identity
     */
    public CpuConcurrencyBudget budget() { return budget; }
    /**
     * Returns the admission capacity supplied by the borrowed budget.
     *
     * @return the immutable positive shared capacity
     */
    public int capacity() { return budget.capacity(); }
    /**
     * Returns the invocation retained by the transferred provider owner. Native work may use it
     * only from a callback admitted by {@link #execute(int, int, Runnable)}.
     *
     * @return the retained non-null invocation
     */
    CpuOpenBlasInvocation invocation() { return invocation; }
    /**
     * Reports whether the coordinator remains in an open configured or unconfigured state.
     *
     * @return whether configuration and call admission remain possible
     */
    public boolean isOpen() {
        lock.lock();
        try { return state == State.OPEN_UNCONFIGURED || state == State.OPEN_CONFIGURED; }
        finally { lock.unlock(); }
    }

    /**
     * Installs and verifies a positive provider count while coordinated calls are quiescent.
     * @param count positive count no greater than the shared capacity
     * @throws IllegalArgumentException if the count is outside the capacity
     * @throws IllegalStateException if provider/lifecycle state is unusable or restoration fails
     * @throws CpuConcurrencyBudget.CpuCoordinationException if interrupted while waiting for
     *     another writer or active call; interrupt status is restored
     */
    public void configure(int count) {
        if (count <= 0 || count > capacity()) {
            throw new IllegalArgumentException("OpenBLAS thread count exceeds CPU capacity");
        }
        beginWriter();
        boolean restored = false;
        try {
            if (!invocation.isOpen()) throw new IllegalStateException("OpenBLAS provider is closed");
            Integer retainedOriginal = retainedOriginalCount();
            int prior = retainedOriginal == null ? invocation.threadCount() : retainedOriginal;
            if (prior <= 0) throw new IllegalStateException("OpenBLAS original thread count is invalid");
            if (retainedOriginal == null) retainOriginalCount(prior);
            invocation.setThreadCount(count);
            int verified = invocation.threadCount();
            if (verified != count) throw new IllegalStateException(
                    "OpenBLAS configured thread count did not verify");
            finishWriter(State.OPEN_CONFIGURED, count);
            return;
        } catch (RuntimeException | Error failure) {
            try {
                Integer restoreCount = retainedOriginalCount();
                if (restoreCount != null && invocation.isOpen()) {
                    invocation.setThreadCount(restoreCount);
                    restored = invocation.threadCount() == restoreCount;
                    if (!restored) throw new IllegalStateException(
                            "OpenBLAS original thread count did not verify");
                } else {
                    restored = restoreCount == null;
                }
            } catch (RuntimeException | Error restoreFailure) {
                failure.addSuppressed(restoreFailure);
            }
            finishWriter(restored ? State.OPEN_UNCONFIGURED : State.FAILED, null);
            throw failure;
        }
    }

    /**
     * Admits and executes one complete prepared native sequence under its exact fixed demand.
     * @param threadCount plan-selected installed provider count
     * @param permitDemand exact demand, required to equal {@code threadCount}
     * @param action non-null complete copy-in/GEMM/copy-out callback
     * @throws NullPointerException if {@code action} is {@code null}
     * @throws IllegalArgumentException if the counts are non-positive, unequal, or exceed budget
     * @throws IllegalStateException if the coordinator is not open and configured for the count,
     *     or the provider owner is closed
     * @throws CpuConcurrencyBudget.CpuCoordinationException if permit acquisition or call
     *     admission is interrupted; interrupt status is restored
     * @throws RuntimeException if {@code action} throws a runtime exception
     * @throws Error if {@code action} throws an error
     */
    public void execute(int threadCount, int permitDemand, Runnable action) {
        Objects.requireNonNull(action, "action");
        if (threadCount <= 0 || permitDemand != threadCount) {
            throw new IllegalArgumentException("OpenBLAS permit demand must equal thread count");
        }
        try (var lease = budget.acquire(permitDemand)) {
            lock.lock();
            try {
                awaitNoWriter();
                if (state != State.OPEN_CONFIGURED || installedCount == null
                        || installedCount != threadCount || !invocation.isOpen()) {
                    throw new IllegalStateException("OpenBLAS coordinator is not configured for plan");
                }
                activeCalls++;
                admitted.set(true);
            } finally { lock.unlock(); }
            try { action.run(); }
            finally {
                lock.lock();
                try {
                    admitted.remove();
                    activeCalls--;
                    changed.signalAll();
                } finally { lock.unlock(); }
            }
        }
    }

    /**
     * Quiesces coordinated work, restores and verifies captured state, then closes the provider.
     * Repeated calls are safe; interruption cannot abort cleanup and is restored on return.
     *
     * @throws IllegalStateException if called from an admitted callback or restoration fails
     * @throws RuntimeException if the transferred close action fails with a runtime exception
     * @throws Error if restoration or the transferred close action fails with an error
     */
    @Override public void close() {
        if (admitted.get()) throw new IllegalStateException(
                "OpenBLAS coordinator cannot close from an admitted call");
        boolean interrupted = false;
        boolean closeOwner = false;
        Integer restoreCount = null;
        lock.lock();
        try {
            if (state == State.CLOSED) return;
            if (state == State.CLOSING) {
                while (state != State.CLOSED) {
                    try { changed.await(); }
                    catch (InterruptedException failure) { interrupted = true; }
                }
            } else {
                closeOwner = true;
                state = State.CLOSING;
                changed.signalAll();
                while (activeCalls != 0 || writerActive) {
                    try { changed.await(); }
                    catch (InterruptedException failure) { interrupted = true; }
                }
                writerActive = true;
                restoreCount = originalCount;
            }
        } finally { lock.unlock(); }
        if (!closeOwner) {
            if (interrupted) Thread.currentThread().interrupt();
            return;
        }

        Throwable primary = null;
        try {
            if (restoreCount != null && invocation.isOpen()) {
                invocation.setThreadCount(restoreCount);
                if (invocation.threadCount() != restoreCount) throw new IllegalStateException(
                        "OpenBLAS original thread count did not verify during close");
            }
        } catch (RuntimeException | Error failure) { primary = failure; }
        try { closeAction.close(); }
        catch (RuntimeException | Error failure) {
            if (primary == null) primary = failure; else primary.addSuppressed(failure);
        } finally {
            lock.lock();
            try {
                writerActive = false;
                installedCount = null;
                state = State.CLOSED;
                changed.signalAll();
            } finally { lock.unlock(); }
            if (interrupted) Thread.currentThread().interrupt();
        }
        if (primary instanceof RuntimeException runtime) throw runtime;
        if (primary instanceof Error error) throw error;
    }

    private void beginWriter() {
        lock.lock();
        boolean claimed = false;
        try {
            while (writerActive) {
                if (state != State.OPEN_UNCONFIGURED && state != State.OPEN_CONFIGURED) {
                    throw new IllegalStateException("OpenBLAS coordinator is not configurable");
                }
                changed.await();
            }
            if (state != State.OPEN_UNCONFIGURED && state != State.OPEN_CONFIGURED) {
                throw new IllegalStateException("OpenBLAS coordinator is not configurable");
            }
            writerActive = true;
            claimed = true;
            while (activeCalls != 0) changed.await();
        } catch (InterruptedException failure) {
            if (claimed) {
                writerActive = false;
                changed.signalAll();
            }
            Thread.currentThread().interrupt();
            throw new CpuConcurrencyBudget.CpuCoordinationException(
                    "OpenBLAS configuration interrupted", failure);
        } finally { lock.unlock(); }
    }

    private void finishWriter(State next, Integer installed) {
        lock.lock();
        try {
            installedCount = installed;
            if (state != State.CLOSING) state = next;
            writerActive = false;
            changed.signalAll();
        } finally { lock.unlock(); }
    }

    private Integer retainedOriginalCount() {
        lock.lock();
        try { return originalCount; }
        finally { lock.unlock(); }
    }

    private void retainOriginalCount(int count) {
        lock.lock();
        try {
            if (originalCount == null) originalCount = count;
        } finally { lock.unlock(); }
    }

    private void awaitNoWriter() {
        while (writerActive) {
            if (state == State.CLOSING || state == State.CLOSED || state == State.FAILED) {
                throw new IllegalStateException("OpenBLAS coordinator is not admitting calls");
            }
            try { changed.await(); }
            catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new CpuConcurrencyBudget.CpuCoordinationException(
                        "OpenBLAS call admission interrupted", failure);
            }
        }
    }
}
