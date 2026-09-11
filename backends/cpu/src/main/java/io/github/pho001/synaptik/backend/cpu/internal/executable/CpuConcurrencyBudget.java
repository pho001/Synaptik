package io.github.pho001.synaptik.backend.cpu.internal.executable;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Explicit fair permit budget shared by coordinated CPU execution resources.
 *
 * <p>The budget owns no threads and has no lifecycle. Callers borrow the same object when their
 * work must share one capacity, acquire a fixed positive demand interruptibly, and close the
 * returned lease in every outcome. The fair semaphore prevents later acquisitions from barging
 * ahead of queued acquisitions; it does not promise operating-system scheduling order. A lease
 * is idempotent and releases exactly the permits acquired by it.</p>
 */
public final class CpuConcurrencyBudget {
    private final int capacity;
    private final Semaphore permits;

    /**
     * Creates a fair budget.
     *
     * @param capacity positive maximum simultaneous CPU demand
     * @throws IllegalArgumentException if {@code capacity} is not positive
     */
    public CpuConcurrencyBudget(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("CPU budget capacity must be positive");
        this.capacity = capacity;
        permits = new Semaphore(capacity, true);
    }

    /**
     * Returns the maximum simultaneous demand admitted by this budget.
     *
     * @return the immutable positive capacity
     */
    public int capacity() { return capacity; }

    /**
     * Acquires an exact fixed demand, waiting interruptibly.
     *
     * @param demand permit count in {@code [1, capacity]}
     * @return a non-null idempotent lease owning exactly {@code demand} permits
     * @throws IllegalArgumentException if the demand is outside the budget
     * @throws CpuCoordinationException if interrupted while waiting; interrupt status is restored
     */
    public Lease acquire(int demand) {
        if (demand <= 0 || demand > capacity) {
            throw new IllegalArgumentException("CPU permit demand must be within budget capacity");
        }
        try {
            permits.acquire(demand);
            return new Lease(permits, demand);
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new CpuCoordinationException("CPU permit acquisition interrupted", failure);
        }
    }

    /**
     * Stable CPU-private runtime failure for interrupted coordination. The cause retains the
     * interruption that prevented admission, and the throwing thread's interrupt status has
     * already been restored.
     */
    public static final class CpuCoordinationException extends RuntimeException {
        /**
         * Creates one coordination failure retaining its interruption cause.
         * @param message stable diagnostic message; may be {@code null}
         * @param cause interruption that prevented acquisition or admission; may be {@code null}
         */
        public CpuCoordinationException(String message, Throwable cause) { super(message, cause); }
    }

    /**
     * One exact idempotently releasable acquisition. The lease owns only its permit demand, not
     * the borrowed budget or any executing thread.
     */
    public static final class Lease implements AutoCloseable {
        private final Semaphore permits;
        private final int demand;
        private final AtomicBoolean released = new AtomicBoolean();

        private Lease(Semaphore permits, int demand) {
            this.permits = permits;
            this.demand = demand;
        }

        /**
         * Returns the demand that this lease releases on its first close.
         *
         * @return the exact positive permit count owned by this lease
         */
        public int demand() { return demand; }

        /** Releases this lease's exact permits once; repeated calls have no effect. */
        @Override public void close() {
            if (released.compareAndSet(false, true)) permits.release(demand);
        }
    }
}
