package io.github.pho001.synaptik.runtime.execution;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.List;
import java.util.Objects;

/**
 * Owns one immutable reusable Runtime recipe and its persistent prepared resources.
 *
 * <p>The execution keeps the exact {@link PreparedMemoryPlan} and {@link PreparedSchedule}
 * references supplied at construction and requires the schedule to report that same plan by
 * reference identity. It privately snapshots each supplied persistent resource identity in
 * acquisition order and becomes the unique owner after successful construction. Resource
 * equality is never consulted and the resource aggregate is not exposed.
 *
 * <p>Recipe state and retained identities remain immutable. Lifecycle coordination is mutable:
 * {@link #close()} atomically rejects new runs and never waits for admitted runs. Cleanup occurs
 * immediately when no lease is active or is deferred to the last lease otherwise. Physical
 * cleanup runs outside lifecycle synchronization, attempts every resource in reverse order, and
 * is claimed by one thread. The class deliberately inherits identity equality, hashing, and
 * diagnostic text from {@link Object}; a lifecycle owner is not a structural recipe value.
 *
 * <p>Distinct active runs still use isolated mutable run state. This type does not create that
 * state, execute the schedule, allocate per-run representations, or expose backend resources.
 */
public final class PreparedExecution implements AutoCloseable {
    private final PreparedMemoryPlan memoryPlan;
    private final PreparedSchedule schedule;
    private final PreparedResource[] resources;
    private boolean closed;
    private boolean cleanupClaimed;
    private int activeRunCount;

    /**
     * Creates a resource-free immutable reusable prepared execution.
     *
     * <p>This source-compatible constructor delegates to the resource-owning constructor with an
     * empty resource list.
     *
     * @param memoryPlan the exact non-null immutable prepared memory plan to retain
     * @param schedule the exact non-null immutable prepared schedule to retain; it must report
     *     {@code memoryPlan} by reference identity
     * @throws NullPointerException if {@code memoryPlan} or {@code schedule} is {@code null}
     * @throws IllegalArgumentException if {@code schedule.memoryPlan()} is not the exact supplied
     *     {@code memoryPlan} reference
     */
    public PreparedExecution(PreparedMemoryPlan memoryPlan, PreparedSchedule schedule) {
        this(memoryPlan, schedule, List.of());
    }

    /**
     * Creates an immutable reusable prepared execution that uniquely owns persistent resources.
     *
     * <p>Validation occurs in the order {@code memoryPlan}, {@code schedule}, exact schedule-plan
     * identity, {@code resources}, then resource entries in supplied order. The list structure is
     * copied into private storage while retaining each exact resource reference. Repeated exact
     * identities are rejected without consulting {@link Object#equals(Object)}. A failed
     * construction transfers no ownership and closes nothing; successful construction transfers
     * ownership of every listed resource to this execution.
     *
     * @param memoryPlan the exact non-null immutable prepared memory plan to retain
     * @param schedule the exact non-null immutable prepared schedule to retain; it must report
     *     {@code memoryPlan} by reference identity
     * @param resources non-null persistent resources in acquisition order; entries must be
     *     non-null and unique by reference identity
     * @throws NullPointerException if an argument or resource entry is {@code null}
     * @throws IllegalArgumentException if the schedule reports another plan reference or a
     *     resource identity occurs more than once
     */
    public PreparedExecution(
            PreparedMemoryPlan memoryPlan,
            PreparedSchedule schedule,
            List<? extends PreparedResource> resources) {
        this.memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan");
        this.schedule = Objects.requireNonNull(schedule, "schedule");
        if (schedule.memoryPlan() != memoryPlan) {
            throw new IllegalArgumentException(
                    "schedule memory plan does not match prepared execution memory plan");
        }
        Objects.requireNonNull(resources, "resources");
        this.resources = new PreparedResource[resources.size()];
        for (int index = 0; index < this.resources.length; index++) {
            PreparedResource resource =
                    Objects.requireNonNull(resources.get(index), "resources[" + index + "]");
            for (int earlier = 0; earlier < index; earlier++) {
                if (this.resources[earlier] == resource) {
                    throw new IllegalArgumentException(
                            "resource is already owned by this prepared execution");
                }
            }
            this.resources[index] = resource;
        }
    }

    /**
     * Returns the exact prepared memory plan supplied at construction.
     *
     * @return the retained non-null immutable plan reference; never a copy or structural
     *     replacement
     */
    public PreparedMemoryPlan memoryPlan() {
        return memoryPlan;
    }

    /**
     * Returns the exact prepared schedule supplied at construction.
     *
     * @return the retained non-null immutable schedule reference whose plan is exactly
     *     {@link #memoryPlan()} by reference identity
     */
    public PreparedSchedule schedule() {
        return schedule;
    }

    /**
     * Acquires permission for one synchronous run to use this execution's persistent resources.
     *
     * <p>Admission is serialized with {@link #close()}. A successful call increments the private
     * active-run count and returns a fresh opaque lease. The runner must close that lease after
     * its complete synchronous call; lease closure is idempotent and may perform deferred
     * resource cleanup when it releases the last admitted run.
     *
     * @return a non-null opaque lease for exactly one admitted run
     * @throws IllegalStateException if close has begun, with message
     *     {@code prepared execution is closed}
     */
    public synchronized RunLease acquireRunLease() {
        if (closed) {
            throw new IllegalStateException("prepared execution is closed");
        }
        activeRunCount++;
        return new RunLease(this);
    }

    /**
     * Reports whether this execution has begun closing and rejects new run leases.
     *
     * <p>The result becomes {@code true} before any physical cleanup callback is invoked and
     * remains true. It does not imply that deferred physical cleanup has completed.
     *
     * @return {@code true} once the first close transition has occurred; otherwise {@code false}
     */
    public synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Rejects new runs and releases persistent resources when no admitted run remains.
     *
     * <p>This operation is idempotent and non-waiting. The first call marks the execution closed.
     * With active leases it returns without physical cleanup; the last lease performs cleanup.
     * Otherwise this caller performs cleanup immediately. Backend callbacks run outside lifecycle
     * synchronization and every resource is attempted once in reverse acquisition order. The
     * first unchecked exception or error is rethrown after all attempts, later distinct failures
     * are suppressed in encounter order, and the exact primary object is never self-suppressed.
     * Repeated closure does not replay a cleanup failure.
     *
     * @throws RuntimeException if the first physical cleanup failure is an unchecked exception
     * @throws Error if the first physical cleanup failure is an error
     */
    @Override
    public void close() {
        boolean cleanUp = false;
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            if (activeRunCount == 0) {
                cleanupClaimed = true;
                cleanUp = true;
            }
        }
        if (cleanUp) {
            closeResources();
        }
    }

    private void releaseRunLease() {
        boolean cleanUp = false;
        synchronized (this) {
            activeRunCount--;
            if (closed && activeRunCount == 0 && !cleanupClaimed) {
                cleanupClaimed = true;
                cleanUp = true;
            }
        }
        if (cleanUp) {
            closeResources();
        }
    }

    private void closeResources() {
        Throwable primary = null;
        for (int index = resources.length - 1; index >= 0; index--) {
            try {
                resources[index].close();
            } catch (RuntimeException | Error failure) {
                if (primary == null) {
                    primary = failure;
                } else if (failure != primary) {
                    primary.addSuppressed(failure);
                }
            }
        }
        if (primary instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (primary instanceof Error error) {
            throw error;
        }
    }

    /**
     * Represents one admitted synchronous use of a prepared execution.
     *
     * <p>The token is intentionally opaque: it exposes no owner, recipe, resource, count, or
     * backend value. Closing it releases admission exactly once and may perform deferred physical
     * cleanup when this is the last active lease after execution close has begun.
     */
    public static final class RunLease implements AutoCloseable {
        private final PreparedExecution owner;
        private boolean released;

        private RunLease(PreparedExecution owner) {
            this.owner = owner;
        }

        /**
         * Releases this run admission once.
         *
         * <p>Repeated calls are no-ops and do not replay a prior cleanup failure. If this is the
         * last admitted run after the execution closed, this caller performs reverse attempt-all
         * resource cleanup outside lifecycle synchronization.
         *
         * @throws RuntimeException if deferred cleanup's first failure is unchecked
         * @throws Error if deferred cleanup's first failure is an error
         */
        @Override
        public void close() {
            synchronized (this) {
                if (released) {
                    return;
                }
                released = true;
            }
            owner.releaseRunLease();
        }
    }
}
