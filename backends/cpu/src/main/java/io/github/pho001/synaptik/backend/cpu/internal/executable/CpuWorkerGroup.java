package io.github.pho001.synaptik.backend.cpu.internal.executable;

import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Fixed caller-owned CPU-private platform-worker group for synchronous range execution.
 *
 * <p>Construction starts exactly the requested named daemon platform workers. External callers
 * may submit concurrently; each submission is isolated, returns only after all started ranges
 * quiesce, and chooses failures deterministically by ascending range index. A detected failure,
 * interruption, or racing close cancels only unclaimed ranges. The owner must close the group;
 * prepared executables and bound invocations only borrow it.
 *
 * <p>This unsupported internal type is not an executor facade, common pool, virtual-thread
 * scheduler, Runtime service, or shared prepared-execution lifecycle owner.
 */
public final class CpuWorkerGroup implements AutoCloseable {
    /** One already-bound indexed range call submitted without generic task metadata. */
    @FunctionalInterface
    interface RangeCall {
        /**
         * Executes the bound range.
         *
         * @throws Throwable if generated computation or its direct invocation fails
         */
        void invoke() throws Throwable;
    }

    /**
     * Reports CPU-private parallel coordination failure, cancellation, or interruption when no
     * original unchecked worker failure can be rethrown directly.
     */
    public static final class CpuParallelExecutionException extends RuntimeException {
        private CpuParallelExecutionException(String message) { super(message); }
        private CpuParallelExecutionException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    private static final Job STOP = new Job(new RangeCall[0], 0);
    private final BlockingQueue<Job> queue = new LinkedBlockingQueue<>();
    private final Thread[] workers;
    private final Set<Job> active = ConcurrentHashMap.newKeySet();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final Object lifecycle = new Object();

    /**
     * Creates and starts exactly {@code workerCount} named daemon platform workers.
     *
     * @param workerCount positive fixed worker count for this caller-owned lifetime
     * @throws IllegalArgumentException if {@code workerCount} is not positive
     */
    public CpuWorkerGroup(int workerCount) {
        if (workerCount <= 0) throw new IllegalArgumentException("worker count must be positive");
        workers = new Thread[workerCount];
        for (int index = 0; index < workerCount; index++) {
            Thread worker = new Thread(this::workerLoop, "synaptik-cpu-worker-" + index);
            worker.setDaemon(true);
            workers[index] = worker;
            worker.start();
        }
    }

    /**
     * Returns the fixed worker count.
     *
     * @return the immutable positive number of owned workers
     */
    public int workerCount() { return workers.length; }
    /**
     * Reports whether this group still accepts submissions.
     *
     * @return {@code true} before close begins; otherwise {@code false}
     */
    public boolean isOpen() { return !closed.get(); }

    /**
     * Returns whether every owned worker can access a segment selected for parallel execution.
     *
     * @param segment non-null segment whose thread-access contract is queried; not retained
     * @return {@code true} exactly when every worker may access {@code segment}
     * @throws NullPointerException if {@code segment} is {@code null}
     */
    public boolean workersCanAccess(java.lang.foreign.MemorySegment segment) {
        for (Thread worker : workers) if (!segment.isAccessibleBy(worker)) return false;
        return true;
    }

    /**
     * Executes two or more indexed range calls synchronously.
     *
     * <p>The first detected unchecked failure cancels unclaimed calls. After all started work
     * joins, the failure at the lowest failing index is rethrown unchanged and later distinct
     * failures are suppressed in ascending index order. Interruption cancels unclaimed work,
     * joins started work, restores interrupt status, and throws a coordination exception with
     * completed worker failures suppressed.
     *
     * @param calls non-null ordered range calls; the array is cloned and must contain at least two
     * @throws NullPointerException if {@code calls} is {@code null}
     * @throws IllegalArgumentException if fewer than two calls are supplied
     * @throws IllegalStateException if invoked by one of this group's workers
     * @throws CpuParallelExecutionException if the submission is interrupted, cancelled by close,
     *     or a worker throws a checked failure
     */
    void execute(RangeCall[] calls) {
        if (calls.length < 2) throw new IllegalArgumentException("parallel work requires two ranges");
        if (owns(Thread.currentThread())) {
            throw new IllegalStateException("CPU worker must not submit parallel work");
        }
        Job job = new Job(calls.clone(), Math.min(workers.length, calls.length));
        synchronized (lifecycle) {
            if (closed.get()) throw new CpuParallelExecutionException(
                    "CPU parallel execution cancelled by worker-group close");
            active.add(job);
            for (int index = 0; index < job.participants; index++) queue.add(job);
        }
        boolean interrupted = false;
        while (true) {
            try { job.done.await(); break; }
            catch (InterruptedException failure) { interrupted = true; job.cancelled.set(true); }
        }
        active.remove(job);
        Throwable primary = primary(job.failures);
        if (interrupted) {
            Thread.currentThread().interrupt();
            var failure = new CpuParallelExecutionException(
                    "CPU parallel execution interrupted", new InterruptedException());
            suppress(failure, job.failures, null);
            throw failure;
        }
        if (primary != null) {
            suppress(primary, job.failures, primary);
            rethrow(primary);
        }
        if (job.cancelled.get()) throw new CpuParallelExecutionException(
                "CPU parallel execution cancelled by worker-group close");
    }

    /**
     * Rejects new work, cancels unclaimed calls, joins started calls, and terminates every worker.
     * Repeated calls are safe. If joining is interrupted, shutdown still completes and the
     * calling thread's interrupt status is restored before return.
     *
     * @throws IllegalStateException if called by one of this group's workers
     */
    @Override public void close() {
        if (owns(Thread.currentThread())) {
            throw new IllegalStateException("CPU worker must not close its worker group");
        }
        synchronized (lifecycle) {
            if (closed.compareAndSet(false, true)) {
                for (Job job : active) job.cancelled.set(true);
                for (int index = 0; index < workers.length; index++) queue.add(STOP);
            }
        }
        boolean interrupted = false;
        for (Thread worker : workers) while (worker.isAlive()) {
            try { worker.join(); }
            catch (InterruptedException failure) { interrupted = true; }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }

    private void workerLoop() {
        try {
            while (true) {
                Job job = queue.take();
                if (job == STOP) return;
                try {
                    while (!job.cancelled.get()) {
                        int index = job.next.getAndIncrement();
                        if (index >= job.calls.length) break;
                        try { job.calls[index].invoke(); }
                        catch (Throwable failure) {
                            job.failures[index] = failure;
                            job.cancelled.set(true);
                        }
                    }
                } finally { job.done.countDown(); }
            }
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
        }
    }

    private boolean owns(Thread candidate) {
        for (Thread worker : workers) if (worker == candidate) return true;
        return false;
    }

    private static Throwable primary(Throwable[] failures) {
        for (Throwable failure : failures) if (failure != null) return failure;
        return null;
    }

    private static void suppress(Throwable target, Throwable[] failures, Throwable skip) {
        for (Throwable failure : failures) if (failure != null && failure != skip) {
            target.addSuppressed(failure);
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure instanceof Error error) throw error;
        throw new CpuParallelExecutionException("CPU parallel execution failed", failure);
    }

    private static final class Job {
        final RangeCall[] calls;
        final int participants;
        final AtomicInteger next = new AtomicInteger();
        final AtomicBoolean cancelled = new AtomicBoolean();
        final Throwable[] failures;
        final CountDownLatch done;
        Job(RangeCall[] calls, int participants) {
            this.calls = calls;
            this.participants = participants;
            failures = new Throwable[calls.length];
            done = new CountDownLatch(participants);
        }
    }
}
