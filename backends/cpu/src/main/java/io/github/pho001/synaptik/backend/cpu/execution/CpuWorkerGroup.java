package io.github.pho001.synaptik.backend.cpu.execution;

import java.lang.foreign.MemorySegment;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.Objects;
import java.util.Set;

/**
 * Owns a fixed bounded set of daemon platform workers and synchronous range dispatch.
 *
 * <p>Concurrent callers share worker capacity but not call-level range, cancellation, or failure
 * state. A call returns only after all of its started ranges quiesce. Cancellation is cooperative
 * between ranges, and closing the group is thread-safe and idempotent.</p>
 */
final class CpuWorkerGroup implements AutoCloseable {
    private final Object lock = new Object();
    private final Thread[] workers;
    private final ArrayDeque<Call> available = new ArrayDeque<>();
    private final Set<Call> active = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;

    /**
     * Starts exactly {@code workerCount} owned daemon platform threads.
     * @param workerCount positive fixed worker count
     * @throws IllegalArgumentException if {@code workerCount} is not positive
     */
    CpuWorkerGroup(int workerCount) {
        if (workerCount <= 0) throw new IllegalArgumentException("workerCount must be positive");
        workers = new Thread[workerCount];
        for (int index = 0; index < workerCount; index++) {
            int workerIndex = index;
            workers[index] = Thread.ofPlatform().daemon().name("synaptik-cpu-worker-" + index)
                    .unstarted(() -> workerLoop(workerIndex));
            workers[index].start();
        }
    }

    /** @return the positive immutable worker count */
    int workerCount() { return workers.length; }

    /**
     * Tests worker access without reading or retaining the segment.
     *
     * @param segment non-null live or dead segment to test against every owned worker thread
     * @return whether every owned worker is currently permitted to access {@code segment}
     * @throws NullPointerException if {@code segment} is {@code null}
     */
    boolean isAccessibleByEveryWorker(MemorySegment segment) {
        Objects.requireNonNull(segment, "segment");
        for (Thread worker : workers) if (!segment.isAccessibleBy(worker)) return false;
        return true;
    }

    /**
     * Executes a validated half-open range and returns only after all started work quiesces.
     *
     * @param startInclusive non-negative inclusive first element index
     * @param endExclusive exclusive end index, not less than {@code startInclusive}
     * @param minimumRangeSize positive minimum desired elements per predetermined range
     * @param deterministic whether later consumers require stable range-index ordering semantics
     * @param body non-null range body; may run concurrently on owned workers
     * @throws NullPointerException if {@code body} is {@code null}
     * @throws IllegalArgumentException if range geometry or minimum size is invalid
     * @throws IllegalStateException if the group is closed or an owned worker submits recursively
     * @throws CpuParallelExecutionException if the caller is interrupted or close cancels the call
     */
    void execute(long startInclusive, long endExclusive, long minimumRangeSize,
            boolean deterministic, CpuRangeBody body) {
        Objects.requireNonNull(body, "body");
        if (startInclusive < 0) throw new IllegalArgumentException("startInclusive must be non-negative");
        if (endExclusive < startInclusive) {
            throw new IllegalArgumentException("endExclusive must not be less than startInclusive");
        }
        if (minimumRangeSize <= 0) {
            throw new IllegalArgumentException("minimumRangeSize must be positive");
        }
        synchronized (lock) {
            ensureCanSubmit();
            if (startInclusive == endExclusive) return;
            long elementCount = endExclusive - startInclusive;
            long byMinimum = 1L + (elementCount - 1L) / minimumRangeSize;
            int rangeCount = (int) Math.min((long) workers.length, byMinimum);
            Call call = new Call(startInclusive, elementCount, rangeCount, deterministic, body);
            active.add(call);
            available.addLast(call);
            lock.notifyAll();
            boolean interrupted = false;
            while (!call.done()) {
                try { lock.wait(); }
                catch (InterruptedException exception) {
                    if (!interrupted) call.interruption = exception;
                    interrupted = true;
                    cancel(call);
                }
            }
            active.remove(call);
            if (interrupted) {
                Thread.currentThread().interrupt();
                var failure = new CpuParallelExecutionException(
                        "CPU parallel execution interrupted", call.interruption);
                call.suppressFailuresOn(failure);
                throw failure;
            }
            if (call.primary != null) rethrow(call.primary);
            if (call.closedCancellation) {
                var failure = new CpuParallelExecutionException(
                        "CPU parallel execution cancelled by worker-group close");
                call.suppressFailuresOn(failure);
                throw failure;
            }
        }
    }

    /** @return whether shutdown has begun */
    boolean isClosed() { synchronized (lock) { return closed; } }

    /**
     * Cancels queued work, lets running bodies quiesce, and joins every worker exactly once.
     *
     * @throws IllegalStateException if an owned worker attempts to close its group
     * @throws CpuParallelExecutionException if the closing thread is interrupted while joining;
     *         worker shutdown still completes and the interrupt status is restored
     */
    @Override public void close() {
        if (isWorker(Thread.currentThread())) {
            throw new IllegalStateException("CPU worker must not close its worker group");
        }
        synchronized (lock) {
            if (!closed) {
                closed = true;
                for (Call call : new ArrayList<>(active)) {
                    call.closedCancellation = true;
                    cancel(call);
                }
                lock.notifyAll();
            }
        }
        boolean interrupted = false;
        InterruptedException cause = null;
        for (Thread worker : workers) {
            while (worker.isAlive()) {
                try { worker.join(); }
                catch (InterruptedException exception) { if (cause == null) cause = exception; interrupted = true; }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
            throw new CpuParallelExecutionException("CPU worker-group shutdown interrupted", cause);
        }
    }

    /** Rejects submission after shutdown or recursively from an owned worker. */
    private void ensureCanSubmit() {
        if (closed) throw new IllegalStateException("CPU worker group is closed");
        if (isWorker(Thread.currentThread())) {
            throw new IllegalStateException("CPU worker must not submit parallel work");
        }
    }

    /** Returns whether the supplied thread is one exact worker identity owned by this group. */
    private boolean isWorker(Thread thread) {
        for (Thread worker : workers) if (worker == thread) return true;
        return false;
    }

    /** Runs one owned worker's queue/execute/report loop until shutdown drains available work. */
    private void workerLoop(int ignoredIndex) {
        while (true) {
            Call call;
            int rangeIndex;
            synchronized (lock) {
                while (available.isEmpty() && !closed) {
                    try { lock.wait(); }
                    catch (InterruptedException ignored) { /* owned workers are never interrupted */ }
                }
                if (available.isEmpty()) return;
                call = available.removeFirst();
                if (call.cancelled || call.nextRange == call.rangeCount) continue;
                rangeIndex = call.nextRange++;
                call.running++;
                if (!call.cancelled && call.nextRange < call.rangeCount) available.addLast(call);
            }
            Throwable failure = null;
            try {
                long start = call.rangeStart(rangeIndex);
                call.body.execute(start, call.rangeStart(rangeIndex + 1), rangeIndex);
            } catch (RuntimeException | Error thrown) { failure = thrown; }
            synchronized (lock) {
                call.running--;
                if (failure != null) {
                    call.observeFailure(failure);
                    cancel(call);
                }
                lock.notifyAll();
            }
        }
    }

    /** Marks one call cancelled, removes its queued claims, and wakes all waiters. */
    private void cancel(Call call) {
        call.cancelled = true;
        available.removeIf(candidate -> candidate == call);
        lock.notifyAll();
    }

    /** Rethrows the exact unchecked worker failure without wrapping. */
    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) throw runtime;
        throw (Error) failure;
    }

    /** Mutable coordination state owned by one synchronous submission. */
    private static final class Call {
        final long start;
        final long count;
        final int rangeCount;
        final boolean deterministic;
        final CpuRangeBody body;
        int nextRange;
        int running;
        boolean cancelled;
        boolean closedCancellation;
        InterruptedException interruption;
        Throwable primary;
        final Throwable[] laterFailures;
        int laterFailureCount;

        /** Creates one call-level state object and its bounded later-failure storage. */
        Call(long start, long count, int rangeCount, boolean deterministic, CpuRangeBody body) {
            this.start = start; this.count = count; this.rangeCount = rangeCount;
            this.deterministic = deterministic; this.body = body;
            this.laterFailures = new Throwable[Math.max(0, rangeCount - 1)];
        }

        /** Returns one deterministic quotient/remainder boundary for {@code index}. */
        long rangeStart(int index) {
            long quotient = count / rangeCount;
            long remainder = count % rangeCount;
            return start + quotient * index + Math.min((long) index, remainder);
        }

        /** @return whether no started range remains and no further range can begin */
        boolean done() { return running == 0 && (cancelled || nextRange == rangeCount); }

        /** Records the first exact worker failure and suppresses later distinct identities. */
        void observeFailure(Throwable failure) {
            if (primary == null) primary = failure;
            else if (failure != primary) {
                primary.addSuppressed(failure);
                laterFailures[laterFailureCount++] = failure;
            }
        }

        /** Adds observed distinct worker failures to a coordination failure in observation order. */
        void suppressFailuresOn(Throwable target) {
            if (primary != null && primary != target) target.addSuppressed(primary);
            for (int index = 0; index < laterFailureCount; index++) {
                Throwable failure = laterFailures[index];
                if (failure != target && failure != primary) target.addSuppressed(failure);
            }
        }
    }
}
