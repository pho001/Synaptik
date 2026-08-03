package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class CpuWorkerGroupTest {
    @Test void partitionsRangesExactlyAndHandlesEmptyAndConcurrentCalls() throws Exception {
        try (var group = new CpuWorkerGroup(3)) {
            var ranges = new CopyOnWriteArrayList<String>();
            group.execute(5, 15, 3, true,
                    (start, end, index) -> ranges.add(index + ":" + start + "-" + end));
            ranges.sort(String::compareTo);
            assertEquals(List.of("0:5-9", "1:9-12", "2:12-15"), ranges);
            var calls = new AtomicInteger();
            group.execute(0, 0, 1, false, (start, end, index) -> calls.incrementAndGet());
            try (var executor = Executors.newFixedThreadPool(2)) {
                var first = executor.submit(() -> group.execute(0, 100, 10, false,
                        (start, end, index) -> calls.incrementAndGet()));
                var second = executor.submit(() -> group.execute(0, 100, 10, false,
                        (start, end, index) -> calls.incrementAndGet()));
                first.get(); second.get();
            }
            assertEquals(6, calls.get());
        }
    }

    @Test void propagatesPrimaryFailureAndSuppressesLaterDistinctFailure() {
        try (var group = new CpuWorkerGroup(2)) {
            var entered = new CountDownLatch(2);
            var release = new CountDownLatch(1);
            var first = new IllegalStateException("first");
            var second = new IllegalArgumentException("second");
            Throwable thrown = assertThrows(Throwable.class, () -> group.execute(0, 2, 1, true,
                    (start, end, index) -> {
                        entered.countDown();
                        try { entered.await(); } catch (InterruptedException e) { throw new AssertionError(e); }
                        release.countDown();
                        if (index == 0) throw first;
                        try { release.await(); } catch (InterruptedException e) { throw new AssertionError(e); }
                        throw second;
                    }));
            assertTrue(thrown == first || thrown == second);
            assertEquals(1, thrown.getSuppressed().length);
        }
    }

    @Test void rejectsReentrancyRestoresInterruptAndShutsDownIdempotently() throws Exception {
        var group = new CpuWorkerGroup(1);
        var reentrant = new CompletableFuture<Throwable>();
        var reentrantClose = new CompletableFuture<Throwable>();
        group.execute(0, 1, 1, true, (start, end, index) -> {
            try { group.execute(0, 1, 1, true, (a, b, c) -> { }); }
            catch (Throwable failure) { reentrant.complete(failure); }
            try { group.close(); }
            catch (Throwable failure) { reentrantClose.complete(failure); }
        });
        assertEquals("CPU worker must not submit parallel work", reentrant.get().getMessage());
        assertEquals("CPU worker must not close its worker group", reentrantClose.get().getMessage());
        try (Arena arena = Arena.ofConfined()) {
            assertFalse(group.isAccessibleByEveryWorker(arena.allocate(1)));
        }
        var interruptedResult = new CompletableFuture<Boolean>();
        Thread caller = Thread.ofPlatform().start(() -> {
            Thread.currentThread().interrupt();
            var failure = assertThrows(CpuParallelExecutionException.class,
                    () -> group.execute(0, 1, 1, true, (s, e, i) -> { }));
            interruptedResult.complete("CPU parallel execution interrupted".equals(failure.getMessage())
                    && Thread.currentThread().isInterrupted());
        });
        caller.join();
        assertTrue(interruptedResult.get());
        group.close(); group.close();
        assertTrue(group.isClosed());
        assertTrue(Thread.getAllStackTraces().keySet().stream()
                .noneMatch(thread -> thread.isAlive()
                        && thread.getName().startsWith("synaptik-cpu-worker-")));
        assertEquals("CPU worker group is closed", assertThrows(IllegalStateException.class,
                () -> group.execute(0, 1, 1, true, (s, e, i) -> { })).getMessage());
    }

    @Test void concurrentCloseCancelsDispatchAfterRunningRangeQuiesces() throws Exception {
        var group = new CpuWorkerGroup(1);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            Future<Throwable> dispatch = executor.submit(() -> {
                try {
                    group.execute(0, 2, 1, false, (start, end, index) -> {
                        entered.countDown();
                        try { release.await(); }
                        catch (InterruptedException exception) { throw new AssertionError(exception); }
                    });
                    return null;
                } catch (Throwable failure) { return failure; }
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            Future<?> close = executor.submit(group::close);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (!group.isClosed() && System.nanoTime() < deadline) Thread.onSpinWait();
            assertTrue(group.isClosed());
            release.countDown();
            close.get(5, TimeUnit.SECONDS);
            Throwable failure = dispatch.get(5, TimeUnit.SECONDS);
            assertInstanceOf(CpuParallelExecutionException.class, failure);
            assertEquals("CPU parallel execution cancelled by worker-group close",
                    failure.getMessage());
        }
    }
}
