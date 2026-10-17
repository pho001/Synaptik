package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class CpuWorkerGroupTest {
    @Test void executesEveryRangeOnceAndSupportsConcurrentExternalSubmissions() throws Exception {
        try (var group = new CpuWorkerGroup(3)) {
            var count = new AtomicInteger();
            CpuWorkerGroup.RangeCall[] calls = new CpuWorkerGroup.RangeCall[7];
            java.util.Arrays.setAll(calls, ignored -> count::incrementAndGet);
            var first = Thread.ofPlatform().start(() -> group.execute(calls));
            var second = Thread.ofPlatform().start(() -> group.execute(calls));
            first.join(); second.join();
            assertEquals(14, count.get());
        }
    }

    @Test void choosesLowestRangeFailureAndSuppressesLaterFailuresInRangeOrder() {
        var low = new IllegalStateException("low");
        var high = new IllegalArgumentException("high");
        var barrier = new CyclicBarrier(2);
        try (var group = new CpuWorkerGroup(2)) {
            CpuWorkerGroup.RangeCall[] calls = {
                    () -> { barrier.await(); throw low; },
                    () -> { barrier.await(); throw high; }
            };
            Throwable actual = assertThrows(IllegalStateException.class, () -> group.execute(calls));
            assertAll(
                    () -> assertSame(low, actual),
                    () -> assertArrayEquals(new Throwable[] {high}, actual.getSuppressed()));
        }
    }

    @Test void rejectsNestedParallelSubmissionAndOwnedClose() {
        try (var group = new CpuWorkerGroup(2)) {
            CpuWorkerGroup.RangeCall[] nested = {() -> { }, () -> { }};
            var nestedFailure = assertThrows(IllegalStateException.class,
                    () -> group.execute(new CpuWorkerGroup.RangeCall[] {
                            () -> group.execute(nested), () -> { }
                    }));
            assertEquals("CPU worker must not submit parallel work", nestedFailure.getMessage());
            var closeFailure = assertThrows(IllegalStateException.class,
                    () -> group.execute(new CpuWorkerGroup.RangeCall[] {
                            group::close, () -> { }
                    }));
            assertEquals("CPU worker must not close its worker group", closeFailure.getMessage());
        }
    }

    @Test void closeIsIdempotentAndRejectsNewWork() {
        var group = new CpuWorkerGroup(2);
        group.close();
        assertAll(
                () -> assertFalse(group.isOpen()),
                () -> assertDoesNotThrow(group::close),
                () -> assertThrows(CpuWorkerGroup.CpuParallelExecutionException.class,
                        () -> group.execute(new CpuWorkerGroup.RangeCall[] {() -> { }, () -> { }})));
    }

    @Test void interruptCancelsJoinsRestoresStatusAndReportsCoordinationFailure() throws Exception {
        var started = new CountDownLatch(2);
        var release = new CountDownLatch(1);
        var observed = new AtomicReference<Throwable>();
        var interruptedStatus = new AtomicReference<Boolean>();
        var workerFailure = new IllegalStateException("worker failure joined after interrupt");
        try (var group = new CpuWorkerGroup(2)) {
            CpuWorkerGroup.RangeCall failing = () -> {
                started.countDown(); release.await(); throw workerFailure;
            };
            CpuWorkerGroup.RangeCall successful = () -> { started.countDown(); release.await(); };
            var submitter = Thread.ofPlatform().start(() -> {
                try { group.execute(new CpuWorkerGroup.RangeCall[] {failing, successful}); }
                catch (Throwable failure) { observed.set(failure); }
                interruptedStatus.set(Thread.currentThread().isInterrupted());
            });
            assertTrue(started.await(5, TimeUnit.SECONDS));
            submitter.interrupt();
            release.countDown();
            submitter.join(5_000);
            assertAll(
                    () -> assertFalse(submitter.isAlive()),
                    () -> assertInstanceOf(CpuWorkerGroup.CpuParallelExecutionException.class,
                            observed.get()),
                    () -> assertEquals("CPU parallel execution interrupted",
                            observed.get().getMessage()),
                    () -> assertInstanceOf(InterruptedException.class, observed.get().getCause()),
                    () -> assertArrayEquals(new Throwable[] {workerFailure},
                            observed.get().getSuppressed()),
                    () -> assertEquals(Boolean.TRUE, interruptedStatus.get()));
        }
    }

    @Test void racingCloseWithoutWorkerFailureReportsCloseCancellation() throws Exception {
        var started = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var observed = new AtomicReference<Throwable>();
        var group = new CpuWorkerGroup(1);
        var submitter = Thread.ofPlatform().start(() -> {
            try { group.execute(new CpuWorkerGroup.RangeCall[] {
                    () -> { started.countDown(); release.await(); },
                    () -> fail("cancelled unclaimed range must not start")
            }); }
            catch (Throwable failure) { observed.set(failure); }
        });
        assertTrue(started.await(5, TimeUnit.SECONDS));
        var closer = Thread.ofPlatform().start(group::close);
        while (group.isOpen()) Thread.onSpinWait();
        release.countDown();
        submitter.join(5_000);
        closer.join(5_000);
        assertAll(
                () -> assertFalse(submitter.isAlive()),
                () -> assertFalse(closer.isAlive()),
                () -> assertInstanceOf(CpuWorkerGroup.CpuParallelExecutionException.class,
                        observed.get()),
                () -> assertEquals(
                        "CPU parallel execution cancelled by worker-group close",
                        observed.get().getMessage()));
    }

    @Test void racingCloseCancelsUnclaimedWorkJoinsStartedWorkAndPreservesWorkerFailure()
            throws Exception {
        var started = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var completed = new AtomicInteger();
        var workerFailure = new IllegalStateException("worker failed during close");
        var observed = new AtomicReference<Throwable>();
        var group = new CpuWorkerGroup(1);
        var submitter = Thread.ofPlatform().start(() -> {
            try { group.execute(new CpuWorkerGroup.RangeCall[] {
                    () -> { started.countDown(); release.await(); throw workerFailure; },
                    completed::incrementAndGet,
                    completed::incrementAndGet
            }); }
            catch (Throwable failure) { observed.set(failure); }
        });
        assertTrue(started.await(5, TimeUnit.SECONDS));
        var closer = Thread.ofPlatform().start(group::close);
        release.countDown();
        submitter.join(5_000);
        closer.join(5_000);
        assertAll(
                () -> assertFalse(submitter.isAlive()),
                () -> assertFalse(closer.isAlive()),
                () -> assertSame(workerFailure, observed.get()),
                () -> assertEquals(0, completed.get()),
                () -> assertFalse(group.isOpen()));
    }
}
