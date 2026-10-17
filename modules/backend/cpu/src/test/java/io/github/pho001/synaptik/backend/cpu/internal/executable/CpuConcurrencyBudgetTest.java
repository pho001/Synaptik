package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class CpuConcurrencyBudgetTest {
    @Test void validatesCapacityAndDemandAndLeaseReleasesOnce() {
        assertThrows(IllegalArgumentException.class, () -> new CpuConcurrencyBudget(0));
        var budget = new CpuConcurrencyBudget(2);
        assertAll(() -> assertEquals(2, budget.capacity()),
                () -> assertThrows(IllegalArgumentException.class, () -> budget.acquire(0)),
                () -> assertThrows(IllegalArgumentException.class, () -> budget.acquire(3)));
        var lease = budget.acquire(2);
        assertEquals(2, lease.demand());
        lease.close();
        lease.close();
        assertDoesNotThrow(() -> budget.acquire(2).close());
    }

    @Test void interruptionRestoresStatusAndDoesNotLeakPermits() throws Exception {
        var budget = new CpuConcurrencyBudget(1);
        var held = budget.acquire(1);
        var failure = new AtomicReference<Throwable>();
        var interrupted = new AtomicReference<Boolean>();
        Thread waiter = Thread.ofPlatform().start(() -> {
            try { budget.acquire(1); }
            catch (Throwable actual) { failure.set(actual); }
            interrupted.set(Thread.currentThread().isInterrupted());
        });
        waiter.interrupt();
        waiter.join(5_000);
        assertAll(() -> assertInstanceOf(CpuConcurrencyBudget.CpuCoordinationException.class,
                        failure.get()),
                () -> assertEquals(Boolean.TRUE, interrupted.get()));
        held.close();
        assertDoesNotThrow(() -> budget.acquire(1).close());
    }

    @Test void coordinatedWorkerSubmissionChargesActualParticipants() throws Exception {
        var budget = new CpuConcurrencyBudget(2);
        var started = new CountDownLatch(2);
        var release = new CountDownLatch(1);
        try (var group = new CpuWorkerGroup(4, budget)) {
            Thread submitter = Thread.ofPlatform().start(() -> group.execute(
                    new CpuWorkerGroup.RangeCall[] {
                            () -> { started.countDown(); release.await(); },
                            () -> { started.countDown(); release.await(); }
                    }));
            assertTrue(started.await(5, TimeUnit.SECONDS));
            var acquired = new CountDownLatch(1);
            Thread waiter = Thread.ofPlatform().start(() -> {
                try (var ignored = budget.acquire(1)) { acquired.countDown(); }
            });
            assertFalse(acquired.await(100, TimeUnit.MILLISECONDS));
            release.countDown();
            submitter.join(5_000);
            assertTrue(acquired.await(5, TimeUnit.SECONDS));
            waiter.join(5_000);
        }
    }

    @Test void workerFailureReleasesItsWholeParticipantLease() {
        var budget = new CpuConcurrencyBudget(2);
        assertThrows(NullPointerException.class, () -> new CpuWorkerGroup(1, null));
        try (var group = new CpuWorkerGroup(2, budget)) {
            assertThrows(IllegalStateException.class, () -> group.execute(
                    new CpuWorkerGroup.RangeCall[] {
                            () -> { throw new IllegalStateException("worker failed"); },
                            () -> { }
                    }));
            assertDoesNotThrow(() -> budget.acquire(2).close());
        }
    }
}
