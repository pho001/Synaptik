package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class CpuOpenBlasCoordinatorTest {
    @Test void configuresExecutesReconfiguresAndRestoresBeforeClose() {
        var fake = new FakeInvocation(3);
        var closed = new AtomicBoolean();
        var coordinator = new CpuOpenBlasCoordinator(fake, () -> {
            assertEquals(3, fake.threadCount());
            closed.set(true);
            fake.open.set(false);
        }, new CpuConcurrencyBudget(4));
        coordinator.configure(2);
        coordinator.execute(2, 2, () -> assertEquals(2, fake.threads.get()));
        coordinator.configure(1);
        assertThrows(IllegalStateException.class,
                () -> coordinator.execute(2, 2, () -> fail("must not execute")));
        coordinator.close();
        coordinator.close();
        assertAll(() -> assertEquals(3, fake.threads.get()), () -> assertTrue(closed.get()),
                () -> assertFalse(coordinator.isOpen()),
                () -> assertEquals(5, fake.queries.get()),
                () -> assertEquals(3, fake.sets.get()));
    }

    @Test void writerAndCloseWaitForAdmittedCallWithoutHoldingControlLock() throws Exception {
        var fake = new FakeInvocation(4);
        var coordinator = new CpuOpenBlasCoordinator(fake, () -> fake.open.set(false),
                new CpuConcurrencyBudget(4));
        coordinator.configure(2);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        Thread call = Thread.ofPlatform().start(() -> coordinator.execute(2, 2, () -> {
            entered.countDown();
            try { release.await(); } catch (InterruptedException failure) {
                throw new AssertionError(failure);
            }
        }));
        assertTrue(entered.await(5, TimeUnit.SECONDS));
        var configured = new CountDownLatch(1);
        Thread writer = Thread.ofPlatform().start(() -> {
            coordinator.configure(2);
            configured.countDown();
        });
        assertFalse(configured.await(100, TimeUnit.MILLISECONDS));
        var secondEntered = new CountDownLatch(1);
        Thread secondCall = Thread.ofPlatform().start(() -> coordinator.execute(2, 2,
                secondEntered::countDown));
        assertFalse(secondEntered.await(100, TimeUnit.MILLISECONDS));
        release.countDown();
        call.join(5_000);
        assertTrue(configured.await(5, TimeUnit.SECONDS));
        writer.join(5_000);
        assertTrue(secondEntered.await(5, TimeUnit.SECONDS));
        secondCall.join(5_000);
        coordinator.close();
        assertEquals(4, fake.threads.get());
    }

    @Test void fixedDemandsBoundOverlapAndReleaseAfterFailure() throws Exception {
        var fake = new FakeInvocation(4);
        var budget = new CpuConcurrencyBudget(2);
        var coordinator = new CpuOpenBlasCoordinator(fake, () -> fake.open.set(false), budget);
        coordinator.configure(1);
        var entered = new CountDownLatch(2);
        var release = new CountDownLatch(1);
        var active = new AtomicInteger();
        var maximum = new AtomicInteger();
        Runnable call = () -> coordinator.execute(1, 1, () -> {
            int now = active.incrementAndGet();
            maximum.accumulateAndGet(now, Math::max);
            entered.countDown();
            try { release.await(); } catch (InterruptedException failure) {
                throw new AssertionError(failure);
            } finally { active.decrementAndGet(); }
        });
        Thread first = Thread.ofPlatform().start(call);
        Thread second = Thread.ofPlatform().start(call);
        assertTrue(entered.await(5, TimeUnit.SECONDS));
        var thirdEntered = new CountDownLatch(1);
        Thread third = Thread.ofPlatform().start(() -> coordinator.execute(1, 1,
                thirdEntered::countDown));
        assertFalse(thirdEntered.await(100, TimeUnit.MILLISECONDS));
        release.countDown();
        first.join(5_000);
        second.join(5_000);
        assertTrue(thirdEntered.await(5, TimeUnit.SECONDS));
        third.join(5_000);
        assertAll(() -> assertEquals(2, maximum.get()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> coordinator.execute(1, 2, () -> fail("must not execute"))),
                () -> assertThrows(IllegalStateException.class,
                        () -> coordinator.execute(2, 2, () -> fail("must not execute"))));

        coordinator.configure(2);
        assertThrows(IllegalArgumentException.class,
                () -> coordinator.execute(2, 2, () -> { throw new IllegalArgumentException("x"); }));
        assertDoesNotThrow(() -> coordinator.execute(2, 2, () -> { }));
        coordinator.close();
    }

    @Test void failedInstallationRestoresAndFailedRestorationIsSuppressed() {
        var fake = new FakeInvocation(3);
        fake.failVerification.set(true);
        var coordinator = new CpuOpenBlasCoordinator(fake, () -> fake.open.set(false),
                new CpuConcurrencyBudget(4));
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> coordinator.configure(2));
        assertEquals(3, fake.threads.get());
        assertTrue(coordinator.isOpen());
        coordinator.close();

        var broken = new FakeInvocation(3);
        broken.failVerification.set(true);
        broken.failRestore.set(true);
        var failed = new CpuOpenBlasCoordinator(broken, () -> broken.open.set(false),
                new CpuConcurrencyBudget(4));
        Throwable primary = assertThrows(IllegalStateException.class, () -> failed.configure(2));
        assertEquals(1, primary.getSuppressed().length);
        assertFalse(failed.isOpen());
        assertThrows(IllegalStateException.class,
                () -> failed.execute(2, 2, () -> fail("must not execute")));
        assertThrows(IllegalStateException.class, failed::close);
        assertFalse(broken.open.get());
    }

    @Test void qualificationIsExclusiveAndRestoresAfterCallbackFailure() throws Exception {
        var fake = new FakeInvocation(3);
        var coordinator = new CpuOpenBlasCoordinator(fake, () -> fake.open.set(false),
                new CpuConcurrencyBudget(2));
        coordinator.configure(1);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        Thread call = Thread.ofPlatform().start(() -> coordinator.execute(1, 1, () -> {
            entered.countDown();
            try { release.await(); } catch (InterruptedException failure) {
                throw new AssertionError(failure);
            }
        }));
        assertTrue(entered.await(5, TimeUnit.SECONDS));
        var qualified = new CountDownLatch(1);
        var target = CpuOpenBlasQualifier.target("Linux", "x86_64", 64,
                java.nio.ByteOrder.LITTLE_ENDIAN);
        Thread writer = Thread.ofPlatform().start(() -> {
            coordinator.qualify(target, qualified::countDown);
        });
        assertFalse(qualified.await(100, TimeUnit.MILLISECONDS));
        release.countDown();
        call.join(5_000); writer.join(5_000);
        assertTrue(qualified.await(5, TimeUnit.SECONDS));
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> coordinator.qualify(target, () -> {
                    throw new IllegalArgumentException("qualification failed");
                }));
        assertEquals("qualification failed", failure.getMessage());
        assertAll(() -> assertEquals(3, fake.threads.get()),
                () -> assertTrue(coordinator.isOpen()));
        coordinator.close();
    }

    @Test void discoveryTransfersOwnershipExactlyOnceAndRetainsMetadata() {
        var fake = new FakeInvocation(2);
        var selection = new CpuOpenBlasDiscoveryResult.LibraryName("fake");
        var result = new CpuOpenBlasDiscoveryResult(CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME,
                java.util.Optional.empty(),
                java.util.List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                java.util.Optional.of(selection),
                CpuOpenBlasDiscoveryResult.Status.LOADED);
        var session = new CpuOpenBlasDiscoverySession(result, java.util.Optional.of(
                new CpuOpenBlasDiscoverySession.OwnedResource(fake, () -> fake.open.set(false))));
        var coordinator = session.transferToCoordinator(new CpuConcurrencyBudget(2));
        session.close();
        assertAll(() -> assertSame(result, session.result()), () -> assertTrue(fake.open.get()),
                () -> assertThrows(IllegalStateException.class,
                        () -> session.transferToCoordinator(new CpuConcurrencyBudget(2))));
        coordinator.close();
        assertFalse(fake.open.get());

        var unavailableResult = new CpuOpenBlasDiscoveryResult(
                CpuOpenBlasDiscoveryRequest.Mode.DISABLED, java.util.Optional.empty(),
                java.util.List.of(), java.util.Optional.empty(),
                CpuOpenBlasDiscoveryResult.Status.DISABLED);
        var unavailable = new CpuOpenBlasDiscoverySession(unavailableResult,
                java.util.Optional.empty());
        assertThrows(IllegalStateException.class,
                () -> unavailable.transferToCoordinator(new CpuConcurrencyBudget(1)));
    }

    private static final class FakeInvocation implements CpuOpenBlasInvocation {
        final AtomicBoolean open = new AtomicBoolean(true);
        final AtomicInteger threads;
        final AtomicInteger queries = new AtomicInteger();
        final AtomicInteger sets = new AtomicInteger();
        final AtomicBoolean failVerification = new AtomicBoolean();
        final AtomicBoolean failRestore = new AtomicBoolean();
        FakeInvocation(int threads) { this.threads = new AtomicInteger(threads); }
        @Override public boolean isOpen() { return open.get(); }
        @Override public int threadCount() {
            queries.incrementAndGet();
            if (sets.get() > 0 && failVerification.compareAndSet(true, false)) {
                return threads.get() + 1;
            }
            return threads.get();
        }
        @Override public void setThreadCount(int count) {
            sets.incrementAndGet();
            if (count == 3 && failRestore.get()) throw new IllegalStateException("restore failed");
            threads.set(count);
        }
        @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                MemorySegment b, float beta, MemorySegment c) { }
        @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                MemorySegment b, double beta, MemorySegment c) { }
    }
}
