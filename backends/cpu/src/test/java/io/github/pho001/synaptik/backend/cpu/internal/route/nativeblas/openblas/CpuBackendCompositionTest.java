package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Native-free ownership-transfer, fallback, and close checks for supported composition. */
final class CpuBackendCompositionTest {
    @Test
    void qualificationFailureClosesPartialOwnershipAndReturnsPortableComposition() {
        AtomicInteger closes = new AtomicInteger();
        CpuOpenBlasInvocation failing = new CpuOpenBlasInvocation() {
            @Override public boolean isOpen() { return true; }
            @Override public int threadCount() { throw new IllegalStateException("qualification"); }
            @Override public void setThreadCount(int count) { }
            @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                    MemorySegment b, float beta, MemorySegment c) { }
            @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                    MemorySegment b, double beta, MemorySegment c) { }
        };
        try (CpuBackendComposition composition = CpuBackendComposition.open(
                loadedSession(failing, closes))) {
            assertTrue(composition.availabilitySnapshot().devices().size() == 1);
        }
        assertEquals(1, closes.get());
    }

    @Test
    void fatalQualificationFailureClosesTransferredOwnershipAndPreservesFailureTree() {
        var primary = new AssertionError("qualification");
        var restoreFailure = new IllegalStateException("restore cleanup");
        var ownerCloseFailure = new AssertionError("owner cleanup");
        AtomicInteger restoreAttempts = new AtomicInteger();
        AtomicInteger closes = new AtomicInteger();
        int[] threads = {3};
        CpuOpenBlasInvocation failing = new CpuOpenBlasInvocation() {
            @Override public boolean isOpen() { return true; }
            @Override public int threadCount() { return threads[0]; }
            @Override public void setThreadCount(int count) {
                if (count == 3 && restoreAttempts.incrementAndGet() == 2) {
                    throw restoreFailure;
                }
                threads[0] = count;
            }
            @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                    MemorySegment b, float beta, MemorySegment c) {
                throw primary;
            }
            @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                    MemorySegment b, double beta, MemorySegment c) { }
        };
        CpuOpenBlasDiscoverySession session = loadedSession(failing, () -> {
            closes.incrementAndGet();
            throw ownerCloseFailure;
        });

        AssertionError thrown = assertThrows(
                AssertionError.class, () -> CpuBackendComposition.open(session));

        assertAll(
                () -> assertSame(primary, thrown),
                () -> assertArrayEquals(new Throwable[] {restoreFailure}, thrown.getSuppressed()),
                () -> assertArrayEquals(
                        new Throwable[] {ownerCloseFailure}, restoreFailure.getSuppressed()),
                () -> assertEquals(2, restoreAttempts.get()),
                () -> assertEquals(1, closes.get()));
        session.close();
        assertEquals(1, closes.get());
    }

    @Test
    void successfulQualificationTransfersAndClosesProviderOwnershipExactlyOnce() {
        AtomicInteger closes = new AtomicInteger();
        ComputingInvocation invocation = new ComputingInvocation();
        CpuBackendComposition composition = CpuBackendComposition.open(
                loadedSession(invocation, closes));
        assertEquals(1, invocation.threads);
        composition.close();
        composition.close();
        assertEquals(1, closes.get());
        assertEquals(3, invocation.threads);
    }

    private static CpuOpenBlasDiscoverySession loadedSession(CpuOpenBlasInvocation invocation,
            AtomicInteger closes) {
        return loadedSession(invocation, closes::incrementAndGet);
    }

    private static CpuOpenBlasDiscoverySession loadedSession(CpuOpenBlasInvocation invocation,
            CpuOpenBlasDiscoverySession.CloseAction closeAction) {
        var selection = new CpuOpenBlasDiscoveryResult.LibraryName("test-openblas");
        var result = new CpuOpenBlasDiscoveryResult(
                CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME, Optional.empty(),
                List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                Optional.of(selection), CpuOpenBlasDiscoveryResult.Status.LOADED);
        return new CpuOpenBlasDiscoverySession(result, Optional.of(
                new CpuOpenBlasDiscoverySession.OwnedResource(invocation, closeAction)));
    }

    private static final class ComputingInvocation implements CpuOpenBlasInvocation {
        int threads = 3;
        boolean open = true;
        @Override public boolean isOpen() { return open; }
        @Override public int threadCount() { return threads; }
        @Override public void setThreadCount(int count) { threads = count; }
        @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                MemorySegment b, float beta, MemorySegment c) {
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
                float sum = 0;
                for (int inner = 0; inner < k; inner++) sum +=
                        a.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, row * k + inner)
                                * b.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,
                                        inner * n + column);
                long index = (long) row * n + column;
                c.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, index,
                        alpha * sum + beta * c.getAtIndex(
                                java.lang.foreign.ValueLayout.JAVA_FLOAT, index));
            }
        }
        @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                MemorySegment b, double beta, MemorySegment c) {
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
                double sum = 0;
                for (int inner = 0; inner < k; inner++) sum +=
                        a.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE, row * k + inner)
                                * b.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,
                                        inner * n + column);
                long index = (long) row * n + column;
                c.setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE, index,
                        alpha * sum + beta * c.getAtIndex(
                                java.lang.foreign.ValueLayout.JAVA_DOUBLE, index));
            }
        }
    }
}
