package io.github.pho001.synaptik.backend.metal.internal;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class MetalFoundationTest {
    @Test
    void createsFreshExactGeometryAndRoundTripsOnlyRequestedBytes() {
        var api = new FakeNativeApi();
        var context = MetalDeviceContext.open(api);
        var zero = context.createBuffer(0);
        var first = context.createBuffer(6);
        var second = context.createBuffer(6);
        var workspace = context.createWorkspace(3);

        assertEquals(0, zero.byteSize());
        assertEquals(6, first.byteSize());
        assertEquals(6, second.byteSize());
        assertEquals(3, workspace.byteSize());
        assertEquals(List.of(0L, 6L, 6L, 3L), api.createdSizes);
        assertEquals(4, api.distinctBufferHandleCount());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(8);
            MemorySegment destination = arena.allocate(8);
            for (int index = 0; index < 8; index++) {
                source.setAtIndex(JAVA_BYTE, index, (byte) (10 + index));
                destination.setAtIndex(JAVA_BYTE, index, (byte) -1);
            }
            first.upload(1, source, 2, 4);
            first.download(1, destination, 3, 4);
            assertArrayEquals(
                    new byte[] {-1, -1, -1, 12, 13, 14, 15, -1},
                    destination.toArray(JAVA_BYTE));
            assertEquals(1, api.uploadCalls.get());
            assertEquals(1, api.downloadCalls.get());

            zero.upload(0, source, 8, 0);
            zero.download(0, destination, 8, 0);
            assertEquals(2, api.uploadCalls.get());
            assertEquals(2, api.downloadCalls.get());
        }

        workspace.close();
        second.close();
        first.close();
        zero.close();
        context.close();
        assertEquals(4, api.bufferReleaseCalls.get());
        assertEquals(1, api.contextReleaseCalls.get());
        assertEquals(1, api.closeCalls.get());
    }

    @Test
    void validatesAllRangesAndSegmentStateBeforeNativeAccess() {
        var api = new FakeNativeApi();
        var context = MetalDeviceContext.open(api);
        var buffer = context.createBuffer(4);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bytes = arena.allocate(4);
            assertThrows(IllegalArgumentException.class, () -> buffer.upload(-1, bytes, 0, 0));
            assertThrows(IllegalArgumentException.class, () -> buffer.upload(5, bytes, 0, 0));
            assertThrows(IllegalArgumentException.class, () -> buffer.upload(0, bytes, -1, 0));
            assertThrows(IllegalArgumentException.class, () -> buffer.upload(0, bytes, 0, 5));
            assertThrows(IllegalArgumentException.class, () -> buffer.download(0, bytes, 3, 2));
            assertThrows(IllegalArgumentException.class,
                    () -> buffer.download(0, bytes.asReadOnly(), 0, 1));
            assertEquals(0, api.uploadCalls.get());
            assertEquals(0, api.downloadCalls.get());
        }

        MemorySegment dead;
        try (Arena arena = Arena.ofConfined()) {
            dead = arena.allocate(1);
        }
        assertThrows(IllegalStateException.class, () -> buffer.upload(0, dead, 0, 1));

        buffer.close();
        context.close();
    }

    @Test
    void contextCloseIsVisibleImmediatelyAndNativeReleaseWaitsForChildren() {
        var api = new FakeNativeApi();
        var context = MetalDeviceContext.open(api);
        var buffer = context.createBuffer(2);
        var workspace = context.createWorkspace(2);

        context.close();
        assertTrue(context.isClosed());
        assertEquals(0, api.contextReleaseCalls.get());
        assertThrows(IllegalStateException.class, () -> context.createBuffer(1));
        try (Arena arena = Arena.ofConfined()) {
            assertThrows(IllegalStateException.class,
                    () -> buffer.upload(0, arena.allocate(1), 0, 1));
        }

        buffer.close();
        assertEquals(0, api.contextReleaseCalls.get());
        workspace.close();
        assertEquals(1, api.contextReleaseCalls.get());
        assertEquals(1, api.closeCalls.get());
        context.close();
        assertEquals(1, api.contextReleaseCalls.get());
    }

    @Test
    void concurrentResourceAndContextCloseConsumeEveryHandleOnce() throws Exception {
        var api = new FakeNativeApi();
        var context = MetalDeviceContext.open(api);
        var buffer = context.createBuffer(1);
        try (var executor = Executors.newFixedThreadPool(8)) {
            var futures = new ArrayList<java.util.concurrent.Future<?>>();
            for (int index = 0; index < 20; index++) {
                futures.add(executor.submit(index % 2 == 0 ? buffer::close : context::close));
            }
            for (var future : futures) {
                future.get(5, TimeUnit.SECONDS);
            }
        }
        assertTrue(buffer.isClosed());
        assertTrue(context.isClosed());
        assertEquals(1, api.bufferReleaseCalls.get());
        assertEquals(1, api.contextReleaseCalls.get());
        assertEquals(1, api.closeCalls.get());
    }

    @Test
    void closeWaitsForAnAdmittedAccess() throws Exception {
        var api = new FakeNativeApi();
        api.blockUploads = true;
        var context = MetalDeviceContext.open(api);
        var buffer = context.createBuffer(1);
        try (Arena arena = Arena.ofShared(); var executor = Executors.newFixedThreadPool(2)) {
            MemorySegment source = arena.allocate(1);
            var access = executor.submit(() -> buffer.upload(0, source, 0, 1));
            assertTrue(api.uploadEntered.await(5, TimeUnit.SECONDS));
            CountDownLatch closeStarted = new CountDownLatch(1);
            var close = executor.submit(() -> {
                closeStarted.countDown();
                buffer.close();
            });
            assertTrue(closeStarted.await(5, TimeUnit.SECONDS));
            assertEquals(0, api.bufferReleaseCalls.get());
            api.continueUpload.countDown();
            access.get(5, TimeUnit.SECONDS);
            close.get(5, TimeUnit.SECONDS);
        }
        context.close();
        assertEquals(1, api.bufferReleaseCalls.get());
    }

    @Test
    void distinctBufferAccessesMayOverlap() throws Exception {
        var api = new FakeNativeApi();
        api.blockUploads = true;
        api.uploadEntered = new CountDownLatch(2);
        var context = MetalDeviceContext.open(api);
        var first = context.createBuffer(1);
        var second = context.createBuffer(1);
        try (Arena arena = Arena.ofShared(); var executor = Executors.newFixedThreadPool(2)) {
            MemorySegment source = arena.allocate(1);
            var firstAccess = executor.submit(() -> first.upload(0, source, 0, 1));
            var secondAccess = executor.submit(() -> second.upload(0, source, 0, 1));
            try {
                assertTrue(api.uploadEntered.await(5, TimeUnit.SECONDS));
            } finally {
                api.continueUpload.countDown();
            }
            firstAccess.get(5, TimeUnit.SECONDS);
            secondAccess.get(5, TimeUnit.SECONDS);
        }
        first.close();
        second.close();
        context.close();
        assertEquals(2, api.uploadCalls.get());
    }

    @Test
    void contextClosePreservesAdmittedAccessAndRejectsLaterAccessBeforeNativeCall()
            throws Exception {
        var api = new FakeNativeApi();
        api.blockUploads = true;
        var context = MetalDeviceContext.open(api);
        var admittedBuffer = context.createBuffer(1);
        var laterBuffer = context.createBuffer(1);
        try (Arena arena = Arena.ofShared(); var executor = Executors.newFixedThreadPool(2)) {
            MemorySegment source = arena.allocate(1);
            var admitted = executor.submit(() -> admittedBuffer.upload(0, source, 0, 1));
            assertTrue(api.uploadEntered.await(5, TimeUnit.SECONDS));
            var close = executor.submit(context::close);
            try {
                close.get(5, TimeUnit.SECONDS);
                assertTrue(context.isClosed());
                assertThrows(IllegalStateException.class,
                        () -> laterBuffer.upload(0, source, 0, 1));
                assertEquals(1, api.uploadCalls.get());
            } finally {
                api.continueUpload.countDown();
            }
            admitted.get(5, TimeUnit.SECONDS);
        }
        admittedBuffer.close();
        laterBuffer.close();
        assertEquals(1, api.uploadCalls.get());
        assertEquals(1, api.contextReleaseCalls.get());
    }

    @Test
    void cleanupPreservesPrimaryAndDeterministicDistinctSuppressedFailures() {
        var api = new FakeNativeApi();
        var context = MetalDeviceContext.open(api);
        var buffer = context.createBuffer(1);
        var primary = new IllegalStateException("buffer release");
        var contextFailure = new IllegalStateException("context release");
        var lookupFailure = new IllegalStateException("lookup close");
        api.bufferReleaseFailure = primary;
        api.contextReleaseFailure = contextFailure;
        api.closeFailure = lookupFailure;

        context.close();
        var actual = assertThrows(IllegalStateException.class, buffer::close);
        assertSame(primary, actual);
        assertArrayEquals(new Throwable[] {contextFailure, lookupFailure}, actual.getSuppressed());
        buffer.close();
        context.close();
        assertEquals(1, api.bufferReleaseCalls.get());
        assertEquals(1, api.contextReleaseCalls.get());
        assertEquals(1, api.closeCalls.get());
    }

    @Test
    void failedContextCreationClosesTheAcquiredNativeApiAndSuppressesCleanup() {
        var api = new FakeNativeApi();
        var primary = new IllegalStateException("create");
        var cleanup = new IllegalStateException("close");
        api.contextCreateFailure = primary;
        api.closeFailure = cleanup;

        var actual = assertThrows(IllegalStateException.class, () -> MetalDeviceContext.open(api));
        assertSame(primary, actual);
        assertArrayEquals(new Throwable[] {cleanup}, actual.getSuppressed());
        assertEquals(1, api.closeCalls.get());
        assertEquals(0, api.contextReleaseCalls.get());
    }

    @Test
    void nativeFailureRetainsKnownAndUnknownStatusFacts() {
        var known = new MetalNativeApi.NativeFailure("upload", 5);
        var unknown = new MetalNativeApi.NativeFailure("upload", 91);
        assertEquals("upload", known.operation());
        assertEquals(5, known.statusCode());
        assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, known.status());
        assertEquals(91, unknown.statusCode());
        assertEquals(null, unknown.status());
        assertTrue(unknown.getMessage().contains("unknown native status 91"));
    }

    @Test
    void negExecutableCreatePreflightRejectsMalformedAbiWithoutNativeInvocation() {
        var api = new FakeNativeApi();
        MetalNativeApi.Handle context = new MetalNativeApi.Handle(MemorySegment.ofAddress(97));
        NegCreate valid = validNegCreate();

        assertInvalidCreate(api, context, valid.withRanks(new int[0]));
        assertInvalidCreate(api, context, valid.withDimensions(new long[47]));
        assertInvalidCreate(api, context, valid.withNodeInputs(new int[0]));
        assertInvalidCreate(api, context, valid.withNodeOutputs(new int[] {1}));
        assertInvalidCreate(api, context, valid.withFeeds(new int[0]));
        assertInvalidCreate(api, context, valid.withTargets(new int[0]));

        int[] zeroRank = valid.ranks().clone();
        zeroRank[0] = 0;
        assertInvalidCreate(api, context, valid.withRanks(zeroRank));
        int[] excessiveRank = valid.ranks().clone();
        excessiveRank[0] = 17;
        assertInvalidCreate(api, context, valid.withRanks(excessiveRank));
        long[] nonzeroPadding = valid.dimensions().clone();
        nonzeroPadding[2] = 1;
        assertInvalidCreate(api, context, valid.withDimensions(nonzeroPadding));

        assertInvalidCreate(api, context, valid.withFeeds(new int[] {-1}));
        assertInvalidCreate(api, context, valid.withFeeds(new int[] {3}));
        assertInvalidCreate(api, context, valid.withFeeds(new int[] {0, 0}));
        assertInvalidCreate(api, context, valid.withTargets(new int[] {-1}));
        assertInvalidCreate(api, context, valid.withTargets(new int[] {3}));
        assertInvalidCreate(api, context, valid.withTargets(new int[] {0}));
        assertInvalidCreate(api, context, valid.withTargets(new int[] {2, 2}));

        assertInvalidCreate(api, context, valid.withNodeInputs(new int[] {3, 1}));
        assertInvalidCreate(api, context, valid.withNodeInputs(new int[] {1, 0}));
        assertInvalidCreate(api, context, valid.withNodeOutputs(new int[] {3, 2}));
        assertInvalidCreate(api, context, valid.withNodeOutputs(new int[] {0, 2}));
        assertInvalidCreate(api, context, valid.withNodeOutputs(new int[] {1, 1}));

        int[] mismatchedRank = valid.ranks().clone();
        mismatchedRank[1] = 1;
        assertInvalidCreate(api, context, valid.withRanks(mismatchedRank));
        long[] mismatchedShape = valid.dimensions().clone();
        mismatchedShape[16] = 7;
        assertInvalidCreate(api, context, valid.withDimensions(mismatchedShape));
        long[] zeroDimension = valid.dimensions().clone();
        zeroDimension[0] = 0;
        assertInvalidCreate(api, context, valid.withDimensions(zeroDimension));
        long[] negativeDimension = valid.dimensions().clone();
        negativeDimension[0] = -1;
        assertInvalidCreate(api, context, valid.withDimensions(negativeDimension));
        long[] overflowingGeometry = valid.dimensions().clone();
        overflowingGeometry[0] = Long.MAX_VALUE;
        overflowingGeometry[1] = 2;
        assertInvalidCreate(api, context, valid.withDimensions(overflowingGeometry));

        int[] ranksWithUnused = {2, 2, 2, 2};
        long[] dimensionsWithUnused = new long[64];
        for (int value = 0; value < 4; value++) {
            dimensionsWithUnused[value * 16] = 2;
            dimensionsWithUnused[value * 16 + 1] = 3;
        }
        assertInvalidCreate(api, context,
                new NegCreate(ranksWithUnused, dimensionsWithUnused,
                        valid.nodeInputs(), valid.nodeOutputs(), valid.feeds(), valid.targets()));

        assertEquals(0, api.executableCreateCalls.get());
        assertTrue(api.createNegExecutable(context, valid.ranks(), valid.dimensions(),
                valid.nodeInputs(), valid.nodeOutputs(), valid.feeds(), valid.targets()) != null);
        assertEquals(1, api.executableCreateCalls.get());
    }

    @Test
    void nativeFoundationRoundTrip() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        var context = MetalDeviceContext.open(library);
        var zero = context.createBuffer(0);
        var first = context.createBuffer(8);
        var second = context.createBuffer(8);
        var workspace = context.createWorkspace(5);
        assertNotEquals(first, second);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(8);
            MemorySegment destination = arena.allocate(8);
            for (int index = 0; index < 8; index++) {
                source.setAtIndex(JAVA_BYTE, index, (byte) (index + 1));
            }
            first.upload(2, source, 1, 5);
            first.download(2, destination, 2, 5);
            assertArrayEquals(new byte[] {0, 0, 2, 3, 4, 5, 6, 0},
                    destination.toArray(JAVA_BYTE));
            zero.upload(0, source, 8, 0);
            zero.download(0, destination, 8, 0);
            assertThrows(IllegalArgumentException.class, () -> first.upload(7, source, 0, 2));

            context.close();
            assertThrows(IllegalStateException.class,
                    () -> second.download(0, destination, 0, 1));
        } finally {
            workspace.close();
            second.close();
            first.close();
            zero.close();
            context.close();
        }
    }

    private static void assertInvalidCreate(
            FakeNativeApi api, MetalNativeApi.Handle context, NegCreate input) {
        int before = api.executableCreateCalls.get();
        assertThrows(IllegalArgumentException.class,
                () -> api.createNegExecutable(context, input.ranks(), input.dimensions(),
                        input.nodeInputs(), input.nodeOutputs(), input.feeds(), input.targets()));
        assertEquals(before, api.executableCreateCalls.get());
    }

    private static NegCreate validNegCreate() {
        int[] ranks = {2, 2, 2};
        long[] dimensions = new long[48];
        for (int value = 0; value < 3; value++) {
            dimensions[value * 16] = 2;
            dimensions[value * 16 + 1] = 3;
        }
        return new NegCreate(ranks, dimensions, new int[] {0, 1}, new int[] {1, 2},
                new int[] {0}, new int[] {2});
    }

    private record NegCreate(
            int[] ranks, long[] dimensions, int[] nodeInputs, int[] nodeOutputs,
            int[] feeds, int[] targets) {
        NegCreate withRanks(int[] replacement) {
            return new NegCreate(replacement, dimensions, nodeInputs, nodeOutputs, feeds, targets);
        }

        NegCreate withDimensions(long[] replacement) {
            return new NegCreate(ranks, replacement, nodeInputs, nodeOutputs, feeds, targets);
        }

        NegCreate withNodeInputs(int[] replacement) {
            return new NegCreate(ranks, dimensions, replacement, nodeOutputs, feeds, targets);
        }

        NegCreate withNodeOutputs(int[] replacement) {
            return new NegCreate(ranks, dimensions, nodeInputs, replacement, feeds, targets);
        }

        NegCreate withFeeds(int[] replacement) {
            return new NegCreate(ranks, dimensions, nodeInputs, nodeOutputs, replacement, targets);
        }

        NegCreate withTargets(int[] replacement) {
            return new NegCreate(ranks, dimensions, nodeInputs, nodeOutputs, feeds, replacement);
        }
    }

    private static final class FakeNativeApi extends MetalNativeApi {
        private long nextHandle = 1L;
        private final Map<Long, byte[]> buffers = Collections.synchronizedMap(new HashMap<>());
        private final List<Long> bufferHandles = new ArrayList<>();
        private final List<Long> createdSizes = Collections.synchronizedList(new ArrayList<>());
        private final AtomicInteger bufferReleaseCalls = new AtomicInteger();
        private final AtomicInteger contextReleaseCalls = new AtomicInteger();
        private final AtomicInteger uploadCalls = new AtomicInteger();
        private final AtomicInteger downloadCalls = new AtomicInteger();
        private final AtomicInteger closeCalls = new AtomicInteger();
        private final AtomicInteger executableCreateCalls = new AtomicInteger();
        private CountDownLatch uploadEntered = new CountDownLatch(1);
        private final CountDownLatch continueUpload = new CountDownLatch(1);
        private volatile boolean blockUploads;
        private RuntimeException contextCreateFailure;
        private RuntimeException bufferReleaseFailure;
        private RuntimeException contextReleaseFailure;
        private RuntimeException closeFailure;

        @Override
        synchronized Handle createContext() {
            if (contextCreateFailure != null) {
                throw contextCreateFailure;
            }
            return handle();
        }

        @Override
        void releaseContext(Handle context) {
            contextReleaseCalls.incrementAndGet();
            if (contextReleaseFailure != null) {
                throw contextReleaseFailure;
            }
        }

        @Override
        synchronized Handle createBuffer(Handle context, long logicalByteSize) {
            if (logicalByteSize > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("fake size too large");
            }
            Handle handle = handle();
            buffers.put(handle.carrier().address(), new byte[(int) logicalByteSize]);
            bufferHandles.add(handle.carrier().address());
            createdSizes.add(logicalByteSize);
            return handle;
        }

        @Override
        synchronized void releaseBuffer(Handle buffer) {
            bufferReleaseCalls.incrementAndGet();
            buffers.remove(buffer.carrier().address());
            if (bufferReleaseFailure != null) {
                throw bufferReleaseFailure;
            }
        }

        @Override
        void upload(
                Handle buffer, long bufferOffset, MemorySegment source, long byteCount) {
            uploadCalls.incrementAndGet();
            if (blockUploads) {
                uploadEntered.countDown();
                try {
                    if (!continueUpload.await(5, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("timed out waiting to continue upload");
                    }
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(interrupted);
                }
            }
            byte[] target = buffers.get(buffer.carrier().address());
            for (long index = 0; index < byteCount; index++) {
                target[Math.toIntExact(bufferOffset + index)] = source.getAtIndex(JAVA_BYTE, index);
            }
        }

        @Override
        synchronized void download(
                Handle buffer, long bufferOffset, MemorySegment destination, long byteCount) {
            downloadCalls.incrementAndGet();
            byte[] source = buffers.get(buffer.carrier().address());
            for (long index = 0; index < byteCount; index++) {
                destination.setAtIndex(
                        JAVA_BYTE, index, source[Math.toIntExact(bufferOffset + index)]);
            }
        }

        @Override
        synchronized NativeCreateResult createNegExecutableNative(
                Handle context,
                int[] valueRanks,
                long[] valueDimensions,
                int[] nodeInputValueIndices,
                int[] nodeOutputValueIndices,
                int[] feedValueIndices,
                int[] targetValueIndices) {
            executableCreateCalls.incrementAndGet();
            return new NativeCreateResult(0, handle());
        }

        @Override
        int releaseExecutableNative(Handle executable) {
            // Foundation-only tests do not create an executable.
            return 0;
        }

        @Override
        int runExecutableNative(Handle executable, int inputCount, MemorySegment inputBuffers,
                int outputCount, MemorySegment outputBuffers) {
            // Foundation-only tests do not execute a graph.
            return 0;
        }

        @Override
        public void close() {
            closeCalls.incrementAndGet();
            if (closeFailure != null) {
                throw closeFailure;
            }
        }

        synchronized int distinctBufferHandleCount() {
            return (int) bufferHandles.stream().distinct().count();
        }

        private Handle handle() {
            return new Handle(MemorySegment.ofAddress(nextHandle++));
        }
    }
}
