package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CreatedBuffer;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;

/** Exercises the package-private composition seam and Engine lifecycle gate. */
final class AdvancedEngineLifecycleTest {
    private static final BackendId BACKEND = new BackendId("test");

    @Test
    void nestedOperationsKeepTheirOuterAdmissionAndRejectOnlyTheirOwnEngineClose() throws Exception {
        var executor = Executors.newSingleThreadExecutor(Thread.ofPlatform().daemon().factory());
        try {
            executor.submit(() -> {
                AtomicReference<AdvancedEngine> owner = new AtomicReference<>();
                List<String> order = new ArrayList<>();
                RecordingComposition unrelatedComposition = new RecordingComposition(emptyExecution());
                AdvancedEngine unrelated = new AdvancedEngine(
                        unrelatedComposition, unrelatedComposition::prepare);
                RecordingComposition composition = new RecordingComposition(
                        createdBufferExecution((bytes, alignment) -> {
                            AdvancedEngine engine = owner.get();
                            assertThrows(IllegalStateException.class, engine::close);
                            assertFalse(engine.isClosed());
                            compile(engine, leaf().abs());
                            assertThrows(NullPointerException.class, () -> engine.prepare(null));
                            assertThrows(IllegalStateException.class, engine::close);
                            assertFalse(engine.isClosed());
                            unrelated.close();
                            assertTrue(unrelated.isClosed());
                            return new TestBuffer(null, order, "result");
                        }));
                try (AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare)) {
                    owner.set(engine);
                    try (var result = engine.run(
                            engine.prepare(compile(engine, leaf().neg())), List.of())) {
                        assertFalse(result.isClosed());
                    }
                    assertEquals(List.of("result"), order);
                    assertFalse(engine.isClosed());
                }
                return null;
            }).get(10, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void admittedSelfCloseRejectsEvenWhileAnotherCloserWaits() throws Exception {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch attemptClose = new CountDownLatch(1);
        AtomicReference<AdvancedEngine> owner = new AtomicReference<>();
        RecordingComposition composition = new RecordingComposition(createdBufferExecution(
                (bytes, alignment) -> {
                    entered.countDown();
                    try {
                        assertTrue(attemptClose.await(10, TimeUnit.SECONDS));
                    } catch (InterruptedException failure) {
                        Thread.currentThread().interrupt();
                        throw new AssertionError(failure);
                    }
                    assertThrows(IllegalStateException.class, owner.get()::close);
                    return new TestBuffer(null, new ArrayList<>(), "result");
                }));
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        owner.set(engine);
        var prepared = engine.prepare(compile(engine, leaf().neg()));
        var executor = Executors.newFixedThreadPool(2, Thread.ofPlatform().daemon().factory());
        try {
            var run = executor.submit(() -> engine.run(prepared, List.of()));
            assertTrue(entered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> { engine.close(); return null; });
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (!engine.isClosed() && System.nanoTime() < deadline) Thread.onSpinWait();
            assertTrue(engine.isClosed());
            assertFalse(close.isDone());
            attemptClose.countDown();
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> run.get(10, TimeUnit.SECONDS));
            assertTrue(failure.getCause() instanceof IllegalStateException);
            close.get(10, TimeUnit.SECONDS);
        } finally {
            attemptClose.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void cleanupOwnerRecursionReturnsButOtherClosersWaitAndReplayFailure() throws Exception {
        AtomicReference<AdvancedEngine> owner = new AtomicReference<>();
        CountDownLatch recursiveCloseReturned = new CountDownLatch(1);
        CountDownLatch finishCleanup = new CountDownLatch(1);
        RuntimeException expected = new RuntimeException("prepared cleanup");
        PreparedResource resource = () -> {
            owner.get().close();
            recursiveCloseReturned.countDown();
            try {
                assertTrue(finishCleanup.await(10, TimeUnit.SECONDS));
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new AssertionError(failure);
            }
            throw expected;
        };
        RecordingComposition composition = new RecordingComposition(resourceExecution(resource));
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        owner.set(engine);
        engine.prepare(compile(engine, leaf().neg()));
        var executor = Executors.newFixedThreadPool(2, Thread.ofPlatform().daemon().factory());
        try {
            var first = executor.submit(() -> assertSame(expected,
                    assertThrows(RuntimeException.class, engine::close)));
            assertTrue(recursiveCloseReturned.await(10, TimeUnit.SECONDS));
            CountDownLatch otherStarted = new CountDownLatch(1);
            var other = executor.submit(() -> {
                Thread.currentThread().interrupt();
                otherStarted.countDown();
                assertSame(expected, assertThrows(RuntimeException.class, engine::close));
                assertTrue(Thread.currentThread().isInterrupted());
                Thread.interrupted();
            });
            assertTrue(otherStarted.await(10, TimeUnit.SECONDS));
            assertFalse(other.isDone());
            finishCleanup.countDown();
            first.get(10, TimeUnit.SECONDS);
            other.get(10, TimeUnit.SECONDS);
            assertSame(expected, assertThrows(RuntimeException.class, engine::close));
        } finally {
            finishCleanup.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void compileDelegatesOnceInExactOrderAndPreservesFailures() {
        RecordingComposition composition = new RecordingComposition(emptyExecution());
        try (AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare)) {
            Tensor input = leaf();
            AdvancedCompiledGraph compiled = compile(engine, input.neg());
            assertSame(engine, compiled.owner());
            assertEquals(1, composition.supportCount.get());

            RuntimeException expected = new RuntimeException("capability");
            composition.capabilityFailure = expected;
            RuntimeException observed = assertThrows(RuntimeException.class,
                    () -> compile(engine, input.abs()));
            assertSame(expected, observed);
        }
    }

    @Test
    void validatesOwnerAndStateBeforeCompositionCallbacks() {
        RecordingComposition firstComposition = new RecordingComposition(emptyExecution());
        RecordingComposition secondComposition = new RecordingComposition(emptyExecution());
        AdvancedEngine first = new AdvancedEngine(firstComposition, firstComposition::prepare);
        AdvancedEngine second = new AdvancedEngine(secondComposition, secondComposition::prepare);
        AdvancedCompiledGraph compiled = compile(first, leaf().neg());
        AdvancedPreparedExecution prepared = first.prepare(compiled);

        assertThrows(IllegalArgumentException.class, () -> second.prepare(compiled));
        assertThrows(IllegalArgumentException.class, () -> second.run(prepared, List.of()));
        assertEquals(0, secondComposition.prepareCount.get());

        first.close();
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class, () -> first.prepare(null)).getMessage());
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class, () -> first.prepare(compiled)).getMessage());
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class,
                        () -> first.run(null, null)).getMessage());
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class,
                        () -> first.run(prepared, null)).getMessage());
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class, () -> first.run(prepared, List.of())).getMessage());
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class, () -> first.borrow(null)).getMessage());
        second.close();
    }

    @Test
    void closeWaitsForAdmittedRunAndRollsItsResultBack() throws Exception {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        TestBuffer owned = new TestBuffer(null, new ArrayList<>(), "result");
        PreparedExecution execution = blockingExecution(entered, release, owned);
        RecordingComposition composition = new RecordingComposition(execution);
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedPreparedExecution prepared = engine.prepare(compile(engine, leaf().neg()));

        try (var executor = Executors.newFixedThreadPool(2)) {
            var run = executor.submit(() -> engine.run(prepared, List.of()));
            assertTrue(entered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> { engine.close(); return null; });
            while (!engine.isClosed()) Thread.onSpinWait();
            release.countDown();
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> run.get(10, TimeUnit.SECONDS));
            assertEquals("advanced engine is closed", failure.getCause().getMessage());
            close.get(10, TimeUnit.SECONDS);
        }
        assertEquals(1, owned.closeCount.get());
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void failedAdvancedResultPublicationRollsBackAndReleasesAdmission() throws Exception {
        List<String> order = new ArrayList<>();
        Error publicationFailure = new AssertionError("advanced result publication");
        RuntimeException cleanupFailure = new RuntimeException("result cleanup");
        TestBuffer buffer = new TestBuffer(cleanupFailure, order, "result");
        TestPreparedResource resource = new TestPreparedResource(null, order, "prepared");
        RecordingComposition composition = new RecordingComposition(
                resourceAndCreatedBufferExecution(resource,
                        (requestedBytes, requestedAlignment) -> buffer));
        composition.closeOrder = order;
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedPreparedExecution prepared = engine.prepare(compile(engine, leaf().neg()));
        replaceField(engine, "openResults", rejectingPublications(publicationFailure));

        assertSame(publicationFailure, assertThrows(AssertionError.class,
                () -> engine.run(prepared, List.of())));
        assertArrayEquals(new Throwable[] {cleanupFailure}, publicationFailure.getSuppressed());
        assertEquals(List.of("result"), order);
        assertEquals(1, buffer.closeCount.get());

        // Do not close in teardown: the unfixed publication path strands Engine admission.
        // A daemon closer also bounds a regression that cleans up but still strands admission.
        var closer = Executors.newSingleThreadExecutor(Thread.ofPlatform().daemon().factory());
        try {
            closer.submit(() -> { engine.close(); return null; }).get(10, TimeUnit.SECONDS);
        } finally {
            closer.shutdownNow();
        }
        assertTrue(engine.isClosed());
        assertEquals(List.of("result", "prepared", "backend"), order);
        assertEquals(1, buffer.closeCount.get());
        assertEquals(1, resource.closeCount.get());
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void closesResultsInReverseOrderThenBackendAndRetainsFailureTree() {
        List<String> order = new ArrayList<>();
        RuntimeException firstFailure = new RuntimeException("second result");
        Error laterFailure = new AssertionError("first result");
        RuntimeException backendFailure = new RuntimeException("backend");
        AtomicInteger sequence = new AtomicInteger();
        RecordingComposition composition = new RecordingComposition(
                createdBufferExecution((requestedBytes, requestedAlignment) -> sequence.getAndIncrement() == 0
                        ? new TestBuffer(laterFailure, order, "first")
                        : new TestBuffer(firstFailure, order, "second")));
        composition.closeFailure = backendFailure;
        composition.closeOrder = order;
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedPreparedExecution prepared = engine.prepare(compile(engine, leaf().neg()));
        AdvancedRunResult first = engine.run(prepared, List.of());
        AdvancedRunResult second = engine.run(prepared, List.of());

        RuntimeException observed = assertThrows(RuntimeException.class, engine::close);
        assertSame(firstFailure, observed);
        assertArrayEquals(new Throwable[] {laterFailure, backendFailure}, observed.getSuppressed());
        assertEquals(List.of("second", "first", "backend"), order);
        assertTrue(first.isClosed());
        assertTrue(second.isClosed());
        assertSame(observed, assertThrows(RuntimeException.class, engine::close));
    }

    @Test
    void concurrentEngineAndResultCloseAreExactlyOnce() throws Exception {
        List<String> order = java.util.Collections.synchronizedList(new ArrayList<>());
        TestBuffer buffer = new TestBuffer(null, order, "result");
        RecordingComposition composition = new RecordingComposition(createdBufferExecution((requestedBytes, requestedAlignment) -> buffer));
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedRunResult result = engine.run(
                engine.prepare(compile(engine, leaf().neg())), List.of());
        try (var executor = Executors.newFixedThreadPool(4)) {
            var tasks = List.of(
                    executor.submit(() -> { result.close(); return null; }),
                    executor.submit(() -> { result.close(); return null; }),
                    executor.submit(() -> { engine.close(); return null; }),
                    executor.submit(() -> { engine.close(); return null; }));
            for (var task : tasks) task.get(10, TimeUnit.SECONDS);
        }
        assertEquals(1, buffer.closeCount.get());
        assertEquals(1, composition.closeCount.get());
        assertTrue(engine.isClosed());
    }

    @Test
    void preservesPrimaryErrorIdentityAndSuppressesLaterBackendFailure() {
        List<String> order = new ArrayList<>();
        Error resultFailure = new AssertionError("result");
        RuntimeException backendFailure = new RuntimeException("backend");
        RecordingComposition composition = new RecordingComposition(
                createdBufferExecution((requestedBytes, requestedAlignment) -> new TestBuffer(resultFailure, order, "result")));
        composition.closeFailure = backendFailure;
        composition.closeOrder = order;
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        engine.run(engine.prepare(compile(engine, leaf().neg())), List.of());

        Error observed = assertThrows(Error.class, engine::close);
        assertSame(resultFailure, observed);
        assertArrayEquals(new Throwable[] {backendFailure}, observed.getSuppressed());
        assertEquals(List.of("result", "backend"), order);
        assertSame(observed, assertThrows(Error.class, engine::close));
    }

    @Test
    void preparedCloseIsExactOnceRetainsFailureAndUnregisters() throws Exception {
        RuntimeException expected = new RuntimeException("prepared");
        TestPreparedResource resource = new TestPreparedResource(expected, new ArrayList<>(), "prepared");
        RecordingComposition composition = new RecordingComposition(
                () -> resourceExecution(resource));
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedPreparedExecution prepared = engine.prepare(compile(engine, leaf().neg()));

        try (var executor = Executors.newFixedThreadPool(3)) {
            var first = executor.submit(() -> { prepared.close(); return null; });
            var second = executor.submit(() -> { prepared.close(); return null; });
            for (var future : List.of(first, second)) {
                var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                        () -> future.get(10, TimeUnit.SECONDS));
                assertSame(expected, failure.getCause());
            }
        }
        assertTrue(prepared.isClosed());
        assertEquals(1, resource.closeCount.get());
        engine.close();
        assertEquals(1, resource.closeCount.get());
    }

    @Test
    void engineClosesResultsThenPreparationsInReversePublicationOrderThenComposition() {
        List<String> order = new ArrayList<>();
        AtomicInteger preparationIndex = new AtomicInteger();
        RecordingComposition composition = new RecordingComposition(() -> {
            int index = preparationIndex.incrementAndGet();
            return resourceAndCreatedBufferExecution(
                    new TestPreparedResource(null, order, "prepared-" + index),
                    (requestedBytes, requestedAlignment) -> new TestBuffer(null, order, "result-" + index));
        });
        composition.closeOrder = order;
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedCompiledGraph compiled = compile(engine, leaf().neg());
        AdvancedPreparedExecution first = engine.prepare(compiled);
        AdvancedPreparedExecution second = engine.prepare(compiled);
        engine.run(first, List.of());
        engine.run(second, List.of());

        engine.close();

        assertEquals(List.of("result-2", "result-1", "prepared-2", "prepared-1", "backend"),
                order);
        assertTrue(first.isClosed());
        assertTrue(second.isClosed());
    }

    @Test
    void shutdownDrainsRetainedOwnersWithoutAllocatingSnapshotsAndReplaysCleanupFailure() {
        for (boolean failingCleanup : List.of(false, true)) {
            var order = new ArrayList<String>();
            var snapshotFailure = new OutOfMemoryError("shutdown snapshot allocation");
            var primary = new OutOfMemoryError("second result cleanup");
            var preparedFailure = new RuntimeException("first prepared cleanup");
            var backendFailure = new AssertionError("backend cleanup");
            AtomicInteger preparationIndex = new AtomicInteger();
            RecordingComposition composition = new RecordingComposition(() -> {
                int index = preparationIndex.incrementAndGet();
                return resourceAndCreatedBufferExecution(
                        new TestPreparedResource(
                                failingCleanup && index == 1 ? preparedFailure : null,
                                order, "prepared-" + index),
                        (bytes, alignment) -> new TestBuffer(
                                failingCleanup && index == 2 ? primary : null,
                                order, "result-" + index));
            });
            composition.closeOrder = order;
            composition.closeFailure = failingCleanup ? backendFailure : null;
            AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
            ArrayList<AdvancedRunResult> results = unsnapshotableOwners(snapshotFailure);
            ArrayList<AutoCloseable> preparations = unsnapshotableOwners(snapshotFailure);
            replaceField(engine, "openResults", results);
            replaceField(engine, "openPreparations", preparations);
            var compiled = compile(engine, leaf().neg());
            var firstPrepared = engine.prepare(compiled);
            var secondPrepared = engine.prepare(compiled);
            var firstResult = engine.run(firstPrepared, List.of());
            var secondResult = engine.run(secondPrepared, List.of());

            if (failingCleanup) {
                assertSame(primary, assertThrows(OutOfMemoryError.class, engine::close));
                assertArrayEquals(new Throwable[] {preparedFailure, backendFailure},
                        primary.getSuppressed());
                assertSame(primary, assertThrows(OutOfMemoryError.class, engine::close));
            } else {
                engine.close();
                engine.close();
            }
            assertEquals(List.of("result-2", "result-1", "prepared-2", "prepared-1", "backend"),
                    order);
            assertTrue(results.isEmpty());
            assertTrue(preparations.isEmpty());
            assertTrue(firstResult.isClosed());
            assertTrue(secondResult.isClosed());
            assertTrue(firstPrepared.isClosed());
            assertTrue(secondPrepared.isClosed());
        }
    }

    @Test
    void closeRacingPreparedPublicationRollsBackTheUnpublishedExecution() throws Exception {
        CountDownLatch prepared = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        TestPreparedResource resource = new TestPreparedResource(null, new ArrayList<>(), "prepared");
        RecordingComposition composition = new RecordingComposition(() -> {
            prepared.countDown();
            try {
                assertTrue(release.await(10, TimeUnit.SECONDS));
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new AssertionError(failure);
            }
            return resourceExecution(resource);
        });
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedCompiledGraph compiled = compile(engine, leaf().neg());
        try (var executor = Executors.newFixedThreadPool(2)) {
            var preparation = executor.submit(() -> engine.prepare(compiled));
            assertTrue(prepared.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> { engine.close(); return null; });
            while (!engine.isClosed()) Thread.onSpinWait();
            release.countDown();
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> preparation.get(10, TimeUnit.SECONDS));
            assertEquals("advanced engine is closed", failure.getCause().getMessage());
            close.get(10, TimeUnit.SECONDS);
        }
        assertEquals(1, resource.closeCount.get());
    }

    @Test
    void directPreparedCloseDelegatesRunRaceToRuntimeLease() throws Exception {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        TestPreparedResource resource = new TestPreparedResource(null, new ArrayList<>(), "prepared");
        RecordingComposition composition = new RecordingComposition(
                blockingExecution(entered, release,
                        new TestBuffer(null, new ArrayList<>(), "result"), resource));
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedPreparedExecution prepared = engine.prepare(compile(engine, leaf().neg()));

        try (var executor = Executors.newFixedThreadPool(2)) {
            var run = executor.submit(() -> engine.run(prepared, List.of()));
            assertTrue(entered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> { prepared.close(); return null; });
            close.get(10, TimeUnit.SECONDS);
            assertEquals(0, resource.closeCount.get());
            release.countDown();
            run.get(10, TimeUnit.SECONDS).close();
        }
        assertEquals(1, resource.closeCount.get());
        assertEquals("prepared execution is closed",
                assertThrows(IllegalStateException.class,
                        () -> engine.run(prepared, List.of())).getMessage());
        engine.close();
    }

    @Test
    void directPreparedCloseWinningOutwardBoundaryRejectsConcurrentRun() throws Exception {
        TestPreparedResource resource =
                new TestPreparedResource(null, new ArrayList<>(), "prepared");
        RecordingComposition composition = new RecordingComposition(
                () -> resourceExecution(resource));
        AdvancedEngine engine = new AdvancedEngine(composition, composition::prepare);
        AdvancedPreparedExecution prepared = engine.prepare(compile(engine, leaf().neg()));
        PreparedExecution inward = prepared.execution();
        AtomicReference<Throwable> closeFailure = new AtomicReference<>();
        AtomicReference<Throwable> runFailure = new AtomicReference<>();

        Thread closeThread;
        Thread runThread;
        synchronized (inward) {
            closeThread = Thread.ofPlatform().start(() -> {
                try {
                    prepared.close();
                } catch (Throwable failure) {
                    closeFailure.set(failure);
                }
            });
            awaitBlocked(closeThread);

            runThread = Thread.ofPlatform().start(() -> {
                try {
                    engine.run(prepared, List.of());
                } catch (Throwable failure) {
                    runFailure.set(failure);
                }
            });
            awaitBlocked(runThread);
            assertEquals(0, resource.closeCount.get());
        }

        closeThread.join(10_000);
        runThread.join(10_000);
        assertFalse(closeThread.isAlive());
        assertFalse(runThread.isAlive());
        assertSame(null, closeFailure.get());
        IllegalStateException rejected =
                (IllegalStateException) runFailure.get();
        assertEquals("prepared execution is closed", rejected.getMessage());
        assertTrue(prepared.isClosed());
        assertTrue(inward.isClosed());
        assertEquals(1, resource.closeCount.get());
        engine.close();
        assertEquals(1, resource.closeCount.get());
    }

    /**
     * Replaces the publication collection at the existing test-local transaction seam.
     *
     * @param target non-null fixture owner with the named declared field
     * @param name exact declared field name
     * @param value test replacement, not a production mutation API
     * @throws AssertionError if reflective fixture setup fails
     */
    private static void replaceField(Object target, String name, Object value) {
        try {
            var field = target.getClass().getDeclaredField(name);
            field.setAccessible(true);
            field.set(target, value);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    /**
     * Rejects registration after execution has produced a result requiring rollback.
     *
     * @param failure exact fatal publication failure
     * @param <T> publication owner type
     * @return empty collection rejecting additions while permitting cleanup/removal
     */
    private static <T> ArrayList<T> rejectingPublications(Error failure) {
        return new ArrayList<>() {
            @Override public boolean add(T value) { throw failure; }
        };
    }

    /**
     * Models exhausted allocation at a retained-owner snapshot boundary while preserving the
     * existing publication and removal operations.
     *
     * @param failure exact error raised if shutdown attempts a collection snapshot
     * @param <T> retained owner type
     * @return empty mutable ownership ledger that cannot be snapshotted
     */
    private static <T> ArrayList<T> unsnapshotableOwners(Error failure) {
        return new ArrayList<>() {
            @Override public Object[] toArray() { throw failure; }
            @Override public <U> U[] toArray(U[] destination) { throw failure; }
        };
    }

    private static void awaitBlocked(Thread thread) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (thread.getState() != Thread.State.BLOCKED) {
            if (!thread.isAlive()) {
                throw new AssertionError("thread terminated before reaching blocked state");
            }
            if (System.nanoTime() >= deadline) {
                throw new AssertionError("timed out waiting for blocked thread state");
            }
            Thread.onSpinWait();
        }
    }

    private static AdvancedCompiledGraph compile(AdvancedEngine engine, Tensor output) {
        return engine.compile(CompileMode.FORWARD_ONLY, List.of(output), Optional.empty(),
                GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral());
    }

    private static Tensor leaf() {
        Shape shape = Shape.of(4);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
    }

    private static PreparedExecution emptyExecution() {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(), List.of());
        return new PreparedExecution(plan, new PreparedSchedule(plan, List.of()));
    }

    private static PreparedExecution resourceExecution(PreparedResource resource) {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(), List.of());
        return new PreparedExecution(plan, new PreparedSchedule(plan, List.of()), List.of(resource));
    }

    private static PreparedExecution resourceAndCreatedBufferExecution(
            PreparedResource resource, PreparedRepresentationPlan.BufferCreator supplier) {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 4, 4)), List.of());
        var creation = new PreparedRepresentationPlan(plan,
                List.of(List.of(new CreatedBuffer(supplier))), List.of());
        return new PreparedExecution(plan, new PreparedSchedule(plan,
                List.of(new PreparedSchedule.RepresentationCreationStep(creation))),
                List.of(resource));
    }

    private static PreparedExecution createdBufferExecution(
            PreparedRepresentationPlan.BufferCreator supplier) {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 4, 4)), List.of());
        var creation = new PreparedRepresentationPlan(plan,
                List.of(List.of(new CreatedBuffer(supplier))), List.of());
        return new PreparedExecution(plan, new PreparedSchedule(plan,
                List.of(new PreparedSchedule.RepresentationCreationStep(creation))));
    }

    private static PreparedExecution blockingExecution(CountDownLatch entered,
            CountDownLatch release, BufferRepresentation owned) {
        return blockingExecution(entered, release, owned, null);
    }

    private static PreparedExecution blockingExecution(CountDownLatch entered,
            CountDownLatch release, BufferRepresentation owned, PreparedResource resource) {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 4, 4)), List.of());
        var creation = new PreparedRepresentationPlan(plan,
                List.of(List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> owned))), List.of());
        PreparedExecutable executable = new BlockingExecutable(plan, entered, release);
        return new PreparedExecution(plan, new PreparedSchedule(plan, List.of(
                new PreparedSchedule.RepresentationCreationStep(creation),
                new PreparedSchedule.ExecutionStep(executable))),
                resource == null ? List.of() : List.of(resource));
    }

    private static final class RecordingComposition implements EngineBackendComposition {
        private final Supplier<PreparedExecution> execution;
        private final AtomicInteger supportCount = new AtomicInteger();
        private final AtomicInteger prepareCount = new AtomicInteger();
        private final AtomicInteger closeCount = new AtomicInteger();
        private RuntimeException capabilityFailure;
        private Throwable closeFailure;
        private List<String> closeOrder;

        private RecordingComposition(PreparedExecution execution) { this(() -> execution); }
        private RecordingComposition(Supplier<PreparedExecution> execution) {
            this.execution = execution;
        }

        @Override public List<BackendCapabilityProvider> capabilityProviders() {
            return List.of(new BackendCapabilityProvider() {
                @Override public BackendId backendId() { return BACKEND; }
                @Override public boolean supports(OperationCapabilityQuery query) {
                    supportCount.incrementAndGet();
                    if (capabilityFailure != null) throw capabilityFailure;
                    return true;
                }
            });
        }
        @Override public List<BackendAvailabilitySnapshot> availabilitySnapshots() {
            return List.of(new BackendAvailabilitySnapshot(BACKEND,
                    Map.of(new BackendDeviceId(BACKEND, "device"), DeviceClass.CPU)));
        }
        @Override public io.github.pho001.synaptik.prepare.PartitionPreparation<?, ?>
                partitionPreparation() {
            throw new AssertionError("unexpected partition preparation");
        }
        @Override public io.github.pho001.synaptik.prepare.PreparedScheduleContributor
                scheduleContributor() {
            throw new AssertionError("unexpected schedule contribution");
        }
        @Override public io.github.pho001.synaptik.prepare.PreparedScheduleAssembler
                scheduleAssembler() {
            throw new AssertionError("unexpected schedule assembly");
        }
        @Override public boolean supportsTransferTo(
                EngineBackendComposition destination, TensorDescriptor descriptor) {
            return false;
        }
        @Override public io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer
                prepareTransferTo(
                        EngineBackendComposition destination,
                        PreparedMemoryPlan memoryPlan,
                        int bufferIndex,
                        int sourceRepresentationIndex,
                        int destinationRepresentationIndex,
                        TensorDescriptor descriptor) {
            throw new AssertionError("unexpected transfer preparation");
        }
        public PreparedExecution prepare(CompileArtifacts artifacts) {
            prepareCount.incrementAndGet();
            return execution.get();
        }
        @Override public BufferRepresentation borrow(HostTensorStorage storage) {
            throw new AssertionError("unexpected borrow");
        }
        @Override public byte[] copyToCanonicalHostBytes(
                BufferRepresentation representation,
                TensorDescriptor descriptor,
                long maximumBytes) {
            throw new AssertionError("unexpected copy");
        }
        @Override public void close() {
            closeCount.incrementAndGet();
            if (closeOrder != null) closeOrder.add("backend");
            if (closeFailure instanceof RuntimeException runtime) throw runtime;
            if (closeFailure instanceof Error error) throw error;
        }
    }

    private static final class TestPreparedResource implements PreparedResource {
        private final Throwable failure;
        private final List<String> order;
        private final String name;
        private final AtomicInteger closeCount = new AtomicInteger();

        private TestPreparedResource(Throwable failure, List<String> order, String name) {
            this.failure = failure;
            this.order = order;
            this.name = name;
        }

        @Override public void close() {
            closeCount.incrementAndGet();
            order.add(name);
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
        }
    }

    private static final class TestBuffer implements BufferRepresentation {
        private final Throwable failure;
        private final List<String> order;
        private final String name;
        private final AtomicInteger closeCount = new AtomicInteger();

        private TestBuffer(Throwable failure, List<String> order, String name) {
            this.failure = failure;
            this.order = order;
            this.name = name;
        }
        @Override public void close() {
            closeCount.incrementAndGet();
            order.add(name);
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
        }
    }

    private static final class BlockingExecutable extends PreparedExecutable {
        private final CountDownLatch entered;
        private final CountDownLatch release;

        private BlockingExecutable(PreparedMemoryPlan plan, CountDownLatch entered,
                CountDownLatch release) {
            super(plan, List.of(), List.of(), List.of());
            this.entered = entered;
            this.release = release;
        }
        @Override protected boolean acceptsBufferRepresentation(int index,
                BufferRepresentation representation) { return false; }
        @Override protected boolean acceptsWorkspaceRepresentation(int index,
                WorkspaceRepresentation representation) { return false; }
        @Override protected BoundInvocation bindCompatible(RunState state,
                BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
            return new BoundInvocation(state) {
                @Override protected void executeBound() {
                    entered.countDown();
                    try {
                        assertTrue(release.await(10, TimeUnit.SECONDS));
                    } catch (InterruptedException failure) {
                        Thread.currentThread().interrupt();
                        throw new AssertionError(failure);
                    }
                }
            };
        }
    }
}
