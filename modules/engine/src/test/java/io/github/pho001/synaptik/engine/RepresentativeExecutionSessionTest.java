package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import io.github.pho001.synaptik.tools.tuning.WorkloadTuningRequest;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.InitializedBuffer;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercises representative binding, isolated execution, cleanup, and fallback state. */
final class RepresentativeExecutionSessionTest {
    private static final BackendId BACKEND = new BackendId("representative-test");

    @Test
    void admissionAndOwnershipPrecedeRepresentativeInspection() {
        RecordingComposition firstComposition = new RecordingComposition();
        RecordingComposition secondComposition = new RecordingComposition();
        Engine first = engine(firstComposition);
        Engine second = engine(secondComposition);
        Tensor input = leaf(1, 2);
        CompiledGraph compiled = first.compile(List.of(input.contiguous()));

        assertThrows(IllegalArgumentException.class,
                () -> secondComposition.lifecycleOwner.openRepresentativeExecutionSession(
                        second, compiled, null));
        assertEquals(0, secondComposition.borrowCount.get());

        second.close();
        IllegalStateException closed = assertThrows(IllegalStateException.class,
                () -> secondComposition.lifecycleOwner.openRepresentativeExecutionSession(
                        null, null, null));
        assertEquals("advanced engine is closed", closed.getMessage());
        first.close();
    }

    @Test
    void engineCloseWaitsForAdmittedExecutionAndCleanup() throws Exception {
        RecordingComposition composition = new RecordingComposition();
        Engine engine = engine(composition);
        Tensor input = leaf(1, 2);
        CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
        RepresentativeExecutionSession session =
                openSession(composition, compiled, List.of(input));
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        var execution = execution(1, 1, () -> {
            entered.countDown();
            await(release);
            return new TestBuffer("result", null, composition.closeOrder);
        });

        try (var executor = Executors.newFixedThreadPool(2)) {
            var run = executor.submit(() -> {
                try (session) {
                    session.execute(execution);
                }
                return null;
            });
            assertTrue(entered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> {
                engine.close();
                return null;
            });
            while (!engine.isClosed()) Thread.onSpinWait();
            assertFalse(close.isDone());
            release.countDown();
            run.get(10, TimeUnit.SECONDS);
            close.get(10, TimeUnit.SECONDS);
        }
        assertEquals(List.of("result", "borrow-0"), composition.closeOrder);
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void closeRaceRejectsFinalFallbackRecipeAfterAdmittedPreparation() throws Exception {
        RecordingComposition composition = new RecordingComposition();
        Engine engine = engine(composition);
        Tensor input = leaf(1, 2);
        CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
        RepresentativeExecutionSession session =
                openSession(composition, compiled, List.of(input));
        RuntimeException tuning = new RuntimeException("tuning");
        composition.prepareEntered = new CountDownLatch(1);
        composition.prepareRelease = new CountDownLatch(1);

        try (var executor = Executors.newFixedThreadPool(2)) {
            var fallback = executor.submit(() ->
                    AdvancedEngine.prepareAfterRecoverableTuningFailure(
                            session, tuning, true));
            assertTrue(composition.prepareEntered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> {
                engine.close();
                return null;
            });
            while (!engine.isClosed()) Thread.onSpinWait();
            assertFalse(close.isDone());
            composition.prepareRelease.countDown();
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> fallback.get(10, TimeUnit.SECONDS));
            assertEquals("advanced engine is closed", failure.getCause().getMessage());
            assertArrayEquals(new Throwable[] {tuning}, failure.getCause().getSuppressed());
            close.get(10, TimeUnit.SECONDS);
        }
        assertEquals(1, composition.prepareCount.get());
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void snapshotsAllStorageThenBorrowsOnceInCompiledOrder() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(1, 2);
            Tensor second = leaf(3, 4);
            HostTensorStorage firstStorage = first.hostStorage().orElseThrow();
            HostTensorStorage secondStorage = second.hostStorage().orElseThrow();
            Tensor replacement = leaf(5, 6);
            CompiledGraph compiled = engine.compile(
                    List.of(first.contiguous(), second.contiguous()));
            composition.afterFirstBorrow = () -> second.replaceHostStorage(
                    replacement.hostStorage().orElseThrow());

            try (var session = openSession(
                    composition, compiled, List.of(second, first))) {
                assertEquals(List.of(firstStorage, secondStorage), composition.borrowedStorages);
                assertEquals(2, composition.borrowCount.get());
            }

            assertEquals(List.of("borrow-1", "borrow-0"), composition.closeOrder);
            assertTrue(firstStorage.isAlive());
            assertTrue(secondStorage.isAlive());
        }
    }

    @Test
    void validatesCompleteBindingBeforeFirstBorrow() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(1, 2);
            Tensor second = leaf(3, 4);
            CompiledGraph compiled = engine.compile(
                    List.of(first.contiguous(), second.contiguous()));

            assertThrows(IllegalArgumentException.class,
                    () -> openSession(composition, compiled, List.of(first)));
            assertEquals(0, composition.borrowCount.get());

            Tensor wrong = TensorFactory.fromFlatArray(new TensorDescriptor(
                    DataType.FLOAT64, Shape.of(2),
                    Optional.of(LayoutDescriptor.contiguous(Shape.of(2))), false),
                    Optional.empty(), new double[] {1, 2});
            assertThrows(IllegalArgumentException.class,
                    () -> openSession(composition, compiled, List.of(first, wrong)));
            assertEquals(0, composition.borrowCount.get());
        }
    }

    @Test
    void validatesEveryStorageBeforeFirstBorrow() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(1, 2);
            Tensor second = leaf(3, 4);
            CompiledGraph compiled = engine.compile(
                    List.of(first.contiguous(), second.contiguous()));
            second.clearHostStorage();

            assertThrows(IllegalStateException.class,
                    () -> openSession(composition, compiled, List.of(first, second)));
            assertEquals(0, composition.borrowCount.get());
        }
    }

    @Test
    void partialBorrowFailureClosesAcquiredWrappersInReverse() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException expected = new RuntimeException("borrow");
        RuntimeException cleanup = new RuntimeException("borrow cleanup");
        composition.borrowFailureIndex = 1;
        composition.borrowFailure = expected;
        composition.borrowCloseFailures.add(cleanup);
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(1, 2);
            Tensor second = leaf(3, 4);
            CompiledGraph compiled = engine.compile(
                    List.of(first.contiguous(), second.contiguous()));

            RuntimeException observed = assertThrows(RuntimeException.class,
                    () -> openSession(composition, compiled, List.of(first, second)));
            assertSame(expected, observed);
            assertArrayEquals(new Throwable[] {cleanup}, observed.getSuppressed());
            assertEquals(List.of("borrow-0"), composition.closeOrder);
        }
    }

    @Test
    void repeatedActionsUseFreshRunStateCompletePublicationsAndCloseResults() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var created = new ArrayList<TestBuffer>();
            PreparedRepresentationPlan.BufferCreator creator = () -> {
                TestBuffer buffer = new TestBuffer("result-" + created.size(), null,
                        composition.closeOrder);
                created.add(buffer);
                return buffer;
            };
            var firstExecution = execution(1, 1, creator);
            var secondExecution = execution(1, 1, creator);
            assertNotSame(firstExecution, secondExecution);

            try (var session = openSession(composition, compiled, List.of(input))) {
                session.execute(firstExecution);
                session.execute(secondExecution);
            }

            assertEquals(2, created.size());
            assertNotSame(created.get(0), created.get(1));
            assertEquals(1, created.get(0).closeCount.get());
            assertEquals(1, created.get(1).closeCount.get());
            assertEquals(List.of("result-0", "result-1", "borrow-0"),
                    composition.closeOrder);
        }
    }

    @Test
    void capturesDetachedReferenceAndMatchesExactCanonicalBytesAfterResultCleanup() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            byte[] source = {0, 0, 0, 0, 0x3f, (byte) 0x80, 0, 0};
            TestBuffer[] capturedBuffer = new TestBuffer[1];

            try (var session = openSession(composition, compiled, List.of(input))) {
                var reference = session.captureCorrectnessReference(
                        execution(1, 1, () -> capturedBuffer[0] = new TestBuffer(
                                "reference", null, composition.closeOrder, source)),
                        source.length);
                assertEquals(1, capturedBuffer[0].closeCount.get());
                source[0] = 1;

                assertEquals(RepresentativePlanCorrectness.Comparison.MATCH,
                        session.compareCorrectness(reference,
                                executionWithBytes(1, composition.closeOrder,
                                        new byte[] {0, 0, 0, 0, 0x3f, (byte) 0x80, 0, 0})));
            }

            assertEquals(List.of("reference", "result-0", "borrow-0"),
                    composition.closeOrder);
            assertEquals(2, composition.copyCount.get());
        }
    }

    @Test
    void comparesEveryOccurrenceInOrderIncludingAliasesAndEmptyPayloads() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(1, 2).contiguous();
            Tensor second = leaf(3, 4).contiguous();
            CompiledGraph compiled = engine.compile(List.of(first, second));
            Tensor firstInput = first.provenance().orElseThrow().inputs().getFirst();
            Tensor secondInput = second.provenance().orElseThrow().inputs().getFirst();
            byte[] payload = {1, 2, 3, 4, 5, 6, 7, 8};

            try (var session = openSession(
                    composition, compiled, List.of(firstInput, secondInput))) {
                var reference = session.captureCorrectnessReference(
                        aliasedExecution(2, 2, composition.closeOrder, payload), 16);
                assertSame(composition.copiedRepresentations.get(0),
                        composition.copiedRepresentations.get(1));
                assertEquals(RepresentativePlanCorrectness.Comparison.MATCH,
                        session.compareCorrectness(reference,
                                executionWithBytes(2, composition.closeOrder,
                                        payload, payload)));
                assertEquals(RepresentativePlanCorrectness.Comparison.MISMATCH,
                        session.compareCorrectness(reference,
                                executionWithBytes(2, composition.closeOrder,
                                        payload, new byte[] {1, 2, 3, 4, 5, 6, 7, 9})));
                assertTrue(session.canResolveRecoverableTuningFailure());
            }
        }

        RecordingComposition emptyComposition = new RecordingComposition();
        try (Engine engine = engine(emptyComposition)) {
            Tensor empty = emptyLeaf();
            CompiledGraph compiled = engine.compile(List.of(empty.contiguous()));
            try (var session = openSession(emptyComposition, compiled, List.of(empty))) {
                var reference = session.captureCorrectnessReference(
                        executionWithBytes(1, emptyComposition.closeOrder, new byte[0]), 0);
                assertEquals(RepresentativePlanCorrectness.Comparison.MATCH,
                        session.compareCorrectness(reference,
                                executionWithBytes(1, emptyComposition.closeOrder, new byte[0])));
            }
        }
    }

    @Test
    void exactComparisonDistinguishesSignedZeroAndNanPayloadBits() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            byte[] positiveZeroAndNan = {0, 0, 0, 0, 0x7f, (byte) 0xc0, 0, 1};
            byte[] negativeZeroAndNan = {(byte) 0x80, 0, 0, 0, 0x7f, (byte) 0xc0, 0, 1};
            byte[] otherNanPayload = {0, 0, 0, 0, 0x7f, (byte) 0xc0, 0, 2};

            try (var session = openSession(composition, compiled, List.of(input))) {
                var reference = session.captureCorrectnessReference(
                        executionWithBytes(1, composition.closeOrder, positiveZeroAndNan), 8);
                assertEquals(RepresentativePlanCorrectness.Comparison.MISMATCH,
                        session.compareCorrectness(reference,
                                executionWithBytes(1, composition.closeOrder,
                                        negativeZeroAndNan)));
                assertEquals(RepresentativePlanCorrectness.Comparison.MISMATCH,
                        session.compareCorrectness(reference,
                                executionWithBytes(1, composition.closeOrder,
                                        otherNanPayload)));
                assertEquals(RepresentativePlanCorrectness.Comparison.MATCH,
                        session.compareCorrectness(reference,
                                executionWithBytes(1, composition.closeOrder,
                                        positiveZeroAndNan)));
            }
        }
    }

    @Test
    void opaqueModelReportsLengthMismatchWithoutExposingPayload() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            try (var session = openSession(composition, compiled, List.of(input))) {
                var reference = RepresentativePlanCorrectness.capture(
                        session, new byte[][] {new byte[] {1}});
                assertEquals(RepresentativePlanCorrectness.Comparison.MISMATCH,
                        RepresentativePlanCorrectness.compare(
                                reference, session, new byte[][] {new byte[] {1, 0}}));
            }
        }
    }

    @Test
    void correctnessPreflightCompletesBeforeExecutionAndAllowsRetry() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            AtomicInteger creations = new AtomicInteger();
            var execution = execution(1, 1, () -> {
                creations.incrementAndGet();
                return new TestBuffer("result", null, composition.closeOrder, new byte[8]);
            });

            try (var session = openSession(composition, compiled, List.of(input))) {
                assertThrows(IllegalArgumentException.class,
                        () -> session.captureCorrectnessReference(execution, -1));
                assertThrows(IllegalArgumentException.class,
                        () -> session.captureCorrectnessReference(execution, 7));
                assertEquals(0, creations.get());
                assertTrue(session.canResolveRecoverableTuningFailure());
                session.captureCorrectnessReference(execution, 8);
                assertEquals(1, creations.get());
            }
        }
    }

    @Test
    void sharedPreflightRejectsEveryDescriptorAndAggregateBoundary() {
        Shape dynamicShape = Shape.ofDimensions(new DynamicDimension("N"));
        TensorDescriptor dynamic = new TensorDescriptor(
                DataType.FLOAT32, dynamicShape, Optional.empty(), false);
        Shape staticShape = Shape.of(2);
        TensorDescriptor unresolved = new TensorDescriptor(
                DataType.FLOAT32, staticShape, Optional.empty(), false);
        Shape oversizedShape = Shape.of(Integer.MAX_VALUE);
        TensorDescriptor oversized = new TensorDescriptor(
                DataType.FLOAT32, oversizedShape,
                Optional.of(LayoutDescriptor.contiguous(oversizedShape)), false);
        TensorDescriptor ordinary = new TensorDescriptor(
                DataType.FLOAT32, staticShape,
                Optional.of(LayoutDescriptor.contiguous(staticShape)), false);

        assertThrows(IllegalArgumentException.class,
                () -> AdvancedEngine.preflightCanonicalDescriptorByteCounts(List.of(), 0));
        assertThrows(IllegalArgumentException.class,
                () -> AdvancedEngine.preflightCanonicalDescriptorByteCounts(
                        List.of(ordinary), -1));
        assertThrows(IllegalArgumentException.class,
                () -> AdvancedEngine.preflightCanonicalDescriptorByteCounts(
                        List.of(dynamic), Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class,
                () -> AdvancedEngine.preflightCanonicalDescriptorByteCounts(
                        List.of(unresolved), Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class,
                () -> AdvancedEngine.preflightCanonicalDescriptorByteCounts(
                        List.of(oversized), Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class,
                () -> AdvancedEngine.preflightCanonicalDescriptorByteCounts(
                        List.of(ordinary, ordinary), 15));
        assertArrayEquals(new long[] {8, 8},
                AdvancedEngine.preflightCanonicalDescriptorByteCounts(
                        List.of(ordinary, ordinary), 16));
    }

    @Test
    void rejectsRecaptureForeignReferenceComparisonBeforeCaptureAndUseAfterClose() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var first = openSession(composition, compiled, List.of(input));
            var second = openSession(composition, compiled, List.of(input));
            var firstReference = first.captureCorrectnessReference(
                    executionWithBytes(1, composition.closeOrder, new byte[8]), 8);

            assertThrows(IllegalStateException.class,
                    () -> first.captureCorrectnessReference(
                            executionWithBytes(1, composition.closeOrder, new byte[8]), 8));
            assertThrows(IllegalStateException.class,
                    () -> second.compareCorrectness(firstReference,
                            executionWithBytes(1, composition.closeOrder, new byte[8])));
            var secondReference = second.captureCorrectnessReference(
                    executionWithBytes(1, composition.closeOrder, new byte[8]), 8);
            assertThrows(IllegalArgumentException.class,
                    () -> first.compareCorrectness(secondReference,
                            executionWithBytes(1, composition.closeOrder, new byte[8])));
            first.close();
            assertThrows(IllegalStateException.class,
                    () -> first.compareCorrectness(firstReference,
                            executionWithBytes(1, composition.closeOrder, new byte[8])));
            second.close();
        }
    }

    @Test
    void engineCloseWaitsForCorrectnessCopyAndSessionCleanup() throws Exception {
        RecordingComposition composition = new RecordingComposition();
        Engine engine = engine(composition);
        Tensor input = leaf(1, 2);
        CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
        RepresentativeExecutionSession session =
                openSession(composition, compiled, List.of(input));
        composition.copyEntered = new CountDownLatch(1);
        composition.copyRelease = new CountDownLatch(1);

        try (var executor = Executors.newFixedThreadPool(2)) {
            var capture = executor.submit(() -> {
                try (session) {
                    return session.captureCorrectnessReference(
                            executionWithBytes(1, composition.closeOrder, new byte[8]), 8);
                }
            });
            assertTrue(composition.copyEntered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> {
                engine.close();
                return null;
            });
            while (!engine.isClosed()) Thread.onSpinWait();
            assertFalse(close.isDone());
            composition.copyRelease.countDown();
            assertTrue(capture.get(10, TimeUnit.SECONDS) != null);
            close.get(10, TimeUnit.SECONDS);
        } finally {
            engine.close();
        }
        assertEquals(List.of("result-0", "borrow-0"), composition.closeOrder);
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void copyFailureKeepsPrimarySuppressesResultCleanupAndPoisonsOnce() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException copyFailure = new RuntimeException("copy");
        RuntimeException resultCleanup = new RuntimeException("result cleanup");
        Error wrapperCleanup = new AssertionError("wrapper cleanup");
        composition.copyFailure = copyFailure;
        composition.borrowCloseFailures.add(wrapperCleanup);
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));
            var execution = execution(1, 1, () -> new TestBuffer(
                    "result", resultCleanup, composition.closeOrder, new byte[8]));

            RuntimeException observed = assertThrows(RuntimeException.class,
                    () -> session.captureCorrectnessReference(execution, 8));
            assertSame(copyFailure, observed);
            assertArrayEquals(new Throwable[] {resultCleanup, wrapperCleanup},
                    observed.getSuppressed());
            assertFalse(session.canResolveRecoverableTuningFailure());
            assertEquals(List.of("result", "borrow-0"), composition.closeOrder);
            assertThrows(IllegalStateException.class,
                    () -> session.captureCorrectnessReference(execution, 8));
            RuntimeException closeFailure = assertThrows(RuntimeException.class, session::close);
            assertSame(resultCleanup, closeFailure);
            assertArrayEquals(new Throwable[] {wrapperCleanup},
                    closeFailure.getSuppressed());
            assertEquals(List.of("result", "borrow-0"), composition.closeOrder);
        }
    }

    @Test
    void correctnessModelAddsNoPublicOrProtectedSurface() {
        assertFalse(Modifier.isPublic(RepresentativePlanCorrectness.class.getModifiers()));
        assertTrue(Modifier.isFinal(RepresentativePlanCorrectness.class.getModifiers()));
        assertFalse(Modifier.isPublic(
                RepresentativePlanCorrectness.Reference.class.getModifiers()));
        assertEquals(List.of(RepresentativePlanCorrectness.Comparison.MATCH,
                        RepresentativePlanCorrectness.Comparison.MISMATCH),
                List.of(RepresentativePlanCorrectness.Comparison.values()));
        for (var method : RepresentativeExecutionSession.class.getDeclaredMethods()) {
            if (method.getName().equals("captureCorrectnessReference")
                    || method.getName().equals("compareCorrectness")) {
                assertFalse(Modifier.isPublic(method.getModifiers()));
                assertFalse(Modifier.isProtected(method.getModifiers()));
            }
        }
    }

    @Test
    void countMismatchPoisonsAndCleansSession() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));
            assertTrue(session.canResolveRecoverableTuningFailure());

            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> session.execute(execution(1, 0, null)));
            assertTrue(session.isRepresentativeExecutionFailure(failure));
            assertFalse(session.canResolveRecoverableTuningFailure());
            assertTrue(failure.getMessage().contains("expected=1, actual=0"));
            assertEquals(List.of("borrow-0"), composition.closeOrder);
            assertThrows(IllegalStateException.class,
                    () -> session.execute(execution(1, 0, null)));
            session.close();
            assertEquals(List.of("borrow-0"), composition.closeOrder);
        }
    }

    @Test
    void trialFailureKeepsPrimaryAndDeterministicCleanupSuppression() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException firstWrapper = new RuntimeException("wrapper one");
        Error secondWrapper = new AssertionError("wrapper two");
        composition.borrowCloseFailures.add(firstWrapper);
        composition.borrowCloseFailures.add(secondWrapper);
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(1, 2);
            Tensor second = leaf(3, 4);
            CompiledGraph compiled = engine.compile(
                    List.of(first.contiguous(), second.contiguous()));
            var session = openSession(composition, compiled, List.of(first, second));

            IllegalStateException observed = assertThrows(IllegalStateException.class,
                    () -> session.execute(execution(2, 0, null)));
            assertArrayEquals(new Throwable[] {secondWrapper, firstWrapper},
                    observed.getSuppressed());
            assertEquals(List.of("borrow-1", "borrow-0"), composition.closeOrder);
            Error closeFailure = assertThrows(Error.class, session::close);
            assertSame(secondWrapper, closeFailure);
            assertArrayEquals(new Throwable[] {firstWrapper}, closeFailure.getSuppressed());
            assertSame(closeFailure, assertThrows(Error.class, session::close));
        }
    }

    @Test
    void resultCleanupFailureIsPrimaryAndPoisonCleanupIsNotDuplicated() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException resultCleanup = new RuntimeException("result cleanup");
        Error wrapperCleanup = new AssertionError("wrapper cleanup");
        composition.borrowCloseFailures.add(wrapperCleanup);
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));
            var execution = execution(1, 1,
                    () -> new TestBuffer("result", resultCleanup, composition.closeOrder));

            RuntimeException observed = assertThrows(RuntimeException.class,
                    () -> session.execute(execution));
            assertSame(resultCleanup, observed);
            assertArrayEquals(new Throwable[] {wrapperCleanup}, observed.getSuppressed());
            assertEquals(List.of("result", "borrow-0"), composition.closeOrder);
            assertSame(observed, assertThrows(RuntimeException.class, session::close));
            assertArrayEquals(new Throwable[] {wrapperCleanup}, observed.getSuppressed());
        }
    }

    @Test
    void selectedPreparationCannotBeginUntilCleanupSucceeds() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException cleanup = new RuntimeException("cleanup");
        composition.borrowCloseFailures.add(cleanup);
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));

            RuntimeException observed = assertThrows(RuntimeException.class,
                    () -> AdvancedEngine.prepareSelectedAfterRepresentativeCleanup(
                            session, null, null, null));
            assertSame(cleanup, observed);
            assertEquals(List.of("borrow-0"), composition.closeOrder);
        }


        RecordingComposition successfulComposition = new RecordingComposition();
        try (Engine engine = engine(successfulComposition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(successfulComposition, compiled, List.of(input));

            assertThrows(NullPointerException.class,
                    () -> AdvancedEngine.prepareSelectedAfterRepresentativeCleanup(
                            session, null, null, null));
            assertEquals(List.of("borrow-0"), successfulComposition.closeOrder);
        }
    }

    @Test
    void requiredFailureNeverFallsBackAndAllowedFailurePreparesOnce() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            RuntimeException required = new RuntimeException("required");
            var strictSession = openSession(composition, compiled, List.of(input));

            RuntimeException strictObserved = assertThrows(RuntimeException.class,
                    () -> AdvancedEngine.prepareAfterRecoverableTuningFailure(
                            strictSession, required, false));
            assertSame(required, strictObserved);
            assertEquals(0, composition.prepareCount.get());

            RuntimeException recoverable = new RuntimeException("recoverable");
            var fallbackSession = openSession(composition, compiled, List.of(input));
            var prepared = AdvancedEngine.prepareAfterRecoverableTuningFailure(
                    fallbackSession, recoverable, true);
            assertSame(composition.preparedExecution, prepared);
            assertEquals(1, composition.prepareCount.get());
        }
    }

    @Test
    void failedFallbackPreparationIsPrimaryWithTuningFailureSuppressed() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException tuning = new RuntimeException("tuning");
        Error preparation = new AssertionError("preparation");
        composition.prepareFailure = preparation;
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));

            Error observed = assertThrows(Error.class,
                    () -> AdvancedEngine.prepareAfterRecoverableTuningFailure(
                            session, tuning, true));
            assertSame(preparation, observed);
            assertArrayEquals(new Throwable[] {tuning}, observed.getSuppressed());
            assertEquals(1, composition.prepareCount.get());
        }
    }

    @Test
    void cleanupFailureForbidsFallbackAndRemainsSuppressedOnTuningFailure() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException tuning = new RuntimeException("tuning");
        Error cleanup = new AssertionError("cleanup");
        composition.borrowCloseFailures.add(cleanup);
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));

            RuntimeException observed = assertThrows(RuntimeException.class,
                    () -> AdvancedEngine.prepareAfterRecoverableTuningFailure(
                            session, tuning, true));
            assertSame(tuning, observed);
            assertArrayEquals(new Throwable[] {cleanup}, observed.getSuppressed());
            assertEquals(0, composition.prepareCount.get());
        }
    }

    @Test
    void poisonedSessionForbidsFallbackEvenAfterSuccessfulCleanup() {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException tuning = new RuntimeException("tuning");
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));
            assertThrows(IllegalStateException.class,
                    () -> session.execute(execution(1, 0, null)));

            RuntimeException observed = assertThrows(RuntimeException.class,
                    () -> AdvancedEngine.prepareAfterRecoverableTuningFailure(
                            session, tuning, true));
            assertSame(tuning, observed);
            assertEquals(1, observed.getSuppressed().length);
            assertTrue(observed.getSuppressed()[0].getMessage().contains("poisoned"));
            assertEquals(0, composition.prepareCount.get());
        }
    }

    @Test
    void syntheticMissPublishesCacheMapsEvidenceAndSecondCallHitsWithoutTrials(
            @TempDir Path directory) {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            Path cache = directory.resolve("workloads.bin");
            SyntheticTuning tuning = new SyntheticTuning();
            var request = tuningRequest(input, cache,
                    ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT, 1, 3);

            var first = composition.lifecycleOwner.prepareTunedOrdinary(
                    engine, compiled, request, tuning);

            assertEquals(ModelAutotuningPreparation.Outcome.TUNED, first.outcome());
            assertSame(compiled, first.preparedExecution().compiledGraph());
            assertEquals(8, tuning.trialPrepareCount.get());
            assertEquals(8, tuning.trialRunCount.get());
            assertEquals(8, tuning.trialBuffers.size());
            var distinctTrialBuffers = java.util.Collections.newSetFromMap(
                    new java.util.IdentityHashMap<TestBuffer, Boolean>());
            distinctTrialBuffers.addAll(tuning.trialBuffers);
            assertEquals(8, distinctTrialBuffers.size());
            assertEquals(4, tuning.candidateRuns[0].get());
            assertEquals(4, tuning.candidateRuns[1].get());
            assertEquals(1, tuning.selectedPrepareCount.get());
            assertNotSame(tuning.lastTrial, first.preparedExecution().execution());
            var evidence = first.evidence().orElseThrow();
            assertSame(request.modelIdentity(), evidence.modelIdentity());
            assertSame(request.config().representativeProfile(), evidence.representativeProfile());
            assertSame(request.config().budget(), evidence.budget());
            var workload = evidence.workloads().getFirst();
            assertEquals(ModelAutotuningPreparation.Source.MEASURED, workload.source());
            assertEquals(List.of(new ModelAutotuningPreparation.CandidateIdentity(new byte[] {1}),
                            new ModelAutotuningPreparation.CandidateIdentity(new byte[] {2})),
                    workload.candidates().stream().map(
                            ModelAutotuningPreparation.CandidateEvidence::identity).toList());
            assertEquals(0, workload.occurrences().getFirst().occurrenceIndex());
            assertEquals(0, workload.occurrences().getFirst().partitionIndex());
            assertEquals(1, workload.occurrences().getFirst().weight());
            assertArrayEquals(new byte[8],
                    workload.occurrences().getFirst().contextIdentity().bytes());
            var complete = evidence.completePlan();
            assertSame(request.config().completePlanBudget(), complete.budget());
            assertEquals(ModelAutotuningPreparation.ReuseScope.SESSION,
                    complete.compatibility().reuseScope());
            assertEquals(ModelAutotuningPreparation.Source.MEASURED, complete.source());
            assertEquals(2, complete.candidates().size());
            assertEquals(ModelAutotuningPreparation.CorrectnessAction.REFERENCE_CAPTURED,
                    complete.candidates().getFirst().correctnessAction());
            assertEquals(ModelAutotuningPreparation.CorrectnessAction.MATCH,
                    complete.candidates().get(1).correctnessAction());
            assertEquals(List.of(3, 3), complete.candidates().stream()
                    .map(candidate -> candidate.elapsedSamplesNanos().size()).toList());
            assertTrue(complete.candidates().stream().anyMatch(candidate ->
                    complete.winnerIdentity().equals(candidate.identity())));
            assertEquals(10, tuning.completeTrialPrepareCount.get());
            assertEquals(10, tuning.completeTrialRunCount.get());
            assertEquals(1, tuning.completeCompatibilityCount.get());
            assertArrayEquals(
                    new byte[] {(byte) tuning.completePhaseOneDecision.candidate()},
                    workload.winnerIdentity().bytes());
            assertFalse(java.nio.file.Files.exists(
                    request.config().modelPlanCache()));

            int trialsBeforeHit = tuning.trialPrepareCount.get();
            var second = composition.lifecycleOwner.prepareTunedOrdinary(
                    engine, compiled, request, tuning);
            assertEquals(trialsBeforeHit, tuning.trialPrepareCount.get());
            assertEquals(ModelAutotuningPreparation.Source.CACHE_HIT,
                    second.evidence().orElseThrow().workloads().getFirst().source());
            assertEquals(List.of(),
                    second.evidence().orElseThrow().workloads().getFirst().candidates());
            assertEquals(ModelAutotuningPreparation.Source.MEASURED,
                    second.evidence().orElseThrow().completePlan().source());
            assertEquals(2, tuning.selectedPrepareCount.get());
            assertEquals(2, tuning.completeCompatibilityCount.get());
            assertNotSame(first.preparedExecution().execution(),
                    second.preparedExecution().execution());
            assertTrue(java.nio.file.Files.isRegularFile(cache));
        }
    }

    @Test
    void syntheticCheckedFailuresNormalizeAndPrepareTrialFailureMayFallback(
            @TempDir Path directory) {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            SyntheticTuning tuning = new SyntheticTuning();
            IOException io = new IOException("io");
            tuning.trialPreparationFailure = io;
            var strict = tuningRequest(input, directory.resolve("strict.bin"),
                    ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT, 0, 1);
            UncheckedIOException observed = assertThrows(UncheckedIOException.class,
                    () -> composition.lifecycleOwner.prepareTunedOrdinary(
                            engine, compiled, strict, tuning));
            assertSame(io, observed.getCause());
            assertEquals(0, composition.prepareCount.get());

            Exception checked = new Exception("checked");
            tuning.trialPreparationFailure = checked;
            var checkedRequest = tuningRequest(input, directory.resolve("checked.bin"),
                    ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT, 0, 1);
            IllegalStateException normalized = assertThrows(IllegalStateException.class,
                    () -> composition.lifecycleOwner.prepareTunedOrdinary(
                            engine, compiled, checkedRequest, tuning));
            assertSame(checked, normalized.getCause());

            RuntimeException prepareFailure = new RuntimeException("prepare trial");
            tuning.trialPreparationFailure = prepareFailure;
            var allowed = tuningRequest(input, directory.resolve("allowed.bin"),
                    ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);
            var fallback = composition.lifecycleOwner.prepareTunedOrdinary(
                    engine, compiled, allowed, tuning);
            assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                    fallback.outcome());
            assertEquals(Optional.empty(), fallback.evidence());
            assertEquals(1, composition.prepareCount.get());
        }
    }

    @Test
    void syntheticExecutionFailuresKeepIdentityNeverFallbackAndCleanupIsSuppressed(
            @TempDir Path directory) {
        for (Throwable failure : List.of(
                new RuntimeException("runtime"), new AssertionError("error"))) {
            RecordingComposition composition = new RecordingComposition();
            RuntimeException cleanup = new RuntimeException("cleanup");
            composition.borrowCloseFailures.add(cleanup);
            try (Engine engine = engine(composition)) {
                Tensor input = leaf(1, 2);
                CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
                SyntheticTuning tuning = new SyntheticTuning();
                tuning.trialExecutionFailure = failure;
                var request = tuningRequest(input, directory.resolve(failure.getClass().getSimpleName()),
                        ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);

                Throwable observed = assertThrows(failure.getClass(), () ->
                        composition.lifecycleOwner.prepareTunedOrdinary(
                                engine, compiled, request, tuning));
                assertSame(failure, observed);
                assertArrayEquals(new Throwable[] {cleanup}, observed.getSuppressed());
                assertEquals(0, composition.prepareCount.get());
                assertEquals(0, tuning.selectedPrepareCount.get());
            }
        }
    }

    @Test
    void selectedPreparationFailureIsNotFallbackEligible(@TempDir Path directory) {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            SyntheticTuning tuning = new SyntheticTuning();
            RuntimeException selectedFailure = new RuntimeException("selected");
            tuning.selectedPreparationFailure = selectedFailure;
            var request = tuningRequest(input, directory.resolve("selected.bin"),
                    ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);

            RuntimeException observed = assertThrows(RuntimeException.class, () ->
                    composition.lifecycleOwner.prepareTunedOrdinary(
                            engine, compiled, request, tuning));
            assertSame(selectedFailure, observed);
            assertEquals(0, composition.prepareCount.get());
            assertEquals(1, tuning.selectedPrepareCount.get());
        }
    }

    @Test
    void completePlanMismatchNormalizesAndUsesOnlyAllowedOrdinaryFallback(
            @TempDir Path directory) {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            SyntheticTuning tuning = new SyntheticTuning();
            tuning.completeBytes[1][7] = 1;
            var request = tuningRequest(input, directory.resolve("mismatch.bin"),
                    ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);

            var fallback = composition.lifecycleOwner.prepareTunedOrdinary(
                    engine, compiled, request, tuning);

            assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                    fallback.outcome());
            assertEquals(1, composition.prepareCount.get());
            assertEquals(0, tuning.selectedPrepareCount.get());
            assertEquals(2, tuning.completeTrialPrepareCount.get());
            assertEquals(2, tuning.completeTrialRunCount.get());
        }
    }

    @Test
    void missingCompletePlanHandoffAndPreExecutionPreparationFailureAreRecoverable(
            @TempDir Path directory) {
        RecordingComposition absentComposition = new RecordingComposition();
        try (Engine engine = engine(absentComposition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            SyntheticTuning tuning = new SyntheticTuning();
            tuning.completePlanHandoffPresent = false;
            var request = tuningRequest(input, directory.resolve("absent-plan.bin"),
                    ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);

            assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                    absentComposition.lifecycleOwner.prepareTunedOrdinary(
                            engine, compiled, request, tuning).outcome());
            assertEquals(1, absentComposition.prepareCount.get());
            assertEquals(0, tuning.completeTrialPrepareCount.get());
        }

        RecordingComposition failureComposition = new RecordingComposition();
        try (Engine engine = engine(failureComposition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            SyntheticTuning tuning = new SyntheticTuning();
            RuntimeException preparation = new RuntimeException("complete prepare");
            tuning.completeTrialPreparationFailure = preparation;
            var request = tuningRequest(input, directory.resolve("prepare-plan.bin"),
                    ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT, 0, 1);

            RuntimeException observed = assertThrows(RuntimeException.class, () ->
                    failureComposition.lifecycleOwner.prepareTunedOrdinary(
                            engine, compiled, request, tuning));
            assertSame(preparation, observed);
            assertEquals(0, failureComposition.prepareCount.get());
        }
    }

    @Test
    void completePlanExecutionFailurePoisonsAndNeverFallsBack(@TempDir Path directory) {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException failure = new RuntimeException("complete execution");
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            SyntheticTuning tuning = new SyntheticTuning();
            tuning.completeTrialExecutionFailure = failure;
            var request = tuningRequest(input, directory.resolve("execute-plan.bin"),
                    ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);

            RuntimeException observed = assertThrows(RuntimeException.class, () ->
                    composition.lifecycleOwner.prepareTunedOrdinary(
                            engine, compiled, request, tuning));
            assertSame(failure, observed);
            assertEquals(0, composition.prepareCount.get());
            assertEquals(0, tuning.selectedPrepareCount.get());
        }
    }

    @Test
    void syntheticPersistentCompletePlanHitSkipsEveryPhaseTwoTrial(@TempDir Path directory) {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            SyntheticTuning tuning = new SyntheticTuning();
            tuning.completeReuseScope = io.github.pho001.synaptik.tools.tuning
                    .CompletePlanTuningRequest.ReuseScope.PERSISTENT;
            Path workloadCache = directory.resolve("persistent-workload.bin");
            var request = tuningRequest(input, workloadCache,
                    ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT, 0, 1);

            var first = composition.lifecycleOwner.prepareTunedOrdinary(
                    engine, compiled, request, tuning);
            int completeTrials = tuning.completeTrialPrepareCount.get();
            var second = composition.lifecycleOwner.prepareTunedOrdinary(
                    engine, compiled, request, tuning);

            assertEquals(10, completeTrials);
            assertEquals(completeTrials, tuning.completeTrialPrepareCount.get());
            assertEquals(ModelAutotuningPreparation.Source.MEASURED,
                    first.evidence().orElseThrow().completePlan().source());
            assertEquals(ModelAutotuningPreparation.Source.CACHE_HIT,
                    second.evidence().orElseThrow().completePlan().source());
            assertEquals(List.of(),
                    second.evidence().orElseThrow().completePlan().candidates());
            assertTrue(java.nio.file.Files.isRegularFile(request.config().modelPlanCache()));
            assertEquals(2, tuning.selectedPrepareCount.get());
        }
    }

    @Test
    void selectedFinalPublicationCloseRaceReturnsNoPartialResult(@TempDir Path directory)
            throws Exception {
        RecordingComposition composition = new RecordingComposition();
        Engine engine = engine(composition);
        Tensor input = leaf(1, 2);
        CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
        SyntheticTuning tuning = new SyntheticTuning();
        tuning.selectedEntered = new CountDownLatch(1);
        tuning.selectedRelease = new CountDownLatch(1);
        var request = tuningRequest(input, directory.resolve("race.bin"),
                ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);

        try (var executor = Executors.newFixedThreadPool(2)) {
            var preparation = executor.submit(() -> composition.lifecycleOwner.prepareTunedOrdinary(
                    engine, compiled, request, tuning));
            assertTrue(tuning.selectedEntered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> {
                engine.close();
                return null;
            });
            while (!engine.isClosed()) Thread.onSpinWait();
            assertFalse(close.isDone());
            tuning.selectedRelease.countDown();
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> preparation.get(10, TimeUnit.SECONDS));
            assertEquals("advanced engine is closed", failure.getCause().getMessage());
            assertEquals(0, composition.prepareCount.get());
            assertEquals(1, tuning.selectedPrepareCount.get());
            close.get(10, TimeUnit.SECONDS);
        } finally {
            engine.close();
        }
    }

    @Test
    void facadeOwnsAdmissionValidationRollbackAndPostSessionAdapterAccess(
            @TempDir Path directory) {
        RecordingComposition firstComposition = new RecordingComposition();
        SyntheticTuning firstTuning = new SyntheticTuning(firstComposition);
        Engine first = engine(firstComposition, firstTuning);
        RecordingComposition secondComposition = new RecordingComposition();
        SyntheticTuning secondTuning = new SyntheticTuning(secondComposition);
        Engine second = engine(secondComposition, secondTuning);
        try {
            Tensor firstInput = leaf(1, 2);
            CompiledGraph compiled = first.compile(List.of(firstInput.contiguous()));
            var request = new ModelAutotuningRequest(
                    tuningRequest(firstInput, directory.resolve("unused.bin"),
                            ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT, 0, 1)
                            .config(),
                    new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}),
                    List.of(firstInput));

            assertThrows(IllegalArgumentException.class,
                    () -> second.prepareTuned(compiled, null));
            assertEquals(0, secondTuning.handoffCount.get());
            assertEquals(0, secondComposition.borrowCount.get());

            assertThrows(NullPointerException.class,
                    () -> first.prepareTuned(compiled, null));
            assertEquals(0, firstTuning.handoffCount.get());
            assertSame(compiled, first.prepare(compiled).compiledGraph());

            var prepared = first.prepareTuned(compiled, request);
            assertEquals(ModelAutotuningPreparation.Outcome.TUNED, prepared.outcome());
            assertEquals(1, firstTuning.handoffCount.get());
            assertEquals(1, firstTuning.borrowCountAtFirstHandoff);

            second.close();
            IllegalStateException closed = assertThrows(IllegalStateException.class,
                    () -> second.prepareTuned(null, null));
            assertEquals("advanced engine is closed", closed.getMessage());
        } finally {
            first.close();
            second.close();
        }

        RecordingComposition failingComposition = new RecordingComposition();
        failingComposition.borrowFailureIndex = 1;
        failingComposition.borrowFailure = new RuntimeException("borrow");
        SyntheticTuning failingTuning = new SyntheticTuning(failingComposition);
        try (Engine engine = engine(failingComposition, failingTuning)) {
            Tensor left = leaf(1, 2);
            Tensor right = leaf(3, 4);
            CompiledGraph compiled = engine.compile(
                    List.of(left.contiguous(), right.contiguous()));
            var base = tuningRequest(left, directory.resolve("binding.bin"),
                    ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT, 0, 1);
            var request = new ModelAutotuningRequest(base.config(), base.modelIdentity(),
                    List.of(left, right));
            assertThrows(RuntimeException.class, () -> engine.prepareTuned(compiled, request));
            assertEquals(List.of("borrow-0"), failingComposition.closeOrder);
            assertEquals(0, failingTuning.handoffCount.get());
        }
        assertEquals(1, failingComposition.closeCount.get());
    }

    @Test
    void productionAdapterConstructionFailureCleansAdmittedSessionAndNeverFallsBack(
            @TempDir Path directory) {
        RecordingComposition composition = new RecordingComposition();
        RuntimeException cleanup = new RuntimeException("adapter cleanup");
        composition.borrowCloseFailures.add(cleanup);
        Engine engine = engine(composition);
        try {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var request = tuningRequest(input, directory.resolve("adapter.bin"),
                    ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC, 0, 1);

            ClassCastException observed = assertThrows(ClassCastException.class,
                    () -> engine.prepareTuned(compiled, request));

            assertArrayEquals(new Throwable[] {cleanup}, observed.getSuppressed());
            assertEquals(List.of("borrow-0"), composition.closeOrder);
            assertEquals(0, composition.prepareCount.get());
            org.junit.jupiter.api.Assertions.assertTimeout(
                    java.time.Duration.ofSeconds(10), engine::close);
        } finally {
            engine.close();
        }
        assertEquals(1, composition.closeCount.get());
    }

    @Test
    void authenticatedSelectionClosesFallbackBoundaryByObservation() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(1, 2);
            CompiledGraph compiled = engine.compile(List.of(input.contiguous()));
            var session = openSession(composition, compiled, List.of(input));
            assertTrue(AdvancedEngine.canResolveRecoverableTuningFailure(session, false));
            assertFalse(AdvancedEngine.canResolveRecoverableTuningFailure(session, true));
            session.close();
        }
    }

    private static ModelAutotuningRequest tuningRequest(
            Tensor input, Path cache, ModelAutotuningConfig.FallbackPolicy fallback,
            int warmups, int samples) {
        var config = new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(1, 2, warmups, samples),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(2, new byte[] {3, 4}),
                fallback,
                cache,
                new ModelAutotuningConfig.CompletePlanBudget(2, 1, 3, 10L, 8L),
                cache.resolveSibling(cache.getFileName() + ".model-plan"));
        return new ModelAutotuningRequest(config,
                new ModelAutotuningRequest.ModelIdentity(3, new byte[] {5, 6}), List.of(input));
    }

    private static Engine engine(RecordingComposition composition) {
        composition.lifecycleOwner = new AdvancedEngine(composition);
        composition.ordinaryOwner = new Engine(composition.lifecycleOwner);
        return composition.ordinaryOwner;
    }

    private static Engine engine(
            RecordingComposition composition,
            AdvancedEngine.ModelAutotuningTuning<?, ?, ?, ?, ?, ?> tuning) {
        composition.lifecycleOwner = new AdvancedEngine(composition, tuning);
        composition.ordinaryOwner = new Engine(composition.lifecycleOwner);
        return composition.ordinaryOwner;
    }

    private static RepresentativeExecutionSession openSession(
            RecordingComposition composition,
            CompiledGraph compiled,
            List<Tensor> inputs) {
        return composition.lifecycleOwner.openRepresentativeExecutionSession(
                composition.ordinaryOwner, compiled, inputs);
    }

    private static Tensor leaf(float first, float second) {
        Shape shape = Shape.of(2);
        return TensorFactory.fromFlatArray(new TensorDescriptor(
                DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false),
                Optional.empty(), new float[] {first, second});
    }

    private static Tensor emptyLeaf() {
        Shape shape = Shape.of(0);
        return TensorFactory.fromFlatArray(new TensorDescriptor(
                DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false),
                Optional.empty(), new float[0]);
    }

    private static io.github.pho001.synaptik.runtime.execution.PreparedExecution
            executionWithBytes(
                    int inputCount,
                    List<String> closeOrder,
                    byte[]... publications) {
        AtomicInteger index = new AtomicInteger();
        return execution(inputCount, publications.length, () -> {
            int current = index.getAndIncrement();
            return new TestBuffer("result-" + current, null, closeOrder,
                    publications[current]);
        });
    }

    private static io.github.pho001.synaptik.runtime.execution.PreparedExecution
            aliasedExecution(
                    int inputCount,
                    int publicationCount,
                    List<String> closeOrder,
                    byte[] payload) {
        int bufferCount = inputCount + 1;
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var preparations =
                new ArrayList<List<PreparedRepresentationPlan.BufferPreparation>>();
        for (int index = 0; index < bufferCount; index++) {
            entries.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 8, 4));
            preparations.add(index < inputCount
                    ? List.of(new CallerInput())
                    : List.of(new InitializedBuffer(() -> new TestBuffer(
                            "alias-result", null, closeOrder, payload))));
        }
        PreparedMemoryPlan plan = new PreparedMemoryPlan(entries, List.of());
        var creation = new PreparedRepresentationPlan(plan, preparations, List.of());
        var steps = new ArrayList<PreparedSchedule.Step>();
        steps.add(new PreparedSchedule.RepresentationCreationStep(creation));
        for (int index = 0; index < publicationCount; index++) {
            steps.add(new PreparedSchedule.PublicationStep(
                    new PreparedPublication(plan, inputCount, 0, index)));
        }
        return new io.github.pho001.synaptik.runtime.execution.PreparedExecution(
                plan, new PreparedSchedule(plan, steps));
    }

    private static io.github.pho001.synaptik.runtime.execution.PreparedExecution execution(
            int inputCount,
            int publicationCount,
            PreparedRepresentationPlan.BufferCreator outputCreator) {
        int bufferCount = inputCount + publicationCount;
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var preparations =
                new ArrayList<List<PreparedRepresentationPlan.BufferPreparation>>();
        for (int index = 0; index < bufferCount; index++) {
            entries.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 8, 4));
            if (index < inputCount) {
                preparations.add(List.of(new CallerInput()));
            } else {
                preparations.add(List.of(new InitializedBuffer(outputCreator)));
            }
        }
        PreparedMemoryPlan plan = new PreparedMemoryPlan(entries, List.of());
        var creation = new PreparedRepresentationPlan(plan, preparations, List.of());
        var steps = new ArrayList<PreparedSchedule.Step>();
        steps.add(new PreparedSchedule.RepresentationCreationStep(creation));
        for (int index = 0; index < publicationCount; index++) {
            steps.add(new PreparedSchedule.PublicationStep(
                    new PreparedPublication(plan, inputCount + index, 0, index)));
        }
        return new io.github.pho001.synaptik.runtime.execution.PreparedExecution(
                plan, new PreparedSchedule(plan, steps));
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(10, TimeUnit.SECONDS)) {
                throw new AssertionError("timed out waiting for test latch");
            }
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new AssertionError(failure);
        }
    }

    private record SyntheticBatch() implements BackendTuningCandidateBatch { }
    private record SyntheticDecision(int candidate) implements BackendTuningDecision { }
    private record SyntheticPlanBatch() implements BackendTuningCandidateBatch { }
    private record SyntheticPlanDecision(int candidate) implements BackendTuningDecision { }

    private static final class SyntheticTuning implements AdvancedEngine.ModelAutotuningTuning<
            SyntheticBatch, SyntheticDecision, Integer,
            SyntheticPlanBatch, SyntheticPlanDecision, Integer> {
        private final SyntheticBatch batch = new SyntheticBatch();
        private final SyntheticPlanBatch planBatch = new SyntheticPlanBatch();
        private final AtomicInteger trialPrepareCount = new AtomicInteger();
        private final AtomicInteger trialRunCount = new AtomicInteger();
        private final AtomicInteger selectedPrepareCount = new AtomicInteger();
        private final AtomicInteger completeTrialPrepareCount = new AtomicInteger();
        private final AtomicInteger completeTrialRunCount = new AtomicInteger();
        private final AtomicInteger completeCompatibilityCount = new AtomicInteger();
        private final AtomicInteger[] candidateRuns = {
                new AtomicInteger(), new AtomicInteger()};
        private final List<TestBuffer> trialBuffers = new ArrayList<>();
        private final AtomicInteger handoffCount = new AtomicInteger();
        private final RecordingComposition observedComposition;
        private int borrowCountAtFirstHandoff = -1;
        private Throwable trialPreparationFailure;
        private Throwable trialExecutionFailure;
        private Throwable selectedPreparationFailure;
        private Throwable completeTrialPreparationFailure;
        private Throwable completeTrialExecutionFailure;
        private boolean completePlanHandoffPresent = true;
        private SyntheticDecision completePhaseOneDecision;
        private final byte[][] completeBytes = {new byte[8], new byte[8]};
        private io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest.ReuseScope
                completeReuseScope = io.github.pho001.synaptik.tools.tuning
                        .CompletePlanTuningRequest.ReuseScope.SESSION;
        private io.github.pho001.synaptik.runtime.execution.PreparedExecution lastTrial;
        private CountDownLatch selectedEntered;
        private CountDownLatch selectedRelease;

        private SyntheticTuning() {
            this(null);
        }

        private SyntheticTuning(RecordingComposition observedComposition) {
            this.observedComposition = observedComposition;
        }

        @Override
        public Optional<BackendPartitionTuningHandoff<SyntheticBatch, SyntheticDecision>>
                candidateHandoff(CompileArtifacts artifacts) {
            if (handoffCount.getAndIncrement() == 0 && observedComposition != null) {
                borrowCountAtFirstHandoff = observedComposition.borrowCount.get();
            }
            return Optional.of(new BackendPartitionTuningHandoff<>(
                    artifacts.partitions().getFirst(), batch, Optional.empty()));
        }

        @Override public List<Integer> candidates(SyntheticBatch ignored) {
            return List.of(1, 2);
        }

        @Override public WorkloadTuningRequest.WorkloadCompatibility compatibility(
                SyntheticBatch ignored) {
            return new WorkloadTuningRequest.WorkloadCompatibility(7, new byte[] {8, 9},
                    WorkloadTuningRequest.ReuseScope.PERSISTENT);
        }

        @Override public WorkloadTuningRequest.CandidateIdentity candidateIdentity(
                Integer candidate) {
            return new WorkloadTuningRequest.CandidateIdentity(
                    new byte[] {candidate.byteValue()});
        }

        @Override public SyntheticDecision selectedDecision(
                SyntheticBatch ignored, Integer candidate) {
            return new SyntheticDecision(candidate);
        }

        @Override public byte[] encodeDecision(SyntheticDecision decision) {
            return new byte[] {(byte) decision.candidate()};
        }

        @Override public Optional<SyntheticDecision> decodeCompatibleDecision(
                SyntheticBatch ignored, byte[] encodedDecision) {
            if (encodedDecision.length != 1
                    || (encodedDecision[0] != 1 && encodedDecision[0] != 2)) {
                return Optional.empty();
            }
            return Optional.of(new SyntheticDecision(encodedDecision[0]));
        }

        @Override
        public io.github.pho001.synaptik.runtime.execution.PreparedExecution prepareTrial(
                SyntheticBatch ignored, Integer candidate) throws Exception {
            trialPrepareCount.incrementAndGet();
            if (trialPreparationFailure instanceof Exception exception) throw exception;
            rethrow(trialPreparationFailure);
            lastTrial = execution(1, 1, () -> {
                trialRunCount.incrementAndGet();
                candidateRuns[candidate - 1].incrementAndGet();
                rethrow(trialExecutionFailure);
                TestBuffer buffer = new TestBuffer(
                        "trial-" + candidate, null, new ArrayList<>());
                trialBuffers.add(buffer);
                return buffer;
            });
            return lastTrial;
        }

        @Override public Optional<BackendPartitionTuningHandoff<
                SyntheticPlanBatch, SyntheticPlanDecision>> completePlanCandidateHandoff(
                        CompileArtifacts artifacts, SyntheticDecision phaseOneDecision) {
            completePhaseOneDecision = phaseOneDecision;
            if (!completePlanHandoffPresent) return Optional.empty();
            if (phaseOneDecision.candidate() != 1 && phaseOneDecision.candidate() != 2) {
                return Optional.empty();
            }
            return Optional.of(new BackendPartitionTuningHandoff<>(
                    artifacts.partitions().getFirst(), planBatch, Optional.empty()));
        }

        @Override public List<Integer> completePlanCandidates(SyntheticPlanBatch ignored) {
            return List.of(1, 2);
        }

        @Override public io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest
                .PlanCompatibility completePlanCompatibility(SyntheticPlanBatch ignored) {
            completeCompatibilityCount.incrementAndGet();
            return new io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest
                    .PlanCompatibility(
                            new io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest
                                    .ProducerIdentity(1, new byte[] {1}),
                            new io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest
                                    .DecisionCodecIdentity(1, new byte[] {2}),
                            1, new byte[] {3}, completeReuseScope);
        }

        @Override public io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest
                .CandidateIdentity completePlanCandidateIdentity(Integer candidate) {
            return new io.github.pho001.synaptik.tools.tuning.CompletePlanTuningRequest
                    .CandidateIdentity(new byte[] {candidate.byteValue()});
        }

        @Override public SyntheticPlanDecision completePlanSelectedDecision(
                SyntheticPlanBatch ignored, Integer candidate) {
            return new SyntheticPlanDecision(candidate);
        }

        @Override public byte[] encodeCompletePlanDecision(SyntheticPlanDecision decision) {
            return new byte[] {(byte) decision.candidate()};
        }

        @Override public Optional<SyntheticPlanDecision> decodeCompatibleCompletePlanDecision(
                SyntheticPlanBatch ignored, byte[] encodedDecision) {
            if (encodedDecision.length != 1
                    || encodedDecision[0] != 1 && encodedDecision[0] != 2) {
                return Optional.empty();
            }
            return Optional.of(new SyntheticPlanDecision(encodedDecision[0]));
        }

        @Override public io.github.pho001.synaptik.runtime.execution.PreparedExecution
                prepareCompletePlanTrial(SyntheticPlanBatch ignored, Integer candidate) {
            completeTrialPrepareCount.incrementAndGet();
            rethrow(completeTrialPreparationFailure);
            return execution(1, 1, () -> {
                completeTrialRunCount.incrementAndGet();
                rethrow(completeTrialExecutionFailure);
                return new TestBuffer("complete-" + candidate, null, new ArrayList<>(),
                        completeBytes[candidate - 1]);
            });
        }

        @Override
        public io.github.pho001.synaptik.runtime.execution.PreparedExecution
                prepareCompletePlanSelected(
                        SyntheticPlanBatch ignored, SyntheticPlanDecision decision) {
            selectedPrepareCount.incrementAndGet();
            if (selectedEntered != null) selectedEntered.countDown();
            if (selectedRelease != null) await(selectedRelease);
            rethrow(selectedPreparationFailure);
            return execution(1, 1,
                    () -> new TestBuffer("selected-" + decision.candidate(), null,
                            new ArrayList<>()));
        }
    }

    private static final class RecordingComposition implements EngineBackendComposition {
        private final AtomicInteger borrowCount = new AtomicInteger();
        private final AtomicInteger prepareCount = new AtomicInteger();
        private final AtomicInteger closeCount = new AtomicInteger();
        private final AtomicInteger copyCount = new AtomicInteger();
        private final List<HostTensorStorage> borrowedStorages = new ArrayList<>();
        private final List<String> closeOrder = new ArrayList<>();
        private final List<BufferRepresentation> copiedRepresentations = new ArrayList<>();
        private final List<Throwable> borrowCloseFailures = new ArrayList<>();
        private final io.github.pho001.synaptik.runtime.execution.PreparedExecution
                preparedExecution = execution(0, 0, null);
        private int borrowFailureIndex = -1;
        private Throwable borrowFailure;
        private Throwable prepareFailure;
        private Throwable copyFailure;
        private Runnable afterFirstBorrow;
        private CountDownLatch prepareEntered;
        private CountDownLatch prepareRelease;
        private CountDownLatch copyEntered;
        private CountDownLatch copyRelease;
        private AdvancedEngine lifecycleOwner;
        private Engine ordinaryOwner;

        @Override
        public List<BackendCapabilityProvider> capabilityProviders() {
            return List.of(new BackendCapabilityProvider() {
                @Override public BackendId backendId() { return BACKEND; }
                @Override public boolean supports(OperationCapabilityQuery query) { return true; }
            });
        }

        @Override
        public List<BackendAvailabilitySnapshot> availabilitySnapshots() {
            return List.of(new BackendAvailabilitySnapshot(BACKEND,
                    Map.of(new BackendDeviceId(BACKEND, "device"), DeviceClass.CPU)));
        }

        @Override
        public io.github.pho001.synaptik.runtime.execution.PreparedExecution prepare(
                CompileArtifacts artifacts) {
            prepareCount.incrementAndGet();
            if (prepareEntered != null) prepareEntered.countDown();
            if (prepareRelease != null) await(prepareRelease);
            rethrow(prepareFailure);
            return preparedExecution;
        }

        @Override
        public BufferRepresentation borrow(HostTensorStorage storage) {
            int index = borrowCount.getAndIncrement();
            if (index == borrowFailureIndex) rethrow(borrowFailure);
            borrowedStorages.add(storage);
            if (index == 0 && afterFirstBorrow != null) afterFirstBorrow.run();
            Throwable closeFailure = index < borrowCloseFailures.size()
                    ? borrowCloseFailures.get(index) : null;
            return new TestBuffer("borrow-" + index, closeFailure, closeOrder);
        }

        @Override
        public byte[] copyToCanonicalHostBytes(
                BufferRepresentation representation,
                TensorDescriptor descriptor,
                long maximumBytes) {
            copyCount.incrementAndGet();
            copiedRepresentations.add(representation);
            if (copyEntered != null) copyEntered.countDown();
            if (copyRelease != null) await(copyRelease);
            rethrow(copyFailure);
            TestBuffer buffer = (TestBuffer) representation;
            if (buffer.canonicalBytes == null) {
                throw new AssertionError("test result has no canonical bytes");
            }
            return buffer.canonicalBytes.clone();
        }

        @Override
        public void close() {
            closeCount.incrementAndGet();
        }
    }

    private static final class TestBuffer implements BufferRepresentation {
        private final String name;
        private final Throwable failure;
        private final List<String> closeOrder;
        private final byte[] canonicalBytes;
        private final AtomicInteger closeCount = new AtomicInteger();

        private TestBuffer(String name, Throwable failure, List<String> closeOrder) {
            this(name, failure, closeOrder, null);
        }

        private TestBuffer(
                String name,
                Throwable failure,
                List<String> closeOrder,
                byte[] canonicalBytes) {
            this.name = name;
            this.failure = failure;
            this.closeOrder = closeOrder;
            this.canonicalBytes = canonicalBytes;
        }

        @Override
        public void close() {
            if (closeCount.getAndIncrement() != 0) return;
            closeOrder.add(name);
            rethrow(failure);
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (failure instanceof Error error) throw error;
    }
}
