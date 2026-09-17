package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
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
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.InitializedBuffer;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.nio.ByteOrder;
import java.nio.ReadOnlyBufferException;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Exercises ordinary identity binding, publication metadata, ownership, and cleanup. */
final class EngineTypedLifecycleTest {
    private static final BackendId BACKEND = new BackendId("typed-test");

    @Test
    void compileBuildsFinalOrderedInputMetadataWithoutStorage() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(false, false);
            Tensor second = leaf(false, false);
            Tensor output = first.add(second).add(first);
            CompiledGraph compiled = engine.compile(List.of(output));

            assertEquals(List.of(first.id(), second.id()), compiled.inputs().stream()
                    .map(CompiledGraph.Input::tensorId).toList());
            assertEquals(List.of(first.descriptor(), second.descriptor()), compiled.inputs().stream()
                    .map(CompiledGraph.Input::descriptor).toList());
            assertThrows(UnsupportedOperationException.class,
                    () -> compiled.inputs().add(new CompiledGraph.Input(first.id(), first.descriptor())));
            assertTrue(composition.compileQueries.get() > 0);
            assertEquals(0, composition.borrowCount.get());
        }
    }

    @Test
    void compileChecksLocalListStructureAndBuildsExplicitFirstOrderRoles() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor left = leaf(true, true);
            Tensor right = leaf(true, true);
            Tensor seed = leaf(false, true);
            Tensor output = left.add(right);

            assertThrows(NullPointerException.class, () -> engine.compile(null));
            assertThrows(IllegalArgumentException.class, () -> engine.compile(List.of()));
            assertThrows(IllegalArgumentException.class, () -> engine.compile(List.of(output, output)));
            assertThrows(IllegalArgumentException.class,
                    () -> engine.compile(List.of(output), List.of(), List.of(left)));
            assertThrows(IllegalArgumentException.class,
                    () -> engine.compile(List.of(output), List.of(seed), List.of(left, left)));

            CompiledGraph compiled = engine.compile(
                    List.of(output), List.of(seed), List.of(left, right));
            assertEquals(List.of(left.id(), right.id(), seed.id()), compiled.inputs().stream()
                    .map(CompiledGraph.Input::tensorId).toList());
            assertEquals(3, compiled.publicationSpecs().size());
            assertEquals(List.of(RunResult.Role.FORWARD, RunResult.Role.GRADIENT,
                            RunResult.Role.GRADIENT),
                    compiled.publicationSpecs().stream().map(spec -> spec.role).toList());
            assertEquals(List.of(output.id(), left.id(), right.id()),
                    compiled.publicationSpecs().stream().map(spec -> spec.tensorId).toList());
        }
    }

    @Test
    void preparesFreshOwnerBoundHandlesAndRejectsAnotherOwner() {
        RecordingComposition composition = new RecordingComposition();
        Engine first = engine(composition);
        Engine second = engine(new RecordingComposition());
        CompiledGraph compiled = first.compile(List.of(leaf(false, true).neg()));
        io.github.pho001.synaptik.engine.PreparedExecution prepared = first.prepare(compiled);
        assertSame(compiled, prepared.compiledGraph());
        assertNotSame(prepared, first.prepare(compiled));
        assertThrows(IllegalArgumentException.class, () -> second.prepare(compiled));
        first.close();
        second.close();
    }

    @Test
    void runMatchesIdsInArbitraryOrderAndClosesResultThenBorrowWrappers() {
        List<String> closeOrder = new ArrayList<>();
        RecordingComposition composition = new RecordingComposition();
        composition.closeOrder = closeOrder;
        Engine engine = engine(composition);
        Tensor left = leaf(false, true);
        Tensor right = leaf(false, true);
        Tensor output = left.add(right);
        var prepared = engine.prepare(engine.compile(List.of(output)));

        RunResult result = engine.run(prepared, List.of(right, left));
        assertEquals(2, composition.borrowCount.get());
        assertEquals(List.of(left.hostStorage().orElseThrow(), right.hostStorage().orElseThrow()),
                composition.borrowedStorages);
        assertEquals(1, result.resultCount());
        assertSame(result.publications(), result.publications());
        RunResult.Publication publication = result.publications().getFirst();
        assertEquals(output.id(), publication.tensorId());
        assertEquals(output.descriptor(), publication.descriptor());
        assertEquals(RunResult.Role.FORWARD, publication.role());
        assertFalse(publication.isClosed());

        result.close();
        assertTrue(result.isClosed());
        assertTrue(publication.isClosed());
        assertEquals(output.id(), publication.tensorId());
        assertEquals(List.of("runtime-result", "borrow-1", "borrow-0"), closeOrder);
        engine.close();
        assertEquals(List.of("runtime-result", "borrow-1", "borrow-0", "backend"), closeOrder);
    }

    @Test
    void rejectsLogicalBindingBeforeBorrowAndUsesClosedPrecedence() {
        RecordingComposition composition = new RecordingComposition();
        Engine engine = engine(composition);
        Tensor required = leaf(false, true);
        var prepared = engine.prepare(engine.compile(List.of(required.neg())));

        assertThrows(IllegalArgumentException.class, () -> engine.run(prepared, List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> engine.run(prepared, List.of(required, required)));
        assertThrows(IllegalArgumentException.class,
                () -> engine.run(prepared, List.of(leaf(false, true))));
        assertEquals(0, composition.borrowCount.get());

        engine.close();
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class, () -> engine.run(null, null)).getMessage());
        assertEquals("advanced engine is closed",
                assertThrows(IllegalStateException.class, () -> engine.compile(null)).getMessage());
    }

    @Test
    void rejectsAbsentStorageAndAnotherPreparedOwnerBeforeBorrow() {
        RecordingComposition shared = new RecordingComposition();
        AdvancedEngine delegate = new AdvancedEngine(shared);
        Engine first = new Engine(delegate);
        Engine second = new Engine(delegate);
        Tensor absent = leaf(false, false);
        var prepared = first.prepare(first.compile(List.of(absent.neg())));

        assertThrows(IllegalStateException.class, () -> first.run(prepared, List.of(absent)));
        assertThrows(IllegalArgumentException.class, () -> second.run(prepared, List.of(absent)));
        assertEquals(0, shared.borrowCount.get());
        first.close();
    }

    @Test
    void partialBorrowFailureClosesEarlierWrappersAndPreservesOriginalFailure() {
        List<String> order = new ArrayList<>();
        RuntimeException expected = new RuntimeException("second borrow");
        RecordingComposition composition = new RecordingComposition();
        composition.closeOrder = order;
        composition.borrowFailureIndex = 1;
        composition.borrowFailure = expected;
        Engine engine = engine(composition);
        Tensor first = leaf(false, true);
        Tensor second = leaf(false, true);
        var prepared = engine.prepare(engine.compile(List.of(first.add(second))));

        assertSame(expected,
                assertThrows(RuntimeException.class, () -> engine.run(prepared, List.of(first, second))));
        assertEquals(List.of("borrow-0"), order);
        engine.close();
    }

    @Test
    void cleanupAttemptsRuntimeThenReverseBorrowsAndSuppressesDistinctFailures() {
        List<String> order = new ArrayList<>();
        RuntimeException runtimeFailure = new RuntimeException("runtime cleanup");
        Error borrowFailure = new AssertionError("borrow cleanup");
        RecordingComposition composition = new RecordingComposition();
        composition.closeOrder = order;
        composition.runtimeCloseFailure = runtimeFailure;
        composition.borrowCloseFailureIndex = 0;
        composition.borrowCloseFailure = borrowFailure;
        Engine engine = engine(composition);
        Tensor input = leaf(false, true);
        RunResult result = engine.run(engine.prepare(engine.compile(List.of(input.neg()))),
                List.of(input));

        assertSame(runtimeFailure, assertThrows(RuntimeException.class, result::close));
        assertArrayEquals(new Throwable[] {borrowFailure}, runtimeFailure.getSuppressed());
        assertEquals(List.of("runtime-result", "borrow-0"), order);
        assertSame(runtimeFailure, assertThrows(RuntimeException.class, result::close));
        engine.close();
    }

    @Test
    void hostValueDefensivelyCopiesAndReturnsFreshIndependentViews() {
        Shape shape = Shape.of(2);
        byte[] source = {1, 2, 3, 4, 5, 6, 7, 8};
        HostTensorValue value = new HostTensorValue(DataType.FLOAT32, shape, source);
        source[0] = 99;
        var first = value.bytes();
        var second = value.bytes();

        assertAll(
                () -> assertSame(DataType.FLOAT32, value.dataType()),
                () -> assertSame(shape, value.shape()),
                () -> assertEquals(2, value.elementCount()),
                () -> assertEquals(8, value.byteSize()),
                () -> assertNotSame(first, second),
                () -> assertTrue(first.isReadOnly()),
                () -> assertFalse(first.hasArray()),
                () -> assertEquals(ByteOrder.BIG_ENDIAN, first.order()),
                () -> assertEquals(0, first.position()),
                () -> assertEquals(8, first.limit()),
                () -> assertEquals(8, first.capacity()),
                () -> assertEquals(1, first.get(0)),
                () -> assertThrows(ReadOnlyBufferException.class, () -> first.put(0, (byte) 7)));
        first.position(3).limit(4);
        assertEquals(0, value.bytes().position());
        assertNotEquals(new HostTensorValue(DataType.FLOAT32, shape,
                new byte[] {1, 2, 3, 4, 5, 6, 7, 8}), value);
    }

    @Test
    void hostValueValidatesConstructorArgumentsCountsAndEmptyShapes() {
        Shape empty = Shape.of(Long.MAX_VALUE, 0, Long.MAX_VALUE);
        HostTensorValue emptyValue = new HostTensorValue(DataType.INT64, empty, new byte[0]);
        assertAll(
                () -> assertEquals("dataType", assertThrows(NullPointerException.class,
                        () -> new HostTensorValue(null, null, null)).getMessage()),
                () -> assertEquals("shape", assertThrows(NullPointerException.class,
                        () -> new HostTensorValue(DataType.FLOAT32, null, null)).getMessage()),
                () -> assertEquals("canonicalBytes", assertThrows(NullPointerException.class,
                        () -> new HostTensorValue(DataType.FLOAT32, Shape.scalar(), null)).getMessage()),
                () -> assertEquals(0, emptyValue.elementCount()),
                () -> assertEquals(0, emptyValue.byteSize()),
                () -> assertEquals(1, new HostTensorValue(
                        DataType.BOOL, Shape.scalar(), new byte[] {1}).elementCount()),
                () -> assertEquals(
                        "backend canonical byte count does not match descriptor: expected=4, actual=0",
                        assertThrows(IllegalStateException.class, () -> new HostTensorValue(
                                DataType.FLOAT32, Shape.scalar(), new byte[0])).getMessage()),
                () -> assertThrows(ArithmeticException.class, () -> new HostTensorValue(
                        DataType.INT64, Shape.of(Long.MAX_VALUE), new byte[0])));
    }

    @Test
    void materializesExactOccurrenceOnceWithoutCachingAndPreservesBackendFailure() {
        RecordingComposition composition = new RecordingComposition();
        composition.copyBytes = new byte[] {0, 0, 0, 1, 0, 0, 0, 2};
        try (Engine engine = engine(composition)) {
            Tensor input = leaf(false, true);
            RunResult result = engine.run(engine.prepare(engine.compile(List.of(input.contiguous()))),
                    List.of(input));
            RunResult.Publication publication = result.publications().getFirst();
            HostTensorValue first = result.materialize(publication, 8);
            composition.copyBytes[0] = 99;
            HostTensorValue second = result.materialize(publication, 8);

            assertAll(
                    () -> assertEquals(2, composition.copyCount.get()),
                    () -> assertSame(publication.descriptor(), composition.copyDescriptors.get(0)),
                    () -> assertSame(composition.copyRepresentations.get(0),
                            composition.copyRepresentations.get(1)),
                    () -> assertEquals(List.of(8L, 8L), composition.copyLimits),
                    () -> assertNotSame(first, second),
                    () -> assertEquals(0, first.bytes().get(0)),
                    () -> assertEquals(99, second.bytes().get(0)));

            RuntimeException expected = new RuntimeException("copy failed");
            composition.copyFailure = expected;
            assertSame(expected, assertThrows(RuntimeException.class,
                    () -> result.materialize(publication, 8)));
            composition.copyFailure = null;
            composition.copyBytes = new byte[7];
            assertEquals(
                    "backend canonical byte count does not match descriptor: expected=8, actual=7",
                    assertThrows(IllegalStateException.class,
                            () -> result.materialize(publication, 8)).getMessage());
            composition.returnNullBytes = true;
            assertEquals("canonicalBytes", assertThrows(NullPointerException.class,
                    () -> result.materialize(publication, 8)).getMessage());
            assertFalse(result.isClosed());
            result.close();
        }
    }

    @Test
    void mapsAliasedOccurrencesByDenseIndexWithoutMergingThem() {
        RecordingComposition composition = new RecordingComposition();
        composition.aliasPublications = true;
        composition.copyBytes = new byte[8];
        try (Engine engine = engine(composition)) {
            Tensor left = leaf(false, true);
            Tensor right = leaf(false, true);
            var prepared = engine.prepare(engine.compile(
                    List.of(left.contiguous(), right.contiguous())));
            try (RunResult result = engine.run(prepared, List.of(right, left))) {
                HostTensorValue first = result.materialize(result.publications().get(0), 8);
                HostTensorValue second = result.materialize(result.publications().get(1), 8);
                assertAll(
                        () -> assertEquals(2, composition.copyCount.get()),
                        () -> assertSame(composition.copyRepresentations.get(0),
                                composition.copyRepresentations.get(1)),
                        () -> assertSame(result.publications().get(0).descriptor(),
                                composition.copyDescriptors.get(0)),
                        () -> assertSame(result.publications().get(1).descriptor(),
                                composition.copyDescriptors.get(1)),
                        () -> assertNotSame(first, second));
            }
        }
    }

    @Test
    void materializeUsesExactPublicationOwnershipAndValidationPrecedence() {
        RecordingComposition composition = new RecordingComposition();
        composition.copyBytes = new byte[8];
        Engine engine = engine(composition);
        Tensor input = leaf(false, true);
        var prepared = engine.prepare(engine.compile(List.of(input.contiguous())));
        RunResult first = engine.run(prepared, List.of(input));
        RunResult second = engine.run(prepared, List.of(input));
        RunResult.Publication publication = first.publications().getFirst();

        assertAll(
                () -> assertEquals("publication", assertThrows(NullPointerException.class,
                        () -> first.materialize(null, -1)).getMessage()),
                () -> assertEquals("publication does not belong to this run result",
                        assertThrows(IllegalArgumentException.class,
                                () -> first.materialize(second.publications().getFirst(), -1))
                                .getMessage()),
                () -> assertEquals("maximumBytes must be non-negative: -1",
                        assertThrows(IllegalArgumentException.class,
                                () -> first.materialize(publication, -1)).getMessage()),
                () -> assertEquals(
                        "canonical byte count exceeds maximumBytes: required=8, maximum=7",
                        assertThrows(IllegalArgumentException.class,
                                () -> first.materialize(publication, 7)).getMessage()),
                () -> assertEquals(0, composition.copyCount.get()));
        first.close();
        assertEquals("run result is closed", assertThrows(IllegalStateException.class,
                () -> first.materialize(null, -1)).getMessage());
        engine.close();
        assertEquals("advanced engine is closed", assertThrows(IllegalStateException.class,
                () -> second.materialize(null, -1)).getMessage());
    }

    @Test
    void materializeValidatesDescriptorCountsBeforePhysicalAccess() {
        RecordingComposition composition = new RecordingComposition();
        AdvancedEngine advanced = new AdvancedEngine(composition);
        Engine engine = new Engine(advanced);
        TestBuffer representation = new TestBuffer(null, null, "publication");

        Shape dynamicShape = Shape.ofDimensions(new DynamicDimension("N"));
        RunResult dynamic = syntheticResult(advanced,
                new TensorDescriptor(DataType.FLOAT32, dynamicShape, Optional.empty(), false),
                representation);
        assertEquals("host snapshot requires a fully static shape: Shape[N]",
                assertThrows(IllegalArgumentException.class,
                        () -> dynamic.materialize(dynamic.publications().getFirst(), Long.MAX_VALUE))
                        .getMessage());

        Shape staticShape = Shape.of(2);
        RunResult unresolved = syntheticResult(advanced,
                new TensorDescriptor(DataType.FLOAT32, staticShape, Optional.empty(), false),
                representation);
        assertEquals("host snapshot requires a resolved layout",
                assertThrows(IllegalArgumentException.class,
                        () -> unresolved.materialize(
                                unresolved.publications().getFirst(), Long.MAX_VALUE)).getMessage());

        Shape hugeShape = Shape.of(600_000_000L);
        RunResult huge = syntheticResult(advanced,
                new TensorDescriptor(DataType.FLOAT32, hugeShape,
                        Optional.of(LayoutDescriptor.contiguous(hugeShape)), false), representation);
        assertEquals("canonical byte count exceeds maximumBytes: required=2400000000, maximum=1",
                assertThrows(IllegalArgumentException.class,
                        () -> huge.materialize(huge.publications().getFirst(), 1)).getMessage());
        assertEquals("canonical byte count exceeds JVM byte[] limit: 2400000000",
                assertThrows(IllegalArgumentException.class,
                        () -> huge.materialize(
                                huge.publications().getFirst(), Long.MAX_VALUE)).getMessage());

        Shape shape = Shape.of(2);
        TensorDescriptor stridedDescriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {2}, 1, true)), false);
        composition.copyBytes = new byte[8];
        RunResult strided = syntheticResult(advanced, stridedDescriptor, representation);
        assertEquals(8, strided.materialize(strided.publications().getFirst(), 8).byteSize());
        assertSame(stridedDescriptor, composition.copyDescriptors.getFirst());
        assertEquals(1, composition.copyCount.get());

        dynamic.close();
        unresolved.close();
        huge.close();
        strided.close();
        engine.close();
    }

    @Test
    void sameResultCopySerializesAndResultAndEngineCloseWait() throws Exception {
        RecordingComposition composition = new RecordingComposition();
        composition.copyBytes = new byte[8];
        composition.copyEntered = new CountDownLatch(1);
        composition.copyRelease = new CountDownLatch(1);
        Engine engine = engine(composition);
        Tensor input = leaf(false, true);
        RunResult result = engine.run(engine.prepare(engine.compile(List.of(input.contiguous()))),
                List.of(input));
        RunResult.Publication publication = result.publications().getFirst();

        try (var executor = Executors.newFixedThreadPool(3)) {
            var copy = executor.submit(() -> result.materialize(publication, 8));
            assertTrue(composition.copyEntered.await(10, TimeUnit.SECONDS));
            var secondCopy = executor.submit(() -> result.materialize(publication, 8));
            var close = executor.submit(() -> { engine.close(); return null; });
            while (!engine.isClosed()) Thread.onSpinWait();
            assertEquals(1, composition.copyCount.get());
            composition.copyRelease.countDown();
            HostTensorValue value = copy.get(10, TimeUnit.SECONDS);
            close.get(10, TimeUnit.SECONDS);
            HostTensorValue secondValue = secondCopy.get(10, TimeUnit.SECONDS);
            assertEquals(8, value.byteSize());
            assertEquals(8, secondValue.byteSize());
            assertEquals(2, composition.copyCount.get());
        }
    }

    @Test
    void differentResultsCopyConcurrentlyAndResultCloseWaitsForItsCopy() throws Exception {
        RecordingComposition composition = new RecordingComposition();
        composition.copyBytes = new byte[8];
        composition.copyEntered = new CountDownLatch(2);
        composition.copyRelease = new CountDownLatch(1);
        Engine engine = engine(composition);
        Tensor input = leaf(false, true);
        var prepared = engine.prepare(engine.compile(List.of(input.contiguous())));
        RunResult first = engine.run(prepared, List.of(input));
        RunResult second = engine.run(prepared, List.of(input));

        try (var executor = Executors.newFixedThreadPool(3)) {
            var firstCopy = executor.submit(
                    () -> first.materialize(first.publications().getFirst(), 8));
            var secondCopy = executor.submit(
                    () -> second.materialize(second.publications().getFirst(), 8));
            assertTrue(composition.copyEntered.await(10, TimeUnit.SECONDS));
            CountDownLatch closeStarted = new CountDownLatch(1);
            var firstClose = executor.submit(() -> {
                closeStarted.countDown();
                first.close();
                return null;
            });
            assertTrue(closeStarted.await(10, TimeUnit.SECONDS));
            assertFalse(firstClose.isDone());
            composition.copyRelease.countDown();
            assertEquals(8, firstCopy.get(10, TimeUnit.SECONDS).byteSize());
            assertEquals(8, secondCopy.get(10, TimeUnit.SECONDS).byteSize());
            firstClose.get(10, TimeUnit.SECONDS);
        }
        assertTrue(first.isClosed());
        assertFalse(second.isClosed());
        engine.close();
    }

    @Test
    void computeValidatesArgumentsInOrderAndPreflightsAggregateBeforeCopy() {
        RecordingComposition composition = new RecordingComposition();
        composition.closeOrder = new ArrayList<>();
        composition.copyBytes = new byte[8];
        Engine engine = engine(composition);
        Tensor left = leaf(false, true);
        Tensor right = leaf(false, true);
        Tensor leftOutput = left.contiguous();
        Tensor rightOutput = right.contiguous();

        assertAll(
                () -> assertEquals("output", assertThrows(NullPointerException.class,
                        () -> engine.compute((Tensor) null, -1)).getMessage()),
                () -> assertEquals("maximumTotalBytes must be non-negative: -1",
                        assertThrows(IllegalArgumentException.class,
                                () -> engine.compute(leftOutput, -1)).getMessage()),
                () -> assertEquals("outputs", assertThrows(NullPointerException.class,
                        () -> engine.compute((List<Tensor>) null, -1)).getMessage()),
                () -> assertEquals("outputs[0]", assertThrows(NullPointerException.class,
                        () -> engine.compute(Collections.singletonList(null), -1))
                                .getMessage()),
                () -> assertEquals("outputs must not be empty",
                        assertThrows(IllegalArgumentException.class,
                                () -> engine.compute(List.of(), -1)).getMessage()),
                () -> assertEquals("outputs[1] duplicates outputs[0]",
                        assertThrows(IllegalArgumentException.class, () -> engine.compute(
                                List.of(leftOutput, leftOutput), -1)).getMessage()));

        assertEquals(
                "total canonical byte count exceeds maximumTotalBytes: required=16, maximum=15",
                assertThrows(IllegalArgumentException.class, () -> engine.compute(
                        List.of(leftOutput, rightOutput), 15)).getMessage());
        assertEquals(0, composition.copyCount.get());
        assertEquals(List.of("runtime-result", "runtime-result", "borrow-1", "borrow-0"),
                composition.closeOrder);

        List<HostTensorValue> values = engine.compute(List.of(leftOutput, rightOutput), 16);
        assertAll(
                () -> assertEquals(2, values.size()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> values.add(values.getFirst())),
                () -> assertNotSame(values.get(0), values.get(1)),
                () -> assertEquals(List.of(8L, 8L), composition.copyLimits),
                () -> assertEquals(2, composition.copyCount.get()));
        engine.close();
    }

    @Test
    void computeDiscoversSharedAndDistinctLeavesAndUsesCompilerBindingOrder() {
        RecordingComposition composition = new RecordingComposition();
        composition.copyBytes = new byte[8];
        try (Engine engine = engine(composition)) {
            Tensor first = leaf(false, true);
            Tensor second = leaf(false, true);
            Tensor shared = first.neg();
            Tensor output = shared.add(shared).add(second).contiguous();

            HostTensorValue singleton = engine.compute(output);
            List<HostTensorValue> ordered = engine.compute(List.of(output));

            assertAll(
                    () -> assertEquals(8, singleton.byteSize()),
                    () -> assertEquals(1, ordered.size()),
                    () -> assertEquals(List.of(
                                    first.hostStorage().orElseThrow(),
                                    second.hostStorage().orElseThrow(),
                                    first.hostStorage().orElseThrow(),
                                    second.hostStorage().orElseThrow()),
                            composition.borrowedStorages),
                    () -> assertEquals(4, composition.borrowCount.get()),
                    () -> assertEquals(2, composition.copyCount.get()));
        }
    }

    @Test
    void leafInventoryIsIterativeAndCompiledSelectionFiltersAndRejectsMissingBindings() {
        RecordingComposition composition = new RecordingComposition();
        try (Engine engine = engine(composition)) {
            Tensor selected = leaf(false, true);
            Tensor extra = leaf(false, false);
            Tensor deep = selected;
            for (int depth = 0; depth < 10_000; depth++) {
                deep = deep.neg();
            }
            Tensor output = deep.add(deep).contiguous();

            Map<io.github.pho001.synaptik.model.tensor.TensorId, Tensor> inventory =
                    AdvancedEngine.inventoryReachableLeaves(List.of(output));
            assertEquals(Map.of(selected.id(), selected), inventory);

            CompiledGraph compiled = engine.compile(List.of(selected.contiguous()));
            var inventoryWithUnselected = new java.util.HashMap<>(inventory);
            inventoryWithUnselected.put(extra.id(), extra);
            assertEquals(List.of(selected),
                    AdvancedEngine.selectCompiledInputs(compiled, inventoryWithUnselected));
            assertEquals(
                    "compiled input has no reachable provenance-free Tensor: " + selected.id(),
                    assertThrows(IllegalStateException.class,
                            () -> AdvancedEngine.selectCompiledInputs(compiled, Map.of()))
                            .getMessage());
            assertEquals(0, composition.borrowCount.get());
        }
    }

    @Test
    void computeSingleUsesSameLifecycleAndSuppressesCleanupFailureOnCopyFailure() {
        RuntimeException copyFailure = new RuntimeException("second copy failed");
        Error cleanupFailure = new AssertionError("temporary result cleanup failed");
        RecordingComposition composition = new RecordingComposition();
        composition.closeOrder = new ArrayList<>();
        composition.copyBytes = new byte[8];
        composition.copyFailure = copyFailure;
        composition.runtimeCloseFailure = cleanupFailure;
        Engine engine = engine(composition);
        Tensor input = leaf(false, true);

        RuntimeException actual = assertThrows(RuntimeException.class,
                () -> engine.compute(input.contiguous(), 8));
        assertSame(copyFailure, actual);
        assertArrayEquals(new Throwable[] {cleanupFailure}, actual.getSuppressed());
        assertEquals(List.of("runtime-result", "borrow-0"), composition.closeOrder);
        engine.close();
    }

    @Test
    void engineCloseWaitsForAdmittedComputeAndDetachedValueReturnsAfterClosureStarts()
            throws Exception {
        RecordingComposition composition = new RecordingComposition();
        composition.copyBytes = new byte[8];
        composition.copyEntered = new CountDownLatch(1);
        composition.copyRelease = new CountDownLatch(1);
        Engine engine = engine(composition);
        Tensor input = leaf(false, true);

        try (var executor = Executors.newFixedThreadPool(2)) {
            var compute = executor.submit(() -> engine.compute(input.contiguous(), 8));
            assertTrue(composition.copyEntered.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> { engine.close(); return null; });
            while (!engine.isClosed()) Thread.onSpinWait();
            assertFalse(close.isDone());
            composition.copyRelease.countDown();
            HostTensorValue value = compute.get(10, TimeUnit.SECONDS);
            close.get(10, TimeUnit.SECONDS);
            assertEquals(8, value.byteSize());
            assertEquals("advanced engine is closed", assertThrows(IllegalStateException.class,
                    () -> engine.compute((Tensor) null, -1)).getMessage());
        }
    }

    private static Engine engine(RecordingComposition composition) {
        return new Engine(new AdvancedEngine(composition));
    }

    private static RunResult syntheticResult(
            AdvancedEngine owner, TensorDescriptor descriptor, BufferRepresentation representation) {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 1, 1)), List.of());
        RunState state = new RunState(plan, List.of(List.of(new BufferRepresentationBinding(
                representation, RunResourceOwnership.BORROWED))), List.of());
        state.setBufferRepresentationValid(0, 0, true);
        var publication = new PreparedPublication(plan, 0, 0, 0).bind(state);
        publication.publish();
        var delegate = new io.github.pho001.synaptik.runtime.run.RunResult(
                state, List.of(publication));
        AdvancedRunResult resultOwner = new AdvancedRunResult(owner, delegate);
        return new RunResult(resultOwner, List.of(new CompiledGraph.PublicationSpec(
                new io.github.pho001.synaptik.model.tensor.TensorId(1), descriptor,
                RunResult.Role.FORWARD, 0, -1)));
    }

    private static Tensor leaf(boolean requiresGrad, boolean storage) {
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
        if (!storage) return TensorFactory.create(descriptor);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), new float[] {1, 2});
    }

    private static final class RecordingComposition implements EngineBackendComposition {
        private final AtomicInteger compileQueries = new AtomicInteger();
        private final AtomicInteger borrowCount = new AtomicInteger();
        private final List<HostTensorStorage> borrowedStorages = new ArrayList<>();
        private List<String> closeOrder;
        private int borrowFailureIndex = -1;
        private Throwable borrowFailure;
        private int borrowCloseFailureIndex = -1;
        private Throwable borrowCloseFailure;
        private Throwable runtimeCloseFailure;
        private final AtomicInteger copyCount = new AtomicInteger();
        private final List<BufferRepresentation> copyRepresentations =
                Collections.synchronizedList(new ArrayList<>());
        private final List<TensorDescriptor> copyDescriptors =
                Collections.synchronizedList(new ArrayList<>());
        private final List<Long> copyLimits = Collections.synchronizedList(new ArrayList<>());
        private byte[] copyBytes;
        private Throwable copyFailure;
        private CountDownLatch copyEntered;
        private CountDownLatch copyRelease;
        private boolean returnNullBytes;
        private boolean aliasPublications;

        @Override
        public List<BackendCapabilityProvider> capabilityProviders() {
            return List.of(new BackendCapabilityProvider() {
                @Override public BackendId backendId() { return BACKEND; }
                @Override public boolean supports(OperationCapabilityQuery query) {
                    compileQueries.incrementAndGet();
                    return true;
                }
            });
        }

        @Override
        public List<BackendAvailabilitySnapshot> availabilitySnapshots() {
            return List.of(new BackendAvailabilitySnapshot(BACKEND,
                    Map.of(new BackendDeviceId(BACKEND, "device"), DeviceClass.CPU)));
        }

        @Override
        public PreparedExecution prepare(CompileArtifacts artifacts) {
            int inputCount = artifacts.constants().bindableInputs().size();
            int publicationCount = artifacts.publication().forwardBindings().size()
                    + artifacts.publication().gradientBindings().size();
            int publicationBufferCount = aliasPublications && publicationCount > 0
                    ? 1 : publicationCount;
            int bufferCount = inputCount + publicationBufferCount;
            var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
            var preparations = new ArrayList<List<PreparedRepresentationPlan.BufferPreparation>>();
            for (int index = 0; index < bufferCount; index++) {
                entries.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 8, 4));
                if (index < inputCount) {
                    preparations.add(List.of(new CallerInput()));
                } else {
                    preparations.add(List.of(new InitializedBuffer(() -> new TestBuffer(
                            runtimeCloseFailure, closeOrder, "runtime-result"))));
                }
            }
            PreparedMemoryPlan plan = new PreparedMemoryPlan(entries, List.of());
            var creation = new PreparedRepresentationPlan(plan, preparations, List.of());
            var steps = new ArrayList<PreparedSchedule.Step>();
            steps.add(new PreparedSchedule.RepresentationCreationStep(creation));
            for (int index = 0; index < publicationCount; index++) {
                steps.add(new PreparedSchedule.PublicationStep(
                        new PreparedPublication(plan,
                                inputCount + (aliasPublications ? 0 : index), 0, index)));
            }
            return new PreparedExecution(plan, new PreparedSchedule(plan, steps));
        }

        @Override
        public BufferRepresentation borrow(HostTensorStorage storage) {
            int index = borrowCount.getAndIncrement();
            if (index == borrowFailureIndex) {
                rethrow(borrowFailure);
            }
            borrowedStorages.add(storage);
            Throwable failure = index == borrowCloseFailureIndex ? borrowCloseFailure : null;
            return new TestBuffer(failure, closeOrder, "borrow-" + index);
        }

        @Override
        public byte[] copyToCanonicalHostBytes(
                BufferRepresentation representation,
                TensorDescriptor descriptor,
                long maximumBytes) {
            copyCount.incrementAndGet();
            copyRepresentations.add(representation);
            copyDescriptors.add(descriptor);
            copyLimits.add(maximumBytes);
            if (copyEntered != null) copyEntered.countDown();
            if (copyRelease != null) {
                try {
                    assertTrue(copyRelease.await(10, TimeUnit.SECONDS));
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(failure);
                }
            }
            rethrow(copyFailure);
            if (returnNullBytes) return null;
            if (copyBytes != null) return copyBytes;
            return new byte[Math.toIntExact(Math.multiplyExact(
                    descriptor.shape().knownElementCount().orElseThrow(),
                    descriptor.dataType().byteWidth()))];
        }

        @Override
        public void close() {
            if (closeOrder != null) closeOrder.add("backend");
        }
    }

    private static final class TestBuffer implements BufferRepresentation {
        private final Throwable failure;
        private final List<String> order;
        private final String name;
        private boolean closed;

        private TestBuffer(Throwable failure, List<String> order, String name) {
            this.failure = failure;
            this.order = order;
            this.name = name;
        }

        @Override
        public void close() {
            if (closed) return;
            closed = true;
            if (order != null) order.add(name);
            rethrow(failure);
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure instanceof Error error) throw error;
    }
}
