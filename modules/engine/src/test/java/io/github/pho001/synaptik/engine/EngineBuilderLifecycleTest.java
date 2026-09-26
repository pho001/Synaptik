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
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;

/** Locks explicit Builder transfer, snapshot, ordering, terminal-failure, and cleanup behavior. */
final class EngineBuilderLifecycleTest {
    private static final BackendId FIRST = new BackendId("first");
    private static final BackendId SECOND = new BackendId("second");
    private static final BackendId THIRD = new BackendId("third");

    @Test
    void numericalProfileDefaultsReassignsAndCapturesAcrossCompile() {
        var defaultQuery = new AtomicReference<OperationCapabilityQuery>();
        RecordingEntry defaultEntry = new RecordingEntry(
                FIRST, query -> {
                    defaultQuery.set(query);
                    return true;
                }, new ArrayList<>());
        try (Engine engine = Engine.builder().takeOwnership(defaultEntry).build()) {
            CompiledGraph compiled = engine.compile(List.of(leaf().neg()));
            assertSame(NumericalProfile.STRICT_IEEE, engine.numericalProfile());
            assertSame(NumericalProfile.STRICT_IEEE,
                    compiled.artifacts().numericalProfile());
            assertSame(NumericalProfile.STRICT_IEEE,
                    defaultQuery.get().numericalProfile());
        }

        var explicitQuery = new AtomicReference<OperationCapabilityQuery>();
        RecordingEntry explicitEntry = new RecordingEntry(
                SECOND, query -> {
                    explicitQuery.set(query);
                    return true;
                }, new ArrayList<>());
        Engine.Builder builder = Engine.builder()
                .numericalProfile(NumericalProfile.ACCELERATOR)
                .takeOwnership(explicitEntry);
        assertSame(builder, builder.numericalProfile(NumericalProfile.STRICT_IEEE));
        assertSame(builder, builder.numericalProfile(NumericalProfile.ACCELERATOR));
        try (Engine engine = builder.build()) {
            CompiledGraph compiled = engine.compile(List.of(leaf().neg()));
            assertSame(NumericalProfile.ACCELERATOR, engine.numericalProfile());
            assertSame(NumericalProfile.ACCELERATOR,
                    compiled.artifacts().numericalProfile());
            assertSame(NumericalProfile.ACCELERATOR,
                    explicitQuery.get().numericalProfile());
        }
    }

    @Test
    void numericalProfileValidatesNullBeforeStateAndRejectsSpentMutation() {
        Engine.Builder closed = Engine.builder();
        closed.close();
        assertEquals("numericalProfile", assertThrows(NullPointerException.class,
                () -> closed.numericalProfile(null)).getMessage());
        assertEquals("engine builder is spent or closed",
                assertThrows(IllegalStateException.class,
                        () -> closed.numericalProfile(NumericalProfile.ACCELERATOR))
                        .getMessage());

        Engine.Builder spent = Engine.builder();
        assertThrows(IllegalArgumentException.class, spent::build);
        assertEquals("numericalProfile", assertThrows(NullPointerException.class,
                () -> spent.numericalProfile(null)).getMessage());
        assertThrows(IllegalStateException.class,
                () -> spent.numericalProfile(NumericalProfile.ACCELERATOR));
    }

    @Test
    void duplicateRegistrationClosesOnlyCandidateAndKeepsFirstOwned() {
        RecordingEntry first = new RecordingEntry(FIRST, true, new ArrayList<>());
        RecordingEntry duplicate = new RecordingEntry(new BackendId("first"), true,
                new ArrayList<>());
        RuntimeException cleanupFailure = new RuntimeException("duplicate close");
        duplicate.closeFailure = cleanupFailure;
        Engine.Builder builder = Engine.builder().takeOwnership(first);

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> builder.takeOwnership(duplicate));

        assertEquals("duplicate backend ID: first", failure.getMessage());
        assertArrayEquals(new Throwable[] {cleanupFailure}, failure.getSuppressed());
        assertEquals(0, first.closeCount.get());
        assertEquals(1, duplicate.closeCount.get());
        builder.close();
        assertEquals(1, first.closeCount.get());
    }

    @Test
    void registrationCapturesProviderAndSnapshotOnceAndCompileUsesRegistrationOrder() {
        var closeOrder = new ArrayList<String>();
        RecordingEntry first = new RecordingEntry(FIRST, true, closeOrder);
        RecordingEntry second = new RecordingEntry(SECOND, true, closeOrder);
        Engine.Builder builder = Engine.builder();
        builder.takeOwnership(first);
        first.replaceFacts(SECOND, false);
        builder.takeOwnership(second);

        try (Engine engine = builder.build()) {
            Tensor input = leaf();
            CompiledGraph compiled = engine.compile(List.of(input.neg()));
            assertEquals(FIRST, compiled.artifacts().partitions().getFirst().owner());
            assertEquals(1, first.providerReads.get());
            assertEquals(1, first.snapshotReads.get());
            assertEquals(1, second.providerReads.get());
            assertEquals(1, second.snapshotReads.get());
        }
        assertEquals(List.of("second", "first"), closeOrder);
    }

    @Test
    void interrogationFailureClosesCandidateOnceAndLeavesEarlierRegistrationUsable() {
        var closeOrder = new ArrayList<String>();
        RecordingEntry accepted = new RecordingEntry(FIRST, true, closeOrder);
        RecordingEntry rejected = new RecordingEntry(SECOND, true, closeOrder);
        RuntimeException primary = new RuntimeException("provider interrogation");
        RuntimeException cleanup = new RuntimeException("candidate cleanup");
        rejected.providerFailure = primary;
        rejected.closeFailure = cleanup;
        Engine.Builder builder = Engine.builder().takeOwnership(accepted);

        RuntimeException observed = assertThrows(
                RuntimeException.class, () -> builder.takeOwnership(rejected));

        assertSame(primary, observed);
        assertArrayEquals(new Throwable[] {cleanup}, observed.getSuppressed());
        assertEquals(1, rejected.closeCount.get());
        assertEquals(0, accepted.closeCount.get());
        try (Engine engine = builder.build()) {
            assertEquals(FIRST, engine.compile(List.of(leaf().neg()))
                    .artifacts().partitions().getFirst().owner());
        }
        assertEquals(1, accepted.closeCount.get());
        assertEquals(List.of("second", "first"), closeOrder);
    }

    @Test
    void failedBuildRollsBackAcceptedEntriesInReverseWithDistinctSuppressionOnly() {
        var closeOrder = new ArrayList<String>();
        RuntimeException primary = new RuntimeException("construction");
        RuntimeException cleanup = new RuntimeException("first cleanup");
        RecordingEntry first = new RecordingEntry(FIRST, true, closeOrder);
        RecordingEntry second = new RecordingEntry(SECOND, true, closeOrder);
        first.closeFailure = cleanup;
        second.closeFailure = primary;
        Engine.Builder builder = Engine.builder().takeOwnership(first).takeOwnership(second);

        RuntimeException observed = assertThrows(RuntimeException.class,
                () -> builder.build(ignored -> {
                    throw primary;
                }));

        assertSame(primary, observed);
        assertArrayEquals(new Throwable[] {cleanup}, observed.getSuppressed());
        assertEquals(List.of("second", "first"), closeOrder);
        assertEquals(1, first.closeCount.get());
        assertEquals(1, second.closeCount.get());
        RecordingEntry late = new RecordingEntry(
                new BackendId("late"), true, new ArrayList<>());
        assertThrows(IllegalStateException.class, () -> builder.takeOwnership(late));
        assertEquals(1, late.closeCount.get());
        builder.close();
    }

    @Test
    void engineCloseAttemptsAllInReverseAndRetainsExactFailure() {
        var closeOrder = new ArrayList<String>();
        RuntimeException firstFailure = new RuntimeException("first close");
        RuntimeException secondFailure = new RuntimeException("second close");
        RecordingEntry first = new RecordingEntry(FIRST, true, closeOrder);
        RecordingEntry second = new RecordingEntry(SECOND, true, closeOrder);
        first.closeFailure = firstFailure;
        second.closeFailure = secondFailure;
        Engine engine = Engine.builder().takeOwnership(first).takeOwnership(second).build();

        RuntimeException observed = assertThrows(RuntimeException.class, engine::close);

        assertSame(secondFailure, observed);
        assertArrayEquals(new Throwable[] {firstFailure}, observed.getSuppressed());
        assertEquals(List.of("second", "first"), closeOrder);
        assertSame(observed, assertThrows(RuntimeException.class, engine::close));
        assertEquals(1, first.closeCount.get());
        assertEquals(1, second.closeCount.get());
    }

    @Test
    void builderCloseAttemptsAllInReverseAndRetainsExactFailure() {
        var closeOrder = new ArrayList<String>();
        RuntimeException firstFailure = new RuntimeException("first close");
        RuntimeException secondFailure = new RuntimeException("second close");
        RecordingEntry first = new RecordingEntry(FIRST, true, closeOrder);
        RecordingEntry second = new RecordingEntry(SECOND, true, closeOrder);
        first.closeFailure = firstFailure;
        second.closeFailure = secondFailure;
        Engine.Builder builder = Engine.builder().takeOwnership(first).takeOwnership(second);

        RuntimeException observed = assertThrows(RuntimeException.class, builder::close);

        assertSame(secondFailure, observed);
        assertArrayEquals(new Throwable[] {firstFailure}, observed.getSuppressed());
        assertEquals(List.of("second", "first"), closeOrder);
        assertSame(observed, assertThrows(RuntimeException.class, builder::close));
        assertEquals(1, first.closeCount.get());
        assertEquals(1, second.closeCount.get());
    }

    @Test
    void emptyBuildIsTerminalAndLaterNonNullRegistrationIsImmediatelyClosed() {
        Engine.Builder builder = Engine.builder();
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, builder::build);
        assertEquals("at least one backend integration is required", failure.getMessage());
        RecordingEntry candidate = new RecordingEntry(FIRST, true, new ArrayList<>());

        IllegalStateException rejected = assertThrows(IllegalStateException.class,
                () -> builder.takeOwnership(candidate));

        assertEquals("engine builder is spent or closed", rejected.getMessage());
        assertEquals(1, candidate.closeCount.get());
        builder.close();
    }

    @Test
    void successfulBuildSpendsBuilderAndLaterCandidateStillTransfersAndCloses() {
        RecordingEntry accepted = new RecordingEntry(FIRST, true, new ArrayList<>());
        Engine.Builder builder = Engine.builder().takeOwnership(accepted);
        Engine engine = builder.build();
        RecordingEntry late = new RecordingEntry(SECOND, true, new ArrayList<>());
        try {
            IllegalStateException rejected = assertThrows(IllegalStateException.class,
                    () -> builder.takeOwnership(late));
            assertEquals(1, late.closeCount.get());
            builder.close();
            assertEquals(0, accepted.closeCount.get());
        } finally {
            engine.close();
        }
        assertEquals(1, accepted.closeCount.get());
    }

    @Test
    void identityMismatchPreservesPrimaryAndClosesTransferredEntry() {
        RecordingEntry mismatched = new RecordingEntry(FIRST, true, new ArrayList<>());
        mismatched.replaceFacts(SECOND, true);
        mismatched.snapshotBackendId = FIRST;
        Engine.Builder builder = Engine.builder();

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> builder.takeOwnership(mismatched));

        assertEquals("provider and availability snapshot backend IDs must be equal",
                failure.getMessage());
        assertEquals(1, mismatched.closeCount.get());
        builder.close();
    }

    @Test
    void emptyAndUnsupportedMixedOwnerPlansRejectBeforeAnyBackendPreparation() {
        RecordingEntry all = new RecordingEntry(FIRST, true, new ArrayList<>());
        try (Engine engine = Engine.builder().takeOwnership(all).build()) {
            CompiledGraph empty = engine.compile(List.of(leaf()));
            assertEquals("preparation requires a non-empty partition plan",
                    assertThrows(IllegalArgumentException.class,
                            () -> engine.prepare(empty)).getMessage());
            assertEquals(0, all.analysisCount.get());
        }

        RecordingEntry neg = new RecordingEntry(
                FIRST,
                query -> query.operation().kind() == UnaryElementwiseKind.NEG,
                new ArrayList<>());
        RecordingEntry add = new RecordingEntry(
                SECOND,
                query -> query.operation().kind() == BinaryArithmeticKind.ADD,
                new ArrayList<>());
        try (Engine engine = Engine.builder()
                .takeOwnership(neg)
                .takeOwnership(add)
                .build()) {
            Tensor input = leaf();
            CompiledGraph mixed = engine.compile(List.of(input.neg().add(input)));
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> engine.prepare(mixed));
            assertTrue(failure.getMessage().startsWith(
                    "unsupported cross-owner transfer from first to second for "));
            assertEquals(0, neg.analysisCount.get());
            assertEquals(0, add.analysisCount.get());
        }
    }

    @Test
    void unsupportedTransferGeometryRejectsBeforeEitherBackendAnalysis() {
        List<Shape> unsupportedShapes = List.of(
                Shape.of(Long.MAX_VALUE),
                Shape.of(0),
                Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
        for (Shape shape : unsupportedShapes) {
            RecordingEntry neg = new RecordingEntry(
                    FIRST,
                    query -> query.operation().kind() == UnaryElementwiseKind.NEG,
                    new ArrayList<>());
            RecordingEntry add = new RecordingEntry(
                    SECOND,
                    query -> query.operation().kind() == BinaryArithmeticKind.ADD,
                    new ArrayList<>());
            neg.transferSupport = CpuMetalPreparedBufferTransfer::supports;
            add.transferSupport = CpuMetalPreparedBufferTransfer::supports;
            try (Engine engine = Engine.builder()
                    .takeOwnership(neg)
                    .takeOwnership(add)
                    .build()) {
                TensorDescriptor descriptor = new TensorDescriptor(
                        DataType.FLOAT32,
                        shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)),
                        false);
                Tensor input = TensorFactory.create(
                        descriptor, Optional.empty(), Optional.empty());
                CompiledGraph mixed = engine.compile(List.of(input.neg().add(input)));

                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class, () -> engine.prepare(mixed));

                assertTrue(failure.getMessage().startsWith(
                        "unsupported cross-owner transfer from first to second for "));
                assertEquals(0, neg.analysisCount.get());
                assertEquals(0, add.analysisCount.get());
            }
        }
    }

    @Test
    void publicGraphPartitionAdmissionCoversSelectAndSliceStorageBoundaries() {
        Shape inputShape = Shape.of(3, 4);
        TensorDescriptor inputDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                inputShape,
                Optional.of(LayoutDescriptor.of(
                        inputShape, new long[] {6, 1}, 2L, true)),
                false);
        Tensor input = TensorFactory.create(
                inputDescriptor, Optional.empty(), Optional.empty());
        Tensor selected = input.select(0, 1);
        Tensor sliced = selected.sliceByLength(
                new long[] {0}, new long[] {2}, new int[] {0}, new long[] {2});
        Tensor output = sliced.neg();
        assertTrue(CpuMetalPreparedBufferTransfer.supports(selected.descriptor()));
        assertTrue(CpuMetalPreparedBufferTransfer.supports(sliced.descriptor()));

        Shape rejectionShape = Shape.of(2, 2);
        assertFalse(CpuMetalPreparedBufferTransfer.supports(new TensorDescriptor(
                DataType.FLOAT32, rejectionShape, Optional.empty(), false)));
        assertFalse(CpuMetalPreparedBufferTransfer.supports(new TensorDescriptor(
                DataType.FLOAT32,
                rejectionShape,
                Optional.of(LayoutDescriptor.of(
                        rejectionShape, new long[] {0, 1}, 0L, true)),
                false)));
        assertFalse(CpuMetalPreparedBufferTransfer.supports(new TensorDescriptor(
                DataType.FLOAT32,
                rejectionShape,
                Optional.of(LayoutDescriptor.of(
                        rejectionShape, new long[] {1, 1}, 0L, true)),
                false)));
        Shape emptyShape = Shape.of(0, 2);
        assertFalse(CpuMetalPreparedBufferTransfer.supports(new TensorDescriptor(
                DataType.FLOAT32,
                emptyShape,
                Optional.of(LayoutDescriptor.contiguous(emptyShape)),
                false)));

        RecordingEntry select = new RecordingEntry(
                FIRST,
                query -> query.operation().kind() == SelectKind.SELECT,
                new ArrayList<>());
        RecordingEntry slice = new RecordingEntry(
                SECOND,
                query -> query.operation().kind() == SliceKind.SLICE,
                new ArrayList<>());
        RecordingEntry neg = new RecordingEntry(
                THIRD,
                query -> query.operation().kind() == UnaryElementwiseKind.NEG,
                new ArrayList<>());
        select.transferSupport = CpuMetalPreparedBufferTransfer::supports;
        slice.transferSupport = CpuMetalPreparedBufferTransfer::supports;
        neg.transferSupport = CpuMetalPreparedBufferTransfer::supports;
        try (Engine engine = Engine.builder()
                .takeOwnership(select)
                .takeOwnership(slice)
                .takeOwnership(neg)
                .build()) {
            CompiledGraph compiled = engine.compile(List.of(output));
            assertThrows(AssertionError.class, () -> engine.prepare(compiled));
            assertEquals(1, select.analysisCount.get() + slice.analysisCount.get()
                    + neg.analysisCount.get());
        }
    }

    @Test
    void missingExactOwnerRejectsBeforeRegisteredAdapterPreparation() {
        RecordingEntry selected = new RecordingEntry(SECOND, true, new ArrayList<>());
        CompileArtifacts artifacts;
        try (Engine engine = Engine.builder().takeOwnership(selected).build()) {
            artifacts = engine.compile(List.of(leaf().neg())).artifacts();
        }
        RecordingEntry registered = new RecordingEntry(FIRST, true, new ArrayList<>());
        EngineBackendRegistry registry = new EngineBackendRegistry(List.of(registered));
        try {
            assertEquals("no registered backend owns plan: second",
                    assertThrows(IllegalArgumentException.class,
                            () -> registry.selectedAdapter(artifacts)).getMessage());
            assertEquals(0, registered.analysisCount.get());
        } finally {
            registry.close();
        }
    }

    private static Tensor leaf() {
        Shape shape = Shape.of(2);
        return TensorFactory.fromFlatArray(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false),
                Optional.empty(), new float[] {1.0f, -2.0f});
    }

    private static final class RecordingEntry implements EngineBackendComposition {
        private final String name;
        private final List<String> closeOrder;
        private final AtomicInteger closeCount = new AtomicInteger();
        private final AtomicInteger providerReads = new AtomicInteger();
        private final AtomicInteger snapshotReads = new AtomicInteger();
        private final AtomicInteger analysisCount = new AtomicInteger();
        private BackendId providerBackendId;
        private BackendId snapshotBackendId;
        private Predicate<OperationCapabilityQuery> support;
        private Predicate<TensorDescriptor> transferSupport = ignored -> false;
        private RuntimeException closeFailure;
        private RuntimeException providerFailure;

        private RecordingEntry(BackendId backendId, boolean supports, List<String> closeOrder) {
            this(backendId, ignored -> supports, closeOrder);
        }

        private RecordingEntry(
                BackendId backendId,
                Predicate<OperationCapabilityQuery> support,
                List<String> closeOrder) {
            this.name = backendId.value();
            this.closeOrder = closeOrder;
            this.providerBackendId = backendId;
            this.snapshotBackendId = backendId;
            this.support = support;
        }

        private void replaceFacts(BackendId backendId, boolean supports) {
            this.providerBackendId = backendId;
            this.snapshotBackendId = backendId;
            this.support = ignored -> supports;
        }

        @Override
        public List<BackendCapabilityProvider> capabilityProviders() {
            providerReads.incrementAndGet();
            if (providerFailure != null) throw providerFailure;
            BackendId capturedId = providerBackendId;
            Predicate<OperationCapabilityQuery> capturedSupport = support;
            return List.of(new BackendCapabilityProvider() {
                @Override
                public BackendId backendId() {
                    return capturedId;
                }

                @Override
                public boolean supports(OperationCapabilityQuery query) {
                    return capturedSupport.test(query);
                }
            });
        }

        @Override
        public List<BackendAvailabilitySnapshot> availabilitySnapshots() {
            snapshotReads.incrementAndGet();
            BackendId capturedId = snapshotBackendId;
            return List.of(new BackendAvailabilitySnapshot(capturedId, Map.of(
                    new BackendDeviceId(capturedId, "device"), DeviceClass.CPU)));
        }

        @Override
        public io.github.pho001.synaptik.prepare.PartitionPreparation<?, ?>
                partitionPreparation() {
            analysisCount.incrementAndGet();
            throw new AssertionError("unexpected partition preparation");
        }

        @Override
        public io.github.pho001.synaptik.prepare.PreparedScheduleContributor
                scheduleContributor() {
            throw new AssertionError("unexpected schedule contribution");
        }

        @Override
        public io.github.pho001.synaptik.prepare.PreparedScheduleAssembler scheduleAssembler() {
            throw new AssertionError("unexpected schedule assembly");
        }

        @Override
        public boolean supportsTransferTo(
                EngineBackendComposition destination, TensorDescriptor descriptor) {
            return transferSupport.test(descriptor);
        }

        @Override
        public io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer
                prepareTransferTo(
                        EngineBackendComposition destination,
                        io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan memoryPlan,
                        int bufferIndex,
                        int sourceRepresentationIndex,
                        int destinationRepresentationIndex,
                        TensorDescriptor descriptor) {
            throw new AssertionError("unexpected transfer preparation");
        }


        @Override
        public BufferRepresentation borrow(HostTensorStorage storage) {
            throw new AssertionError("unexpected borrow");
        }

        @Override
        public byte[] copyToCanonicalHostBytes(
                BufferRepresentation representation,
                TensorDescriptor descriptor,
                long maximumBytes) {
            throw new AssertionError("unexpected materialization");
        }

        @Override
        public void close() {
            closeCount.incrementAndGet();
            closeOrder.add(name);
            if (closeFailure != null) throw closeFailure;
        }
    }
}
