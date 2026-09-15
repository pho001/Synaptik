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
import io.github.pho001.synaptik.model.shape.Shape;
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
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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

    private static Engine engine(RecordingComposition composition) {
        return new Engine(new AdvancedEngine(composition));
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
            int bufferCount = inputCount + publicationCount;
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
                        new PreparedPublication(plan, inputCount + index, 0, index)));
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
