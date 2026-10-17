package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Admission and transactional validation through an owner with no NN dependency. */
final class TrainingParametersTest {
    @Test
    void entryRejectsNullAndBlankBeforeStorageInspection() {
        Tensor tensor = TensorFactory.scalar(1f, Optional.empty(), true);
        assertThrows(NullPointerException.class, () -> new TrainingParameters.Entry(null, tensor));
        assertThrows(NullPointerException.class, () -> new TrainingParameters.Entry("p", null));
        for (String path : List.of("", " ", "\t\n")) {
            assertThrows(IllegalArgumentException.class, () -> new TrainingParameters.Entry(path, tensor));
        }
    }

    @Test
    void malformedGenericMembershipRejectsBeforeCompileOrCopyMutation() {
        Tensor tensor = TensorFactory.scalar(1f, Optional.empty(), true);
        var entry = new TrainingParameters.Entry("p", tensor);
        var storage = tensor.hostStorage().orElseThrow();
        try (Engine engine = Engine.standard(); Arena arena = Arena.ofShared()) {
            assertThrows(NullPointerException.class,
                    () -> TrainingSession.open(engine, null, tensor, new Sgd(0.1)));
            assertThrows(NullPointerException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(null, arena));
            for (List<TrainingParameters.Entry> entries : Arrays.asList(
                    null, Arrays.asList(entry, null))) {
                TrainingParameters source = unchecked(entries);
                assertThrows(NullPointerException.class,
                        () -> TrainingSession.open(engine, source, tensor, new Sgd(0.1)));
                assertThrows(NullPointerException.class,
                        () -> TrainingParameterStorage.copyToSharedNative(source, arena));
            }
            TrainingParameters empty = unchecked(List.of());
            assertThrows(IllegalStateException.class,
                    () -> TrainingSession.open(engine, empty, tensor, new Sgd(0.1)));
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(empty, arena));
            TrainingParameters duplicate = unchecked(List.of(entry, entry));
            assertEquals("duplicate parameter path: p", assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(engine, duplicate, tensor, new Sgd(0.1))).getMessage());
            assertEquals("duplicate parameter path: p", assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(duplicate, arena)).getMessage());
            Controlled invalid = new Controlled(List.of(entry));
            invalid.failAt = 1;
            assertSame(invalid.failure, assertThrows(IllegalStateException.class,
                    () -> TrainingSession.open(engine, invalid, tensor, new Sgd(0.1))));
            invalid.failAt = 2;
            assertSame(invalid.failure, assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(invalid, arena)));
            assertSame(storage, tensor.hostStorage().orElseThrow());
        }
    }

    @Test
    void encounterOrderAndSameSourceCopyThenOpenArePreserved() {
        Tensor z = TensorFactory.scalar(2f, Optional.empty(), true);
        Tensor a = TensorFactory.scalar(3f, Optional.empty(), true);
        Controlled source = new Controlled(List.of(new TrainingParameters.Entry("z", z),
                new TrainingParameters.Entry("a", a)));
        assertThrows(UnsupportedOperationException.class, () -> source.entries().clear());
        var oldZ = z.hostStorage().orElseThrow();
        var oldA = a.hostStorage().orElseThrow();
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            TrainingParameterStorage.copyToSharedNative(source, arena);
            source.validateUnchanged();
            try (TrainingSession session = TrainingSession.open(engine, source, z.add(a),
                    new Adam(0.1, 0.9, 0.999, 1e-8, 0))) {
                assertEquals(List.of("z", "a"), session.state().parameters().stream()
                        .map(TrainingState.ParameterState::path).toList());
                session.run(List.of(), GradientMode.RESET_AND_STEP);
                assertEquals(1, session.optimizerSteps());
            }
            assertTrue(arena.scope().isAlive());
            assertTrue(oldZ.isAlive());
            assertTrue(oldA.isAlive());
            assertEquals(2f, oldZ.segment().get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
            assertEquals(3f, oldA.segment().get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
        }
    }

    @Test
    void liveFailureBeforeRunAndAtEveryRunPrecommitPreservesAllPublishedState() {
        for (GradientMode mode : GradientMode.values()) {
            for (boolean low : List.of(false, true)) {
                try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                    Tensor p = nativeParameter(arena, low);
                    Controlled source = new Controlled(List.of(new TrainingParameters.Entry("p", p)));
                    boolean[] armed = {false};
                    int[] executed = {0};
                    TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
                        @Override void afterExecution() {
                            executed[0]++;
                            if (armed[0]) source.failAt = source.calls + 1;
                        }
                    };
                    Tensor objective = low ? p.cast(DataType.FLOAT32) : p;
                    try (TrainingSession session = TrainingSession.openForTesting(engine, source,
                            objective.mul(objective), new AdamW(0.01, 0.9, 0.999, 1e-8, 0), hooks)) {
                        session.run(List.of(), GradientMode.RESET_AND_STEP);
                        session.run(List.of(), GradientMode.ACCUMULATE);
                        TrainingState before = session.state();
                        int priorExecutions = executed[0];
                        source.failAt = source.calls + 1;
                        assertSame(source.failure, assertThrows(IllegalStateException.class,
                                () -> session.run(List.of(), mode)));
                        assertEquals(priorExecutions, executed[0]);
                        source.failAt = 0;
                        assertState(before, session.state());
                        armed[0] = true;
                        assertSame(source.failure, assertThrows(IllegalStateException.class,
                                () -> session.run(List.of(), mode)));
                        assertEquals(priorExecutions + 1, executed[0]);
                        source.failAt = 0;
                        assertState(before, session.state());
                    }
                }
            }
        }
    }

    @Test
    void liveStateAndBothRestoreValidationSitesAreNotFrozenListChecks() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor p = nativeParameter(arena, true);
            Controlled source = new Controlled(List.of(new TrainingParameters.Entry("p", p)));
            try (TrainingSession session = TrainingSession.open(engine, source,
                    p.cast(DataType.FLOAT32), new Adam(0.01, 0.9, 0.999, 1e-8, 0))) {
                TrainingState initial = session.state();
                session.run(List.of(), GradientMode.RESET_AND_STEP);
                session.run(List.of(), GradientMode.ACCUMULATE);
                TrainingState before = session.state();
                source.live.set(0, new TrainingParameters.Entry("changed", p));
                assertThrows(IllegalStateException.class, session::state);
                assertThrows(IllegalStateException.class, () -> session.restore(initial));
                source.live.set(0, source.entries().getFirst());
                assertState(before, session.state());
                for (int site : List.of(1, 2)) {
                    source.failAt = source.calls + site;
                    assertSame(source.failure, assertThrows(IllegalStateException.class,
                            () -> session.restore(initial)));
                    source.failAt = 0;
                    assertState(before, session.state());
                }
            }
        }
    }

    @Test
    void trainingSnapshotsEvenANonconformingMutableEntriesList() {
        Tensor p = TensorFactory.scalar(1f, Optional.empty(), true);
        List<TrainingParameters.Entry> mutable = new ArrayList<>(List.of(new TrainingParameters.Entry("p", p)));
        List<TrainingParameters.Entry> snapshot = TrainingParameter.captureEntries(unchecked(mutable));
        mutable.clear();
        assertEquals("p", snapshot.getFirst().path());
        assertThrows(UnsupportedOperationException.class, snapshot::clear);
    }

    private static Tensor nativeParameter(Arena arena, boolean low) {
        DataType type = low ? DataType.FLOAT16 : DataType.FLOAT32;
        var descriptor = new TensorDescriptor(type, Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), true);
        MemorySegment segment = arena.allocate(type.byteWidth(), type.byteWidth());
        if (low) segment.set(ValueLayout.JAVA_SHORT, 0, (short) 0x4000);
        else segment.set(ValueLayout.JAVA_FLOAT, 0, 2f);
        return TensorFactory.create(descriptor, Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, 1, segment)));
    }

    private static TrainingParameters unchecked(List<TrainingParameters.Entry> entries) {
        return new TrainingParameters() {
            @Override public List<Entry> entries() { return entries; }
            @Override public void validateUnchanged() { }
        };
    }

    private static void assertState(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.parameterGroups(), actual.parameterGroups());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int index = 0; index < expected.parameters().size(); index++) {
            var a = expected.parameters().get(index);
            var b = actual.parameters().get(index);
            assertEquals(a.path(), b.path());
            assertEquals(a.dataType(), b.dataType());
            assertEquals(a.shape(), b.shape());
            assertEquals(a.parameterBytes(), b.parameterBytes());
            assertEquals(a.masterParameterBytes(), b.masterParameterBytes());
            assertEquals(a.firstMomentBytes(), b.firstMomentBytes());
            assertEquals(a.secondMomentBytes(), b.secondMomentBytes());
            assertEquals(a.accumulatedGradientBytes(), b.accumulatedGradientBytes());
        }
    }

    private static final class Controlled implements TrainingParameters {
        private final List<Entry> snapshot;
        private final List<Entry> live;
        private final IllegalStateException failure = new IllegalStateException("live owner rejected");
        private int calls;
        private int failAt;

        private Controlled(List<Entry> entries) {
            snapshot = List.copyOf(entries);
            live = new ArrayList<>(entries);
        }

        @Override public List<Entry> entries() { return snapshot; }

        @Override public void validateUnchanged() {
            if (++calls == failAt) throw failure;
            if (snapshot.size() != live.size()) throw new IllegalStateException("count changed");
            for (int index = 0; index < live.size(); index++) {
                if (!snapshot.get(index).path().equals(live.get(index).path())
                        || snapshot.get(index).tensor() != live.get(index).tensor()) {
                    throw new IllegalStateException("membership changed");
                }
            }
        }
    }
}
