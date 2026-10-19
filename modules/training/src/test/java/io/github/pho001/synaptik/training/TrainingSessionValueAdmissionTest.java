package io.github.pho001.synaptik.training;

import static io.github.pho001.synaptik.training.TrainingValueFixture.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/** Exact descriptor, physical storage and Training-specific overlap admission. */
final class TrainingSessionValueAdmissionTest {
    /**
     * Rejects each descriptor/storage failure before any committed state change.
     * @throws Exception if concurrent validation or private result inspection fails
     */
    @Test
    void rejectsMalformedDescriptorsAndStorageWithoutChangingPendingState() throws Exception {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            Tensor target = scalar(arena, 0, false);
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            try (TrainingSession session = TrainingSession.open(engine, source,
                    weight.mul(target), new Sgd(0.1))) {
                session.runValues(List.of(scalar(arena, 3, false)), GradientMode.ACCUMULATE);
                TrainingState before = session.state();
                assertThrows(NullPointerException.class, () -> session.runValues(null, null));
                assertThrows(NullPointerException.class, () -> session.runValues(List.of(target), null));
                assertThrows(NullPointerException.class, () -> session.runValues(
                        Collections.singletonList(null), GradientMode.RESET_AND_STEP));
                assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(List.of(), GradientMode.RESET_AND_STEP));
                assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(List.of(target, target), GradientMode.RESET_AND_STEP));
                var scalar = Shape.scalar();
                var vector = Shape.of(1);
                // One representative of every full descriptor component, not a dtype matrix.
                for (TensorDescriptor mismatch : List.of(
                        new TensorDescriptor(DataType.FLOAT64, scalar,
                                Optional.of(LayoutDescriptor.contiguous(scalar)), false),
                        new TensorDescriptor(DataType.FLOAT32, vector,
                                Optional.of(LayoutDescriptor.contiguous(vector)), false),
                        new TensorDescriptor(DataType.FLOAT32, scalar, Optional.empty(), false),
                        new TensorDescriptor(DataType.FLOAT32, scalar,
                                Optional.of(LayoutDescriptor.of(scalar, new long[0], 1, true)), false),
                        new TensorDescriptor(DataType.FLOAT32, scalar,
                                Optional.of(LayoutDescriptor.contiguous(scalar)), true))) {
                    assertThrows(IllegalArgumentException.class, () -> session.runValues(
                            List.of(TensorFactory.create(mismatch)), GradientMode.RESET_AND_STEP));
                }
                assertThrows(IllegalStateException.class, () -> session.runValues(
                        List.of(TensorFactory.create(target.descriptor())), GradientMode.RESET_AND_STEP));
                assertThrows(IllegalArgumentException.class, () -> session.runValues(
                        List.of(stored(target.descriptor(), arena.allocate(0, 4))), GradientMode.RESET_AND_STEP));
                // Model itself prevents attaching storage with a different data type.
                assertThrows(IllegalArgumentException.class, () -> TensorFactory.create(
                        target.descriptor(), Optional.empty(), Optional.of(new MemorySegmentStorage(
                                DataType.INT32, 1, arena.allocate(4, 4)))));
                try (Arena confined = Arena.ofConfined();
                        var executor = Executors.newSingleThreadExecutor()) {
                    Tensor inaccessible = scalar(confined, 1, false);
                    assertTrue(executor.submit(() -> assertThrows(IllegalStateException.class,
                            () -> session.runValues(List.of(inaccessible), GradientMode.RESET_AND_STEP)))
                            .get(10, TimeUnit.SECONDS).getMessage().contains("not accessible"));
                }
                Tensor dead;
                try (Arena temporary = Arena.ofShared()) {
                    dead = scalar(temporary, 1, false);
                }
                assertTrue(assertThrows(IllegalStateException.class, () -> session.runValues(
                        List.of(dead), GradientMode.RESET_AND_STEP)).getMessage().contains("not alive"));
                assertState(before, session.state());
                assertEquals(0, openResults(engine));
                session.runValues(List.of(scalar(arena, 4, false)), GradientMode.ACCUMULATE_AND_STEP);
                assertEquals(1.3f, value(weight), 1e-6f);
            }
        }
    }

    /** Rejects a positional swap when the two external descriptors are incompatible. */
    @Test
    void positionsWithIncompatibleDescriptorsCannotBeSwapped() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            Tensor batch = scalar(arena, 1, false);
            var shape = Shape.of(1);
            Tensor target = stored(new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false), arena.allocate(4, 4));
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            try (TrainingSession session = TrainingSession.open(engine, source,
                    weight.mul(batch).sub(target.sum()), new Sgd(0.1))) {
                var swapped = values(session, batch, target, target, batch);
                assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(swapped, GradientMode.RESET_AND_STEP));
                assertEquals(0, session.executions());
            }
        }
    }

    /**
     * Preserves ordered overlap diagnostics for repeated, sliced, parameter and seed storage.
     * @throws Exception if private seed or result fields cannot be inspected
     */
    @Test
    void rejectsRepeatedSlicesParameterAndSeedAliasesInCompiledDiagnosticOrder() throws Exception {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            Tensor batch = scalar(arena, 1, false);
            Tensor target = scalar(arena, 0, false);
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            try (TrainingSession session = TrainingSession.open(engine, source,
                    objective(weight, batch, target), new Sgd(0.1))) {
                Tensor fresh = scalar(arena, 3, false);
                String expected = "training input storages overlap: input "
                        + session.inputs().get(0).tensorId() + " and input "
                        + session.inputs().get(1).tensorId();
                assertEquals(expected, assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(List.of(fresh, fresh), GradientMode.RESET_AND_STEP))
                        .getMessage());
                Tensor alias = stored(fresh.descriptor(), fresh.hostStorage().orElseThrow()
                        .segment().asSlice(0, 4));
                assertEquals(expected, assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(List.of(fresh, alias), GradientMode.RESET_AND_STEP))
                        .getMessage());
                Tensor parameterAlias = stored(batch.descriptor(),
                        weight.hostStorage().orElseThrow().segment());
                assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(values(session, batch, target, parameterAlias, target),
                                GradientMode.RESET_AND_STEP)).getMessage().contains("parameter weight"));
                Tensor seed = (Tensor) field(session, "cotangentSeed");
                assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(values(session, batch, target, seed, target),
                                GradientMode.RESET_AND_STEP)).getMessage().contains("cotangent seed"));
                assertEquals(0, session.executions());
                assertEquals(0, openResults(engine));
            }
        }
    }

    /** Allows repeated zero-byte external values without relaxing nonempty overlap rejection. */
    @Test
    void zeroByteRepeatedValuesKeepTheOverlapExemption() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            var shape = Shape.of(0);
            var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false);
            Tensor left = stored(descriptor, arena.allocate(0, 4));
            Tensor right = stored(descriptor, arena.allocate(0, 4));
            Tensor empty = stored(descriptor, arena.allocate(0, 4));
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            try (TrainingSession session = TrainingSession.open(engine, source,
                    weight.mul(left.sum().add(right.sum())), new Sgd(0.1))) {
                assertEquals(0, session.runValues(List.of(empty, empty), GradientMode.RESET_AND_STEP)
                        .objective().bytes().getFloat());
                assertEquals(2, value(weight));
            }
        }
    }

    /**
     * Isolates unresolved positional capacity admission from CPU compile capability and old run.
     * @throws Exception if test-only boundary metadata substitution or binding inspection fails
     */
    @Test
    void unresolvedGeometryUsesEngineChecksWithoutStrengtheningOriginalTrainingBinding() throws Exception {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor weight = scalar(arena, 2, true);
            var descriptor = new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
            Tensor prototype = scalar(arena, 3, false);
            var source = new TestParameterSource();
            source.parameter("weight", weight);
            try (TrainingSession session = TrainingSession.open(engine, source,
                    weight.mul(prototype), new Sgd(0.1))) {
                // CPU compile requires resolved input layouts. As in EngineTypedLifecycleTest,
                // substitute only boundary metadata after preparing a valid CPU recipe. This
                // tests absent-layout admission, not a new backend compilation capability.
                replaceField(prototype, "descriptor", descriptor);
                prototype.replaceHostStorage(new MemorySegmentStorage(DataType.FLOAT32, 0,
                        arena.allocate(0, 4)));
                replaceField(session, "externalInputs", List.of(new CompiledGraph.Input(prototype.id(), descriptor)));
                replaceField(session.compiledGraph(), "inputs", session.compiledGraph().inputs().stream()
                        .map(input -> input.tensorId().equals(prototype.id())
                                ? new CompiledGraph.Input(prototype.id(), descriptor) : input).toList());
                // The existing precomputed span is the same one checked canonical static
                // geometry gives this scalar; the old layout-span-or-zero check stays weaker.
                Method bind = TrainingSession.class.getDeclaredMethod("bindInputs", List.class);
                bind.setAccessible(true);
                assertNotNull(bind.invoke(session, List.of(prototype)));
                assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> session.runValues(List.of(stored(descriptor, arena.allocate(0, 4))),
                                GradientMode.RESET_AND_STEP)).getMessage().contains("capacity"));
                Tensor enough = stored(descriptor, MemorySegment.ofArray(new float[] {7}));
                session.runValues(List.of(enough), GradientMode.ACCUMULATE);
                assertEquals(7, session.state().parameters().getFirst()
                        .accumulatedGradientBytes().getFloat());
            }
        }
    }

    /**
     * Substitutes test-only boundary metadata without adding a production accessor or backend.
     * @param owner exact object declaring the immutable field
     * @param name declared field name
     * @param value replacement used only in the isolated admission test
     * @throws ReflectiveOperationException if the existing fields change
     */
    private static void replaceField(Object owner, String name, Object value) throws ReflectiveOperationException {
        Field field = owner.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(owner, value);
    }
}
