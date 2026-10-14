package io.github.pho001.synaptik.training.external;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.training.Adam;
import io.github.pho001.synaptik.training.AdamW;
import io.github.pho001.synaptik.training.GradientMode;
import io.github.pho001.synaptik.training.Optimizer;
import io.github.pho001.synaptik.training.ParameterGroup;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ReadOnlyBufferException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;

/** Exercises detached reconstruction exclusively through the public API from another package. */
final class TrainingStateReconstructionTest {
    private static final List<DataType> TYPES = List.of(
            DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16);

    @Test
    void reconstructsEveryTypeAndDefaultAlgorithmAndMixedOrderedGroups() {
        for (DataType type : TYPES) {
            for (int configuration = 0; configuration < 4; configuration++) {
                Optimizer optimizer = optimizers().get(configuration == 3 ? 1 : configuration);
                List<ParameterGroup> groups = configuration == 3 ? List.of(
                        new ParameterGroup(List.of("d", "b"), optimizers().get(0)),
                        new ParameterGroup(List.of("c"), optimizers().get(2))) : List.of();
                TrainingState saved;
                TrainingState rebuilt;
                try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                    Parameters module = new Parameters(arena, type);
                    try (TrainingSession session = TrainingSession.open(
                            engine, module, module.objective(), optimizer, groups)) {
                        TrainingState initial = session.state();
                        assertEquals(List.of("a", "b", "c", "d"), initial.parameters().stream()
                                .map(TrainingState.ParameterState::path).toList());
                        assertEquals((byte) 0x80, initial.parameters().getFirst().parameterBytes().get(0));
                        TrainingState reconstructedInitial = reconstruct(initial);
                        session.restore(reconstructedInitial);
                        assertStateEquals(initial, session.state());
                        session.run(List.of(), GradientMode.RESET_AND_STEP);
                        session.run(List.of(), GradientMode.ACCUMULATE);
                        saved = session.state();
                        assertEquals(2, saved.executions());
                        assertEquals(1, saved.optimizerSteps());
                        assertEquals(1, saved.accumulatedGradientRuns());
                        assertEquals(1.0d, type == DataType.FLOAT64
                                ? saved.parameters().getFirst().accumulatedGradientBytes().getDouble()
                                : saved.parameters().getFirst().accumulatedGradientBytes().getFloat());
                        rebuilt = reconstruct(saved);
                        session.run(List.of(), GradientMode.ACCUMULATE_AND_STEP);
                        session.restore(rebuilt);
                        assertStateEquals(saved, session.state());
                    }
                }
                assertStateEquals(saved, rebuilt);
                assertViews(rebuilt);
            }
        }
    }

    @Test
    void preservesSignedZeroSlotsAndNonzeroPendingBytesWithZeroPendingCount() {
        for (DataType type : TYPES) {
            for (Optimizer optimizer : List.of(new Sgd(0.1d, -0.0d, -0.0d, false),
                    optimizers().get(1), optimizers().get(2))) {
                try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                    Parameters module = new Parameters(arena, type);
                    try (TrainingSession session = TrainingSession.open(
                            engine, module, module.objective(), optimizer)) {
                        var entries = new ArrayList<TrainingState.ParameterState>();
                        byte[] negativeZero = optimizerBytes(type, -0.0d, -0.0d);
                        byte[] pending = optimizerBytes(type, 1.0d, -0.0d);
                        for (var entry : session.state().parameters()) {
                            entries.add(entry.secondMomentBytes().isPresent()
                                    ? new TrainingState.ParameterState(entry.path(), type, entry.shape(),
                                            bytes(entry.parameterBytes()), bytes(entry.masterParameterBytes()),
                                            negativeZero, negativeZero, pending)
                                    : new TrainingState.ParameterState(entry.path(), type, entry.shape(),
                                            bytes(entry.parameterBytes()), bytes(entry.masterParameterBytes()),
                                            negativeZero, pending));
                        }
                        TrainingState state = new TrainingState(optimizer, List.of(), 0, 0, 0, entries);
                        TrainingState rebuilt = reconstruct(state);
                        session.restore(rebuilt);
                        assertStateEquals(state, session.state());
                        assertEquals(0, session.accumulatedGradientRuns());
                        for (var entry : session.state().parameters()) {
                            assertArrayEquals(negativeZero, bytes(entry.firstMomentBytes()));
                            entry.secondMomentBytes().ifPresent(
                                    view -> assertArrayEquals(negativeZero, bytes(view)));
                            assertArrayEquals(pending, bytes(entry.accumulatedGradientBytes()));
                        }
                    }
                }
            }
        }
    }

    @Test
    void rejectsRepresentativeStructuralErrorsButDoesNotPreflightGroupMembership() {
        byte[] payload = optimizerBytes(DataType.FLOAT32, 1.0d, 2.0d);
        assertThrows(NullPointerException.class, () -> new TrainingState.ParameterState(
                null, DataType.FLOAT32, Shape.of(2), payload, payload, payload, payload));
        assertThrows(IllegalArgumentException.class, () -> new TrainingState.ParameterState(
                "a", DataType.FLOAT32, Shape.of(2), new byte[7], payload, payload, payload));
        var entry = new TrainingState.ParameterState(
                "a", DataType.FLOAT32, Shape.of(2), payload, payload, payload, payload);
        assertThrows(IllegalArgumentException.class, () -> new TrainingState(
                new Sgd(0.1d), List.of(), 0, 1, 0, List.of(entry)));
        var foreignGroup = new ParameterGroup(List.of("not-in-schema"), new Sgd(0.2d));
        TrainingState state = new TrainingState(
                new Sgd(0.1d), List.of(foreignGroup), 0, 0, 0, List.of(entry));
        assertEquals(List.of(foreignGroup), state.parameterGroups());
    }

    @Test
    void invalidLastEntryRejectsRestoreBeforeChangingAnyLiveState() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Parameters module = new Parameters(arena, DataType.FLOAT32);
            try (TrainingSession session = TrainingSession.open(
                    engine, module, module.objective(), optimizers().get(1))) {
                session.run(List.of(), GradientMode.RESET_AND_STEP);
                session.run(List.of(), GradientMode.ACCUMULATE);
                TrainingState before = session.state();
                List<byte[]> liveBefore = module.values.stream()
                        .map(TrainingStateReconstructionTest::liveBytes).toList();
                var entries = new ArrayList<TrainingState.ParameterState>();
                byte[] changed = optimizerBytes(DataType.FLOAT32, 7.0d, 8.0d);
                for (int index = 0; index < before.parameters().size(); index++) {
                    var entry = before.parameters().get(index);
                    byte[] second = index == before.parameters().size() - 1
                            ? optimizerBytes(DataType.FLOAT32, -1.0d, 0.0d) : changed;
                    entries.add(new TrainingState.ParameterState(entry.path(), entry.dataType(),
                            entry.shape(), changed, changed, changed, second, changed));
                }
                TrainingState invalid = new TrainingState(before.optimizer(), before.parameterGroups(),
                        5, 3, 2, entries);
                assertThrows(IllegalArgumentException.class, () -> session.restore(invalid));
                assertStateEquals(before, session.state());
                assertEquals(before.executions(), session.executions());
                assertEquals(before.optimizerSteps(), session.optimizerSteps());
                assertEquals(before.accumulatedGradientRuns(), session.accumulatedGradientRuns());
                for (int index = 0; index < module.values.size(); index++) {
                    assertArrayEquals(liveBefore.get(index), liveBytes(module.values.get(index)));
                }
            }
        }
    }

    private static List<Optimizer> optimizers() {
        return List.of(new Sgd(0.1d, 0.5d, 0.25d, 0.01d, false),
                new Adam(0.1d, 0.5d, 0.25d, 0.01d, -0.0d),
                new AdamW(0.2d, 0.6d, 0.3d, 0.02d, 0.1d));
    }

    private static Optimizer reconstructOptimizer(Optimizer optimizer) {
        return switch (optimizer) {
            case Sgd sgd -> new Sgd(sgd.learningRate(), sgd.momentum(), sgd.dampening(),
                    sgd.weightDecay(), sgd.nesterov());
            case Adam adam -> new Adam(adam.learningRate(), adam.beta1(), adam.beta2(),
                    adam.epsilon(), adam.weightDecay());
            case AdamW adamW -> new AdamW(adamW.learningRate(), adamW.beta1(), adamW.beta2(),
                    adamW.epsilon(), adamW.weightDecay());
        };
    }

    private static TrainingState reconstruct(TrainingState source) {
        var groups = new ArrayList<ParameterGroup>();
        for (var group : source.parameterGroups()) {
            var paths = new ArrayList<>(group.paths());
            groups.add(new ParameterGroup(paths, reconstructOptimizer(group.optimizer())));
            paths.clear();
        }
        var entries = new ArrayList<TrainingState.ParameterState>();
        for (var entry : source.parameters()) {
            byte[] logical = bytes(entry.parameterBytes());
            byte[] master = bytes(entry.masterParameterBytes());
            byte[] first = bytes(entry.firstMomentBytes());
            byte[] pending = bytes(entry.accumulatedGradientBytes());
            byte[] second = entry.secondMomentBytes().map(TrainingStateReconstructionTest::bytes)
                    .orElse(null);
            entries.add(second == null
                    ? new TrainingState.ParameterState(entry.path(), entry.dataType(), entry.shape(),
                            logical, master, first, pending)
                    : new TrainingState.ParameterState(entry.path(), entry.dataType(), entry.shape(),
                            logical, master, first, second, pending));
            for (byte[] array : List.of(logical, master, first, pending)) {
                Arrays.fill(array, (byte) 0x7f);
            }
            if (second != null) Arrays.fill(second, (byte) 0x7f);
        }
        TrainingState rebuilt = new TrainingState(reconstructOptimizer(source.optimizer()), groups,
                source.executions(), source.optimizerSteps(), source.accumulatedGradientRuns(), entries);
        groups.clear();
        entries.clear();
        assertStateEquals(source, rebuilt);
        assertThrows(UnsupportedOperationException.class, () -> rebuilt.parameters().clear());
        assertThrows(UnsupportedOperationException.class, () -> rebuilt.parameterGroups().clear());
        for (var group : rebuilt.parameterGroups()) {
            assertThrows(UnsupportedOperationException.class, () -> group.paths().clear());
        }
        assertViews(rebuilt);
        return rebuilt;
    }

    private static void assertViews(TrainingState state) {
        for (var entry : state.parameters()) {
            var accessors = new ArrayList<Supplier<ByteBuffer>>(List.of(entry::parameterBytes,
                    entry::masterParameterBytes, entry::momentumBytes, entry::firstMomentBytes,
                    entry::accumulatedGradientBytes));
            if (entry.secondMomentBytes().isPresent()) {
                accessors.add(() -> entry.secondMomentBytes().orElseThrow());
            }
            for (var accessor : accessors) {
                ByteBuffer view = accessor.get();
                ByteBuffer independent = accessor.get();
                assertNotSame(view, independent);
                assertTrue(view.isReadOnly());
                assertFalse(view.hasArray());
                assertEquals(ByteOrder.BIG_ENDIAN, view.order());
                assertEquals(0, view.position());
                assertEquals(view.capacity(), view.limit());
                assertThrows(ReadOnlyBufferException.class, () -> view.put(0, (byte) 0));
                view.position(1);
                assertEquals(0, independent.position());
                assertEquals(0, accessor.get().position());
            }
        }
    }

    private static void assertStateEquals(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.parameterGroups(), actual.parameterGroups());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int index = 0; index < expected.parameters().size(); index++) {
            var left = expected.parameters().get(index);
            var right = actual.parameters().get(index);
            assertEquals(left.path(), right.path());
            assertEquals(left.dataType(), right.dataType());
            assertEquals(left.shape(), right.shape());
            assertArrayEquals(bytes(left.parameterBytes()), bytes(right.parameterBytes()));
            assertArrayEquals(bytes(left.masterParameterBytes()), bytes(right.masterParameterBytes()));
            assertArrayEquals(bytes(left.momentumBytes()), bytes(right.momentumBytes()));
            assertArrayEquals(bytes(left.firstMomentBytes()), bytes(right.firstMomentBytes()));
            assertEquals(left.secondMomentBytes().isPresent(), right.secondMomentBytes().isPresent());
            if (left.secondMomentBytes().isPresent()) {
                assertArrayEquals(bytes(left.secondMomentBytes().orElseThrow()),
                        bytes(right.secondMomentBytes().orElseThrow()));
            }
            assertArrayEquals(bytes(left.accumulatedGradientBytes()),
                    bytes(right.accumulatedGradientBytes()));
        }
    }

    private static byte[] bytes(ByteBuffer buffer) {
        byte[] copy = new byte[buffer.remaining()];
        buffer.get(copy);
        return copy;
    }

    private static byte[] optimizerBytes(DataType type, double first, double second) {
        ByteBuffer buffer = ByteBuffer.allocate(type == DataType.FLOAT64 ? 16 : 8)
                .order(ByteOrder.BIG_ENDIAN);
        if (type == DataType.FLOAT64) buffer.putDouble(first).putDouble(second);
        else buffer.putFloat((float) first).putFloat((float) second);
        return buffer.array();
    }

    private static byte[] liveBytes(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
    }

    private static Tensor vector(Arena arena, DataType type) {
        var segment = arena.allocate(2L * type.byteWidth(), type.byteWidth());
        switch (type) {
            case FLOAT64 -> {
                segment.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, -0.0d);
                segment.setAtIndex(ValueLayout.JAVA_DOUBLE, 1, 2.0d);
            }
            case FLOAT32 -> {
                segment.setAtIndex(ValueLayout.JAVA_FLOAT, 0, -0.0f);
                segment.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 2.0f);
            }
            case BFLOAT16, FLOAT16 -> {
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x8000);
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4000);
            }
            default -> throw new AssertionError(type);
        }
        Shape shape = Shape.of(2);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), true), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, 2, segment)));
    }

    private static final class Parameters extends Module {
        private final List<Tensor> values = new ArrayList<>();

        private Parameters(Arena arena, DataType type) {
            for (String path : List.of("a", "b", "c", "d")) {
                Tensor value = vector(arena, type);
                parameter(path, value);
                values.add(value);
            }
        }

        private Tensor objective() {
            Tensor sum = values.getFirst();
            for (int index = 1; index < values.size(); index++) sum = sum.add(values.get(index));
            if (sum.descriptor().dataType() != DataType.FLOAT64) sum = sum.cast(DataType.FLOAT32);
            return sum.sum();
        }
    }
}
