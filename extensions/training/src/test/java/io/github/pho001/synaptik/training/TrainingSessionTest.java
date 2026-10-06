package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

final class TrainingSessionTest {
    @Test
    void appliesPersistentMomentumWithExactStepNumbers() {
        ScalarParameters module = new ScalarParameters(2.0f);
        Tensor target = scalar(0.0f, false);
        Tensor difference = module.first().value().sub(target);
        Tensor objective = difference.mul(difference);

        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, objective, new Sgd(0.1d, 0.5d, 0.0d, false))) {
            assertEquals(List.of(target.id()),
                    session.inputs().stream().map(input -> input.tensorId()).toList());
            assertSame(session.compiledGraph(), session.compiledGraph());

            TrainingStep first = session.run(List.of(target), GradientMode.RESET_AND_STEP);
            assertEquals(1, first.executionNumber());
            assertEquals(1, first.optimizerStep().orElseThrow());
            assertEquals(4.0f, first.objective().bytes().getFloat(), 0.0f);
            assertEquals(1.6f, value(module.first()), 1e-6f);

            TrainingStep second = session.run(List.of(target), GradientMode.RESET_AND_STEP);
            assertEquals(2, second.executionNumber());
            assertEquals(2, second.optimizerStep().orElseThrow());
            assertEquals(1.08f, value(module.first()), 1e-6f);
            assertEquals(5.2f,
                    session.state().parameters().getFirst().momentumBytes().getFloat(), 1e-6f);
            assertEquals(2, session.executions());
            assertEquals(2, session.optimizerSteps());
        }
    }

    @Test
    void appliesDampeningOnlyAfterFirstMomentumStepInFloat32AndFloat64() {
        Sgd optimizer = new Sgd(0.1d, 0.5d, 0.25d, 0.0d, false);
        assertEquals(0.25d, optimizer.dampening(), 0.0d);
        assertEquals(optimizer, new Sgd(0.1d, 0.5d, 0.25d, 0.0d, false));

        ScalarParameters floatModule = new ScalarParameters(2.0f);
        Tensor floatTarget = scalar(0.0f, false);
        Tensor floatDifference = floatModule.first().value().sub(floatTarget);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, floatModule, floatDifference.mul(floatDifference), optimizer)) {
            session.run(List.of(floatTarget), GradientMode.RESET_AND_STEP);
            assertEquals(1.6f, value(floatModule.first()), 1e-6f);
            assertEquals(4.0f,
                    session.state().parameters().getFirst().momentumBytes().getFloat(), 0.0f);
            session.run(List.of(floatTarget), GradientMode.RESET_AND_STEP);
            assertEquals(1.16f, value(floatModule.first()), 1e-6f);
            assertEquals(4.4f,
                    session.state().parameters().getFirst().momentumBytes().getFloat(), 1e-6f);
        }

        Tensor doubleParameter = scalar64(2.0d, true);
        Tensor doubleTarget = scalar64(0.0d, false);
        ScalarParameters doubleModule = new ScalarParameters(doubleParameter);
        Tensor doubleDifference = doubleParameter.sub(doubleTarget);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, doubleModule, doubleDifference.mul(doubleDifference), optimizer)) {
            session.run(List.of(doubleTarget), GradientMode.RESET_AND_STEP);
            assertEquals(1.6d, doubleParameter.hostStorage().orElseThrow().segment()
                    .get(ValueLayout.JAVA_DOUBLE, 0), 1e-12d);
            session.run(List.of(doubleTarget), GradientMode.RESET_AND_STEP);
            assertEquals(1.16d, doubleParameter.hostStorage().orElseThrow().segment()
                    .get(ValueLayout.JAVA_DOUBLE, 0), 1e-12d);
            assertEquals(4.4d,
                    session.state().parameters().getFirst().momentumBytes().getDouble(), 1e-12d);
        }
    }

    @Test
    void mapsMultipleParametersAndImplementsAccumulationAndZeroing() {
        ScalarParameters module = new ScalarParameters(2.0f, -1.0f);
        Tensor firstInput = scalar(3.0f, false);
        Tensor secondInput = scalar(4.0f, false);
        Tensor objective = module.first().value().mul(firstInput)
                .add(module.second().value().mul(secondInput));

        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, objective, new Sgd(0.1d))) {
            assertEquals(List.of(firstInput.id(), secondInput.id()),
                    session.inputs().stream().map(input -> input.tensorId()).toList());

            TrainingStep accumulated = session.run(
                    List.of(secondInput, firstInput), GradientMode.ACCUMULATE);
            assertTrue(accumulated.optimizerStep().isEmpty());
            assertEquals(1, accumulated.accumulatedGradientRuns());
            assertEquals(2.0f, value(module.first()), 0.0f);
            assertEquals(-1.0f, value(module.second()), 0.0f);

            TrainingStep updated = session.run(
                    List.of(firstInput, secondInput), GradientMode.ACCUMULATE_AND_STEP);
            assertEquals(1, updated.optimizerStep().orElseThrow());
            assertEquals(0, updated.accumulatedGradientRuns());
            assertEquals(1.4f, value(module.first()), 1e-6f);
            assertEquals(-1.8f, value(module.second()), 1e-6f);

            session.run(List.of(firstInput, secondInput), GradientMode.ACCUMULATE);
            assertEquals(1, session.accumulatedGradientRuns());
            session.zeroGrad();
            assertEquals(0, session.accumulatedGradientRuns());
            assertEquals(1, session.optimizerSteps());
        }
    }

    @Test
    void usesCoupledWeightDecayAndNesterovInDocumentedOrder() {
        ScalarParameters module = new ScalarParameters(2.0f);
        Tensor target = scalar(0.0f, false);
        Tensor difference = module.first().value().sub(target);
        Tensor objective = difference.mul(difference);

        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine,
                        module,
                        objective,
                        new Sgd(0.1d, 0.5d, 0.1d, true))) {
            session.run(List.of(target), GradientMode.RESET_AND_STEP);
            assertEquals(1.37f, value(module.first()), 1e-6f);
            assertEquals(4.2f,
                    session.state().parameters().getFirst().momentumBytes().getFloat(), 1e-6f);
        }
    }

    @Test
    void snapshotsRestoresAndKeepsFailedRunAtomic() {
        ScalarParameters module = new ScalarParameters(2.0f);
        Tensor target = scalar(0.0f, false);
        Tensor difference = module.first().value().sub(target);
        Tensor objective = difference.mul(difference);

        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, objective, new Sgd(0.1d, 0.5d, 0.0d, false))) {
            session.run(List.of(target), GradientMode.ACCUMULATE);
            TrainingState beforeFailure = session.state();

            assertThrows(IllegalArgumentException.class,
                    () -> session.run(List.of(), GradientMode.ACCUMULATE_AND_STEP));
            TrainingState afterFailure = session.state();
            assertStateEquals(beforeFailure, afterFailure);
            assertEquals(2.0f, value(module.first()), 0.0f);

            session.run(List.of(target), GradientMode.ACCUMULATE_AND_STEP);
            assertEquals(1.2f, value(module.first()), 1e-6f);
            session.restore(beforeFailure);
            assertEquals(2.0f, value(module.first()), 0.0f);
            assertEquals(1, session.executions());
            assertEquals(0, session.optimizerSteps());
            assertEquals(1, session.accumulatedGradientRuns());

            TrainingStep resumed = session.run(
                    List.of(target), GradientMode.ACCUMULATE_AND_STEP);
            assertEquals(1, resumed.optimizerStep().orElseThrow());
            assertEquals(1.2f, value(module.first()), 1e-6f);
        }
    }

    @Test
    void updatesFloat64ParametersAndRejectsUnrepresentableFloat32Configuration() {
        Tensor parameter = scalar64(2.0d, true);
        Tensor target = scalar64(0.0d, false);
        ScalarParameters module = new ScalarParameters(parameter);
        Tensor difference = parameter.sub(target);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, difference.mul(difference), new Sgd(0.1d))) {
            session.run(List.of(target), GradientMode.RESET_AND_STEP);
            assertEquals(1.6d, parameter.hostStorage().orElseThrow().segment()
                    .get(ValueLayout.JAVA_DOUBLE, 0), 1e-12d);
        }

        ScalarParameters floatModule = new ScalarParameters(2.0f);
        Tensor floatParameter = floatModule.first().value();
        assertThrows(IllegalArgumentException.class,
                () -> new Sgd(Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new Sgd(0.1d, 1.0d, 0.0d, false));
        assertThrows(IllegalArgumentException.class,
                () -> new Sgd(0.1d, 0.0d, -0.1d, false));
        assertThrows(IllegalArgumentException.class,
                () -> new Sgd(0.1d, 0.5d, -0.1d, 0.0d, false));
        assertThrows(IllegalArgumentException.class,
                () -> new Sgd(0.1d, 0.5d, 1.1d, 0.0d, false));
        assertThrows(IllegalArgumentException.class,
                () -> new Sgd(0.1d, 0.5d, Double.MIN_VALUE, 0.0d, true));
        assertThrows(IllegalArgumentException.class,
                () -> new Sgd(0.1d, 0.0d, 0.0d, true));
        try (Engine engine = Engine.standard()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(
                            engine,
                            floatModule,
                            floatParameter.mul(floatParameter),
                            new Sgd(Double.MIN_VALUE)));
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(
                            engine,
                            floatModule,
                            floatParameter.mul(floatParameter),
                            new Sgd(0.1d, Double.MIN_VALUE, 0.0d, 0.0d, true)));
            try (TrainingSession ignored = TrainingSession.open(
                    engine,
                    floatModule,
                    floatParameter.mul(floatParameter),
                    new Sgd((double) Float.MIN_VALUE, 0.5d, Double.MIN_VALUE, 0.0d, false))) {
                assertEquals(Double.MIN_VALUE,
                        ((Sgd) ignored.optimizer()).dampening(), 0.0d);
            }
        }
    }

    @Test
    void rejectsDuplicateDeadAndReplacedParameterStorageBeforeMutation() {
        Tensor firstAlias = scalar(2.0f, true);
        Tensor secondAlias = TensorFactory.create(
                firstAlias.descriptor(), Optional.empty(), firstAlias.hostStorage());
        ScalarParameters duplicated = new ScalarParameters(firstAlias, secondAlias);
        Tensor duplicatedObjective = duplicated.first().value().mul(duplicated.second().value());
        try (Engine engine = Engine.standard()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(
                            engine, duplicated, duplicatedObjective, new Sgd(0.1d)));
        }

        Tensor dead;
        try (Arena arena = Arena.ofConfined()) {
            TensorDescriptor descriptor = new TensorDescriptor(
                    DataType.FLOAT32,
                    Shape.scalar(),
                    Optional.of(LayoutDescriptor.contiguous(Shape.scalar())),
                    true);
            MemorySegmentStorage storage = new MemorySegmentStorage(
                    DataType.FLOAT32, 1, arena.allocate(Float.BYTES, Float.BYTES));
            dead = TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
        }
        ScalarParameters deadModule = new ScalarParameters(dead);
        try (Engine engine = Engine.standard()) {
            assertThrows(IllegalStateException.class,
                    () -> TrainingSession.open(
                            engine, deadModule, dead.mul(dead), new Sgd(0.1d)));
        }

        ScalarParameters module = new ScalarParameters(2.0f);
        Tensor target = scalar(0.0f, false);
        Tensor difference = module.first().value().sub(target);
        Tensor objective = difference.mul(difference);
        MemorySegmentStorage original = (MemorySegmentStorage) module.first().value()
                .hostStorage().orElseThrow();
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, objective, new Sgd(0.1d))) {
            MemorySegmentStorage replacement = new MemorySegmentStorage(
                    DataType.FLOAT32, 1, MemorySegment.ofArray(new float[] {2.0f}));
            module.first().value().replaceHostStorage(replacement);
            assertThrows(IllegalStateException.class,
                    () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
            assertEquals(2.0f, original.segment().get(ValueLayout.JAVA_FLOAT, 0), 0.0f);
            assertEquals(0, session.executions());
            assertEquals(0, session.optimizerSteps());
            module.first().value().replaceHostStorage(original);
        }
    }

    @Test
    void rejectsHeapReadOnlyAndThreadConfinedParametersBeforeCompilation() {
        Tensor heap = TensorFactory.scalar(2.0f, Optional.empty(), true);
        ScalarParameters heapModule = new ScalarParameters(heap);
        try (Engine engine = Engine.standard()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(engine, heapModule, heap.mul(heap), new Sgd(0.1d)));
        }

        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            MemorySegment writable = arena.allocate(Float.BYTES, Float.BYTES);
            writable.set(ValueLayout.JAVA_FLOAT, 0, 2.0f);
            Tensor readOnly = tensor(writable.asReadOnly(), true);
            ScalarParameters readOnlyModule = new ScalarParameters(readOnly);
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(
                            engine, readOnlyModule, readOnly.mul(readOnly), new Sgd(0.1d)));
        }

        try (Arena arena = Arena.ofConfined(); Engine engine = Engine.standard()) {
            Tensor confined = scalar(arena, 2.0f, true);
            ScalarParameters confinedModule = new ScalarParameters(confined);
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(
                            engine, confinedModule, confined.mul(confined), new Sgd(0.1d)));
        }
    }

    @Test
    void rejectsMissingDuplicateForeignAndOverlappingInputsWithoutMutation() {
        ScalarParameters module = new ScalarParameters(2.0f);
        Tensor firstInput = scalar(3.0f, false);
        Tensor secondInput = scalar(4.0f, false);
        Tensor objective = module.first().value().mul(firstInput).add(secondInput);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, objective, new Sgd(0.1d))) {
            TrainingState initial = session.state();
            assertThrows(IllegalArgumentException.class,
                    () -> session.run(List.of(firstInput), GradientMode.RESET_AND_STEP));
            assertThrows(IllegalArgumentException.class,
                    () -> session.run(
                            List.of(firstInput, firstInput), GradientMode.RESET_AND_STEP));
            Tensor foreign = scalar(5.0f, false);
            assertThrows(IllegalArgumentException.class,
                    () -> session.run(
                            List.of(firstInput, foreign), GradientMode.RESET_AND_STEP));
            assertStateEquals(initial, session.state());
            assertEquals(2.0f, value(module.first()), 0.0f);
        }

        Tensor parameter = scalar(2.0f, true);
        Tensor overlappingInput = TensorFactory.create(
                new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.scalar(),
                        Optional.of(LayoutDescriptor.contiguous(Shape.scalar())),
                        false),
                Optional.empty(),
                parameter.hostStorage());
        ScalarParameters overlappingModule = new ScalarParameters(parameter);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine,
                        overlappingModule,
                        parameter.mul(overlappingInput),
                        new Sgd(0.1d))) {
            assertThrows(IllegalArgumentException.class,
                    () -> session.run(
                            List.of(overlappingInput), GradientMode.RESET_AND_STEP));
            assertEquals(2.0f, value(overlappingModule.first()), 0.0f);
            assertEquals(0, session.executions());
        }
    }

    @Test
    void rejectsAbsentDeadAndInaccessibleExternalStorageWithoutMutation() throws Exception {
        ScalarParameters absentModule = new ScalarParameters(2.0f);
        Tensor absent = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())),
                false));
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, absentModule, absentModule.first().value().mul(absent),
                        new Sgd(0.1d))) {
            assertThrows(IllegalStateException.class,
                    () -> session.run(List.of(absent), GradientMode.RESET_AND_STEP));
            assertEquals(2.0f, value(absentModule.first()), 0.0f);
            assertEquals(0, session.executions());
        }

        ScalarParameters deadModule = new ScalarParameters(2.0f);
        Arena deadArena = Arena.ofShared();
        Tensor dead = scalar(deadArena, 3.0f, false);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, deadModule, deadModule.first().value().mul(dead),
                        new Sgd(0.1d))) {
            deadArena.close();
            assertThrows(IllegalStateException.class,
                    () -> session.run(List.of(dead), GradientMode.RESET_AND_STEP));
            assertEquals(2.0f, value(deadModule.first()), 0.0f);
            assertEquals(0, session.executions());
        }

        ScalarParameters confinedInputModule = new ScalarParameters(2.0f);
        try (Arena confinedArena = Arena.ofConfined();
                Engine engine = Engine.standard()) {
            Tensor confined = scalar(confinedArena, 3.0f, false);
            try (TrainingSession session = TrainingSession.open(
                    engine,
                    confinedInputModule,
                    confinedInputModule.first().value().mul(confined),
                    new Sgd(0.1d));
                    var executor = Executors.newVirtualThreadPerTaskExecutor()) {
                var failure = executor.submit(() -> assertThrows(
                        IllegalStateException.class,
                        () -> session.run(
                                List.of(confined), GradientMode.RESET_AND_STEP)));
                failure.get(10, TimeUnit.SECONDS);
                assertEquals(2.0f, value(confinedInputModule.first()), 0.0f);
                assertEquals(0, session.executions());
            }
        }
    }

    @Test
    void closeRaceIsIdempotentAndDetachedResultsOutliveOwners() throws Exception {
        ScalarParameters module = new ScalarParameters(2.0f);
        Tensor target = scalar(0.0f, false);
        Tensor difference = module.first().value().sub(target);
        Tensor objective = difference.mul(difference);
        TrainingStep retained;
        TrainingState retainedState;
        Engine engine = Engine.standard();
        TrainingSession session = TrainingSession.open(
                engine, module, objective, new Sgd(0.1d));
        retained = session.run(List.of(target), GradientMode.RESET_AND_STEP);
        retainedState = session.state();

        var start = new CountDownLatch(1);
        boolean racingRunCompleted;
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var run = executor.submit(() -> {
                start.await();
                try {
                    session.run(List.of(target), GradientMode.RESET_AND_STEP);
                    return true;
                } catch (IllegalStateException rejected) {
                    assertTrue(rejected.getMessage().contains("closed"));
                    return false;
                }
            });
            var firstClose = executor.submit(() -> {
                start.await();
                session.close();
                return null;
            });
            var secondClose = executor.submit(() -> {
                start.await();
                session.close();
                return null;
            });
            start.countDown();
            racingRunCompleted = run.get(10, TimeUnit.SECONDS);
            firstClose.get(10, TimeUnit.SECONDS);
            secondClose.get(10, TimeUnit.SECONDS);
        }
        assertEquals(racingRunCompleted ? 2 : 1, session.executions());
        assertEquals(racingRunCompleted ? 1.28f : 1.6f, value(module.first()), 1e-6f);
        assertTrue(session.isClosed());
        session.close();
        assertThrows(IllegalStateException.class,
                () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
        engine.close();

        assertEquals(4.0f, retained.objective().bytes().getFloat(), 0.0f);
        assertEquals(1.6f, retainedState.parameters().getFirst().parameterBytes().getFloat(), 1e-6f);
        assertFalse(retainedState.parameters().getFirst().parameterBytes().hasArray());
    }

    @Test
    void lowPrecisionSnapshotsPreserveLogicalBitsFloatMastersSignedZeroAndSubnormals() {
        for (DataType dataType : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            short negativeZero = (short) 0x8000;
            short smallestSubnormal = 0x0001;
            Tensor parameter =
                    lowVector(dataType, new short[] {negativeZero, smallestSubnormal}, true);
            ScalarParameters module = new ScalarParameters(parameter);
            Tensor objective = parameter.cast(DataType.FLOAT32).sum();

            try (Engine engine = Engine.standard();
                    TrainingSession session =
                            TrainingSession.open(engine, module, objective, new Sgd(0.1d))) {
                TrainingState initial = session.state();
                TrainingState.ParameterState state = initial.parameters().getFirst();
                assertEquals(2 * Short.BYTES, state.parameterBytes().remaining());
                assertEquals(2 * Float.BYTES, state.masterParameterBytes().remaining());
                assertEquals(2 * Float.BYTES, state.momentumBytes().remaining());
                assertEquals(2 * Float.BYTES, state.accumulatedGradientBytes().remaining());

                ByteBuffer logical = state.parameterBytes();
                assertEquals(negativeZero, logical.getShort());
                assertEquals(smallestSubnormal, logical.getShort());
                ByteBuffer master = state.masterParameterBytes();
                assertEquals(
                        Float.floatToRawIntBits(-0.0f),
                        Float.floatToRawIntBits(master.getFloat()));
                assertEquals(decodeLow(dataType, smallestSubnormal), master.getFloat(), 0.0f);

                session.run(List.of(), GradientMode.RESET_AND_STEP);
                session.restore(initial);
                MemorySegment restored = parameter.hostStorage().orElseThrow().segment();
                assertEquals(negativeZero, restored.getAtIndex(ValueLayout.JAVA_SHORT, 0));
                assertEquals(smallestSubnormal, restored.getAtIndex(ValueLayout.JAVA_SHORT, 1));
                assertStateEquals(initial, session.state());
            }
        }
    }

    private static Tensor scalar(float value, boolean requiresGrad) {
        return scalar(Arena.global(), value, requiresGrad);
    }

    private static Tensor scalar(Arena arena, float value, boolean requiresGrad) {
        MemorySegment segment = arena.allocate(Float.BYTES, Float.BYTES);
        segment.set(ValueLayout.JAVA_FLOAT, 0, value);
        return tensor(segment, requiresGrad);
    }

    private static Tensor tensor(MemorySegment segment, boolean requiresGrad) {
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegmentStorage storage =
                new MemorySegmentStorage(DataType.FLOAT32, 1, segment);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }

    private static Tensor lowVector(
            DataType dataType, short[] rawValues, boolean requiresGrad) {
        Shape shape = Shape.of(rawValues.length);
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegment segment = Arena.global().allocate(
                Math.multiplyExact((long) rawValues.length, Short.BYTES),
                Short.BYTES);
        for (int index = 0; index < rawValues.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_SHORT, index, rawValues[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        dataType, rawValues.length, segment)));
    }

    private static float decodeLow(DataType dataType, short bits) {
        return dataType == DataType.BFLOAT16
                ? BFloat16Bits.toFloat(bits)
                : Float16Bits.toFloat(bits);
    }

    private static Tensor scalar64(double value, boolean requiresGrad) {
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT64,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegment segment = Arena.global().allocate(Double.BYTES, Double.BYTES);
        segment.set(ValueLayout.JAVA_DOUBLE, 0, value);
        MemorySegmentStorage storage =
                new MemorySegmentStorage(DataType.FLOAT64, 1, segment);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }

    private static float value(Parameter parameter) {
        return parameter.value().hostStorage().orElseThrow().segment()
                .get(ValueLayout.JAVA_FLOAT, 0);
    }

    private static byte[] bytes(ByteBuffer buffer) {
        byte[] result = new byte[buffer.remaining()];
        buffer.get(result);
        return result;
    }

    private static void assertStateEquals(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int index = 0; index < expected.parameters().size(); index++) {
            TrainingState.ParameterState left = expected.parameters().get(index);
            TrainingState.ParameterState right = actual.parameters().get(index);
            assertEquals(left.path(), right.path());
            assertEquals(left.dataType(), right.dataType());
            assertEquals(left.shape(), right.shape());
            assertArrayEquals(bytes(left.parameterBytes()), bytes(right.parameterBytes()));
            assertArrayEquals(
                    bytes(left.masterParameterBytes()),
                    bytes(right.masterParameterBytes()));
            assertArrayEquals(bytes(left.momentumBytes()), bytes(right.momentumBytes()));
            assertArrayEquals(
                    bytes(left.accumulatedGradientBytes()),
                    bytes(right.accumulatedGradientBytes()));
        }
    }

    private static final class ScalarParameters extends Module {
        private final Parameter first;
        private final Parameter second;

        private ScalarParameters(float first) {
            this(scalar(first, true), null);
        }

        private ScalarParameters(float first, float second) {
            this(scalar(first, true), scalar(second, true));
        }

        private ScalarParameters(Tensor first) {
            this(first, null);
        }

        private ScalarParameters(Tensor first, Tensor second) {
            this.first = parameter("first", first);
            this.second = second == null ? null : parameter("second", second);
        }

        private Parameter first() {
            return first;
        }

        private Parameter second() {
            if (second == null) {
                throw new IllegalStateException("second parameter is absent");
            }
            return second;
        }
    }
}
