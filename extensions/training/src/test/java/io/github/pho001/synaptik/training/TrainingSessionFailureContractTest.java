package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
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
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

final class TrainingSessionFailureContractTest {
    @Test
    void rejectsZeroElementParameterBeforeCompilationWithoutExternalMutation() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Shape shape = Shape.of(0);
            TensorDescriptor descriptor = new TensorDescriptor(
                    DataType.FLOAT32,
                    shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)),
                    true);
            MemorySegmentStorage storage = new MemorySegmentStorage(
                    DataType.FLOAT32, 0, arena.allocate(0, 1));
            Tensor empty = TensorFactory.create(
                    descriptor, Optional.empty(), Optional.of(storage));
            NamedParameters module = new NamedParameters("empty", empty);

            IllegalArgumentException rejected = assertThrows(
                    IllegalArgumentException.class,
                    () -> TrainingSession.open(
                            engine, module, empty.sum(), new Sgd(0.1d)));

            assertTrue(rejected.getMessage().contains("at least one element"));
            assertSame(empty, module.first().value());
            assertSame(storage, empty.hostStorage().orElseThrow());
            assertEquals(0, storage.byteSize());
            assertTrue(storage.isAlive());
            assertFalse(engine.isClosed());
        }
    }

    @Test
    void lifecycleAdmissionPrecedesInvalidArgumentsAndEngineClosureWins() {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor target = scalar32(0.0f, false);
        Engine engine = Engine.standard();
        TrainingSession session = TrainingSession.open(
                engine, module, squaredDifference(module.first().value(), target), new Sgd(0.1d));

        session.close();
        assertEquals("training session is closed", assertThrows(
                IllegalStateException.class, () -> session.run(null, null)).getMessage());
        assertEquals("training session is closed", assertThrows(
                IllegalStateException.class,
                () -> session.run(List.of(target), null)).getMessage());
        assertEquals("training session is closed", assertThrows(
                IllegalStateException.class, () -> session.restore(null)).getMessage());

        engine.close();
        assertEquals("training engine is closed", assertThrows(
                IllegalStateException.class, () -> session.run(null, null)).getMessage());
        assertEquals("training engine is closed", assertThrows(
                IllegalStateException.class,
                () -> session.run(List.of(target), null)).getMessage());
        assertEquals("training engine is closed", assertThrows(
                IllegalStateException.class, () -> session.restore(null)).getMessage());
    }

    @Test
    void actualBusyAdmissionPrecedesInvalidArguments() throws Exception {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor target = scalar32(0.0f, false);
        CountDownLatch admitted = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicBoolean blockFirstRun = new AtomicBoolean();
        TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
            @Override
            void afterAdmission(TrainingSession.Operation operation) {
                if (operation == TrainingSession.Operation.RUN
                        && blockFirstRun.compareAndSet(false, true)) {
                    admitted.countDown();
                    await(release);
                }
            }
        };

        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.openForTesting(
                        engine,
                        module,
                        squaredDifference(module.first().value(), target),
                        new Sgd(0.1d),
                        hooks);
                var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var first = executor.submit(
                    () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
            assertTrue(admitted.await(10, TimeUnit.SECONDS));
            try {
                assertEquals("training session already has an operation in flight", assertThrows(
                        IllegalStateException.class,
                        () -> session.run(List.of(target), null)).getMessage());
                assertEquals("training session already has an operation in flight", assertThrows(
                        IllegalStateException.class, () -> session.run(null, null)).getMessage());
            } finally {
                release.countDown();
            }
            assertEquals(1, first.get(10, TimeUnit.SECONDS).executionNumber());
            assertEquals(1, session.executions());
        }
    }

    @Test
    void closeWaitsForAdmittedStepWhichCompletesBeforeCleanup() throws Exception {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor target = scalar32(0.0f, false);
        CountDownLatch admitted = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
            @Override
            void afterAdmission(TrainingSession.Operation operation) {
                if (operation == TrainingSession.Operation.RUN) {
                    admitted.countDown();
                    await(release);
                }
            }
        };

        Engine engine = Engine.standard();
        TrainingSession session = TrainingSession.openForTesting(
                engine,
                module,
                squaredDifference(module.first().value(), target),
                new Sgd(0.1d),
                hooks);
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var run = executor.submit(
                    () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
            assertTrue(admitted.await(10, TimeUnit.SECONDS));
            var close = executor.submit(() -> {
                session.close();
                return null;
            });
            while (!session.isClosed()) Thread.onSpinWait();
            assertFalse(close.isDone());
            assertEquals(0, session.executions());
            assertEquals(2.0f, value32(module.first()), 0.0f);
            release.countDown();
            assertEquals(1, run.get(10, TimeUnit.SECONDS).executionNumber());
            close.get(10, TimeUnit.SECONDS);
        } finally {
            release.countDown();
            session.close();
            engine.close();
        }
        assertEquals(1, session.executions());
        assertEquals(1.6f, value32(module.first()), 1e-6f);
    }

    @Test
    void postMaterializationAndPostExecutionFailuresAreAtomic() {
        assertInjectedRunFailureIsAtomic(new TrainingSession.Hooks() {
            @Override
            void afterObjectiveMaterialization() {
                throw new InjectedFailure("materialization phase");
            }
        }, "materialization phase");
        assertInjectedRunFailureIsAtomic(new TrainingSession.Hooks() {
            @Override
            void afterExecution() {
                throw new InjectedFailure("post execution");
            }
        }, "post execution");
    }

    @Test
    void lateBindingValidationFailureAfterExecutionIsAtomic() {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor original = module.first().value();
        Tensor replacement = scalar32(2.0f, true);
        Tensor target = scalar32(0.0f, false);
        AtomicBoolean replaceOnce = new AtomicBoolean();
        TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
            @Override
            void afterExecution() {
                if (replaceOnce.compareAndSet(false, true)) {
                    module.first().replace(replacement);
                }
            }
        };

        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.openForTesting(
                        engine,
                        module,
                        squaredDifference(original, target),
                        new Sgd(0.1d),
                        hooks)) {
            TrainingState initial = session.state();
            assertThrows(IllegalStateException.class,
                    () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
            assertEquals(2.0f, original.hostStorage().orElseThrow().segment()
                    .get(ValueLayout.JAVA_FLOAT, 0), 0.0f);
            assertEquals(2.0f, replacement.hostStorage().orElseThrow().segment()
                    .get(ValueLayout.JAVA_FLOAT, 0), 0.0f);
            assertEquals(0, session.executions());
            assertEquals(0, session.optimizerSteps());
            module.first().replace(original);
            assertStateEquals(initial, session.state());
        }
    }

    @Test
    void laterParameterWriteFailureRollsBackEarlierWriteAndAllState() {
        NamedParameters module = new NamedParameters(
                scalar32(2.0f, true), scalar32(-1.0f, true));
        Tensor firstInput = scalar32(3.0f, false);
        Tensor secondInput = scalar32(4.0f, false);
        Tensor objective = module.first().value().mul(firstInput)
                .add(module.second().value().mul(secondInput));
        InjectedFailure observed = new InjectedFailure("second parameter write");
        TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
            @Override
            void beforeParameterWrite(int parameterIndex) {
                if (parameterIndex == 1) throw observed;
            }
        };

        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.openForTesting(
                        engine, module, objective, new Sgd(0.1d, 0.5d, 0.0d, false), hooks)) {
            TrainingState initial = session.state();
            assertSame(observed, assertThrows(InjectedFailure.class,
                    () -> session.run(
                            List.of(firstInput, secondInput), GradientMode.RESET_AND_STEP)));
            assertEquals(2.0f, value32(module.first()), 0.0f);
            assertEquals(-1.0f, value32(module.second()), 0.0f);
            assertStateEquals(initial, session.state());
        }
    }

    @Test
    void retainedCloseFailureIsReplayedWhileAllCleanupIsAttempted() {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor target = scalar32(0.0f, false);
        InjectedFailure observed = new InjectedFailure("owned cleanup");
        TrainingSession.Hooks hooks = new TrainingSession.Hooks() {
            @Override
            void beforeOwnedResourceClose() {
                throw observed;
            }
        };
        Engine engine = Engine.standard();
        TrainingSession session = TrainingSession.openForTesting(
                engine,
                module,
                squaredDifference(module.first().value(), target),
                new Sgd(0.1d),
                hooks);
        try {
            assertSame(observed, assertThrows(InjectedFailure.class, session::close));
            assertTrue(session.isClosed());
            assertSame(observed, assertThrows(InjectedFailure.class, session::close));
        } finally {
            engine.close();
        }
    }

    @Test
    void restoreRejectsDifferentSessionSchemaAndDampeningConfiguration() {
        NamedParameters sourceModule = new NamedParameters("source", scalar32(2.0f, true));
        NamedParameters pathMismatch = new NamedParameters("other", scalar32(2.0f, true));
        NamedParameters optimizerMismatch = new NamedParameters("source", scalar32(2.0f, true));
        Tensor sourceTarget = scalar32(0.0f, false);
        Tensor pathTarget = scalar32(0.0f, false);
        Tensor optimizerTarget = scalar32(0.0f, false);
        Sgd sourceOptimizer = new Sgd(0.1d, 0.5d, 0.25d, 0.0d, false);

        try (Engine engine = Engine.standard();
                TrainingSession source = TrainingSession.open(
                        engine,
                        sourceModule,
                        squaredDifference(sourceModule.first().value(), sourceTarget),
                        sourceOptimizer);
                TrainingSession wrongPath = TrainingSession.open(
                        engine,
                        pathMismatch,
                        squaredDifference(pathMismatch.first().value(), pathTarget),
                        sourceOptimizer);
                TrainingSession wrongOptimizer = TrainingSession.open(
                        engine,
                        optimizerMismatch,
                        squaredDifference(optimizerMismatch.first().value(), optimizerTarget),
                        new Sgd(0.1d, 0.5d, 0.5d, 0.0d, false))) {
            source.run(List.of(sourceTarget), GradientMode.RESET_AND_STEP);
            TrainingState state = source.state();
            TrainingState wrongPathInitial = wrongPath.state();
            TrainingState wrongOptimizerInitial = wrongOptimizer.state();

            assertThrows(IllegalArgumentException.class, () -> wrongPath.restore(state));
            assertThrows(IllegalArgumentException.class, () -> wrongOptimizer.restore(state));
            assertStateEquals(wrongPathInitial, wrongPath.state());
            assertStateEquals(wrongOptimizerInitial, wrongOptimizer.state());
        }
    }

    @Test
    void counterOverflowFailsBeforeExecutionAndLeavesRestoredStateUnchanged() {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor target = scalar32(0.0f, false);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine,
                        module,
                        squaredDifference(module.first().value(), target),
                        new Sgd(0.1d))) {
            TrainingState base = session.state();
            TrainingState maximum = new TrainingState(
                    base.optimizer(), Long.MAX_VALUE, Long.MAX_VALUE, 0, base.parameters());
            session.restore(maximum);

            assertThrows(ArithmeticException.class,
                    () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
            assertStateEquals(maximum, session.state());
            assertEquals(2.0f, value32(module.first()), 0.0f);
        }
    }

    @Test
    void rejectsRawNonFiniteParametersAndAcceptsSignedZeroForBothPrecisions() {
        assertInitialParameterRejected(scalar32(Float.NaN, true));
        assertInitialParameterRejected(scalar32(Float.POSITIVE_INFINITY, true));
        assertInitialParameterRejected(scalar64(Double.NaN, true));
        assertInitialParameterRejected(scalar64(Double.NEGATIVE_INFINITY, true));

        Tensor floatZero = scalar32(-0.0f, true);
        NamedParameters floatModule = new NamedParameters("zero", floatZero);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, floatModule, floatZero.mul(floatZero), new Sgd(0.1d))) {
            session.run(List.of(), GradientMode.RESET_AND_STEP);
            assertTrue(Float.isFinite(value32(floatModule.first())));
            assertEquals(0.0f, value32(floatModule.first()), 0.0f);
        }

        Tensor doubleZero = scalar64(-0.0d, true);
        NamedParameters doubleModule = new NamedParameters("zero", doubleZero);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, doubleModule, doubleZero.mul(doubleZero), new Sgd(0.1d))) {
            session.run(List.of(), GradientMode.RESET_AND_STEP);
            double value = doubleZero.hostStorage().orElseThrow().segment()
                    .get(ValueLayout.JAVA_DOUBLE, 0);
            assertTrue(Double.isFinite(value));
            assertEquals(0.0d, value, 0.0d);
        }
    }

    @Test
    void decodedNonFiniteGradientsAndCandidatesFailAtomicallyForBothPrecisions() {
        assertGradientFailureIsAtomic(
                scalar32(1.0f, true), scalar32(Float.NaN, false), new Sgd(0.1d));
        assertGradientFailureIsAtomic(
                scalar64(1.0d, true), scalar64(Double.POSITIVE_INFINITY, false), new Sgd(0.1d));

        assertGradientFailureIsAtomic(
                scalar32(Float.MAX_VALUE, true), scalar32(-Float.MAX_VALUE, false), new Sgd(1.0d));
        assertGradientFailureIsAtomic(
                scalar64(Double.MAX_VALUE, true), scalar64(-Double.MAX_VALUE, false), new Sgd(1.0d));
    }

    @Test
    void rejectsCurrentParameterMadeNonFiniteAfterOpenWithoutPublishingState() {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor parameter = module.first().value();
        Tensor target = scalar32(0.0f, false);
        MemorySegment segment = parameter.hostStorage().orElseThrow().segment();
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine,
                        module,
                        squaredDifference(parameter, target),
                        new Sgd(0.1d, 0.5d, 0.0d, false))) {
            segment.set(ValueLayout.JAVA_FLOAT, 0, Float.NEGATIVE_INFINITY);
            assertThrows(IllegalStateException.class,
                    () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
            assertEquals(0, session.executions());
            assertEquals(0, session.optimizerSteps());
            assertEquals(Float.NEGATIVE_INFINITY,
                    segment.get(ValueLayout.JAVA_FLOAT, 0), 0.0f);
            segment.set(ValueLayout.JAVA_FLOAT, 0, 2.0f);
            TrainingState state = session.state();
            assertEquals(0.0f, state.parameters().getFirst().momentumBytes().getFloat(), 0.0f);
            assertEquals(0.0f,
                    state.parameters().getFirst().accumulatedGradientBytes().getFloat(), 0.0f);
        }
    }

    @Test
    void accumulationAndMomentumOverflowLeavePublishedStateAndCountersUnchanged() {
        assertAccumulationOverflowIsAtomic(
                scalar32(0.0f, true), scalar32(Float.MAX_VALUE, false));
        assertAccumulationOverflowIsAtomic(
                scalar64(0.0d, true), scalar64(Double.MAX_VALUE, false));

        NamedParameters module = new NamedParameters("value", scalar32(0.0f, true));
        Tensor input = scalar32(Float.MAX_VALUE, false);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine,
                        module,
                        module.first().value().mul(input),
                        new Sgd((double) Float.MIN_VALUE, 0.9d, 0.0d, false))) {
            session.run(List.of(input), GradientMode.RESET_AND_STEP);
            TrainingState beforeOverflow = session.state();
            assertThrows(ArithmeticException.class,
                    () -> session.run(List.of(input), GradientMode.RESET_AND_STEP));
            assertStateEquals(beforeOverflow, session.state());
            assertEquals(1, session.executions());
            assertEquals(1, session.optimizerSteps());
        }
    }

    @Test
    void restoreRejectsNonFiniteMomentumAndAccumulationPayloadsAtomically() {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor target = scalar32(0.0f, false);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine,
                        module,
                        squaredDifference(module.first().value(), target),
                        new Sgd(0.1d, 0.5d, 0.0d, false))) {
            TrainingState initial = session.state();
            TrainingState.ParameterState parameter = initial.parameters().getFirst();
            TrainingState.ParameterState badMomentum = new TrainingState.ParameterState(
                    parameter.path(),
                    parameter.dataType(),
                    parameter.shape(),
                    bytes(parameter.parameterBytes()),
                    floatBytes(Float.NaN),
                    bytes(parameter.accumulatedGradientBytes()));
            TrainingState corruptMomentum = new TrainingState(
                    initial.optimizer(), 0, 0, 0, List.of(badMomentum));
            assertThrows(IllegalArgumentException.class,
                    () -> session.restore(corruptMomentum));
            assertStateEquals(initial, session.state());

            TrainingState.ParameterState badAccumulation = new TrainingState.ParameterState(
                    parameter.path(),
                    parameter.dataType(),
                    parameter.shape(),
                    bytes(parameter.parameterBytes()),
                    bytes(parameter.momentumBytes()),
                    floatBytes(Float.POSITIVE_INFINITY));
            TrainingState corruptAccumulation = new TrainingState(
                    initial.optimizer(), 1, 0, 1, List.of(badAccumulation));
            assertThrows(IllegalArgumentException.class,
                    () -> session.restore(corruptAccumulation));
            assertStateEquals(initial, session.state());
        }
    }

    private static void assertInjectedRunFailureIsAtomic(
            TrainingSession.Hooks hooks, String message) {
        NamedParameters module = new NamedParameters("value", scalar32(2.0f, true));
        Tensor target = scalar32(0.0f, false);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.openForTesting(
                        engine,
                        module,
                        squaredDifference(module.first().value(), target),
                        new Sgd(0.1d, 0.5d, 0.0d, false),
                        hooks)) {
            TrainingState initial = session.state();
            InjectedFailure failure = assertThrows(InjectedFailure.class,
                    () -> session.run(List.of(target), GradientMode.RESET_AND_STEP));
            assertEquals(message, failure.getMessage());
            assertStateEquals(initial, session.state());
            assertEquals(2.0f, value32(module.first()), 0.0f);
        }
    }

    private static void assertInitialParameterRejected(Tensor parameter) {
        NamedParameters module = new NamedParameters("value", parameter);
        try (Engine engine = Engine.standard()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(
                            engine, module, parameter.mul(parameter), new Sgd(0.1d)));
            assertFalse(engine.isClosed());
        }
    }

    private static void assertGradientFailureIsAtomic(
            Tensor parameter, Tensor input, Sgd optimizer) {
        NamedParameters module = new NamedParameters("value", parameter);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, parameter.mul(input), optimizer)) {
            TrainingState initial = session.state();
            assertThrows(ArithmeticException.class,
                    () -> session.run(List.of(input), GradientMode.RESET_AND_STEP));
            assertStateEquals(initial, session.state());
            assertEquals(0, session.executions());
            assertEquals(0, session.optimizerSteps());
        }
    }

    private static void assertAccumulationOverflowIsAtomic(Tensor parameter, Tensor input) {
        NamedParameters module = new NamedParameters("value", parameter);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, parameter.mul(input), new Sgd(0.1d))) {
            session.run(List.of(input), GradientMode.ACCUMULATE);
            TrainingState beforeOverflow = session.state();
            assertThrows(ArithmeticException.class,
                    () -> session.run(List.of(input), GradientMode.ACCUMULATE));
            assertStateEquals(beforeOverflow, session.state());
            assertEquals(1, session.executions());
            assertEquals(1, session.accumulatedGradientRuns());
        }
    }

    private static Tensor squaredDifference(Tensor parameter, Tensor target) {
        Tensor difference = parameter.sub(target);
        return difference.mul(difference);
    }

    private static Tensor scalar32(float value, boolean requiresGrad) {
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad);
        MemorySegment segment = Arena.global().allocate(Float.BYTES, Float.BYTES);
        segment.set(ValueLayout.JAVA_FLOAT, 0, value);
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, segment)));
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
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT64, 1, segment)));
    }

    private static float value32(Parameter parameter) {
        return parameter.value().hostStorage().orElseThrow().segment()
                .get(ValueLayout.JAVA_FLOAT, 0);
    }

    private static byte[] floatBytes(float value) {
        return ByteBuffer.allocate(Float.BYTES).order(ByteOrder.BIG_ENDIAN).putFloat(value).array();
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
            assertArrayEquals(bytes(left.momentumBytes()), bytes(right.momentumBytes()));
            assertArrayEquals(
                    bytes(left.accumulatedGradientBytes()),
                    bytes(right.accumulatedGradientBytes()));
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(10, TimeUnit.SECONDS)) {
                throw new AssertionError("timed out waiting for deterministic test release");
            }
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new AssertionError(failure);
        }
    }

    private static final class NamedParameters extends Module {
        private final Parameter first;
        private final Parameter second;

        private NamedParameters(String name, Tensor value) {
            first = parameter(name, value);
            second = null;
        }

        private NamedParameters(Tensor first, Tensor second) {
            this.first = parameter("first", first);
            this.second = parameter("second", second);
        }

        private Parameter first() {
            return first;
        }

        private Parameter second() {
            if (second == null) throw new IllegalStateException("second parameter is absent");
            return second;
        }
    }

    private static final class InjectedFailure extends RuntimeException {
        private InjectedFailure(String message) {
            super(message);
        }
    }
}
