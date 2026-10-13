package io.github.pho001.synaptik.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
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
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ReadOnlyBufferException;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class AdamOptimizerTest {
    @Test
    void bothAlgorithmsFollowIndependentMultiStepOracleAndAllGradientModes() {
        for (boolean decoupled : List.of(false, true)) {
            Optimizer optimizer = config(decoupled, 0.1d, 0.5d, 0.25d, 0.01d, 0.2d);
            Tensor parameter = scalar(DataType.FLOAT64, 2.0d);
            One module = new One(parameter);
            try (Engine engine = Engine.standard();
                    TrainingSession session = TrainingSession.open(
                            engine, module, parameter.mul(parameter), optimizer)) {
                assertEquals(optimizer, session.state().optimizer());
                Oracle oracle = new Oracle(2.0d, decoupled);
                session.run(List.of(), GradientMode.RESET_AND_STEP);
                oracle.step(4.0d);
                assertOracle(oracle, session.state(), 1e-12d);

                double currentGradient = 2.0d * oracle.parameter;
                session.run(List.of(), GradientMode.ACCUMULATE);
                session.run(List.of(), GradientMode.ACCUMULATE);
                assertEquals(1, session.optimizerSteps());
                assertEquals(2, session.accumulatedGradientRuns());
                assertOracle(oracle, session.state(), 1e-12d);
                session.zeroGrad();
                assertEquals(0, session.accumulatedGradientRuns());
                session.run(List.of(), GradientMode.ACCUMULATE);
                session.run(List.of(), GradientMode.ACCUMULATE_AND_STEP);
                oracle.step(2.0d * currentGradient);
                assertOracle(oracle, session.state(), 1e-12d);
                assertEquals(2, session.optimizerSteps());
                assertEquals(0, session.accumulatedGradientRuns());

                session.run(List.of(), GradientMode.RESET_AND_STEP);
                oracle.step(2.0d * oracle.parameter);
                assertOracle(oracle, session.state(), 1e-12d);
                assertEquals(3, session.optimizerSteps());
            }
        }
    }

    @Test
    void adaptiveSlotsAndFinalLowNarrowingCoverEveryParameterDtype() throws Exception {
        for (DataType dtype : List.of(DataType.FLOAT64, DataType.FLOAT32,
                DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor parameter = scalar(dtype, 2.0d);
            Tensor objective = dtype == DataType.BFLOAT16 || dtype == DataType.FLOAT16
                    ? parameter.cast(DataType.FLOAT32).sum() : parameter.mul(parameter);
            try (Engine engine = Engine.standard();
                    TrainingSession session = TrainingSession.open(engine, new One(parameter),
                            objective, new AdamW(0.1d, 0.5d, 0.25d, 0.01d, 0.2d))) {
                TrainingParameter binding = binding(session);
                String width = dtype == DataType.FLOAT64 ? "64" : "32";
                assertTrue(field(binding, "secondMoment" + width) != null);
                assertNull(field(binding, "secondMomentCandidate" + width));
                TrainingState before = session.state();
                var entry = before.parameters().getFirst();
                assertEquals(dtype == DataType.FLOAT64 ? 8 : 4,
                        entry.secondMomentBytes().orElseThrow().remaining());
                assertArrayEquals(bytes(entry.momentumBytes()), bytes(entry.firstMomentBytes()));
                assertTrue(entry.secondMomentBytes().orElseThrow().isReadOnly());
                assertThrows(ReadOnlyBufferException.class,
                        () -> entry.secondMomentBytes().orElseThrow().put(0, (byte) 1));
                session.run(List.of(), GradientMode.RESET_AND_STEP);
                assertTrue(field(binding, "secondMomentCandidate" + width) != null);
                TrainingState stepped = session.state();
                var result = stepped.parameters().getFirst();
                assertEquals(dtype == DataType.FLOAT64 ? 8 : 4,
                        result.secondMomentBytes().orElseThrow().remaining());
                if (dtype == DataType.BFLOAT16 || dtype == DataType.FLOAT16) {
                    float master = result.masterParameterBytes().getFloat();
                    short expected = dtype == DataType.BFLOAT16
                            ? BFloat16Bits.fromFloat(master) : Float16Bits.fromFloat(master);
                    assertEquals(expected, result.parameterBytes().getShort());
                }
                session.restore(before);
                assertStateBytes(before, session.state());
            }
        }
    }

    @Test
    void float32AdaptiveArithmeticTracksOracleWithRepresentedCoefficients() {
        for (boolean decoupled : List.of(false, true)) {
            Tensor parameter = scalar(DataType.FLOAT32, 2.0d);
            Optimizer optimizer = config(decoupled, 0.1d, 0.5d, 0.25d, 0.01d, 0.2d);
            try (Engine engine = Engine.standard();
                    TrainingSession session = TrainingSession.open(engine, new One(parameter),
                            parameter.mul(parameter), optimizer)) {
                Oracle oracle = new Oracle(2.0d, decoupled);
                for (int step = 0; step < 3; step++) {
                    double represented = session.state().parameters().getFirst()
                            .parameterBytes().getFloat();
                    session.run(List.of(), GradientMode.RESET_AND_STEP);
                    oracle.parameter = represented;
                    oracle.step(2.0d * represented);
                    var state = session.state().parameters().getFirst();
                    assertEquals(oracle.parameter, state.parameterBytes().getFloat(), 2e-6d);
                    assertEquals(oracle.first, state.firstMomentBytes().getFloat(), 2e-6d);
                    assertEquals(oracle.second,
                            state.secondMomentBytes().orElseThrow().getFloat(), 4e-6d);
                }
            }
        }
    }

    @Test
    void mixedAssignmentRestoreIdentityAndLaterFailureAreAtomic() {
        class Three extends Module {
            final Parameter a = parameter("a", scalar(DataType.FLOAT32, 2.0d));
            final Parameter b = parameter("b", scalar(DataType.FLOAT32, 3.0d));
            final Parameter c = parameter("c", scalar(DataType.FLOAT32, 4.0d));
        }
        Three module = new Three();
        Tensor objective = module.a.value().add(module.b.value()).add(module.c.value());
        Adam defaultAdam = new Adam(0.1d, 0.5d, 0.25d, 0.01d, 0.1d);
        List<ParameterGroup> groups = List.of(
                new ParameterGroup(List.of("b"), new Sgd(0.2d)),
                new ParameterGroup(List.of("c"), new AdamW(0.1d, 0.5d, 0.25d, 0.01d, 0.1d)));
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(
                        engine, module, objective, defaultAdam, groups)) {
            TrainingState initial = session.state();
            assertEquals(defaultAdam, initial.optimizer());
            assertTrue(initial.parameters().get(0).secondMomentBytes().isPresent());
            assertTrue(initial.parameters().get(1).secondMomentBytes().isEmpty());
            assertTrue(initial.parameters().get(2).secondMomentBytes().isPresent());
            session.run(List.of(), GradientMode.RESET_AND_STEP);
            TrainingState stepped = session.state();
            assertEquals(1, stepped.optimizerSteps());
            assertEquals(2.8f, stepped.parameters().get(1).parameterBytes().getFloat(), 1e-6f);
            session.restore(initial);
            assertStateBytes(initial, session.state());
            session.run(List.of(), GradientMode.RESET_AND_STEP);
            assertStateBytes(stepped, session.state());
            for (List<ParameterGroup> mismatched : List.of(
                    List.of(groups.get(1), groups.get(0)),
                    List.of(new ParameterGroup(List.of("b"), new Adam(0.2d, 0.5d, 0.25d,
                            0.01d, 0.1d)), groups.get(1)),
                    List.of(groups.get(0), new ParameterGroup(List.of("c"),
                            new AdamW(0.1d, 0.5d, 0.25d, 0.01d, 0.2d))))) {
                assertThrows(IllegalArgumentException.class, () -> session.restore(new TrainingState(
                        stepped.optimizer(), mismatched, stepped.executions(),
                        stepped.optimizerSteps(), stepped.accumulatedGradientRuns(),
                        stepped.parameters())));
                assertStateBytes(stepped, session.state());
            }
            assertThrows(IllegalArgumentException.class, () -> session.restore(new TrainingState(
                    new AdamW(0.1d, 0.5d, 0.25d, 0.01d, 0.1d), groups,
                    stepped.executions(), stepped.optimizerSteps(),
                    stepped.accumulatedGradientRuns(), stepped.parameters())));
            assertStateBytes(stepped, session.state());
        }

        Three failingModule = new Three();
        Tensor failingObjective = failingModule.a.value().add(failingModule.b.value())
                .add(failingModule.c.value());
        TrainingSession.Hooks failure = new TrainingSession.Hooks() {
            @Override void beforeParameterWrite(int index) {
                if (index == 2) throw new IllegalStateException("injected later write failure");
            }
        };
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.openForTesting(engine, failingModule,
                        failingObjective, defaultAdam, groups, failure)) {
            session.run(List.of(), GradientMode.ACCUMULATE);
            TrainingState before = session.state();
            assertThrows(IllegalStateException.class,
                    () -> session.run(List.of(), GradientMode.ACCUMULATE_AND_STEP));
            assertStateBytes(before, session.state());
        }
    }

    @Test
    void restoreRejectsMissingExtraNegativeAndNonFiniteSecondMomentsBeforeMutation() {
        Tensor parameter = scalar(DataType.FLOAT32, 2.0d);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(engine, new One(parameter),
                        parameter.mul(parameter), new Adam(0.1d, 0.5d, 0.25d, 0.01d, 0.1d))) {
            TrainingState before = session.state();
            var state = before.parameters().getFirst();
            for (byte[] second : List.of(
                    ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putFloat(-1.0f).array(),
                    ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putFloat(Float.NaN).array(),
                    ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN)
                            .putFloat(Float.POSITIVE_INFINITY).array())) {
                var candidate = new TrainingState.ParameterState(state.path(), state.dataType(),
                        state.shape(), bytes(state.parameterBytes()),
                        bytes(state.masterParameterBytes()), bytes(state.firstMomentBytes()),
                        second, bytes(state.accumulatedGradientBytes()));
                TrainingState invalid = new TrainingState(before.optimizer(), before.executions(),
                        before.optimizerSteps(), before.accumulatedGradientRuns(), List.of(candidate));
                assertThrows(IllegalArgumentException.class, () -> session.restore(invalid));
                assertStateBytes(before, session.state());
            }
            var missing = new TrainingState.ParameterState(state.path(), state.dataType(),
                    state.shape(), bytes(state.parameterBytes()), bytes(state.masterParameterBytes()),
                    bytes(state.firstMomentBytes()), bytes(state.accumulatedGradientBytes()));
            assertThrows(IllegalArgumentException.class, () -> session.restore(new TrainingState(
                    before.optimizer(), 0, 0, 0, List.of(missing))));
            assertThrows(IllegalArgumentException.class, () -> new TrainingState.ParameterState(
                    state.path(), state.dataType(), state.shape(), bytes(state.parameterBytes()),
                    bytes(state.masterParameterBytes()), bytes(state.firstMomentBytes()),
                    new byte[3], bytes(state.accumulatedGradientBytes())));
            assertStateBytes(before, session.state());
        }
        Tensor sgdParameter = scalar(DataType.FLOAT32, 2.0d);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(engine, new One(sgdParameter),
                        sgdParameter.mul(sgdParameter), new Sgd(0.1d))) {
            TrainingState before = session.state();
            var entry = before.parameters().getFirst();
            assertTrue(entry.secondMomentBytes().isEmpty());
            var extra = new TrainingState.ParameterState(entry.path(), entry.dataType(),
                    entry.shape(), bytes(entry.parameterBytes()), bytes(entry.masterParameterBytes()),
                    bytes(entry.firstMomentBytes()), new byte[Float.BYTES],
                    bytes(entry.accumulatedGradientBytes()));
            assertThrows(IllegalArgumentException.class, () -> session.restore(new TrainingState(
                    before.optimizer(), 0, 0, 0, List.of(extra))));
            assertStateBytes(before, session.state());
        }
    }

    @Test
    void adaptiveSnapshotRoundTripsSignedZeroSlotsAndRemainsDetached() {
        Tensor parameter = scalar(DataType.FLOAT64, 2.0d);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(engine, new One(parameter),
                        parameter.mul(parameter), new AdamW(0.1d, 0.5d, 0.25d, 0.01d, 0.0d))) {
            TrainingState initial = session.state();
            var entry = initial.parameters().getFirst();
            byte[] negativeZero = ByteBuffer.allocate(Double.BYTES).order(ByteOrder.BIG_ENDIAN)
                    .putDouble(-0.0d).array();
            var signed = new TrainingState.ParameterState(entry.path(), entry.dataType(),
                    entry.shape(), bytes(entry.parameterBytes()), bytes(entry.masterParameterBytes()),
                    negativeZero, negativeZero, bytes(entry.accumulatedGradientBytes()));
            TrainingState toRestore = new TrainingState(initial.optimizer(), 0, 0, 0,
                    List.of(signed));
            session.restore(toRestore);
            TrainingState restored = session.state();
            assertArrayEquals(negativeZero,
                    bytes(restored.parameters().getFirst().firstMomentBytes()));
            assertArrayEquals(negativeZero,
                    bytes(restored.parameters().getFirst().secondMomentBytes().orElseThrow()));
            assertArrayEquals(negativeZero,
                    bytes(restored.parameters().getFirst().momentumBytes()));
            assertFalse(restored.parameters().getFirst().firstMomentBytes().hasArray());
            assertFalse(restored.parameters().getFirst().secondMomentBytes().orElseThrow().hasArray());
            session.run(List.of(), GradientMode.RESET_AND_STEP);
            assertArrayEquals(negativeZero,
                    bytes(restored.parameters().getFirst().secondMomentBytes().orElseThrow()));
            session.restore(restored);
            assertStateBytes(restored, session.state());
        }
    }

    @Test
    void invalidCoefficientsAndOverflowRejectWithoutMutation() {
        assertFalse(new Adam(0.1d, 0.5d, 0.5d, 0.01d, 0.0d)
                .equals(new AdamW(0.1d, 0.5d, 0.5d, 0.01d, 0.0d)));
        assertFalse(new Adam(0.1d, 0.5d, 0.5d, 0.01d, -0.0d)
                .equals(new Adam(0.1d, 0.5d, 0.5d, 0.01d, 0.0d)));
        for (double invalid : List.of(Double.NaN, Double.POSITIVE_INFINITY, 0.0d, -1.0d)) {
            assertThrows(IllegalArgumentException.class,
                    () -> new Adam(invalid, 0.5d, 0.5d, 0.01d, 0.0d));
            assertThrows(IllegalArgumentException.class,
                    () -> new AdamW(0.1d, 0.5d, 0.5d, invalid, 0.0d));
        }
        assertThrows(IllegalArgumentException.class,
                () -> new Adam(0.1d, 1.0d, 0.5d, 0.01d, 0.0d));
        assertThrows(IllegalArgumentException.class,
                () -> new AdamW(0.1d, 0.5d, -0.1d, 0.01d, 0.0d));
        Tensor parameter = scalar(DataType.FLOAT32, 2.0d);
        try (Engine engine = Engine.standard()) {
            for (Optimizer invalid : List.of(
                    new Adam(Double.MIN_VALUE, 0.5d, 0.5d, 0.01d, 0.0d),
                    new AdamW(0.1d, 0.5d, 0.5d, Double.MIN_VALUE, 0.0d),
                    new Adam(0.1d, Math.nextDown(1.0d), 0.5d, 0.01d, 0.0d),
                    new AdamW(0.1d, 0.5d, Math.nextDown(1.0d), 0.01d, 0.0d),
                    new Adam(0.1d, 0.5d, 0.5d, 0.01d, Double.MAX_VALUE))) {
                assertThrows(IllegalArgumentException.class, () -> TrainingSession.open(engine,
                        new One(parameter), parameter.mul(parameter), invalid));
            }
            try (TrainingSession session = TrainingSession.open(engine, new One(parameter),
                    parameter.mul(parameter), new Adam(0.1d, 0.5d, 0.5d, 0.01d,
                            Float.MAX_VALUE))) {
                TrainingState before = session.state();
                assertThrows(ArithmeticException.class,
                        () -> session.run(List.of(), GradientMode.RESET_AND_STEP));
                assertStateBytes(before, session.state());
            }
        }
        Tensor squaredOverflow = scalar(DataType.FLOAT32, 1e20d);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(engine, new One(squaredOverflow),
                        squaredOverflow.mul(squaredOverflow),
                        new AdamW(0.1d, 0.5d, 0.5d, 0.01d, 0.0d))) {
            TrainingState before = session.state();
            assertThrows(ArithmeticException.class,
                    () -> session.run(List.of(), GradientMode.RESET_AND_STEP));
            assertStateBytes(before, session.state());
        }
        Tensor lowOverflow = scalar(DataType.FLOAT16, 65504.0d);
        try (Engine engine = Engine.standard();
                TrainingSession session = TrainingSession.open(engine, new One(lowOverflow),
                        lowOverflow.cast(DataType.FLOAT32).neg().sum(),
                        new AdamW(1000.0d, 0.0d, 0.0d, 1e-8d, 0.0d))) {
            TrainingState before = session.state();
            assertThrows(ArithmeticException.class,
                    () -> session.run(List.of(), GradientMode.RESET_AND_STEP));
            assertStateBytes(before, session.state());
        }
    }

    private static Optimizer config(boolean decoupled, double lr, double b1, double b2,
            double eps, double decay) {
        return decoupled ? new AdamW(lr, b1, b2, eps, decay)
                : new Adam(lr, b1, b2, eps, decay);
    }

    private static void assertOracle(Oracle oracle, TrainingState actual, double tolerance) {
        TrainingState.ParameterState state = actual.parameters().getFirst();
        assertEquals(oracle.parameter, state.parameterBytes().getDouble(), tolerance);
        assertEquals(oracle.first, state.firstMomentBytes().getDouble(), tolerance);
        assertEquals(oracle.first, state.momentumBytes().getDouble(), tolerance);
        assertEquals(oracle.second, state.secondMomentBytes().orElseThrow().getDouble(), tolerance);
    }

    private static final class Oracle {
        double parameter;
        double first;
        double second;
        int step;
        final boolean decoupled;

        Oracle(double parameter, boolean decoupled) {
            this.parameter = parameter;
            this.decoupled = decoupled;
        }

        void step(double publishedGradient) {
            step++;
            double gradient = decoupled ? publishedGradient
                    : publishedGradient + 0.2d * parameter;
            first = 0.5d * first + 0.5d * gradient;
            second = 0.25d * second + 0.75d * gradient * gradient;
            double correctedFirst = first / (1.0d - Math.pow(0.5d, step));
            double correctedSecond = second / (1.0d - Math.pow(0.25d, step));
            double base = decoupled ? parameter * (1.0d - 0.1d * 0.2d) : parameter;
            parameter = base - 0.1d * correctedFirst / (Math.sqrt(correctedSecond) + 0.01d);
        }
    }

    private static Tensor scalar(DataType dtype, double value) {
        Shape shape = Shape.scalar();
        int width = dtype.byteWidth();
        var segment = Arena.global().allocate(width, width);
        if (dtype == DataType.FLOAT64) {
            segment.set(ValueLayout.JAVA_DOUBLE, 0, value);
        } else if (dtype == DataType.FLOAT32) {
            segment.set(ValueLayout.JAVA_FLOAT, 0, (float) value);
        } else {
            short bits = dtype == DataType.BFLOAT16
                    ? BFloat16Bits.fromFloat((float) value) : Float16Bits.fromFloat((float) value);
            segment.set(ValueLayout.JAVA_SHORT, 0, bits);
        }
        return TensorFactory.create(new TensorDescriptor(dtype, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), true), Optional.empty(),
                Optional.of(new MemorySegmentStorage(dtype, 1, segment)));
    }

    private static TrainingParameter binding(TrainingSession session) throws Exception {
        Field field = TrainingSession.class.getDeclaredField("parameters");
        field.setAccessible(true);
        return (TrainingParameter) ((List<?>) field.get(session)).getFirst();
    }

    private static Object field(TrainingParameter binding, String name) throws Exception {
        Field field = TrainingParameter.class.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(binding);
    }

    private static void assertStateBytes(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.parameterGroups(), actual.parameterGroups());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int index = 0; index < expected.parameters().size(); index++) {
            var left = expected.parameters().get(index);
            var right = actual.parameters().get(index);
            assertArrayEquals(bytes(left.parameterBytes()), bytes(right.parameterBytes()));
            assertArrayEquals(bytes(left.masterParameterBytes()), bytes(right.masterParameterBytes()));
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

    private static byte[] bytes(ByteBuffer view) {
        byte[] bytes = new byte[view.remaining()];
        view.get(bytes);
        return bytes;
    }

    private static final class One extends Module {
        One(Tensor value) { parameter("first", value); }
    }
}
