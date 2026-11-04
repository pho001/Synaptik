package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.Float8E4M3Bits;
import io.github.pho001.synaptik.model.datatype.Float8E5M2Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.training.GradientMode;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingParameterStorage;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Public CPU low-carrier compute, shared Compiler AD, training and cold-value lifecycles.
 * Dyadic equations and independent format decoding check values; conversion helpers only encode
 * fixtures. No native availability, internal plan or route is part of these assertions.
 */
final class CpuLowPrecisionWorkflowIntegrationTest {
    @TempDir Path directory;

    @ParameterizedTest
    @EnumSource(value = DataType.class,
            names = {"BFLOAT16", "FLOAT16", "FLOAT8_E4M3", "FLOAT8_E5M2"})
    void scalarModelComputesAndDifferentiatesOriginalLowParameter(DataType type) {
        Tensor prototype = formal(type, false, Shape.scalar());
        Tensor sourceWeight = value(type, true, Shape.scalar(), 2);
        try (Engine engine = cpu();
                var model = Model.graph(List.of(prototype), List.of(prototype.mul(sourceWeight)),
                        List.of(new GraphCheckpoint.Leaf(
                                "weight", GraphCheckpoint.Role.PARAMETER, sourceWeight)), engine)) {
            Tensor input = value(type, false, Shape.scalar(), 2);
            Tensor ownedWeight = model.stateDictionary().entries().getFirst().value();
            assertLow(model.compute(List.of(input)).getFirst(), type, 4);
            Tensor prediction = model.forward(List.of(input)).getFirst();
            Tensor wide = prediction.cast(DataType.FLOAT32);
            Tensor objective = wide.mul(wide);
            assertEquals(DataType.FLOAT32, objective.descriptor().dataType());
            var backward = engine.backward(objective, List.of(ownedWeight), 1024);
            assertF32(backward.objective(), 16);
            assertLow(backward.gradients().getFirst(), type, 16);
            assertLowTensor(sourceWeight, type, 2);
        }
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class,
            names = {"BFLOAT16", "FLOAT16", "FLOAT8_E4M3", "FLOAT8_E5M2"})
    void linearModelTrainsSavesRawParametersAndStartsFreshTrainingAfterLoad(DataType type)
            throws Exception {
        Tensor externalWeight = value(type, true, Shape.of(1, 2), 1, 2);
        TensorDescriptor signature = descriptor(type, true, Shape.of(1, 2));
        TensorDescriptor targetSignature = descriptor(DataType.FLOAT32, false, Shape.of(1, 1));
        Path file = directory.resolve(type.name() + ".graph");
        byte[] saved;
        HostTensorValue retained;
        try (Engine engine = cpu()) {
            try (var model = Model.defineTensor(signature, topology -> {
                var linear = topology.addModule("linear", new Linear(externalWeight));
                return linear::forward;
            }, engine)) {
                Tensor input = value(type, true, Shape.of(1, 2), 1, 1);
                Tensor ownedWeight = model.stateDictionary().entries().getFirst().value();
                assertLow(model.compute(input), type, 3);
                Tensor prediction = model.forward(input).cast(DataType.FLOAT32);
                Tensor objective = prediction.mul(prediction).sum();
                assertEquals(DataType.FLOAT32, objective.descriptor().dataType());
                var backward = engine.backward(objective, List.of(ownedWeight, input), 1024);
                assertF32(backward.objective(), 9);
                // y = x0*w0 + x1*w1; dW = 2*y*x; dX = 2*y*w.
                assertLow(backward.gradients().get(0), type, 6, 6);
                assertLow(backward.gradients().get(1), type, 6, 12);

                model.configureTraining(targetSignature,
                        CpuLowPrecisionWorkflowIntegrationTest::squaredError, new Sgd(.125));
                Tensor batch = value(type, true, Shape.of(1, 2), 1, 1);
                Tensor zero = value(DataType.FLOAT32, false, Shape.of(1, 1), 0);
                assertF32(model.train(batch, zero).objective(), 9);
                assertLowTensor(ownedWeight, type, .25, 1.25);
                assertLow(model.compute(batch), type, 1.5);
                assertF32(model.train(batch, zero).objective(), 2.25);
                assertLowTensor(ownedWeight, type, -.125, .875);
                retained = model.compute(batch);
                assertLow(retained, type, .75);
                saved = raw(ownedWeight);
                model.save(file);
            }
            try (var loaded = Model.load(file).withEngine(engine)) {
                Tensor restored = loaded.stateDictionary().entries().getFirst().value();
                assertArrayEquals(saved, raw(restored));
                assertLowTensor(restored, type, -.125, .875);
                assertLow(loaded.compute(value(type, true, Shape.of(1, 2), 1, 1)), type, .75);
                loaded.configureTraining(targetSignature,
                        CpuLowPrecisionWorkflowIntegrationTest::squaredError, new Sgd(.125));
                // This is a new optimizer, not an optimizer-continuation checkpoint.
                Tensor nextBatch = value(type, true, Shape.of(1, 2), 2, 2);
                assertF32(loaded.train(nextBatch,
                        value(DataType.FLOAT32, false, Shape.of(1, 1), 0)).objective(), 2.25);
                assertLowTensor(restored, type, -.875, .125);
                assertLow(loaded.compute(nextBatch), type, -1.5);
            }
        }
        assertLow(retained, type, .75);
        assertLowTensor(externalWeight, type, 1, 2);
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class,
            names = {"BFLOAT16", "FLOAT16", "FLOAT8_E4M3", "FLOAT8_E5M2"})
    void repeatedMomentumSgdRetainsF32MasterBelowTheLogicalQuantum(DataType type) {
        Tensor parameter = value(type, true, Shape.scalar(), 2);
        var owner = new Parameters(parameter);
        // Immediately below 2, each carrier has one known dyadic spacing q. A quarter-q
        // learning rate makes the first logical candidate stay 2 while its master moves.
        float quantum = switch (type) {
            case BFLOAT16 -> 0x1.0p-7f;
            case FLOAT16 -> 0x1.0p-10f;
            case FLOAT8_E4M3 -> .125f;
            case FLOAT8_E5M2 -> .25f;
            default -> throw new AssertionError(type);
        };
        float rate = quantum / 4;
        try (Arena arena = Arena.ofShared(); Engine engine = cpu()) {
            TrainingParameterStorage.copyToSharedNative(owner.trainingParameters(), arena);
            try (TrainingSession session = TrainingSession.open(engine, owner.trainingParameters(),
                    parameter.cast(DataType.FLOAT32), new Sgd(rate, .5, 0, false))) {
                float[] objective = {2, 2, 2 - quantum};
                float[] master = {2 - rate, 2 - 2.5f * rate, 2 - 4.25f * rate};
                float[] momentum = {1, 1.5f, 1.75f};
                for (int step = 0; step < 3; step++) {
                    assertF32(session.run(List.of(), GradientMode.RESET_AND_STEP).objective(),
                            objective[step]);
                    var state = session.state();
                    var entry = state.parameters().getFirst();
                    assertEquals(master[step], entry.masterParameterBytes().getFloat());
                    assertEquals(momentum[step], entry.momentumBytes().getFloat());
                    assertLowTensor(parameter, type, step == 0 ? 2 : 2 - quantum);
                    // Exact independent encodings: 2 is 0x4000/0x40; its predecessor is one
                    // raw positive code smaller. Neither assertion re-encodes a result.
                    int twoBits = type.byteWidth() == 1 ? 0x40 : 0x4000;
                    assertEquals(twoBits - (step == 0 ? 0 : 1), rawCode(parameter, 0));
                    assertEquals(type, entry.dataType());
                    assertEquals(step + 1, state.optimizerSteps());
                    assertEquals(step + 1, state.executions());
                }
            }
        }
    }

    @Test
    void e4m3NonfiniteNarrowedCandidateRejectsEveryParameterAndOptimizerState() {
        DataType type = DataType.FLOAT8_E4M3;
        Tensor first = value(type, true, Shape.scalar(), 1);
        Tensor second = value(type, true, Shape.scalar(), 448);
        var owner = new Parameters(first, second);
        Tensor objective = first.cast(DataType.FLOAT32).sub(second.cast(DataType.FLOAT32));
        try (Arena arena = Arena.ofShared(); Engine engine = cpu()) {
            TrainingParameterStorage.copyToSharedNative(owner.trainingParameters(), arena);
            try (TrainingSession session = TrainingSession.open(engine, owner.trainingParameters(),
                    objective, new Sgd(16, .5, 0, false))) {
                assertF32(session.run(List.of(), GradientMode.RESET_AND_STEP).objective(), -447);
                // 448 + 16 = 464 ties toward finite 448. The next master is 488, whose
                // E4M3 representation is NaN despite finite F32 objective/gradient/master.
                assertLowTensor(first, type, -15);
                assertLowTensor(second, type, 448);
                TrainingState before = session.state();
                assertEquals(464, before.parameters().get(1).masterParameterBytes().getFloat());
                assertEquals(-1, before.parameters().get(1).momentumBytes().getFloat());
                byte[] firstBefore = raw(first), secondBefore = raw(second);
                assertThrows(ArithmeticException.class,
                        () -> session.run(List.of(), GradientMode.RESET_AND_STEP));
                assertArrayEquals(firstBefore, raw(first));
                assertArrayEquals(secondBefore, raw(second));
                assertStateEquals(before, session.state());
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class,
            names = {"BFLOAT16", "FLOAT16", "FLOAT8_E4M3", "FLOAT8_E5M2"})
    void lengthAwareScanActualAdAndTrainingIgnoreInactivePoisonedRows(DataType type) {
        Shape frame = Shape.of(2, 1);
        Tensor h = formal(type, true, frame), x = formal(type, true, frame);
        Tensor w = formal(type, true, Shape.of(1));
        Tensor next = h.add(x.mul(w));
        var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
        Tensor initial = value(type, true, frame, 1, -0.0);
        Tensor sequence = value(type, true, Shape.of(2, 2, 1), 2, Double.NaN, 4, Double.NaN);
        Tensor weight = value(type, true, Shape.of(1), .5);
        Tensor lengths = lengths(2, 0);
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), lengths,
                1, List.of(OptionalInt.of(0)), body, false);
        Tensor objective = scan.finalCarry().getFirst().cast(DataType.FLOAT32).sum();
        var owner = new Parameters(weight);
        try (Arena arena = Arena.ofShared(); Engine engine = cpu()) {
            assertLow(engine.compute(scan.finalCarry().getFirst()), type, 4, -0.0);
            assertLow(engine.compute(scan.outputs().getFirst()), type, 2, 0, 4, 0);
            var backward = engine.backward(objective, List.of(initial, sequence, weight), 4096);
            assertF32(backward.objective(), 4);
            assertLow(backward.gradients().get(0), type, 1, 1);
            assertLow(backward.gradients().get(1), type, .5, 0, .5, 0);
            assertLow(backward.gradients().get(2), type, 6);
            TrainingParameterStorage.copyToSharedNative(owner.trainingParameters(), arena);
            try (TrainingSession session = TrainingSession.open(
                    engine, owner.trainingParameters(), objective, new Sgd(.125))) {
                assertF32(session.run(List.of(initial, sequence, lengths),
                        GradientMode.RESET_AND_STEP).objective(), 4);
                assertLowTensor(weight, type, -.25);
                assertEquals(-.25f,
                        session.state().parameters().getFirst().masterParameterBytes().getFloat());
                assertF32(session.run(List.of(initial, sequence, lengths),
                        GradientMode.RESET_AND_STEP).objective(), -.5);
                assertLowTensor(weight, type, -1);
                assertEquals(-1f,
                        session.state().parameters().getFirst().masterParameterBytes().getFloat());
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class,
            names = {"BFLOAT16", "FLOAT16", "FLOAT8_E4M3", "FLOAT8_E5M2"})
    void zeroTimeGeneralScanAndInactiveFixedLstmPreserveStateAndActualAdjoints(DataType type) {
        try (Engine engine = cpu()) {
            Tensor h = formal(type, true, Shape.of(2, 1));
            Tensor x = formal(type, true, Shape.of(2, 1));
            Tensor w = formal(type, true, Shape.of(1));
            Tensor next = h.add(x.mul(w));
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
            Tensor initial = value(type, true, Shape.of(2, 1), -0.0, 1);
            Tensor empty = value(type, true, Shape.of(0, 2, 1));
            Tensor weight = value(type, true, Shape.of(1), 2);
            for (boolean reverse : List.of(false, true)) {
                var scan = Scan.apply(List.of(initial), List.of(empty), List.of(weight), body, reverse);
                assertLow(engine.compute(scan.finalCarry().getFirst()), type, -0.0, 1);
                assertLow(engine.compute(scan.outputs().getFirst()), type);
                var backward = engine.backward(scan.finalCarry().getFirst().cast(DataType.FLOAT32).sum(),
                        List.of(initial, empty, weight), 4096);
                assertF32(backward.objective(), 1);
                assertLow(backward.gradients().get(0), type, 1, 1);
                assertLow(backward.gradients().get(1), type);
                assertLow(backward.gradients().get(2), type, 0);
            }
            for (int time : new int[] {0, 2}) {
                Tensor input = value(type, true, Shape.of(time, 2, 1),
                        time == 0 ? new double[0] : new double[] {Double.NaN, Double.NaN, Double.NaN, Double.NaN});
                Tensor cell = value(type, true, Shape.of(2, 1), 2, -0.0);
                Tensor inputWeight = value(type, true, Shape.of(4, 1), 1, 1, 1, 1);
                Tensor hiddenWeight = value(type, true, Shape.of(4, 1), 1, 1, 1, 1);
                for (RecurrentDirection direction : RecurrentDirection.values()) {
                    var fixed = RecurrentScan.lstm(input, lengths(0, 0), initial, cell,
                            inputWeight, hiddenWeight, direction);
                    assertLow(engine.compute(fixed.outputs()), type, new double[time * 2]);
                    assertLow(engine.compute(fixed.finalHidden()), type, -0.0, 1);
                    assertLow(engine.compute(fixed.finalCell()), type, 2, -0.0);
                    Tensor objective = fixed.finalHidden().cast(DataType.FLOAT32).sum()
                            .add(fixed.finalCell().cast(DataType.FLOAT32).sum());
                    var backward = engine.backward(objective,
                            List.of(initial, cell, input, inputWeight, hiddenWeight), 4096);
                    assertF32(backward.objective(), 3);
                    assertLow(backward.gradients().get(0), type, 1, 1);
                    assertLow(backward.gradients().get(1), type, 1, 1);
                    assertLow(backward.gradients().get(2), type, new double[time * 2]);
                    assertLow(backward.gradients().get(3), type, 0, 0, 0, 0);
                    assertLow(backward.gradients().get(4), type, 0, 0, 0, 0);
                }
            }
        }
    }

    /**
     * Builds a visible F32 objective before any loss arithmetic, not a hidden promoted carrier.
     * @param prediction low Model output
     * @param target F32 target of the same shape
     * @return scalar F32 sum of squared errors
     */
    private static Tensor squaredError(Tensor prediction, Tensor target) {
        Tensor error = prediction.cast(DataType.FLOAT32).sub(target);
        return error.mul(error).sum();
    }

    /** @return an explicitly CPU-only owned Engine without optional native prerequisites */
    private static Engine cpu() {
        try (Engine.Builder builder = Engine.builder()) {
            return builder.takeOwnership(CpuBackendIntegration.open()).build();
        }
    }

    /**
     * @param type logical storage format
     * @param gradient whether the leaf is differentiable
     * @param shape resolved canonical geometry
     * @return immutable descriptor for one fixture
     */
    private static TensorDescriptor descriptor(DataType type, boolean gradient, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
    }

    /**
     * @param type logical storage format
     * @param gradient whether this formal is differentiable
     * @param shape resolved canonical geometry
     * @return storage-free body or Model input formal
     */
    private static Tensor formal(DataType type, boolean gradient, Shape shape) {
        return TensorFactory.create(descriptor(type, gradient, shape));
    }

    /**
     * Encodes fixtures only; assertions decode represented bits independently below.
     * @param type declared format, one of the four low carriers or F32
     * @param gradient whether the source is differentiable
     * @param shape exact row-major geometry
     * @param lanes source numerical values, including deliberate poisoned padding
     * @return fresh automatically managed heap Tensor
     */
    private static Tensor value(DataType type, boolean gradient, Shape shape, double... lanes) {
        TensorDescriptor descriptor = descriptor(type, gradient, shape);
        if (type == DataType.FLOAT32) {
            float[] bits = new float[lanes.length];
            for (int i = 0; i < lanes.length; i++) bits[i] = (float) lanes[i];
            return TensorFactory.fromFlatArray(descriptor, Optional.empty(), bits);
        }
        if (type == DataType.BFLOAT16 || type == DataType.FLOAT16) {
            short[] bits = new short[lanes.length];
            for (int i = 0; i < lanes.length; i++) {
                bits[i] = type == DataType.BFLOAT16
                        ? BFloat16Bits.fromFloat((float) lanes[i]) : Float16Bits.fromFloat((float) lanes[i]);
            }
            return type == DataType.FLOAT16
                    ? TensorFactory.fromFlatFloat16Array(descriptor, Optional.empty(), bits)
                    : TensorFactory.fromFlatArray(descriptor, Optional.empty(), bits);
        }
        byte[] bits = new byte[lanes.length];
        for (int i = 0; i < lanes.length; i++) {
            bits[i] = type == DataType.FLOAT8_E4M3
                    ? Float8E4M3Bits.fromDouble(lanes[i]) : Float8E5M2Bits.fromDouble(lanes[i]);
        }
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), bits);
    }

    /**
     * @param lanes exact runtime valid prefix lengths
     * @return fresh canonical non-gradient INT64 vector
     */
    private static Tensor lengths(long... lanes) {
        return TensorFactory.fromFlatArray(descriptor(DataType.INT64, false, Shape.of(lanes.length)),
                Optional.empty(), lanes);
    }

    /**
     * @param actual detached F32 publication
     * @param expected independently calculated scalar
     */
    private static void assertF32(HostTensorValue actual, double expected) {
        assertEquals(DataType.FLOAT32, actual.dataType());
        assertEquals(Shape.scalar(), actual.shape());
        assertEquals(expected, actual.bytes().getFloat());
    }

    /**
     * @param actual detached canonical low publication
     * @param type required logical format
     * @param expected independent exact represented values, including signed zeros
     */
    private static void assertLow(HostTensorValue actual, DataType type, double... expected) {
        assertEquals(type, actual.dataType());
        assertEquals((long) expected.length * type.byteWidth(), actual.byteSize());
        ByteBuffer bytes = actual.bytes();
        for (int i = 0; i < expected.length; i++) {
            int code = type.byteWidth() == 1 ? Byte.toUnsignedInt(bytes.get()) : Short.toUnsignedInt(bytes.getShort());
            assertEquals(Double.doubleToRawLongBits(expected[i]),
                    Double.doubleToRawLongBits(decode(type, code)), "represented lane " + i);
        }
    }

    /**
     * @param tensor live low source or parameter
     * @param type required logical format
     * @param expected independent exact represented values
     */
    private static void assertLowTensor(Tensor tensor, DataType type, double... expected) {
        assertEquals(type, tensor.descriptor().dataType());
        assertEquals((long) expected.length * type.byteWidth(), raw(tensor).length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(Double.doubleToRawLongBits(expected[i]),
                    Double.doubleToRawLongBits(decode(type, rawCode(tensor, i))), "stored lane " + i);
        }
    }

    /**
     * @param tensor live low Tensor
     * @param lane logical contiguous element index
     * @return unsigned exact raw stored code
     */
    private static int rawCode(Tensor tensor, int lane) {
        var segment = tensor.hostStorage().orElseThrow().segment();
        return tensor.descriptor().dataType().byteWidth() == 1
                ? Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, lane))
                : Short.toUnsignedInt(segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, lane * 2L));
    }

    /**
     * Independent binary-format equation; no production narrowing or decoding helper is an oracle.
     * @param type one of the four low floating formats
     * @param code unsigned stored code
     * @return exact binary64 represented value, preserving signed zero
     */
    private static double decode(DataType type, int code) {
        int fractionBits = switch (type) {
            case BFLOAT16, FLOAT8_E4M3 -> type == DataType.BFLOAT16 ? 7 : 3;
            case FLOAT16 -> 10;
            case FLOAT8_E5M2 -> 2;
            default -> throw new AssertionError(type);
        };
        int exponentBits = type == DataType.BFLOAT16 ? 8 : type == DataType.FLOAT8_E4M3 ? 4 : 5;
        int bias = (1 << (exponentBits - 1)) - 1;
        int fraction = code & ((1 << fractionBits) - 1);
        int exponent = (code >>> fractionBits) & ((1 << exponentBits) - 1);
        double sign = (code & (1 << (fractionBits + exponentBits))) == 0 ? 1 : -1;
        if (type == DataType.FLOAT8_E4M3) {
            if (exponent == 15 && fraction == 7) return Double.NaN;
        } else if (exponent == (1 << exponentBits) - 1) {
            return fraction == 0 ? sign * Double.POSITIVE_INFINITY : Double.NaN;
        }
        double magnitude = exponent == 0
                ? Math.scalb((double) fraction, 1 - bias - fractionBits)
                : Math.scalb((double) ((1 << fractionBits) + fraction), exponent - bias - fractionBits);
        return sign * magnitude;
    }

    /**
     * @param tensor live stored Tensor
     * @return detached exact physical bytes for persistence or rollback comparison
     */
    private static byte[] raw(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE);
    }

    /**
     * Compares every publicly restorable state component after a failed transaction.
     * @param expected detached pre-failure snapshot
     * @param actual detached post-failure snapshot
     */
    private static void assertStateEquals(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.parameterGroups(), actual.parameterGroups());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int i = 0; i < expected.parameters().size(); i++) {
            var left = expected.parameters().get(i);
            var right = actual.parameters().get(i);
            assertEquals(left.path(), right.path());
            assertEquals(left.dataType(), right.dataType());
            assertEquals(left.shape(), right.shape());
            assertEquals(left.parameterBytes(), right.parameterBytes());
            assertEquals(left.masterParameterBytes(), right.masterParameterBytes());
            assertEquals(left.momentumBytes(), right.momentumBytes());
            assertEquals(left.secondMomentBytes(), right.secondMomentBytes());
            assertEquals(left.accumulatedGradientBytes(), right.accumulatedGradientBytes());
        }
    }

    /** Real public NN parameter ownership, without fabricated admission or private session access. */
    private static final class Parameters extends Module {
        /**
         * @param parameters distinct live parameter identities in transaction order
         */
        private Parameters(Tensor... parameters) {
            for (int i = 0; i < parameters.length; i++) parameter("p" + i, parameters[i]);
        }
    }
}
