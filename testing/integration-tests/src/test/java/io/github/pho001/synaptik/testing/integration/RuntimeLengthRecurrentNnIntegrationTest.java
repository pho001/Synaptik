package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.RecurrentScanResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.GruCell;
import io.github.pho001.synaptik.nn.layers.GruSequence;
import io.github.pho001.synaptik.nn.layers.LstmCell;
import io.github.pho001.synaptik.nn.layers.LstmSequence;
import io.github.pho001.synaptik.nn.layers.RnnCell;
import io.github.pho001.synaptik.nn.layers.RnnSequence;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

/** NN-owned trainable parameters through the existing canonical FLOAT32 scalar Engine route. */
class RuntimeLengthRecurrentNnIntegrationTest {
    static Stream<Arguments> variants() {
        return Stream.of(RecurrentScanKind.values()).flatMap(kind -> Stream.of(false, true)
                .flatMap(bias -> Stream.of(RecurrentDirection.values())
                        .map(direction -> Arguments.of(kind, bias, direction))));
    }

    @ParameterizedTest
    @MethodSource("variants")
    void prepareOnceFreshLengthsPreserveResultsAndRepairAfterInvalidRun(
            RecurrentScanKind kind, boolean bias, RecurrentDirection direction) {
        Module sequence = supplied(kind, bias);
        sequence.eval(); // ordinary NN parameters remain trainable even during inference
        float[] firstX = {.2f, .3f, .4f, Float.NaN, .6f, Float.NaN};
        float[] nextX = {Float.NaN, .7f, Float.NaN, .9f, Float.NaN, Float.NaN};
        Tensor x = floats(Shape.of(3, 2, 1), false, firstX);
        Tensor lengths = lengths(3, 1);
        Tensor h = floats(Shape.of(2, 1), false, -.2f, -.0f);
        Tensor c = floats(Shape.of(2, 1), false, .3f, -.0f);
        List<Tensor> roots = scan(sequence, x, lengths, h, c, direction);
        assertTrue(roots.stream().allMatch(value -> value.descriptor().requiresGrad()));
        assertTrue(sequence.parametersRecursively().values().stream()
                .allMatch(parameter -> parameter.value().descriptor().requiresGrad()));
        assertEquals(Shape.of(3, 2, 1), roots.getFirst().descriptor().shape());
        roots.subList(1, roots.size()).forEach(value ->
                assertEquals(Shape.of(2, 1), value.descriptor().shape()));
        var pool = inputPool(sequence, x, lengths, h, c);
        try (Engine engine = Engine.standard()) {
            CompiledGraph compiled = engine.compile(roots);
            var values = ordered(compiled, pool);
            try (var prepared = engine.prepare(compiled);
                    var first = engine.runValues(prepared, fresh(values, x, firstX, lengths, new long[] {3, 1}))) {
                assertResult(first, oracle(kind, bias, direction, firstX, new long[] {3, 1},
                        new float[] {-.2f, -.0f}, new float[] {.3f, -.0f}));
                List<List<Integer>> preserved = raw(first);
                var nextValues = fresh(values, x, nextX, lengths, new long[] {0, 2});
                try (var next = engine.runValues(prepared, nextValues)) {
                    assertResult(next, oracle(kind, bias, direction, nextX, new long[] {0, 2},
                            new float[] {-.2f, -.0f}, new float[] {.3f, -.0f}));
                    assertEquals(preserved, raw(first));
                }
                for (long[] invalid : List.of(new long[] {4, 1}, new long[] {1, -1})) {
                    assertThrows(IllegalArgumentException.class,
                            () -> engine.runValues(prepared, fresh(values, x, firstX, lengths, invalid)));
                    assertEquals(preserved, raw(first));
                }
                try (var repaired = engine.runValues(prepared, nextValues)) {
                    assertResult(repaired, oracle(kind, bias, direction, nextX, new long[] {0, 2},
                            new float[] {-.2f, -.0f}, new float[] {.3f, -.0f}));
                }
                try (var zero = engine.runValues(prepared,
                        fresh(values, x, nextX, lengths, new long[] {0, 0}))) {
                    assertArrayEquals(new float[6], floats(zero, 0));
                    assertEquals(Float.floatToRawIntBits(-.0f),
                            Float.floatToRawIntBits(floats(zero, 1)[1]));
                    if (kind == RecurrentScanKind.LSTM) assertEquals(Float.floatToRawIntBits(-.0f),
                            Float.floatToRawIntBits(floats(zero, 2)[1]));
                }
            }
            Tensor objective = roots.get(1).sum();
            Tensor parameter = sequence.children().get("cell").parameters().getFirst().value();
            assertThrows(IllegalArgumentException.class,
                    () -> engine.backward(objective, List.of(parameter), 10000));
            assertThrows(IllegalArgumentException.class,
                    () -> engine.compile(List.of(objective),
                            List.of(floats(Shape.scalar(), false, 1)), List.of(parameter)));
        }
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void producedLengthsUseFreshSourceValuesWithoutRecomposition(RecurrentScanKind kind) {
        Module sequence = supplied(kind, true);
        float[] data = {.2f, .3f, .4f, Float.NaN, .6f, Float.NaN};
        Tensor x = floats(Shape.of(3, 2, 1), false, data);
        Tensor source = floats(Shape.of(2), false, 3, 1);
        Tensor lengths = source.cast(DataType.INT64);
        Tensor h = floats(Shape.of(2, 1), false, -.0f, .1f);
        Tensor c = floats(Shape.of(2, 1), false, -.0f, .3f);
        var roots = scan(sequence, x, lengths, h, c, RecurrentDirection.REVERSE);
        var pool = inputPool(sequence, x, source, h, c);
        try (Engine engine = Engine.standard()) {
            CompiledGraph compiled = engine.compile(roots);
            var values = ordered(compiled, pool);
            try (var prepared = engine.prepare(compiled); var first = engine.runValues(prepared, values)) {
                assertResult(first, oracle(kind, true, RecurrentDirection.REVERSE, data,
                        new long[] {3, 1}, new float[] {-.0f, .1f}, new float[] {-.0f, .3f}));
                var changed = new ArrayList<>(values);
                changed.set(values.indexOf(source), floats(Shape.of(2), false, 0, 1));
                try (var next = engine.runValues(prepared, changed)) {
                    assertResult(next, oracle(kind, true, RecurrentDirection.REVERSE, data,
                            new long[] {0, 1}, new float[] {-.0f, .1f}, new float[] {-.0f, .3f}));
                    assertEquals(Float.floatToRawIntBits(-.0f),
                            Float.floatToRawIntBits(floats(next, 1)[0]));
                    if (kind == RecurrentScanKind.LSTM) assertEquals(Float.floatToRawIntBits(-.0f),
                            Float.floatToRawIntBits(floats(next, 2)[0]));
                }
                changed.set(values.indexOf(source), floats(Shape.of(2), false, 0, 4));
                assertThrows(IllegalArgumentException.class, () -> engine.runValues(prepared, changed));
                changed.set(values.indexOf(source), floats(Shape.of(2), false, 0, 1));
                try (var repaired = engine.runValues(prepared, changed)) {
                    assertEquals(roots.size(), repaired.resultCount());
                }
            }
        }
    }

    @Test
    void lstmExplicitSymbolicCarryComposesTwoOrdinaryScansWithSharedParameters() {
        LstmSequence sequence = (LstmSequence) supplied(RecurrentScanKind.LSTM, true);
        float[] firstX = {.2f, .3f, .4f, Float.NaN, .6f, Float.NaN};
        float[] nextX = {Float.NaN, .7f, Float.NaN, .9f, Float.NaN, Float.NaN};
        Tensor x = floats(Shape.of(3, 2, 1), false, firstX);
        Tensor nextInput = floats(Shape.of(3, 2, 1), false, nextX);
        Tensor lengths = lengths(3, 1), nextLengths = lengths(0, 2);
        Tensor h = floats(Shape.of(2, 1), false, -.2f, -.0f);
        Tensor c = floats(Shape.of(2, 1), false, .3f, -.0f);
        var first = sequence.forwardScan(x, lengths, h, c, RecurrentDirection.FORWARD);
        var next = sequence.forwardScan(nextInput, nextLengths,
                first.finalHidden(), first.finalCell(), RecurrentDirection.REVERSE);
        var pool = inputPool(sequence, x, lengths, h, c);
        pool.add(nextInput); pool.add(nextLengths);
        var expectedFirst = oracle(RecurrentScanKind.LSTM, true, RecurrentDirection.FORWARD,
                firstX, new long[] {3, 1}, new float[] {-.2f, -.0f}, new float[] {.3f, -.0f});
        var expectedNext = oracle(RecurrentScanKind.LSTM, true, RecurrentDirection.REVERSE,
                nextX, new long[] {0, 2}, expectedFirst.get(1), expectedFirst.get(2));
        try (Engine engine = Engine.standard()) {
            var compiled = engine.compile(List.of(next.outputs(), next.finalHidden(), next.finalCell()));
            try (var prepared = engine.prepare(compiled);
                    var result = engine.runValues(prepared, ordered(compiled, pool))) {
                assertResult(result, expectedNext);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(RecurrentScanKind.class)
    void automaticEmptyScanBindsAndExecutesSemanticInitialStates(RecurrentScanKind kind) {
        for (boolean zeroTime : List.of(true, false)) {
            Module sequence = switch (kind) {
                case RNN_TANH -> new RnnSequence(1, true, DataType.FLOAT32, ParameterInitialization.zeros(), 42);
                case GRU_RESET_AFTER -> new GruSequence(1, true, DataType.FLOAT32, ParameterInitialization.zeros(), 42);
                case LSTM -> new LstmSequence(1, true, DataType.FLOAT32, ParameterInitialization.zeros(), 42);
            };
            int time = zeroTime ? 0 : 3, batch = zeroTime ? 2 : 0;
            Tensor x = floats(Shape.of(time, batch, 1), false, new float[0]);
            Tensor lengths = lengths(new long[batch]);
            float[] hidden = zeroTime ? new float[] {-.0f, .2f} : new float[0];
            float[] cell = zeroTime ? new float[] {.3f, -.0f} : new float[0];
            Tensor h = floats(Shape.of(batch, 1), false, hidden);
            Tensor c = floats(Shape.of(batch, 1), false, cell);
            var roots = scan(sequence, x, lengths, h, c, RecurrentDirection.REVERSE);
            assertEquals(3, sequence.parametersRecursively().size());
            try (Engine engine = Engine.standard()) {
                var compiled = engine.compile(roots);
                try (var prepared = engine.prepare(compiled);
                        var result = engine.runValues(prepared, ordered(compiled,
                                inputPool(sequence, x, lengths, h, c)))) {
                    assertEquals(0, floats(result, 0).length);
                    assertArrayEquals(hidden, floats(result, 1));
                    if (zeroTime) assertEquals(Float.floatToRawIntBits(-.0f),
                            Float.floatToRawIntBits(floats(result, 1)[0]));
                    if (kind == RecurrentScanKind.LSTM) {
                        assertArrayEquals(cell, floats(result, 2));
                        if (zeroTime) assertEquals(Float.floatToRawIntBits(-.0f),
                                Float.floatToRawIntBits(floats(result, 2)[1]));
                    }
                }
            }
        }
    }

    private static Module supplied(RecurrentScanKind kind, boolean biased) {
        int gates = gates(kind);
        float[] w = new float[gates], u = new float[gates], b = new float[gates];
        for (int gate = 0; gate < gates; gate++) {
            w[gate] = .2f * (gate + 1); u[gate] = -.1f; b[gate] = .03f * (gate + 1);
        }
        Tensor inputWeight = floats(Shape.of(gates, 1), true, w);
        Tensor hiddenWeight = floats(Shape.of(gates, 1), true, u);
        Tensor bias = floats(Shape.of(gates), true, b);
        return switch (kind) {
            case RNN_TANH -> new RnnSequence(biased ? new RnnCell(inputWeight, hiddenWeight, bias)
                    : new RnnCell(inputWeight, hiddenWeight));
            case GRU_RESET_AFTER -> new GruSequence(biased ? new GruCell(inputWeight, hiddenWeight, bias)
                    : new GruCell(inputWeight, hiddenWeight));
            case LSTM -> new LstmSequence(biased ? new LstmCell(inputWeight, hiddenWeight, bias)
                    : new LstmCell(inputWeight, hiddenWeight));
        };
    }

    private static List<Tensor> scan(Module sequence, Tensor x, Tensor l, Tensor h, Tensor c,
            RecurrentDirection direction) {
        return switch (sequence) {
            case RnnSequence rnn -> outputs(rnn.forwardScan(x, l, h, direction));
            case GruSequence gru -> outputs(gru.forwardScan(x, l, h, direction));
            case LstmSequence lstm -> {
                var result = lstm.forwardScan(x, l, h, c, direction);
                yield List.of(result.outputs(), result.finalHidden(), result.finalCell());
            }
            default -> throw new AssertionError();
        };
    }

    private static List<Tensor> outputs(RecurrentScanResult result) {
        return List.of(result.outputs(), result.finalHidden());
    }

    private static ArrayList<Tensor> inputPool(Module sequence, Tensor x, Tensor lengths, Tensor h, Tensor c) {
        var values = new ArrayList<>(List.of(x, lengths, h, c));
        sequence.parametersRecursively().values().stream().map(Parameter::value).forEach(values::add);
        return values;
    }

    private static List<Tensor> ordered(CompiledGraph graph, List<Tensor> pool) {
        return graph.inputs().stream().map(input -> pool.stream()
                .filter(tensor -> tensor.id().equals(input.tensorId())).findFirst().orElseThrow()).toList();
    }

    private static List<Tensor> fresh(List<Tensor> prototypes, Tensor input, float[] data,
            Tensor lengths, long[] lengthValues) {
        return prototypes.stream().map(tensor -> tensor == input
                ? TensorFactory.fromFlatArray(tensor.descriptor(), Optional.empty(), data.clone())
                : tensor == lengths
                        ? TensorFactory.fromFlatArray(tensor.descriptor(), Optional.empty(), lengthValues.clone())
                        : TensorFactory.create(tensor.descriptor(), Optional.empty(), tensor.hostStorage())).toList();
    }

    private static Tensor floats(Shape shape, boolean grad, float... values) {
        return TensorFactory.fromFlatArray(descriptor(DataType.FLOAT32, shape, grad), Optional.empty(), values);
    }

    private static Tensor lengths(long... values) {
        return TensorFactory.fromFlatArray(descriptor(DataType.INT64, Shape.of(values.length), false),
                Optional.empty(), values);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static float[] floats(RunResult result, int port) {
        var bytes = result.materialize(result.publications().get(port), 10000).bytes();
        var values = new float[bytes.remaining() / Float.BYTES];
        for (int index = 0; index < values.length; index++) values[index] = bytes.getFloat();
        return values;
    }

    private static List<List<Integer>> raw(RunResult result) {
        var ports = new ArrayList<List<Integer>>();
        for (int port = 0; port < result.resultCount(); port++) {
            var bytes = result.materialize(result.publications().get(port), 10000).bytes();
            var values = new ArrayList<Integer>();
            while (bytes.hasRemaining()) values.add(Byte.toUnsignedInt(bytes.get()));
            ports.add(values);
        }
        return ports;
    }

    private static void assertResult(RunResult result, List<float[]> expected) {
        assertEquals(expected.size(), result.resultCount());
        for (int port = 0; port < expected.size(); port++) {
            float[] actual = floats(result, port);
            assertArrayEquals(expected.get(port), actual, 2e-6f);
            if (port == 0) for (int index = 0; index < actual.length; index++) {
                if (expected.get(port)[index] == 0) assertEquals(0, Float.floatToRawIntBits(actual[index]));
            }
        }
    }

    /** Direct scalar reference with the fixed packed equations and original-time alignment. */
    private static List<float[]> oracle(RecurrentScanKind kind, boolean bias, RecurrentDirection direction,
            float[] x, long[] lengths, float[] initialHidden, float[] initialCell) {
        float[] dense = new float[6], hidden = initialHidden.clone(), cell = initialCell.clone();
        for (int row = 0; row < 2; row++) for (int step = 0; step < lengths[row]; step++) {
            int time = direction == RecurrentDirection.FORWARD ? step : (int) lengths[row] - 1 - step;
            float[] p = new float[gates(kind)];
            for (int gate = 0; gate < p.length; gate++) {
                p[gate] = x[time * 2 + row] * (.2f * (gate + 1));
                if (bias) p[gate] += .03f * (gate + 1);
            }
            float recurrent = hidden[row] * -.1f;
            switch (kind) {
                case RNN_TANH -> hidden[row] = tanh(p[0] + recurrent);
                case GRU_RESET_AFTER -> {
                    float reset = sigmoid(p[0] + recurrent), update = sigmoid(p[1] + recurrent);
                    float candidate = tanh(p[2] + reset * recurrent);
                    hidden[row] = candidate + update * (hidden[row] - candidate);
                }
                case LSTM -> {
                    float input = sigmoid(p[0] + recurrent), forget = sigmoid(p[1] + recurrent);
                    float candidate = tanh(p[2] + recurrent), output = sigmoid(p[3] + recurrent);
                    cell[row] = forget * cell[row] + input * candidate;
                    hidden[row] = output * tanh(cell[row]);
                }
            }
            dense[time * 2 + row] = hidden[row];
        }
        return kind == RecurrentScanKind.LSTM ? List.of(dense, hidden, cell) : List.of(dense, hidden);
    }

    private static float tanh(float value) { return (float) Math.tanh(value); }
    private static float sigmoid(float value) { return (float) (1.0 / (1.0 + Math.exp(-value))); }
    private static int gates(RecurrentScanKind kind) {
        return switch (kind) { case RNN_TANH -> 1; case GRU_RESET_AFTER -> 3; case LSTM -> 4; };
    }
}
