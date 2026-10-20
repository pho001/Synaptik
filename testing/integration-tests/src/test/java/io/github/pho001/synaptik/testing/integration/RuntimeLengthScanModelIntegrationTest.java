package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.nn.module.ModelBuilder;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Public build/save/close/load/compute workflow, qualified against independent scalar equations. */
final class RuntimeLengthScanModelIntegrationTest {
    @TempDir Path directory;

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm", "rnn/rnn", "gru/gru", "lstm/lstm", "rnn/gru/lstm"})
    void freshOrderedLengthsDriveDenseFinalAndLastOutputsBeforeAndAfterColdLoad(String chain) throws Exception {
        for (String terminal : List.of("dense", "final", "last")) {
            String[] kinds = chain.split("/");
            ModelBuilder builder = Model.builder().input(DataType.FLOAT32, 3, 3, 1).seed(41);
            for (String kind : kinds) append(builder, kind);
            if (terminal.equals("final")) builder.finalHidden();
            if (terminal.equals("last")) builder.lastTimeStep();
            Path file = directory.resolve("scan"), resave = directory.resolve("resaved");
            List<List<float[]>> parameters = new ArrayList<>();
            try (var model = builder.buildWithLengths()) {
                for (int step = 0; step < kinds.length; step++) {
                    String prefix = "layers." + step + ".sequence.cell.";
                    parameters.add(model.stateDictionary().entries().stream().filter(e -> e.path().startsWith(prefix))
                            .map(e -> e.value().hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_FLOAT)).toList());
                }
                model.save(file); // No forward/compute is needed; close original before cold load.
            }
            HostTensorValue retained;
            try (Engine borrowed = Engine.standard(); var loaded = Model.loadGraph(file).withEngine(borrowed)) {
                for (long[] length : List.of(new long[]{3, 1, 0}, new long[]{0, 0, 0}, new long[]{1, 3, 2})) {
                    float[] x = {.2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, 1};
                    for (int t = 0; t < 3; t++) for (int b = 0; b < 3; b++) if (t >= length[b]) x[t * 3 + b] = Float.NaN;
                    float[] dense = x, finalHidden = null;
                    for (int step = 0; step < kinds.length; step++) {
                        var expected = oracle(kinds[step], dense, length, new float[3], new float[3], parameters.get(step), false);
                        dense = expected.get(0); finalHidden = expected.get(1);
                    }
                    float[] expected = terminal.equals("final") ? finalHidden
                            : terminal.equals("last") ? Arrays.copyOfRange(dense, 6, 9) : dense;
                    Tensor values = floats(false, new long[]{3, 3, 1}, x), lengths = lengths(length);
                    retained = loaded.compute(values, lengths);
                    assertArrayEquals(expected, floats(retained), 2e-6f);
                    assertArrayEquals(floats(retained), floats(loaded.compute(List.of(values, lengths)).getFirst()));
                    assertEquals(terminal.equals("dense") ? Shape.of(3, 3, 1) : Shape.of(3, 1), retained.shape());
                    assertThrows(UnsupportedOperationException.class, () -> loaded.configureTraining((TensorDescriptor) null, null, null));
                    assertThrows(UnsupportedOperationException.class, () -> loaded.train((List<Tensor>) null, null));
                }
                Tensor zero = floats(false, new long[]{3, 3, 1}, new float[9]);
                assertThrows(IllegalArgumentException.class, () -> loaded.compute(zero, lengths(3, 4, 0)));
                retained = loaded.compute(zero, lengths(3, 1, 0)); // Same path remains usable after invalid lengths.
                loaded.save(resave);
                assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(resave));
                loaded.close(); assertFalse(borrowed.isClosed());
            }
            assertTrue(Float.isFinite(retained.bytes().getFloat()));
        }
    }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void advancedExplicitStatesAndFinalStateOnlyRootsKeepCanonicalPortsAndAliases(String family) throws Exception {
        int gates = family.equals("rnn") ? 1 : family.equals("gru") ? 3 : 4;
        Tensor x = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 3, 1));
        Tensor l = TensorFactory.create(descriptor(DataType.INT64, false, 3));
        Tensor h = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 1));
        Tensor c = TensorFactory.create(h.descriptor());
        float[] w = new float[gates], u = new float[gates], bias = new float[gates];
        for (int i = 0; i < gates; i++) { w[i] = .2f * (i + 1); u[i] = -.1f; bias[i] = .03f * (i + 1); }
        Tensor weight = floats(true, new long[]{gates, 1}, w), hidden = floats(true, new long[]{gates, 1}, u);
        Tensor b = floats(true, new long[]{gates}, bias);
        Tensor root = switch (family) {
            case "rnn" -> RecurrentScan.rnn(x, l, h, weight, hidden, b, RecurrentDirection.REVERSE).finalHidden();
            case "gru" -> RecurrentScan.gru(x, l, h, weight, hidden, b, RecurrentDirection.REVERSE).finalHidden();
            case "lstm" -> RecurrentScan.lstm(x, l, h, c, weight, hidden, b, RecurrentDirection.REVERSE).finalHidden();
            default -> throw new AssertionError(family);
        };
        List<Tensor> inputs = family.equals("lstm") ? List.of(x, l, h, c) : List.of(x, l, h);
        Path file = directory.resolve("explicit");
        try (var model = Model.graph(inputs, List.of(root), List.of(
                new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight),
                new GraphCheckpoint.Leaf("hidden", GraphCheckpoint.Role.PARAMETER, hidden),
                new GraphCheckpoint.Leaf("bias", GraphCheckpoint.Role.PARAMETER, b)))) { model.save(file); }
        try (var model = Model.loadGraph(file)) {
            float[] data = {.2f, .3f, Float.NaN, .4f, Float.NaN, Float.NaN, .6f, Float.NaN, Float.NaN};
            float[] state = {-.2f, .1f, -.0f};
            Tensor values = floats(false, new long[]{3, 3, 1}, data), lengths = lengths(3, 1, 0);
            Tensor shared = floats(false, new long[]{3, 1}, state);
            var fresh = family.equals("lstm") ? List.of(values, lengths, shared, shared) : List.of(values, lengths, shared);
            var output = model.forward(fresh).getFirst();
            var producer = output.provenance().orElseThrow().producer();
            assertSame(output, producer.output(1));
            assertEquals(family.equals("lstm") ? 3 : 2, producer.outputCount());
            if (family.equals("lstm")) assertSame(producer.inputs().get(2), producer.inputs().get(3));
            float[] expected = oracle(family, data, new long[]{3, 1, 0}, state, state, List.of(w, u, bias), true).get(1);
            assertArrayEquals(expected, floats(model.compute(fresh).getFirst()), 2e-6f);
            assertEquals(Float.floatToRawIntBits(-.0f), Float.floatToRawIntBits(floats(model.compute(fresh).getFirst())[2]));
            assertThrows(UnsupportedOperationException.class, () -> model.configureTraining((List<Tensor>) null, null, null));
        }
    }

    @Test void ordinaryTwoInputModelsUseDeclaredPositionsNotLengthRoles() throws Exception {
        Tensor a = TensorFactory.create(descriptor(DataType.FLOAT32, false, 1));
        Tensor b = TensorFactory.create(a.descriptor());
        Path file = directory.resolve("ordinary");
        try (var model = Model.graph(List.of(b, a), List.of(a.add(a).add(b)), List.of())) { model.save(file); }
        try (var model = Model.loadGraph(file)) {
            assertEquals(11, model.compute(floats(false, new long[]{1}, 5), floats(false, new long[]{1}, 3)).bytes().getFloat());
        }
    }

    private static void append(ModelBuilder builder, String family) {
        switch (family) { case "rnn" -> builder.rnnScan(1); case "gru" -> builder.gruScan(1); case "lstm" -> builder.lstmScan(1); default -> throw new AssertionError(); }
    }

    private static TensorDescriptor descriptor(DataType type, boolean grad, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static Tensor floats(boolean grad, long[] axes, float... values) {
        return TensorFactory.fromFlatArray(descriptor(DataType.FLOAT32, grad, axes), Optional.empty(), values);
    }

    private static Tensor lengths(long... values) {
        return TensorFactory.fromFlatArray(descriptor(DataType.INT64, false, values.length), Optional.empty(), values);
    }

    private static float[] floats(HostTensorValue value) {
        float[] values = new float[value.bytes().remaining() / 4];
        for (int i = 0; i < values.length; i++) values[i] = value.bytes().getFloat(i * 4);
        return values;
    }

    /** Independent scalar equations for width-one packed gates; skips suffixes before reading x. */
    private static List<float[]> oracle(String kind, float[] x, long[] lengths, float[] initialH,
            float[] initialC, List<float[]> parameters, boolean reverse) {
        float[] dense = new float[9], hidden = initialH.clone(), cell = initialC.clone();
        float[] w = parameters.get(0), u = parameters.get(1), bias = parameters.get(2);
        for (int b = 0; b < 3; b++) for (int step = 0; step < lengths[b]; step++) {
            int t = reverse ? (int) lengths[b] - 1 - step : step;
            float[] p = new float[w.length], q = new float[u.length];
            for (int gate = 0; gate < p.length; gate++) {
                p[gate] = x[t * 3 + b] * w[gate] + bias[gate]; q[gate] = hidden[b] * u[gate];
            }
            switch (kind) {
                case "rnn" -> hidden[b] = tanh(p[0] + q[0]);
                case "gru" -> {
                    float reset = sigmoid(p[0] + q[0]), update = sigmoid(p[1] + q[1]);
                    float candidate = tanh(p[2] + reset * q[2]);
                    hidden[b] = candidate + update * (hidden[b] - candidate);
                }
                case "lstm" -> {
                    cell[b] = sigmoid(p[1] + q[1]) * cell[b] + sigmoid(p[0] + q[0]) * tanh(p[2] + q[2]);
                    hidden[b] = sigmoid(p[3] + q[3]) * tanh(cell[b]);
                }
                default -> throw new AssertionError(kind);
            }
            dense[t * 3 + b] = hidden[b];
        }
        return List.of(dense, hidden);
    }

    private static float sigmoid(float x) { return (float) (1 / (1 + Math.exp(-x))); }
    private static float tanh(float x) { return (float) Math.tanh(x); }
}
