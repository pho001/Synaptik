package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.nn.module.ModelBuilderTest.*;
import static io.github.pho001.synaptik.nn.module.OwnedModelSourceTest.field;
import static io.github.pho001.synaptik.nn.module.ModelBuilderExecutionTest.sameValues;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Complete runtime-length construction, independent numerical oracles and owned lifecycle checks. */
final class RuntimeLengthScanModelTest {
    @TempDir Path directory;

    static Tensor lengths(long... values) {
        Shape shape = Shape.of(values.length);
        return TensorFactory.fromFlatArray(new TensorDescriptor(DataType.INT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false), Optional.empty(), values);
    }

    static ModelBuilder scan(ModelBuilder builder, String family, long hidden) {
        return switch (family) {
            case "rnn" -> builder.rnnScan(hidden);
            case "gru" -> builder.gruScan(hidden);
            case "lstm" -> builder.lstmScan(hidden);
            default -> throw new AssertionError(family);
        };
    }

    static float[] lanes(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_FLOAT);
    }

    static float[] lanes(HostTensorValue value) {
        float[] lanes = new float[value.bytes().remaining() / 4];
        for (int i = 0; i < lanes.length; i++) lanes[i] = value.bytes().getFloat(i * 4);
        return lanes;
    }

    static List<Tensor> weights(Model<?, ?> model, int step) {
        String prefix = "layers." + step + ".sequence.cell.";
        return model.stateDictionary().entries().stream().filter(e -> e.path().startsWith(prefix))
                .map(StateEntry::value).toList();
    }

    /** Independent scalar FLOAT32 packed-gate equations, no Tensor/Sequence/scan forward calls. */
    static List<float[]> oracle(float[] x, int time, int batch, int features, int hidden,
            long[] lengths, String family, List<Tensor> weights) {
        float[] w = lanes(weights.get(0)), u = lanes(weights.get(1)), bias = lanes(weights.get(2));
        int gates = family.equals("rnn") ? 1 : family.equals("gru") ? 3 : 4;
        float[] dense = new float[time * batch * hidden], finalH = new float[batch * hidden];
        for (int row = 0; row < batch; row++) {
            float[] h = new float[hidden], c = new float[hidden];
            for (int t = 0; t < lengths[row]; t++) {
                float[] p = new float[gates * hidden], q = new float[p.length];
                for (int gate = 0; gate < p.length; gate++) {
                    for (int f = 0; f < features; f++) p[gate] += x[(t * batch + row) * features + f] * w[gate * features + f];
                    p[gate] += bias[gate];
                    for (int f = 0; f < hidden; f++) q[gate] += h[f] * u[gate * hidden + f];
                }
                float[] next = new float[hidden];
                for (int f = 0; f < hidden; f++) {
                    if (gates == 1) next[f] = tanh(p[f] + q[f]);
                    else if (gates == 3) {
                        float r = sigmoid(p[f] + q[f]), z = sigmoid(p[hidden + f] + q[hidden + f]);
                        float n = tanh(p[2 * hidden + f] + r * q[2 * hidden + f]);
                        next[f] = n + z * (h[f] - n);
                    } else {
                        float i = sigmoid(p[f] + q[f]), forget = sigmoid(p[hidden + f] + q[hidden + f]);
                        float g = tanh(p[2 * hidden + f] + q[2 * hidden + f]);
                        float o = sigmoid(p[3 * hidden + f] + q[3 * hidden + f]);
                        c[f] = forget * c[f] + i * g;
                        next[f] = o * tanh(c[f]);
                    }
                    dense[(t * batch + row) * hidden + f] = next[f];
                }
                h = next;
            }
            System.arraycopy(h, 0, finalH, row * hidden, hidden);
        }
        return List.of(dense, finalH);
    }

    private static float tanh(float x) { return (float) Math.tanh(x); }
    private static float sigmoid(float x) { return (float) (1 / (1 + Math.exp(-x))); }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm", "rnn/rnn", "gru/gru", "lstm/lstm", "rnn/gru/lstm"})
    void everyDenseFinalAndLastOutputMatchesEquationsAndColdRoundtrip(String chain) throws Exception {
        String[] kinds = chain.split("/");
        for (String terminal : List.of("dense", "final", "last")) {
            var builder = Model.builder().input(DataType.FLOAT32, 3, 3, 2).seed(37);
            for (String kind : kinds) scan(builder, kind, 2);
            if (terminal.equals("final")) builder.finalHidden();
            if (terminal.equals("last")) builder.lastTimeStep();
            Path file = directory.resolve("scan"), resave = directory.resolve("scan-again");
            List<List<Tensor>> parameters = new ArrayList<>();
            try (var original = builder.buildWithLengths()) {
                for (int i = 0; i < kinds.length; i++) parameters.add(weights(original, i));
                original.save(file); // Save before compute; close before loading the saved source.
                assertNull(field(field(original, "execution"), "engine"));
                float[] input = new float[18];
                for (int lane = 0; lane < input.length; lane++) input[lane] = (lane - 5) * .07f;
                float[] expected = input, finalH = null;
                for (int step = 0; step < kinds.length; step++) {
                    var result = oracle(expected, 3, 3, 2, 2, new long[]{3, 1, 0}, kinds[step], parameters.get(step));
                    expected = result.getFirst(); finalH = result.getLast();
                }
                if (terminal.equals("final")) expected = finalH;
                if (terminal.equals("last")) expected = Arrays.copyOfRange(expected, 12, 18);
                assertArrayEquals(expected, lanes(original.compute(values(false, new long[]{3, 3, 2}, input), lengths(3, 1, 0))), 3e-6f);
            }
            try (var loaded = Model.loadGraph(file)) {
                Object prepared = null;
                for (long[] length : List.of(new long[]{3, 1, 0}, new long[]{0, 0, 0}, new long[]{1, 3, 2}, new long[]{3, 1, 0})) {
                    float[] data = new float[18];
                    for (int t = 0; t < 3; t++) for (int b = 0; b < 3; b++) for (int f = 0; f < 2; f++) {
                        data[(t * 3 + b) * 2 + f] = t < length[b] ? (t * 6 + b * 2 + f - 5) * .07f : Float.NaN;
                    }
                    float[] expected = data, finalH = null;
                    for (int step = 0; step < kinds.length; step++) {
                        var result = oracle(expected, 3, 3, 2, 2, length, kinds[step], parameters.get(step));
                        expected = result.getFirst(); finalH = result.getLast();
                    }
                    if (terminal.equals("final")) expected = finalH;
                    if (terminal.equals("last")) expected = Arrays.copyOfRange(expected, 12, 18);
                    Tensor x = values(false, new long[]{3, 3, 2}, data), l = lengths(length);
                    var actual = loaded.compute(x, l);
                    assertArrayEquals(expected, lanes(actual), 3e-6f, chain + " " + terminal);
                    sameValues(actual, loaded.compute(List.of(x, l)).getFirst(), 0);
                    Object current = field(field(loaded, "execution"), "inference");
                    if (prepared != null) assertSame(prepared, current);
                    prepared = current;
                    for (int lane = 0; lane < expected.length; lane++) if (expected[lane] == 0) {
                        assertEquals(0, Float.floatToRawIntBits(lanes(actual)[lane]));
                    }
                }
                loaded.save(resave);
                assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(resave));
                assertThrows(IllegalArgumentException.class, () -> loaded.compute(values(false, new long[]{3, 3, 2}, new float[18]), lengths(3, -1, 0)));
                assertThrows(IllegalArgumentException.class, () -> loaded.compute(values(false, new long[]{3, 3, 2}, new float[18]), lengths(3, 4, 0)));
                loaded.compute(values(false, new long[]{3, 3, 2}, new float[18]), lengths(3, 1, 0));
                assertSame(prepared, field(field(loaded, "execution"), "inference"));
            }
        }
    }

    @Test void denseInterstepsPrefixHeadsAndStaticRecurrenceAreNotGloballyMasked() throws Exception {
        Tensor weight = values(true, new long[]{1, 1}, .7f), bias = values(true, new long[]{1}, .2f);
        for (boolean staticMiddle : List.of(false, true)) {
            var builder = Model.builder().input(DataType.FLOAT32, 3, 3, 1).linear(weight, bias).gruScan(1)
                    .linear(weight, bias).sigmoid();
            if (staticMiddle) builder.rnn(1);
            builder.lstmScan(1).finalHidden().linear(weight, bias);
            try (var model = builder.buildWithLengths()) {
                Tensor x = values(false, new long[]{3, 3, 1}, .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f);
                long[] length = {3, 1, 0};
                float[] prefix = lanes(x); for (int i = 0; i < prefix.length; i++) prefix[i] = prefix[i] * .7f + .2f;
                float[] middle = oracle(prefix, 3, 3, 1, 1, length, "gru", weights(model, 1)).getFirst();
                for (int i = 0; i < middle.length; i++) middle[i] = sigmoid(middle[i] * .7f + .2f);
                if (staticMiddle) middle = oracle(middle, 3, 3, 1, 1, new long[]{3, 3, 3}, "rnn", weights(model, 4)).getFirst();
                float[] expected = oracle(middle, 3, 3, 1, 1, length, "lstm", weights(model, staticMiddle ? 5 : 4)).getLast();
                for (int i = 0; i < expected.length; i++) expected[i] = expected[i] * .7f + .2f;
                assertArrayEquals(expected, lanes(model.compute(x, lengths(length))), 3e-6f);
                assertEquals(.2f, expected[2]); // Classifier bias after a semantic zero final state.
                assertSame(model.stateDictionary().entries().get(0).value(), model.stateDictionary().entries().get(5).value());
            }
        }
        try (var sigmoid = Model.builder().input(DataType.FLOAT32, 3, 3, 1).rnnScan(1).sigmoid().buildWithLengths()) {
            float[] output = lanes(sigmoid.compute(values(false, new long[]{3, 3, 1}, new float[9]), lengths(3, 1, 0)));
            assertEquals(.5f, output[2]); assertEquals(.5f, output[8]);
        }
    }

    @Test void terminalGuardsFreshBuildsSignaturesAndProducedColdSubstitutions() throws Exception {
        assertThrows(IllegalStateException.class, () -> Model.builder().input(DataType.FLOAT32, 1).relu().buildWithLengths());
        var unavailable = Model.builder().input(DataType.FLOAT32, 2, 1, 1)
                .linear(TensorFactory.create(descriptor(true, 1, 1))).rnnScan(1);
        assertTrue(assertThrows(IllegalStateException.class, unavailable::build).getMessage().contains("buildWithLengths"));
        for (var builder : List.of(Model.builder().input(DataType.FLOAT32, 2, 1, 1),
                Model.builder().input(DataType.FLOAT32, 2, 1, 1).rnn(1),
                Model.builder().input(DataType.FLOAT32, 2, 1, 1).rnnScan(1).relu(),
                Model.builder().input(DataType.FLOAT32, 2, 1, 1).rnnScan(1).finalHidden())) {
            assertThrows(IllegalStateException.class, builder::finalHidden);
        }
        assertThrows(IllegalArgumentException.class, () -> Model.builder().input(DataType.FLOAT32, 1).rnnScan(0));
        assertThrows(IllegalArgumentException.class, () -> Model.builder().input(DataType.FLOAT32, 2, 1).gruScan(1).buildWithLengths());
        assertThrows(IllegalArgumentException.class, () -> Model.builder().input(DataType.FLOAT32, 2, 1, 1).rnnScan(1).flatten().lstmScan(1).buildWithLengths());
        assertThrows(IllegalArgumentException.class, () -> Model.builder().input(DataType.FLOAT32, 3, 2, 3).rnnScan(3).conv1d(1, 1).gruScan(1).buildWithLengths());
        var builder = Model.builder().input(DataType.FLOAT32, 3, 2, 1).gruScan(2).finalHidden();
        try (var a = builder.buildWithLengths(); var b = builder.buildWithLengths()) {
            assertEquals("io.github.pho001.synaptik.nn.module.Model<java.util.List<io.github.pho001.synaptik.model.tensor.Tensor>, java.util.List<io.github.pho001.synaptik.model.tensor.Tensor>>",
                    ModelBuilder.class.getMethod("buildWithLengths").getGenericReturnType().getTypeName());
            Tensor x = input(3, 2, 1).abs().contiguous(), l = lengths(3, 0).reshape(1, 2).reshape(2).contiguous();
            Tensor output = a.forward(List.of(x, l)).getFirst();
            var producer = output.provenance().orElseThrow().producer();
            assertSame(output, producer.output(1));
            assertSame(x, producer.inputs().get(0)); assertSame(l, producer.inputs().get(1));
            assertEquals(List.of("layers.0.sequence.cell.inputWeight", "layers.0.sequence.cell.hiddenWeight", "layers.0.sequence.cell.bias"),
                    a.stateDictionary().entries().stream().map(StateEntry::path).toList());
            assertEquals(2, source(a).inputs().size());
            for (Tensor prototype : source(a).inputs()) {
                assertTrue(prototype.hostStorage().isEmpty());
                assertTrue(source(a).leaves().stream().noneMatch(leaf -> leaf.tensor() == prototype));
            }
            for (int i = 0; i < a.stateDictionary().entries().size(); i++) {
                Tensor first = a.stateDictionary().entries().get(i).value(), second = b.stateDictionary().entries().get(i).value();
                assertNotSame(first, second); assertArrayEquals(bytes(first), bytes(second));
            }
            assertNull(field(field(a, "execution"), "engine"));
            assertThrows(IllegalArgumentException.class, () -> a.forward(List.of(x)));
            assertThrows(NullPointerException.class, () -> a.forward(Arrays.asList(x, null)));
            builder.relu().seed(9);
            assertSame(output.descriptor(), output.provenance().orElseThrow().outputDescriptor());
        }
        for (DataType type : List.of(DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            try (var model = Model.builder().input(type, 2, 1, 1).lstmScan(1).finalHidden().buildWithLengths()) {
                assertEquals(type, source(model).outputs().getFirst().descriptor().dataType());
                assertNull(field(field(model, "execution"), "engine"));
            }
        }
    }

    static void assertTrainingRejected(Model<?, ?> model) throws Exception {
        Object state = field(model, "execution");
        Object engine = field(state, "engine"), cached = field(state, "inference"), frozen = field(state, "frozen");
        var entries = model.stateDictionary().entries();
        var stores = entries.stream().map(e -> e.value().hostStorage().orElseThrow()).toList();
        var before = entries.stream().map(e -> bytes(e.value())).toList();
        AtomicInteger calls = new AtomicInteger();
        assertThrows(UnsupportedOperationException.class, () -> model.configureTraining((TensorDescriptor) null,
                (p, t) -> { calls.incrementAndGet(); throw new AssertionError(); }, null));
        assertThrows(UnsupportedOperationException.class, () -> model.configureTraining((List<Tensor>) null,
                (p, t) -> { calls.incrementAndGet(); throw new AssertionError(); }, null));
        assertThrows(UnsupportedOperationException.class, () -> model.train((Tensor) null, null));
        assertThrows(UnsupportedOperationException.class, () -> model.train((List<Tensor>) null, null));
        assertEquals(0, calls.get()); assertSame(engine, field(state, "engine"));
        assertSame(cached, field(state, "inference")); assertEquals(frozen, field(state, "frozen"));
        assertNull(field(state, "nativeArena")); assertNull(field(state, "session"));
        if (cached != null) assertFalse(((PreparedExecution) cached).isClosed());
        for (int i = 0; i < entries.size(); i++) {
            assertSame(stores.get(i), entries.get(i).value().hostStorage().orElseThrow());
            assertArrayEquals(before.get(i), bytes(entries.get(i).value()));
            assertTrue(entries.get(i).value().descriptor().requiresGrad());
        }
    }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void trainingRejectsAllOverloadsBeforeSetupBeforeAfterComputeAndLoad(String kind) throws Exception {
        for (boolean head : List.of(false, true)) {
            var builder = scan(Model.builder().input(DataType.FLOAT32, 2, 1, 1), kind, 1).finalHidden();
            if (head) builder.linear(1);
            Path file = directory.resolve("training-guard");
            try (var model = builder.buildWithLengths(); Engine borrowed = Engine.standard()) {
                assertTrainingRejected(model);
                model.withEngine(borrowed); model.save(file);
                assertTrainingRejected(model);
                Tensor x = values(false, new long[]{2, 1, 1}, .2f, -.1f), l = lengths(1);
                var expected = model.compute(x, l);
                assertTrainingRejected(model);
                sameValues(expected, model.compute(x, l), 0);
                try (var loaded = Model.loadGraph(file)) {
                    assertTrainingRejected(loaded);
                    sameValues(expected, loaded.compute(x, l), 0);
                    assertTrainingRejected(loaded);
                }
                model.train(); // Mode mutation must outrank unsupported training.
                assertThrows(IllegalStateException.class, () -> model.train((Tensor) null, null));
                assertNull(field(field(model, "execution"), "inference"));
                assertFalse(borrowed.isClosed());
            }
        }
    }

    @Test void scanPresenceComesFromGraphDefinitionAndLoadProvenanceIncludingFinalCellOnly() throws Exception {
        Tensor x = input(2, 1, 1), l = TensorFactory.create(lengths(1).descriptor());
        Tensor h = input(1, 1), c = input(1, 1);
        Tensor w = values(true, new long[]{4, 1}, .2f, .3f, .4f, .5f);
        Tensor u = values(true, new long[]{4, 1}, -.1f, -.2f, -.3f, -.4f);
        Tensor b = values(true, new long[]{4}, .01f, .02f, .03f, .04f);
        var scan = RecurrentScan.lstm(x, l, h, c, w, u, b, RecurrentDirection.FORWARD);
        Path graphFile = directory.resolve("cell-only");
        try (var model = Model.graph(List.of(x, l, h, c), List.of(scan.finalCell()), List.of(
                new GraphCheckpoint.Leaf("w", GraphCheckpoint.Role.PARAMETER, w),
                new GraphCheckpoint.Leaf("u", GraphCheckpoint.Role.PARAMETER, u),
                new GraphCheckpoint.Leaf("b", GraphCheckpoint.Role.PARAMETER, b)))) {
            assertTrue((Boolean) field(field(field(model, "execution"), "owned"), "containsScan"));
            assertTrainingRejected(model); model.save(graphFile);
        }
        try (var loaded = Model.loadGraph(graphFile)) {
            assertTrainingRejected(loaded);
            var fresh = List.of(values(false, new long[]{2, 1, 1}, .2f, .3f), lengths(0),
                    values(false, new long[]{1, 1}, -.2f), values(false, new long[]{1, 1}, .7f));
            assertEquals(.7f, loaded.compute(fresh).getFirst().bytes().getFloat());
            assertTrainingRejected(loaded);
        }
        AtomicInteger callbacks = new AtomicInteger();
        try (var defined = Model.defineTensor(x.descriptor(), topology -> {
            var sequence = topology.addModule("sequence", new io.github.pho001.synaptik.nn.layers.RnnSequence(
                    1, true, DataType.FLOAT32, io.github.pho001.synaptik.nn.initialization.ParameterInitialization.glorotUniform(), 7));
            return input -> {
                callbacks.incrementAndGet();
                return sequence.forwardScan(input, lengths(1), values(false, new long[]{1, 1}, 0),
                        RecurrentDirection.FORWARD).finalHidden().sigmoid();
            };
        })) {
            assertTrainingRejected(defined);
            Path file = directory.resolve("defined"); defined.save(file);
            var result = defined.compute(values(false, new long[]{2, 1, 1}, .2f, .3f));
            assertTrainingRejected(defined); assertEquals(1, callbacks.get());
            try (var loaded = Model.load(file)) {
                assertTrainingRejected(loaded);
                sameValues(result, loaded.compute(values(false, new long[]{2, 1, 1}, .2f, .3f)), 0);
                assertTrainingRejected(loaded);
            }
        }
    }

    @Test void scanRejectionPreservesLifecyclePrecedenceAndAllDeclaredInputsStayOutOfConstants() throws Exception {
        try (var model = Model.builder().input(DataType.FLOAT32, 2, 1, 1).rnnScan(1).buildWithLengths();
                Engine borrowed = Engine.standard()) {
            Object state = field(model, "execution");
            OwnedModelLifecycleTest.set(state, "busy", true);
            try {
                assertTrue(assertThrows(IllegalStateException.class, () -> model.configureTraining((TensorDescriptor) null, null, null)).getMessage().contains("busy"));
                assertTrue(assertThrows(IllegalStateException.class, () -> model.train((List<Tensor>) null, null)).getMessage().contains("busy"));
                assertTrue(assertThrows(IllegalStateException.class, () -> model.compute((Tensor) null, (Tensor) null)).getMessage().contains("busy"));
            } finally { OwnedModelLifecycleTest.set(state, "busy", false); }
            model.withEngine(borrowed); borrowed.close();
            assertTrue(assertThrows(IllegalStateException.class, () -> model.train((Tensor) null, null)).getMessage().contains("selected Engine"));
            model.close();
            assertTrue(assertThrows(IllegalStateException.class, () -> model.configureTraining((List<Tensor>) null, null, null)).getMessage().contains("selected Engine"));
        }
        Tensor x = input(1, 1), unused = TensorFactory.create(lengths(0).descriptor());
        Sequential layers = new Sequential(List.of()); layers.eval();
        try (var model = Model.fromComposition(List.of(x, unused), List.of(x.neg()), layers,
                inputs -> List.of(inputs.getFirst().neg()))) {
            assertTrue(source(model).leaves().isEmpty());
            assertEquals(2, source(model).inputs().size());
        }
    }

    @Test void scanSaveOutsideClosedSliceDoesNotFreezeOrDisableInference() throws Exception {
        Tensor x = input(2, 1, 1), l = TensorFactory.create(lengths(1).descriptor());
        var sequence = new io.github.pho001.synaptik.nn.layers.RnnSequence(1, true, DataType.FLOAT32,
                io.github.pho001.synaptik.nn.initialization.ParameterInitialization.glorotUniform(), 7);
        // A test-only ordinary adapter exposes executable ABS beyond Checkpoint's closed source slice.
        UnaryTensorModule step = new UnaryTensorModule() {
            final io.github.pho001.synaptik.nn.layers.RnnSequence owned = child("sequence", sequence);
            @Override public Tensor forward(Tensor input) {
                return owned.forwardScan(input, l, values(false, new long[]{1, 1}, 0), RecurrentDirection.FORWARD).outputs().abs();
            }
        };
        Sequential layers = new Sequential(List.of(step)); layers.eval();
        Tensor output = step.forward(x);
        try (var model = Model.fromComposition(List.of(x, l), List.of(output), layers,
                inputs -> List.of(ownedScanAbs(sequence, inputs.getFirst(), inputs.get(1)))); Engine borrowed = Engine.standard()) {
            Path target = directory.resolve("unsupported"); Files.writeString(target, "unchanged");
            assertThrows(IllegalArgumentException.class, () -> model.save(target));
            assertEquals("unchanged", Files.readString(target));
            assertNull(field(field(model, "execution"), "engine"));
            assertFalse((Boolean) field(field(model, "execution"), "frozen"));
            model.withEngine(borrowed);
            assertTrue(Float.isFinite(model.compute(values(false, new long[]{2, 1, 1}, .2f, .3f), lengths(1)).bytes().getFloat()));
            assertTrainingRejected(model);
        }
    }

    private static Tensor ownedScanAbs(io.github.pho001.synaptik.nn.layers.RnnSequence sequence, Tensor x, Tensor lengths) {
        return sequence.forwardScan(x, lengths, values(false, new long[]{1, 1}, 0), RecurrentDirection.FORWARD).outputs().abs();
    }

    @Test void ordinaryConvolutionPrefixEstablishesScanBatchAndDenseHeadKeepsItsOwnGeometry() throws Exception {
        try (var model = Model.builder().input(DataType.FLOAT32, 3, 2, 2).seed(19)
                .conv1d(3, 2).relu().rnnScan(1).conv1d(2, 1).buildWithLengths(); Engine direct = Engine.standard()) {
            assertEquals(Shape.of(3), source(model).inputs().get(1).descriptor().shape());
            Tensor x = values(false, new long[]{3, 2, 2}, .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f, .9f, 1, 1.1f, 1.2f);
            assertThrows(IllegalArgumentException.class, () -> model.compute(x, lengths(3, 1)));
            assertFalse((Boolean) field(field(model, "execution"), "frozen"));
            var state = model.stateDictionary().entries();
            Tensor prefix = x.conv1d(state.get(0).value(), state.get(1).value(),
                    new io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs(1, 0, 1, 1)).relu();
            float[] recurrent = oracle(lanes(direct.compute(prefix)), 3, 3, 1, 1, new long[]{3, 1, 0}, "rnn", weights(model, 2)).getFirst();
            Tensor reference = values(true, new long[]{3, 3, 1}, recurrent).conv1d(state.get(5).value(), state.get(6).value(),
                    new io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs(1, 0, 1, 1));
            sameValues(direct.compute(reference), model.compute(x, lengths(3, 1, 0)), 3e-6f);
            assertEquals(Shape.of(3, 2, 1), source(model).outputs().getFirst().descriptor().shape());
        }
        // Bare Conv1d's squeeze leaves a noncanonical read layout; this cold save does not execute it.
        try (var model = Model.builder().input(DataType.FLOAT32, 3, 2, 2).conv1d(3, 2).rnnScan(1).buildWithLengths()) {
            assertEquals(Shape.of(3), source(model).inputs().get(1).descriptor().shape());
            assertNull(field(field(model, "execution"), "engine"));
            model.save(directory.resolve("noncanonical-prefix"));
        }
        try (var model = Model.builder().input(DataType.FLOAT32, 2, 1, 1).seed(7)
                .rnnScan(1).finalHidden().linear(1, false).buildWithLengths()) {
            Tensor expected = io.github.pho001.synaptik.nn.initialization.ParameterInitializers.glorotUniform(
                    Shape.of(1, 1), DataType.FLOAT32, java.util.random.RandomGeneratorFactory.of("L64X128MixRandom")
                            .create(7 + 0x9E3779B97F4A7C15L * 2));
            assertArrayEquals(bytes(expected), bytes(model.stateDictionary().entries().get(3).value()));
        }
    }
}
