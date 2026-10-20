package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.nn.module.ModelBuilderTest.*;
import static io.github.pho001.synaptik.nn.module.OwnedModelSourceTest.field;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Independent equations and explicit SGD: neither Model.forward nor checkpoint reconstruction is an oracle. */
final class ModelBuilderExecutionTest {
    @TempDir Path directory;

    static Tensor objective(Tensor prediction, Tensor target) {
        Tensor error = prediction.sub(target);
        return error.mul(error).sum();
    }

    static List<Tensor> copyWeights(Model<?, ?> model) {
        return model.stateDictionary().entries().stream().map(StateEntry::value)
                .map(t -> TensorFactory.fromFlatArray(t.descriptor(), Optional.empty(),
                        t.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_FLOAT))).toList();
    }

    static void sameValues(HostTensorValue expected, HostTensorValue actual, float tolerance) {
        assertEquals(expected.shape(), actual.shape());
        assertEquals(expected.bytes().remaining(), actual.bytes().remaining());
        for (int i = 0; i < expected.bytes().remaining(); i += 4) {
            assertEquals(expected.bytes().getFloat(i), actual.bytes().getFloat(i), tolerance, "lane " + i / 4);
        }
    }

    static Tensor activate(Tensor x, String kind) {
        return switch (kind) { case "relu" -> x.relu(); case "tanh" -> x.tanh();
            case "sigmoid" -> x.sigmoid(); default -> throw new AssertionError(kind); };
    }

    static ModelBuilder activation(ModelBuilder builder, String kind) {
        return switch (kind) { case "relu" -> builder.relu(); case "tanh" -> builder.tanh();
            case "sigmoid" -> builder.sigmoid(); default -> throw new AssertionError(kind); };
    }

    static Tensor mlp(Tensor x, List<Tensor> w, String kind) {
        return activate(x.linear(w.get(0), w.get(1)), kind).linear(w.get(2), w.get(3));
    }

    /** Uses Engine's independent backward publication and explicit FLOAT32 SGD arithmetic. */
    static void compareStep(Engine engine, Model<Tensor, Tensor> model, Tensor prediction,
            Tensor batch, Tensor target, List<Tensor> direct, float rate) {
        var backward = engine.backward(objective(prediction, target), direct, 1_000_000);
        float actual = model.train(batch, target).objective().bytes().getFloat();
        assertEquals(backward.objective().bytes().getFloat(), actual, 2e-5);
        boolean anyNonzero = false;
        for (int p = 0; p < direct.size(); p++) {
            var memory = direct.get(p).hostStorage().orElseThrow().segment();
            var owned = model.stateDictionary().entries().get(p).value().hostStorage().orElseThrow().segment();
            var gradient = backward.gradients().get(p).bytes();
            for (int lane = 0; lane < memory.byteSize() / 4; lane++) {
                float before = memory.get(ValueLayout.JAVA_FLOAT_UNALIGNED, lane * 4L);
                float g = gradient.getFloat(lane * 4);
                anyNonzero |= Math.abs(g) > 1e-7;
                float after = before - rate * g;
                memory.set(ValueLayout.JAVA_FLOAT_UNALIGNED, lane * 4L, after);
                assertEquals(after, owned.get(ValueLayout.JAVA_FLOAT_UNALIGNED, lane * 4L), 2e-5);
            }
        }
        assertTrue(anyNonzero, "fixture must actually train");
    }

    @ParameterizedTest @ValueSource(strings = {"relu", "tanh", "sigmoid"})
    void mlpComputeTrainAndReloadMatchDirectEquationsAndEverySgdUpdate(String kind) throws Exception {
        List<Tensor> caller = List.of(values(true, new long[]{2, 2}, .4f, -.2f, .15f, .3f),
                values(true, new long[]{2}, .1f, -.05f), values(true, new long[]{1, 2}, .7f, -.4f),
                values(true, new long[]{1}, .2f));
        var stores = caller.stream().map(t -> t.hostStorage().orElseThrow()).toList();
        var original = caller.stream().map(ModelBuilderTest::bytes).toList();
        var builder = activation(Model.builder().input(DataType.FLOAT32, 1, 2)
                .linear(caller.get(0), caller.get(1)), kind).linear(caller.get(2), caller.get(3));
        Path initial = directory.resolve("initial"), trained = directory.resolve("trained");
        try (var model = builder.build(); Engine engine = Engine.standard()) {
            model.save(initial);
            try (var loaded = Model.load(initial)) {
                for (var subject : List.of(model, loaded)) {
                    List<Tensor> direct = copyWeights(subject);
                    subject.configureTraining(descriptor(false, 1, 1), ModelBuilderExecutionTest::objective, new Sgd(.01));
                    var state = field(subject, "execution");
                    var session = field(state, "session");
                    var storage = subject.stateDictionary().entries().stream()
                            .map(e -> e.value().hostStorage().orElseThrow()).toList();
                    for (float[] lanes : List.of(new float[]{1, 2}, new float[]{-.5f, 1})) {
                        Tensor batch = values(false, new long[]{1, 2}, lanes);
                        Tensor target = values(false, new long[]{1, 1}, .3f);
                        sameValues(engine.compute(mlp(batch, direct, kind)), subject.compute(batch), 1e-5f);
                        compareStep(engine, subject, mlp(batch, direct, kind), batch, target, direct, .01f);
                        sameValues(engine.compute(mlp(batch, direct, kind)), subject.compute(batch), 1e-5f);
                        assertSame(session, field(state, "session"));
                        for (int p = 0; p < storage.size(); p++) assertSame(storage.get(p),
                                subject.stateDictionary().entries().get(p).value().hostStorage().orElseThrow());
                    }
                    subject.save(trained);
                    try (var reloaded = Model.load(trained)) {
                        Tensor batch = values(false, new long[]{1, 2}, 2, -1);
                        sameValues(engine.compute(mlp(batch, direct, kind)), reloaded.compute(batch), 1e-5f);
                    }
                }
            }
        }
        for (int p = 0; p < caller.size(); p++) {
            assertSame(stores.get(p), caller.get(p).hostStorage().orElseThrow());
            assertArrayEquals(original.get(p), bytes(caller.get(p)));
        }
    }

    @Test void fiveKindBuilderStillSavesAndReloadsVersionOne() throws Exception {
        Path file = directory.resolve("v1");
        try (var model = Model.builder().input(DataType.FLOAT32, 1, 2).linear(1).build()) {
            Tensor batch = values(false, new long[]{1, 2}, 1, 2);
            model.save(file);
            assertEquals("graph-model.v1", io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile
                    .read(file, ModelExecutionState.DEFAULT_LIMITS).kind());
            try (var loaded = Model.load(file)) {
                sameValues(model.compute(batch), loaded.compute(batch), 0);
                loaded.configureTraining(descriptor(false, 1, 1), ModelBuilderExecutionTest::objective, new Sgd(.01));
                loaded.train(batch, values(false, new long[]{1, 1}, .3f));
            }
        }
    }

    static ModelBuilder recurrent(ModelBuilder builder, String kind, long hidden) {
        return switch (kind) { case "rnn" -> builder.rnn(hidden); case "gru" -> builder.gru(hidden);
            case "lstm" -> builder.lstm(hidden); default -> throw new AssertionError(kind); };
    }

    /** Direct full-batch equations, no Sequence/cell/Model/builder forward sharing. */
    static Tensor unroll(Tensor input, List<Tensor> w, String kind, int time, int batch, int hidden) {
        Tensor h = values(false, new long[]{batch, hidden}, new float[batch * hidden]);
        Tensor c = values(false, new long[]{batch, hidden}, new float[batch * hidden]);
        List<Tensor> outputs = new ArrayList<>();
        for (int t = 0; t < time; t++) {
            Tensor x = input.select(0, t).linear(w.get(0), w.get(2));
            Tensor y = h.linear(w.get(1));
            if (kind.equals("rnn")) h = x.add(y).tanh();
            else if (kind.equals("gru")) {
                Tensor reset = x.sliceAxis(-1, 0, hidden).add(y.sliceAxis(-1, 0, hidden)).sigmoid();
                Tensor update = x.sliceAxis(-1, hidden, 2L * hidden)
                        .add(y.sliceAxis(-1, hidden, 2L * hidden)).sigmoid();
                Tensor candidate = x.sliceAxis(-1, 2L * hidden, 3L * hidden)
                        .add(reset.mul(y.sliceAxis(-1, 2L * hidden, 3L * hidden))).tanh();
                h = candidate.add(update.mul(h.sub(candidate)));
            } else {
                Tensor i = x.sliceAxis(-1, 0, hidden).add(y.sliceAxis(-1, 0, hidden)).sigmoid();
                Tensor f = x.sliceAxis(-1, hidden, 2L * hidden)
                        .add(y.sliceAxis(-1, hidden, 2L * hidden)).sigmoid();
                Tensor g = x.sliceAxis(-1, 2L * hidden, 3L * hidden)
                        .add(y.sliceAxis(-1, 2L * hidden, 3L * hidden)).tanh();
                Tensor o = x.sliceAxis(-1, 3L * hidden, 4L * hidden)
                        .add(y.sliceAxis(-1, 3L * hidden, 4L * hidden)).sigmoid();
                c = f.mul(c).add(i.mul(g));
                h = o.mul(c.tanh());
            }
            outputs.add(h.reshape(1, batch, hidden));
        }
        return Tensor.concat(0, outputs.toArray(Tensor[]::new));
    }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void allRecurrentTimeOutputsChangedBatchesAndExplicitFinalStepMatchDirectEquations(String kind) throws Exception {
        try (var model = recurrent(Model.builder().input(DataType.FLOAT32, 3, 2, 2).seed(19), kind, 2).build();
                var last = recurrent(Model.builder().input(DataType.FLOAT32, 3, 2, 2).seed(19), kind, 2)
                        .lastTimeStep().build(); Engine engine = Engine.standard()) {
            List<Tensor> weights = copyWeights(model);
            Tensor first = values(false, new long[]{3, 2, 2}, .1f, .2f, -.3f, .4f, .5f, -.6f, .7f, .8f, -.9f, 1, .2f, -.1f);
            Tensor changed = values(false, new long[]{3, 2, 2}, -.2f, -.4f, .1f, .3f, .7f, .2f, -.6f, .1f, .1f, -.5f, .8f, .3f);
            var fixed = source(model);
            Object prepared = null;
            for (Tensor batch : List.of(first, first, changed, first)) {
                Tensor expected = unroll(batch, weights, kind, 3, 2, 2);
                sameValues(engine.compute(expected), model.compute(batch), 2e-5f);
                sameValues(engine.compute(expected.select(0, 2)), last.compute(batch), 2e-5f);
                var current = field(field(model, "execution"), "inference");
                if (prepared != null) assertSame(prepared, current);
                prepared = current;
                assertSame(fixed, source(model));
            }
        }
    }

    @Test void validMultiStepRnnTrainingMatchesDirectUnrolledGradientsAndSgd() throws Exception {
        try (var model = Model.builder().input(DataType.FLOAT32, 3, 1, 2).seed(29).rnn(2).build();
                Engine engine = Engine.standard()) {
            List<Tensor> weights = copyWeights(model);
            model.configureTraining(descriptor(false, 3, 1, 2), ModelBuilderExecutionTest::objective, new Sgd(.01));
            var session = field(field(model, "execution"), "session");
            for (float[] lanes : List.of(new float[]{.1f, -.2f, .3f, .4f, -.2f, .6f},
                    new float[]{-.4f, .2f, .5f, -.1f, .7f, .3f})) {
                Tensor batch = values(false, new long[]{3, 1, 2}, lanes);
                Tensor target = values(false, new long[]{3, 1, 2}, .2f, -.1f, .3f, .1f, -.3f, .4f);
                compareStep(engine, model, unroll(batch, weights, "rnn", 3, 1, 2), batch, target, weights, .01f);
                sameValues(engine.compute(unroll(batch, weights, "rnn", 3, 1, 2)), model.compute(batch), 2e-5f);
                assertSame(session, field(field(model, "execution"), "session"));
            }
        }
    }

    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void stackedRecurrentLayersAndLastTimeProjectionMatchIndependentEquations(String kind) throws Exception {
        var allBuilder = recurrent(recurrent(Model.builder().input(DataType.FLOAT32, 3, 2, 2)
                .seed(17), kind, 3), kind, 2);
        var lastBuilder = recurrent(recurrent(Model.builder().input(DataType.FLOAT32, 3, 2, 2)
                .seed(17), kind, 3), kind, 2).lastTimeStep().linear(1);
        try (var all = allBuilder.build(); var last = lastBuilder.build(); Engine engine = Engine.standard()) {
            List<Tensor> w = copyWeights(last);
            for (float sign : new float[]{1, -1, 1}) {
                Tensor batch = values(false, new long[]{3, 2, 2}, sign * .1f, -.2f, .3f, sign * .4f,
                        .5f, -.6f, sign * .7f, .8f, -.9f, 1, sign * .2f, -.1f);
                Tensor first = unroll(batch, w.subList(0, 3), kind, 3, 2, 3);
                Tensor second = unroll(first, w.subList(3, 6), kind, 3, 2, 2);
                sameValues(engine.compute(second), all.compute(batch), 2e-5f);
                Tensor projected = second.select(0, 2).linear(w.get(6), w.get(7));
                sameValues(engine.compute(projected), last.compute(batch), 2e-5f);
            }
        }
    }

    @Test void approvedCnnNumericallyExecutesWithTheDocumentedSpatialGeometry() throws Exception {
        try (var model = Model.builder().input(DataType.FLOAT32, 32, 1, 28, 28).seed(42)
                .conv2d(32, 3, 3).relu().maxPool2d(2, 2).conv2d(64, 3, 3).relu()
                .maxPool2d(2, 2).flatten().linear(128).relu().linear(10).build(); Engine engine = Engine.standard()) {
            List<Tensor> w = copyWeights(model);
            float[] lanes = new float[32 * 28 * 28];
            for (int i = 0; i < lanes.length; i++) lanes[i] = (i % 17 - 8) * .05f;
            Tensor batch = values(false, new long[]{32, 1, 28, 28}, lanes);
            var attrs = new Conv2dAttrs(1, 1, 0, 0, 1, 1, 1);
            var pool = new MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false);
            Tensor x = batch.conv2d(w.get(0), w.get(1), attrs).relu();
            assertEquals(io.github.pho001.synaptik.model.shape.Shape.of(32, 32, 26, 26), x.descriptor().shape());
            x = x.maxPool2d(pool).conv2d(w.get(2), w.get(3), attrs).relu().maxPool2d(pool);
            assertEquals(io.github.pho001.synaptik.model.shape.Shape.of(32, 64, 5, 5), x.descriptor().shape());
            Tensor flattened = x.reshape(32, 1600);
            Tensor expected = flattened.linear(w.get(4), w.get(5)).relu().linear(w.get(6), w.get(7));
            sameValues(engine.compute(expected), model.compute(batch), 2e-5f);
        }
    }

    @Test void conv1dAndConv3dExecuteButConv3dBackwardRemainsRejected() {
        try (var one = Model.builder().input(DataType.FLOAT32, 1, 1, 4).conv1d(2, 2).build();
                var three = Model.builder().input(DataType.FLOAT32, 1, 1, 3, 3, 3).conv3d(2, 2, 2, 2).build();
                Engine engine = Engine.standard()) {
            Tensor x = values(false, new long[]{1, 1, 4}, 1, -2, 3, .5f);
            var w = copyWeights(one);
            sameValues(engine.compute(x.conv1d(w.get(0), w.get(1), new Conv1dAttrs(1, 0, 1, 1))), one.compute(x), 1e-5f);
            float[] lanes = new float[27]; for (int i = 0; i < 27; i++) lanes[i] = (i - 13) * .1f;
            Tensor volume = values(false, new long[]{1, 1, 3, 3, 3}, lanes);
            w = copyWeights(three);
            sameValues(engine.compute(volume.conv3d(w.get(0), w.get(1),
                    new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 1))), three.compute(volume), 1e-5f);
            assertThrows(IllegalArgumentException.class, () -> three.configureTraining(descriptor(false, 1, 2, 2, 2, 2),
                    ModelBuilderExecutionTest::objective, new Sgd(.01)));
        }
    }

    /**
     * Checks closed-before-load recurrence against independent full-batch Tensor equations and SGD.
     * @param kind RNN, GRU or LSTM fixture selector
     * @throws Exception if supported persistence, reflection or execution fails
     */
    @ParameterizedTest @ValueSource(strings = {"rnn", "gru", "lstm"})
    void coldReloadedAllTimeLastAndStackedRecurrentModelsMatchEveryFreshSgdUpdate(String kind) throws Exception {
        for (int depth : new int[]{1, 2}) for (boolean last : List.of(false, true)) {
            var builder = recurrent(Model.builder().input(DataType.FLOAT32, 2, 1, 1).seed(23), kind, 1);
            if (depth == 2) recurrent(builder, kind, 1);
            if (last) builder.lastTimeStep();
            Path file = directory.resolve("recurrent"), trained = directory.resolve("trained-recurrent");
            List<Tensor> weights;
            try (var original = builder.build()) {
                weights = copyWeights(original);
                original.save(file); // Save before compute, then close the original before cold loading.
                assertNull(field(field(original, "execution"), "engine"));
            }
            try (var model = Model.load(file); Engine engine = Engine.standard()) {
                assertNull(field(field(model, "execution"), "engine"));
                Tensor target = last ? values(false, new long[]{1, 1}, .13f)
                        : values(false, new long[]{2, 1, 1}, .13f, -.07f);
                model.configureTraining(target.descriptor(), ModelBuilderExecutionTest::objective, new Sgd(.01));
                Object session = field(field(model, "execution"), "session");
                for (float[] lanes : List.of(new float[]{.2f, -.3f}, new float[]{-.1f, .4f})) {
                    Tensor batch = values(false, new long[]{2, 1, 1}, lanes);
                    Tensor expected = unroll(batch, weights.subList(0, 3), kind, 2, 1, 1);
                    if (depth == 2) expected = unroll(expected, weights.subList(3, 6), kind, 2, 1, 1);
                    if (last) expected = expected.select(0, 1);
                    sameValues(engine.compute(expected), model.compute(batch), 2e-5f);
                    compareStep(engine, model, expected, batch, target, weights, .01f);
                    assertSame(session, field(field(model, "execution"), "session"));
                }
                model.save(trained);
                try (var restored = Model.load(trained)) {
                    var state = model.stateDictionary().entries();
                    for (int p = 0; p < state.size(); p++) assertArrayEquals(bytes(state.get(p).value()),
                            bytes(restored.stateDictionary().entries().get(p).value()));
                    Tensor batch = values(false, new long[]{2, 1, 1}, .4f, -.2f);
                    sameValues(model.compute(batch), restored.compute(batch), 0);
                }
            }
        }
    }

    /**
     * Verifies fresh loaded CNN training, every parameter update and persisted trained raw values.
     * @throws Exception if supported persistence or execution fails
     */
    @Test void coldReloadedConv1dAndPooledConv2dTrainAllParametersAndResaveCurrentValues() throws Exception {
        for (boolean two : List.of(false, true)) {
            var builder = two ? Model.builder().input(DataType.FLOAT32, 1, 1, 4, 4).seed(31)
                    .conv2d(1, 2, 2).maxPool2d(2, 2).flatten().linear(1)
                    : Model.builder().input(DataType.FLOAT32, 1, 1, 4).seed(31).conv1d(1, 2).flatten().linear(1);
            Path file = directory.resolve("cnn"); List<Tensor> w;
            try (var original = builder.build()) { w = copyWeights(original); original.save(file); }
            try (var model = Model.load(file); Engine engine = Engine.standard()) {
                Tensor target = values(false, new long[]{1, 1}, .2f);
                model.configureTraining(target.descriptor(), ModelBuilderExecutionTest::objective, new Sgd(.01));
                for (float sign : new float[]{1, -1}) {
                    float[] lanes = new float[two ? 16 : 4];
                    for (int i = 0; i < lanes.length; i++) lanes[i] = sign * (i + 1) * .07f;
                    Tensor batch = values(false, two ? new long[]{1, 1, 4, 4} : new long[]{1, 1, 4}, lanes);
                    Tensor expected = two ? batch.conv2d(w.get(0), w.get(1), Conv2dAttrs.defaults())
                            .maxPool2d(new MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false)).reshape(1, 1)
                            : batch.conv1d(w.get(0), w.get(1), Conv1dAttrs.defaults()).reshape(1, 3);
                    expected = expected.linear(w.get(2), w.get(3));
                    sameValues(engine.compute(expected), model.compute(batch), 2e-5f);
                    compareStep(engine, model, expected, batch, target, w, .01f);
                }
                model.save(file);
                try (var restored = Model.load(file)) {
                    for (int p = 0; p < w.size(); p++) assertArrayEquals(bytes(model.stateDictionary().entries().get(p).value()),
                            bytes(restored.stateDictionary().entries().get(p).value()));
                }
            }
        }
    }
}
