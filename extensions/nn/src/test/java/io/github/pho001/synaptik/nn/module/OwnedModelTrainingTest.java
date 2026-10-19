package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.nn.module.OwnedModelSourceTest.*;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.training.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Real TrainingSession comparisons, stable identities, native transition and trained persistence. */
final class OwnedModelTrainingTest {
    @TempDir Path directory;

    static Tensor loss(Tensor prediction, Tensor target) {
        Tensor error = prediction.sub(target);
        return error.mul(error);
    }

    @Test void freshTrainMatchesDirectSessionAndComputeSaveSeeSameWeights() throws Exception {
        Tensor external = scalar(2, true), x = prototype();
        TrainingStep retained;
        Path file = directory.resolve("trained");
        try (var model = Model.graph(List.of(x), List.of(external.mul(x)),
                List.of(parameter("weight", external)));
                Arena directArena = Arena.ofShared(); Engine directEngine = Engine.standard()) {
            var direct = new Weight(scalar(2, true));
            TrainingParameterStorage.copyToSharedNative(direct.trainingParameters(), directArena);
            Tensor directX = prototype(), directY = prototype();
            try (var session = TrainingSession.open(directEngine, direct.trainingParameters(),
                    loss(direct.weight.value().mul(directX), directY), new Sgd(0.1))) {
                assertEquals(6, model.compute(scalar(3, false)).bytes().getFloat());
                Object state = field(model, "execution");
                var oldInference = (io.github.pho001.synaptik.engine.PreparedExecution) field(state, "inference");
                Tensor weight = model.stateDictionary().entries().getFirst().value();
                Object oldStore = weight.hostStorage().orElseThrow();
                AtomicInteger calls = new AtomicInteger();
                model.configureTraining(prototype().descriptor(), (prediction, target) -> {
                    calls.incrementAndGet(); return loss(prediction, target);
                }, new Sgd(0.1));
                assertTrue(oldInference.isClosed());
                assertSame(weight, model.stateDictionary().entries().getFirst().value());
                assertNotSame(oldStore, weight.hostStorage().orElseThrow());
                Object nativeStore = weight.hostStorage().orElseThrow();
                var modelSession = (TrainingSession) field(state, "session");
                for (float[] batch : List.of(new float[] {3, 1}, new float[] {2, -3})) {
                    retained = model.train(scalar(batch[0], false), scalar(batch[1], false));
                    var directValues = session.inputs().stream().map(input -> scalar(
                            input.tensorId().equals(directX.id()) ? batch[0] : batch[1], false)).toList();
                    TrainingStep expected = session.runValues(directValues, GradientMode.RESET_AND_STEP);
                    assertEquals(expected.objective().bytes().getFloat(), retained.objective().bytes().getFloat());
                    assertEquals(number(direct.weight.value()), number(weight), 1e-6);
                    assertSame(nativeStore, weight.hostStorage().orElseThrow());
                    assertSame(modelSession, field(state, "session"));
                    assertEquals(number(weight) * 4, model.compute(scalar(4, false)).bytes().getFloat(), 1e-6);
                }
                assertEquals(1, calls.get());
                assertEquals(2, modelSession.executions());
                assertEquals(0, openResults((Engine) field(state, "engine")));
                assertEquals(0, modelSession.accumulatedGradientRuns());
                var snapshot = modelSession.state();
                float before = number(weight);
                assertThrows(ArithmeticException.class, () -> model.train(scalar(Float.NaN, false), scalar(0, false)));
                assertEquals(before, number(weight));
                assertEquals(snapshot.executions(), modelSession.executions());
                assertThrows(IllegalArgumentException.class, () -> model.train(weight, scalar(0, false)));
                assertEquals(snapshot.executions(), modelSession.executions());
                assertThrows(IllegalStateException.class, () -> model.configureTraining(
                        prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1)));
                model.save(file);
                retained = model.train(scalar(0, false), scalar(0, false));
            }
        }
        assertEquals(2, number(external));
        assertEquals(0, retained.objective().bytes().getFloat());
        try (var loaded = Model.load(file)) {
            assertEquals(-5.6f, loaded.compute(scalar(4, false)).bytes().getFloat(), 1e-5);
        }
    }

    @Test void setupFailureFreezesEngineButRetryDoesNotRecopyAdmittedStorage() throws Exception {
        Tensor x = prototype(), w = scalar(2, true);
        try (var model = Model.graph(List.of(x), List.of(w.mul(x)), List.of(parameter("weight", w)))) {
            var state = field(model, "execution");
            RuntimeException injected = new IllegalArgumentException("objective failure");
            assertSame(injected, assertThrows(IllegalArgumentException.class, () -> model.configureTraining(
                    prototype().descriptor(), (p, t) -> { throw injected; }, new Sgd(0.1))));
            assertTrue((Boolean) field(state, "frozen"));
            assertNull(field(state, "engine"));
            assertThrows(IllegalStateException.class, () -> model.withEngine(null));
            model.configureTraining(prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1));
            var nativeStorage = model.stateDictionary().entries().getFirst().value().hostStorage().orElseThrow();
            assertTrue(nativeStorage.segment().isNative());
            assertEquals(25, model.train(scalar(3, false), scalar(1, false)).objective().bytes().getFloat());
        }
    }

    @Test void objectiveExternalLeafAndBadScalarRejectWithoutRetainingCallback() throws Exception {
        Tensor x = prototype(), w = scalar(2, true);
        try (var model = Model.graph(List.of(x), List.of(w.mul(x)), List.of(parameter("weight", w)))) {
            Object storage = model.stateDictionary().entries().getFirst().value().hostStorage().orElseThrow();
            assertThrows(IllegalArgumentException.class, () -> model.configureTraining(prototype().descriptor(),
                    (p, t) -> p.add(scalar(1, false)), new Sgd(0.1)));
            assertSame(storage, model.stateDictionary().entries().getFirst().value().hostStorage().orElseThrow());
            model.configureTraining(prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1));
            assertEquals(ForwardMode.EVALUATION, model.mode());
        }
    }

    @Test void aliasesAndUnresolvedParameterLayoutsStayRejectedWithoutConversion() throws Exception {
        Tensor w = scalar(2, true), x = prototype();
        try (var model = Model.graph(List.of(x), List.of(w.mul(x)),
                List.of(parameter("a", w), parameter("b", w)))) {
            Tensor owned = model.stateDictionary().entries().getFirst().value();
            Object storage = owned.hostStorage().orElseThrow();
            assertThrows(IllegalArgumentException.class, () -> model.configureTraining(
                    prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1)));
            assertSame(storage, owned.hostStorage().orElseThrow());
        }
        var descriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(w.descriptor().dataType(),
                w.descriptor().shape(), java.util.Optional.empty(), true);
        Tensor unresolved = TensorFactory.create(descriptor, java.util.Optional.empty(), w.hostStorage());
        try (var model = Model.graph(List.of(x), List.of(unresolved.mul(x)),
                List.of(parameter("weight", unresolved)))) {
            assertThrows(IllegalArgumentException.class, () -> model.configureTraining(
                    prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1)));
            assertTrue(model.stateDictionary().entries().getFirst().value().descriptor().layout().isEmpty());
        }
    }

    @Test void listTrainingMapsBuffersConstantsReorderedAndEliminatedSlotsOnce() throws Exception {
        Tensor x = prototype(), unused = prototype(), w = scalar(2, true);
        Tensor buffer = scalar(3, false), constant = scalar(4, false);
        Tensor target = prototype(), unusedTarget = prototype();
        try (var model = Model.graph(List.of(unused, x), List.of(w.mul(x).add(buffer).add(constant)),
                List.of(parameter("weight", w),
                        new GraphCheckpoint.Leaf("buffer", GraphCheckpoint.Role.BUFFER, buffer),
                        new GraphCheckpoint.Leaf("constant", GraphCheckpoint.Role.CONSTANT, constant)))) {
            model.configureTraining(List.of(unusedTarget, target),
                    (p, t) -> loss(p.getFirst(), t.getLast()), new Sgd(0.1));
            Object state = field(model, "execution");
            var session = (TrainingSession) field(state, "session");
            var prepared = field(session, "preparedExecution");
            assertEquals(144, model.train(List.of(scalar(99, false), scalar(3, false)),
                    List.of(scalar(99, false), scalar(1, false))).objective().bytes().getFloat());
            assertEquals(-5.2f, number(model.stateDictionary().entries().getFirst().value()), 1e-6);
            assertThrows(IllegalStateException.class, () -> model.train(
                    List.of(unused, scalar(3, false)), List.of(scalar(0, false), scalar(1, false))));
            assertThrows(IllegalStateException.class, () -> model.train(
                    List.of(scalar(0, false), scalar(3, false)), List.of(unusedTarget, scalar(1, false))));
            assertEquals(1, session.executions());
            assertSame(prepared, field(session, "preparedExecution"));
            assertEquals(0, openResults((Engine) field(state, "engine")));
        }
    }

    @Test void failedNativeCopyOfLaterNonfiniteParameterPreservesAllOldAssociations() throws Exception {
        Tensor x = prototype(), first = scalar(2, true), bad = scalar(Float.NaN, true);
        try (var model = Model.graph(List.of(x), List.of(first.mul(x).add(bad)),
                List.of(parameter("first", first), parameter("bad", bad)))) {
            var entries = model.stateDictionary().entries();
            var stores = entries.stream().map(e -> e.value().hostStorage().orElseThrow()).toList();
            assertThrows(IllegalArgumentException.class, () -> model.configureTraining(
                    prototype().descriptor(), OwnedModelTrainingTest::loss, new Sgd(0.1)));
            for (int i = 0; i < stores.size(); i++) assertSame(stores.get(i), entries.get(i).value().hostStorage().orElseThrow());
            assertEquals(2, number(entries.getFirst().value()));
            assertTrue(Float.isNaN(number(entries.getLast().value())));
            assertNull(field(field(model, "execution"), "nativeArena"));
            assertNull(field(field(model, "execution"), "session"));
        }
    }
}
