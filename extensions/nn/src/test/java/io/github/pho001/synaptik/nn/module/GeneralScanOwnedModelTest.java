package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.nn.module.OwnedModelSourceTest.*;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Leaf;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Role;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.nn.layers.RnnCell;
import io.github.pho001.synaptik.nn.layers.RnnSequence;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingStep;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Bounded cold admission and ordinary owned execution, with no second scan execution owner. */
final class GeneralScanOwnedModelTest {
    @TempDir Path directory;

    @Test void coldInventoryIsNotCpuAdmissionAndEveryCanonicalRootRebindsWithoutEngine() throws Exception {
        for (DataType type : List.of(DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64))
            for (int time : List.of(0, 3)) for (boolean reverse : List.of(false, true)) {
                Tensor initial = stored(descriptor(type, false, 2), .25, -.0);
                Tensor sequence = stored(descriptor(type, false, time, 2), new double[time * 2]);
                Tensor weight = stored(descriptor(type, true, 2), .5, -.0);
                Tensor h = TensorFactory.create(initial.descriptor()), x = TensorFactory.create(initial.descriptor());
                Tensor w = TensorFactory.create(weight.descriptor());
                // RELU has cold support independently of the initial general-scan CPU algebra.
                Tensor next = type == DataType.FLOAT32 ? h.add(x).mul(w).relu() : h.add(x).mul(w).tanh();
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
                var oldStore = weight.hostStorage().orElseThrow(); byte[] before = bits(weight);
                for (Tensor root : List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst(),
                        scan.residuals().getLast(), scan.residuals().getLast().neg())) {
                    try (var model = Model.graph(List.of(sequence, initial), List.of(root),
                            List.of(parameter("a.weight", weight), parameter("b.weight", weight)))) {
                        Object state = field(model, "execution"); var source = source(model);
                        assertNull(field(state, "engine"));
                        assertFalse((Boolean) field(field(state, "owned"), "containsScan"));
                        for (int i = 0; i < 2; i++) {
                            assertTrue(source.inputs().get(i).hostStorage().isEmpty());
                            assertEquals(List.of(sequence, initial).get(i).descriptor(), source.inputs().get(i).descriptor());
                        }
                        Tensor owned = source.leaves().getFirst().tensor();
                        assertSame(owned, source.leaves().getLast().tensor()); assertNotSame(weight, owned);
                        assertEquals(weight.descriptor(), owned.descriptor()); assertArrayEquals(before, bits(owned));
                        Tensor produced = TensorFactory.create(sequence.descriptor()).neg().contiguous();
                        Tensor freshInitial = TensorFactory.create(initial.descriptor());
                        Path artifact = directory.resolve("cold-inventory");
                        model.save(artifact);
                        try (var loaded = Model.loadGraph(artifact)) {
                            var loadedState = field(loaded, "execution");
                            assertNull(field(loadedState, "engine"));
                            assertTrue(source(loaded).inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                            Tensor adopted = loaded.stateDictionary().entries().getFirst().value();
                            assertArrayEquals(before, bits(adopted));
                            assertNotSame(owned, adopted);
                            assertSame(adopted, loaded.stateDictionary().entries().getLast().value());
                            assertEquals(root.descriptor(), loaded.forward(List.of(produced, freshInitial)).getFirst().descriptor());
                            assertNull(field(loadedState, "engine"));
                        }
                        Tensor rebound = model.forward(List.of(produced, freshInitial)).getFirst();
                        var producer = scanProducer(rebound);
                        assertSame(produced, producer.inputs().get(1));
                        assertSame(freshInitial, producer.inputs().getFirst()); assertSame(owned, producer.inputs().get(2));
                        assertEquals(root.descriptor(), rebound.descriptor());
                        assertNotSame(body, producer.scanBody().orElseThrow());
                        assertNotSame(h, producer.scanBody().orElseThrow().carry().getFirst());
                        assertEquals(scan.finalCarry().getFirst().provenance().orElseThrow().producer().outputDescriptors(),
                                producer.outputDescriptors());
                        assertNull(field(state, "engine"));
                    }
                }
                assertSame(oldStore, weight.hostStorage().orElseThrow()); assertArrayEquals(before, bits(weight));
            }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64", "BFLOAT16", "FLOAT16"})
    void explicitRnnLengthScanAcquiresRebindsAndPersistsOriginalCarriersWithoutExecution(DataType type) throws Exception {
        for (RecurrentDirection direction : RecurrentDirection.values()) for (boolean biased : List.of(false, true)) {
            Tensor input = TensorFactory.create(descriptor(type, true, 3, 2, 3));
            Tensor initial = TensorFactory.create(descriptor(type, true, 2, 2));
            Tensor lengths = TensorFactory.create(descriptor(DataType.INT64, false, 2));
            Tensor w = stored(descriptor(type, true, 2, 3), .3, -.7, .11, .2, .4, -.6);
            Tensor u = stored(descriptor(type, true, 2, 2), -.2, .5, .7, -.1);
            Tensor b = stored(descriptor(type, true, 2), .13, -.17);
            var cell = biased ? new RnnCell(w, u, b) : new RnnCell(w, u);
            var sequence = new RnnSequence(cell);
            var scan = sequence.forwardGeneralScan(input, lengths, initial, direction);
            var parameters = new ArrayList<>(List.of(parameter("cell.inputWeight", w), parameter("cell.hiddenWeight", u)));
            if (biased) parameters.add(parameter("cell.bias", b));
            var outputs = List.of(scan.outputs(), scan.finalHidden());
            Path artifact = directory.resolve("explicit-rnn-" + type + "-" + direction + "-" + biased);
            try (var model = Model.graph(List.of(input, lengths, initial), outputs, parameters)) {
                assertNull(field(field(model, "execution"), "engine"));
                model.save(artifact);
                try (var loaded = Model.loadGraph(artifact)) {
                    assertNull(field(field(loaded, "execution"), "engine"));
                    for (var parameter : parameters) {
                        Tensor adopted = loaded.stateDictionary().entries().stream()
                                .filter(entry -> entry.path().equals(parameter.name())).findFirst().orElseThrow().value();
                        assertEquals(type, adopted.descriptor().dataType());
                        assertArrayEquals(bits(parameter.tensor()), bits(adopted));
                        assertNotSame(parameter.tensor(), adopted);
                    }
                    // Descriptor-only rebind accepts produced inputs and never reads lengths.
                    Tensor freshInput = TensorFactory.create(input.descriptor()).neg();
                    Tensor freshInitial = TensorFactory.create(initial.descriptor()).neg();
                    Tensor freshLengths = TensorFactory.create(lengths.descriptor());
                    var rebound = loaded.forward(List.of(freshInput, freshLengths, freshInitial));
                    for (int i = 0; i < outputs.size(); i++) assertEquals(outputs.get(i).descriptor(), rebound.get(i).descriptor());
                    var producer = scanProducer(rebound.getFirst());
                    assertSame(freshInitial, producer.inputs().getFirst());
                    assertSame(freshInput, producer.inputs().get(1));
                    assertSame(freshLengths, producer.inputs().getLast());
                    assertEquals(type, producer.scanBody().orElseThrow().nextCarry().getFirst().descriptor().dataType());
                    // A fresh Training objective is built after acquisition, never persisted.
                    Tensor objective = rebound.getLast().cast(DataType.FLOAT32).sum();
                    assertEquals(DataType.FLOAT32, objective.descriptor().dataType());
                    assertTrue(objective.descriptor().requiresGrad());
                    assertNull(field(field(loaded, "execution"), "engine"));
                }
                assertNull(field(field(model, "execution"), "engine"));
            }
        }
    }

    @Test void definitionSnapshotsOnceInstallsPrivateWrappersAndDiscoversConstants() throws Exception {
        try (Engine borrowed = Engine.standard()) {
            for (boolean selectEngine : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                Tensor external = stored(descriptor(DataType.FLOAT32, true, 2), .5, -.25);
                Tensor initial = stored(descriptor(DataType.FLOAT32, false, 2), .1, .2);
                var store = external.hostStorage().orElseThrow();
                AtomicInteger definitions = new AtomicInteger(), forwards = new AtomicInteger();
                AtomicReference<Weight> child = new AtomicReference<>();
                ModelDefinition<Tensor, Tensor> definition = topology -> {
                    definitions.incrementAndGet(); child.set(topology.addModule("layer", new Weight(external)));
                    return sequence -> {
                        forwards.incrementAndGet(); assertEquals(ForwardMode.EVALUATION, child.get().mode());
                        return pointwise(initial, sequence, external, reverse).finalCarry().getFirst();
                    };
                };
                var signature = descriptor(DataType.FLOAT32, false, 3, 2);
                try (var model = selectEngine ? Model.defineTensor(signature, definition, borrowed)
                        : Model.defineTensor(signature, definition)) {
                    Tensor owned = model.stateDictionary().entries().getFirst().value();
                    assertSame(owned, child.get().weight.value()); assertNotSame(external, owned);
                    assertEquals(List.of("layer.weight"), model.stateDictionary().entries().stream().map(StateEntry::path).toList());
                    assertEquals(List.of(Role.PARAMETER, Role.CONSTANT), source(model).leaves().stream().map(Leaf::role).toList());
                    Tensor replacement = TensorFactory.create(signature).neg().contiguous();
                    assertSame(replacement, scanProducer(model.forward(replacement)).inputs().get(1));
                    for (int batch = 0; batch < 2; batch++) model.compute(stored(signature, new double[6]));
                    assertEquals(1, definitions.get()); assertEquals(1, forwards.get());
                    assertSame(store, external.hostStorage().orElseThrow());
                    assertArrayEquals(new double[]{.5, -.25}, numbers(external));
                }
                assertFalse(borrowed.isClosed());
            }
        }
    }

    @Test void unsupportedColdBodyAndContainerFailuresRemainWithTheirOwners() {
        Tensor initial = TensorFactory.create(descriptor(DataType.FLOAT32, false, 2));
        Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 0, 2));
        Tensor weight = TensorFactory.create(descriptor(DataType.FLOAT32, true, 2)); // no readable state
        Tensor h = TensorFactory.create(initial.descriptor()), x = TensorFactory.create(initial.descriptor());
        Tensor w = TensorFactory.create(weight.descriptor());
        var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x).mul(w).abs()), List.of());
        var output = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, false).finalCarry();
        assertThrows(IllegalArgumentException.class, () -> Model.graph(List.of(sequence, initial), output,
                List.of(parameter("weight", weight))));
        assertThrows(NullPointerException.class, () -> Model.graph(null, output, List.of()));
        assertThrows(NullPointerException.class, () -> Model.graph(List.of(sequence),
                java.util.Arrays.asList(output.getFirst(), null), List.of()));
        var supported = pointwise(initial, sequence, weight, false).finalCarry();
        assertThrows(IllegalStateException.class, () -> Model.graph(List.of(sequence, initial), supported,
                List.of(parameter("weight", weight))));
    }

    @Test void repeatedActualPositionsOccurrencesAndAliasedRolesStayDistinct() throws Exception {
        Tensor initial = stored(descriptor(DataType.FLOAT32, true, 2), .25, .5);
        Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 2));
        Tensor h = TensorFactory.create(initial.descriptor()), x = TensorFactory.create(descriptor(DataType.FLOAT32, false, 2));
        Tensor a = TensorFactory.create(initial.descriptor()), b = TensorFactory.create(initial.descriptor());
        Tensor next = h.mul(a).add(x).add(b);
        var body = new ScanBody(List.of(h), List.of(x), List.of(a, b), List.of(next), List.of(next));
        var first = Scan.apply(List.of(initial), List.of(sequence), List.of(initial, initial), body, false);
        var second = Scan.apply(List.of(initial), List.of(sequence), List.of(initial, initial), body, true);
        var inputList = new ArrayList<>(List.of(sequence));
        var outputs = new ArrayList<>(List.of(first.finalCarry().getFirst(), first.outputs().getFirst(),
                second.finalCarry().getFirst()));
        var leaves = new ArrayList<>(List.of(parameter("a", initial), parameter("b", initial)));
        try (var model = Model.graph(inputList, outputs, leaves, Engine.standard())) {
            Engine borrowed = (Engine) field(field(model, "execution"), "engine");
            try {
                inputList.clear(); outputs.clear(); leaves.clear();
                var source = source(model);
                var p = scanProducer(source.outputs().getFirst());
                var q = scanProducer(source.outputs().getLast());
                assertSame(p, scanProducer(source.outputs().get(1))); assertNotSame(p, q);
                assertSame(source.outputs().getFirst(), p.output(0)); assertSame(source.outputs().get(1), p.output(1));
                assertSame(p.inputs().getFirst(), p.inputs().get(2)); assertSame(p.inputs().get(2), p.inputs().get(3));
                assertSame(p.inputs().getFirst(), q.inputs().getFirst());
                assertNotSame(p.scanBody().orElseThrow(), q.scanBody().orElseThrow());
                assertEquals(2, p.scanBody().orElseThrow().invariants().size());
                var fresh = stored(sequence.descriptor(), .1, -.2, .3, .4, -.5, .6);
                assertEquals(3, model.compute(List.of(fresh)).size());
                var store = source.leaves().getFirst().tensor().hostStorage().orElseThrow();
                assertThrows(IllegalArgumentException.class, () -> model.configureTraining(List.of(),
                        (predictions, targets) -> predictions.getFirst().sum(), new Sgd(.01)));
                assertSame(store, source.leaves().getFirst().tensor().hostStorage().orElseThrow());
            } finally { model.close(); assertFalse(borrowed.isClosed()); borrowed.close(); }
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void freshComputeMatchesRepresentedRecurrenceWithEmptyTimeAndStepDomains(DataType type) throws Exception {
        for (boolean loaded : List.of(false, true)) for (boolean reverse : List.of(false, true))
                for (int time : List.of(0, 3)) for (int n : List.of(0, 2)) {
            Tensor initial = TensorFactory.create(descriptor(type, false, n));
            Tensor sequence = TensorFactory.create(descriptor(type, false, time, n));
            double[] weights = new double[n]; java.util.Arrays.fill(weights, .5);
            Tensor external = stored(descriptor(type, true, n), weights);
            var scan = pointwise(initial, sequence, external, reverse);
            HostTensorValue retained;
            try (Engine borrowed = Engine.standard(); var model = graphOrLoaded(List.of(sequence, initial),
                    List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()), List.of(parameter("weight", external)), borrowed, loaded)) {
                Object state = field(model, "execution");
                Object prepared = null, slots = null;
                for (int batch = 1; batch <= 2; batch++) {
                    double[] data = new double[time * n], start = new double[n];
                    for (int i = 0; i < data.length; i++) data[i] = (i + batch) * .0625;
                    java.util.Arrays.fill(start, -.25 * batch);
                    var values = model.compute(List.of(stored(sequence.descriptor(), data), stored(initial.descriptor(), start)));
                    var expected = recurrence(type, start, data, weights, time, reverse);
                    assertArrayEquals(expected.get(0), numbers(values.get(0)));
                    assertArrayEquals(expected.get(1), numbers(values.get(1)));
                    if (batch == 1) {
                        prepared = field(state, "inference"); slots = field(state, "inferenceSlots");
                        poison(borrowed);
                    } else { assertSame(prepared, field(state, "inference")); assertSame(slots, field(state, "inferenceSlots")); }
                    assertEquals(0, openResults(borrowed)); retained = values.getFirst();
                }
                assertTrue(source(model).inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                var last = model.compute(List.of(stored(sequence.descriptor(), new double[time * n]),
                        stored(initial.descriptor(), new double[n]))).getFirst();
                model.close(); assertFalse(borrowed.isClosed()); assertEquals(n, numbers(last).length);
            }
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void listTrainingMatchesUnrolledLossDecodedGradientsAndSgdWithOneNativeTransition(DataType type) throws Exception {
        for (boolean loaded : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
            Tensor initial = TensorFactory.create(descriptor(type, false, 2, 2));
            Tensor sequence = TensorFactory.create(descriptor(type, false, 3, 2, 2));
            Tensor external = stored(descriptor(type, true, 2), .5, -.25);
            Tensor h = TensorFactory.create(initial.descriptor()), x = TensorFactory.create(initial.descriptor());
            Tensor w = TensorFactory.create(external.descriptor());
            Tensor next = h.mul(w).add(x).tanh();
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next.sum(0, false)));
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(external), body, reverse);
            var targetH = TensorFactory.create(descriptor(type, false, 2, 2));
            var targetY = TensorFactory.create(descriptor(type, false, 3, 2));
            byte[] original = bits(external); var originalStore = external.hostStorage().orElseThrow();
            TrainingStep retained;
            try (Engine borrowed = Engine.standard(); Engine oracle = Engine.standard();
                    var model = graphOrLoaded(List.of(sequence, initial), List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()),
                            List.of(parameter("weight", external)), borrowed, loaded)) {
                var state = field(model, "execution");
                var wrapper = model.parametersRecursively().get("weight"); Tensor owned = wrapper.value();
                model.compute(List.of(stored(sequence.descriptor(), new double[12]), stored(initial.descriptor(), .1, .2, .3, .4)));
                var previous = (PreparedExecution) field(state, "inference"); var heap = owned.hostStorage().orElseThrow();
                AtomicInteger objectives = new AtomicInteger();
                model.configureTraining(List.of(targetH, targetY), (p, t) -> {
                    objectives.incrementAndGet(); return loss(p, t);
                }, new Sgd(.01));
                assertTrue(previous.isClosed()); assertNotSame(heap, owned.hostStorage().orElseThrow());
                var nativeStore = owned.hostStorage().orElseThrow(); assertTrue(nativeStore.segment().isNative());
                var session = (TrainingSession) field(state, "session");
                var prepared = field(session, "preparedExecution"); var compiled = field(session, "compiledGraph");
                var slots = field(state, "trainingSlots"); var fixed = field(state, "trainingFixed");
                // Prepare post-transition inference before poisoning future provider lookups.
                model.compute(List.of(stored(sequence.descriptor(), new double[12]), stored(initial.descriptor(), .1, .2, .3, .4)));
                var inference = field(state, "inference"); poison(borrowed);
                for (int batch = 1; batch <= 2; batch++) {
                    double[] data = new double[12]; for (int i = 0; i < data.length; i++) data[i] = (i - 4 * batch) * .03125;
                    Tensor directW = stored(owned.descriptor(), numbers(owned));
                    Tensor directH = stored(initial.descriptor(), .1, -.2, .3, -.4);
                    Tensor directX = stored(sequence.descriptor(), data);
                    Tensor th = stored(targetH.descriptor(), .1 * batch, .2, -.1, .3);
                    Tensor ty = stored(targetY.descriptor(), .1, -.2 * batch, .3, -.4, .2, .1);
                    Tensor unrolled = directH; Tensor[] emissions = new Tensor[3];
                    for (int k = 0; k < 3; k++) {
                        int t = reverse ? 2 - k : k;
                        unrolled = unrolled.mul(directW).add(directX.select(0, t)).tanh();
                        emissions[t] = unrolled.sum(0, false);
                    }
                    var reference = oracle.backward(loss(List.of(unrolled, Tensor.stack(0, emissions)), List.of(th, ty)), List.of(directW), 10000);
                    double[] gradients = numbers(reference.gradients().getFirst());
                    try (Arena batches = Arena.ofShared()) {
                        retained = model.train(List.of(nativeValue(directX, batches), nativeValue(directH, batches)),
                                List.of(nativeValue(th, batches), nativeValue(ty, batches)));
                    }
                    double tolerance = type == DataType.FLOAT64 ? 2e-12 : 3e-5;
                    assertArrayEquals(numbers(reference.objective()), numbers(retained.objective()), tolerance);
                    var parameter = ((List<?>) field(session, "parameters")).getFirst();
                    if (type == DataType.FLOAT64) assertArrayEquals(gradients, (double[]) field(parameter, "gradient64"), tolerance);
                    else {
                        float[] actual = (float[]) field(parameter, "gradient32");
                        for (int i = 0; i < actual.length; i++) assertEquals(gradients[i], actual[i], tolerance);
                    }
                    double[] expected = numbers(directW);
                    for (int i = 0; i < expected.length; i++) expected[i] = type == DataType.FLOAT64
                            ? expected[i] - .01 * gradients[i] : (float) expected[i] - .01f * (float) gradients[i];
                    assertArrayEquals(expected, numbers(owned), tolerance);
                    Tensor updated = stored(directW.descriptor(), numbers(owned));
                    Tensor forward = directH;
                    for (int k = 0; k < 3; k++) forward = forward.mul(updated).add(directX.select(0, reverse ? 2 - k : k)).tanh();
                    assertArrayEquals(numbers(oracle.compute(forward)), numbers(model.compute(List.of(directX, directH)).getFirst()), tolerance);
                    assertSame(wrapper, model.parametersRecursively().get("weight")); assertSame(owned, wrapper.value());
                    assertSame(nativeStore, owned.hostStorage().orElseThrow()); assertSame(session, field(state, "session"));
                    assertSame(prepared, field(session, "preparedExecution")); assertSame(compiled, field(session, "compiledGraph"));
                    assertSame(slots, field(state, "trainingSlots")); assertSame(fixed, field(state, "trainingFixed"));
                    assertSame(inference, field(state, "inference")); assertEquals(0, openResults(borrowed));
                }
                assertEquals(1, objectives.get()); assertEquals(2, session.executions());
                Path trained = directory.resolve("trained");
                model.save(trained);
                try (var restored = Model.loadGraph(trained)) {
                    assertArrayEquals(bits(owned), bits(restored.stateDictionary().entries().getFirst().value()));
                    assertNull(field(field(restored, "execution"), "engine"));
                    assertNull(field(field(restored, "execution"), "session"));
                    assertNotSame(owned, restored.stateDictionary().entries().getFirst().value());
                }
                assertSame(inference, field(state, "inference")); assertSame(session, field(state, "session"));
                retained = model.train(List.of(stored(sequence.descriptor(), new double[12]), stored(initial.descriptor(), .1, .2, .3, .4)),
                        List.of(stored(targetH.descriptor(), new double[4]), stored(targetY.descriptor(), new double[6])));
                model.close(); assertTrue(session.isClosed()); assertTrue(((PreparedExecution) prepared).isClosed());
                assertFalse(nativeStore.isAlive()); assertFalse(borrowed.isClosed()); assertTrue(Double.isFinite(numbers(retained.objective())[0]));
            }
            assertSame(originalStore, external.hostStorage().orElseThrow()); assertArrayEquals(original, bits(external));
        }
    }

    /**
     * Exercises identical workflow assertions against acquired and closed-then-loaded sources.
     * @param inputs declared storage-free signature
     * @param outputs ordinary source roots
     * @param leaves borrowed state registrations
     * @param engine caller-owned execution composition
     * @param loaded whether to close acquisition before cold loading
     * @return independent owned model borrowing the selected Engine
     * @throws Exception if save/load or reflective cold-owner assertions fail
     */
    private Model<List<Tensor>, List<Tensor>> graphOrLoaded(List<Tensor> inputs, List<Tensor> outputs,
            List<Leaf> leaves, Engine engine, boolean loaded) throws Exception {
        if (!loaded) return Model.graph(inputs, outputs, leaves, engine);
        Path path = directory.resolve("cold");
        try (var original = Model.graph(inputs, outputs, leaves)) {
            original.save(path);
            assertNull(field(field(original, "execution"), "engine"));
        }
        var result = Model.loadGraph(path);
        assertNull(field(field(result, "execution"), "engine"));
        assertNull(field(field(result, "execution"), "session"));
        result.forward(inputs);
        assertNull(field(field(result, "execution"), "engine"));
        return result.withEngine(engine);
    }

    @Test void saveBeforeComputeAndAfterTrainingPreservesFreezeCacheAndOwnedState() throws Exception {
        Tensor initial = stored(descriptor(DataType.FLOAT32, false, 2), .1, .2);
        Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 2));
        Tensor weight = stored(descriptor(DataType.FLOAT32, true, 2), .5, .25);
        var scan = pointwise(initial, sequence, weight, false);
        Path sentinel = directory.resolve("sentinel"); byte[] bytes = {1, 3, 7}; Files.write(sentinel, bytes);
        try (var model = Model.graph(List.of(sequence), scan.finalCarry(), List.of(parameter("weight", weight),
                new Leaf("initial", Role.CONSTANT, initial))); Engine borrowed = Engine.standard()) {
            var impossible = new io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits(48, 8, 0, 1, 0, 0);
            assertThrows(IllegalArgumentException.class, () -> model.save(sentinel, impossible));
            assertArrayEquals(bytes, Files.readAllBytes(sentinel));
            model.save(sentinel);
            try (var loaded = Model.load(sentinel)) {
                assertNull(field(field(loaded, "execution"), "engine"));
                assertArrayEquals(bits(model.stateDictionary().entries().getFirst().value()),
                        bits(loaded.stateDictionary().entries().getFirst().value()));
                assertNotSame(model.stateDictionary().entries().getFirst().value(),
                        loaded.stateDictionary().entries().getFirst().value());
            }
            var state = field(model, "execution"); assertNull(field(state, "engine")); assertFalse((Boolean) field(state, "frozen"));
            model.withEngine(borrowed);
            Tensor batch = stored(sequence.descriptor(), .1, .2, .3, -.1, -.2, .4);
            model.compute(batch);
            model.configureTraining(descriptor(DataType.FLOAT32, false, 2), (p, t) -> squared(p, t), new Sgd(.01));
            model.train(batch, stored(descriptor(DataType.FLOAT32, false, 2), .2, .1));
            Object prepared = field(state, "inference"); Object session = field(state, "session");
            Tensor owned = model.stateDictionary().entries().getFirst().value();
            model.save(sentinel);
            assertSame(prepared, field(state, "inference")); assertSame(session, field(state, "session"));
            assertSame(owned, model.stateDictionary().entries().getFirst().value());
            byte[] trained = Files.readAllBytes(sentinel);
            assertThrows(IllegalArgumentException.class, () -> model.save(sentinel, impossible));
            assertArrayEquals(trained, Files.readAllBytes(sentinel));
            try (var loaded = Model.load(sentinel)) {
                assertArrayEquals(bits(owned), bits(loaded.stateDictionary().entries().getFirst().value()));
                assertNull(field(field(loaded, "execution"), "session"));
            }
            model.compute(batch);
        }
    }

    @Test void freshBatchesAllocateNoTensorsAndValidateUnusedDescriptorsAndTrainingOverlap() throws Exception {
        Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 2));
        Tensor initial = TensorFactory.create(descriptor(DataType.FLOAT32, false, 2));
        Tensor unused = TensorFactory.create(initial.descriptor());
        Tensor weight = stored(descriptor(DataType.FLOAT32, true, 2), .5, .25);
        var scan = pointwise(initial, sequence, weight, true);
        try (var model = Model.graph(List.of(unused, sequence, initial), scan.outputs(), List.of(parameter("weight", weight)))) {
            var state = field(model, "execution");
            Tensor batch = stored(sequence.descriptor(), .1, .2, -.2, .1, .4, -.3);
            Tensor start = stored(initial.descriptor(), .2, -.1);
            for (TensorDescriptor invalid : List.of(descriptor(DataType.FLOAT64, false, 2),
                    descriptor(DataType.FLOAT32, true, 2), descriptor(DataType.FLOAT32, false, 1),
                    new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false))) {
                assertThrows(IllegalArgumentException.class, () -> model.compute(List.of(TensorFactory.create(invalid), batch, start)));
            }
            assertNull(field(state, "engine")); assertFalse((Boolean) field(state, "frozen"));
            // Both used input slots have different signatures here; the target shares the sequence signature.
            var values = List.of(stored(unused.descriptor(), 0, 0), batch, start);
            model.compute(values);
            Tensor target = TensorFactory.create(sequence.descriptor());
            model.configureTraining(List.of(target), (p, t) -> squared(p.getFirst(), t.getFirst()), new Sgd(.01));
            model.compute(values); // finish the one post-native inference setup before the probe
            for (int attempt = 0; attempt < 2; attempt++) {
                var fresh = List.of(stored(unused.descriptor(), attempt, 0),
                        stored(sequence.descriptor(), .1, .3, -.1, -.2, .2, .4), stored(initial.descriptor(), .1, .2));
                Tensor targets = stored(target.descriptor(), 0, 0, 0, 0, 0, 0);
                Tensor before = TensorFactory.create(initial.descriptor());
                model.compute(fresh); model.train(fresh, List.of(targets));
                Tensor after = TensorFactory.create(initial.descriptor());
                assertEquals(1, after.id().value() - before.id().value(), "cached compute/train must construct no Tensor, including transient source rebuilds");
            }
            var session = (TrainingSession) field(state, "session"); long executions = session.executions();
            assertThrows(IllegalArgumentException.class, () -> model.train(values, List.of(batch)));
            assertEquals(executions, session.executions());
            assertThrows(IllegalStateException.class, () -> model.train(List.of(unused, batch, start), List.of(stored(target.descriptor(), new double[6]))));
            assertEquals(executions, session.executions());
            assertTrue(source(model).inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
        }
        // Engine permits the same stored value in different exact-descriptor positions; Training does not.
        Tensor h = TensorFactory.create(initial.descriptor()), x = TensorFactory.create(initial.descriptor()), y = TensorFactory.create(initial.descriptor());
        Tensor w = TensorFactory.create(weight.descriptor());
        var body = new ScanBody(List.of(h), List.of(x, y), List.of(w), List.of(h.add(x).add(y.mul(w))), List.of());
        var two = Scan.apply(List.of(initial), List.of(sequence, TensorFactory.create(sequence.descriptor())), List.of(weight), body, false);
        var secondSequence = two.finalCarry().getFirst().provenance().orElseThrow().inputs().get(2);
        try (var model = Model.graph(List.of(sequence, secondSequence, initial), two.finalCarry(), List.of(parameter("weight", weight)))) {
            Tensor shared = stored(sequence.descriptor(), .1, .2, .3, .4, .5, .6), start = stored(initial.descriptor(), 0, 0);
            model.compute(List.of(shared, shared, start));
            model.configureTraining(List.of(TensorFactory.create(initial.descriptor())),
                    (p, t) -> squared(p.getFirst(), t.getFirst()), new Sgd(.01));
            assertThrows(IllegalArgumentException.class, () -> model.train(List.of(shared, shared, start), List.of(stored(initial.descriptor(), 0, 0))));
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void zeroTimeConnectedInitialTrainsButEmptyParametersAndDisconnectedObjectivesDoNot(DataType type) throws Exception {
        for (boolean reverse : List.of(false, true)) {
            Tensor initial = stored(descriptor(type, true, 2), .5, -.25);
            Tensor sequence = TensorFactory.create(descriptor(type, false, 0, 2));
            var scan = pointwise(initial, sequence, initial, reverse);
            try (var model = Model.graph(List.of(sequence), scan.finalCarry(), List.of(parameter("initial", initial)))) {
                model.configureTraining(descriptor(type, false, 2), GeneralScanOwnedModelTest::squared, new Sgd(.125));
                var step = model.train(stored(sequence.descriptor()), stored(descriptor(type, false, 2), 0, 0));
                assertEquals(.3125, numbers(step.objective())[0]);
                assertArrayEquals(new double[]{.375, -.1875}, numbers(model.stateDictionary().entries().getFirst().value()));
            }
            Tensor empty = stored(descriptor(type, true, 0));
            Tensor emptySequence = TensorFactory.create(descriptor(type, false, 0, 0));
            var emptyScan = pointwise(empty, emptySequence, empty, reverse);
            try (var model = Model.graph(List.of(emptySequence), emptyScan.finalCarry(), List.of(parameter("initial", empty)))) {
                assertThrows(IllegalArgumentException.class, () -> model.configureTraining(descriptor(type, false, 0),
                        GeneralScanOwnedModelTest::squared, new Sgd(.01)));
                assertNull(field(field(model, "execution"), "nativeArena"));
            }
        }
    }

    @Test void residualRootAndDisconnectedSetupRetryRetainNativeAdmissionAndFrozenBinding() throws Exception {
        Tensor initial = stored(descriptor(DataType.FLOAT32, false, 2), .1, .2);
        Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 2));
        Tensor weight = stored(descriptor(DataType.FLOAT32, true, 2), .5, .25);
        var scan = pointwise(initial, sequence, weight, false);
        Path residualFile = directory.resolve("residual");
        try (var model = Model.graph(List.of(sequence), List.of(scan.residuals().getLast()),
                List.of(parameter("weight", weight), new Leaf("initial", Role.CONSTANT, initial)))) {
            model.save(residualFile);
            var failure = assertThrows(IllegalArgumentException.class, () -> model.configureTraining(List.of(),
                    (p, t) -> p.getFirst().sum(), new Sgd(.01)));
            assertTrue(failure.getMessage().contains("residual"));
            assertNull(field(field(model, "execution"), "session"));
        }
        try (var loaded = Model.loadGraph(residualFile)) {
            var failure = assertThrows(IllegalArgumentException.class, () -> loaded.configureTraining(List.of(),
                    (p, t) -> p.getFirst().sum(), new Sgd(.01)));
            assertTrue(failure.getMessage().contains("residual"));
            assertNull(field(field(loaded, "execution"), "session"));
        }
        try (var model = Model.graph(List.of(sequence), scan.finalCarry(),
                List.of(parameter("weight", weight), new Leaf("initial", Role.CONSTANT, initial)))) {
            var state = field(model, "execution"); AtomicInteger calls = new AtomicInteger();
            assertThrows(IllegalArgumentException.class, () -> model.configureTraining(descriptor(DataType.FLOAT32, true, 2),
                    (p, t) -> { calls.incrementAndGet(); return t.mul(t).sum(); }, new Sgd(.01)));
            Tensor owned = model.stateDictionary().entries().getFirst().value(); var nativeStore = owned.hostStorage().orElseThrow();
            assertTrue(nativeStore.segment().isNative()); assertNull(field(state, "session"));
            assertThrows(IllegalStateException.class, () -> model.withEngine(null));
            model.configureTraining(descriptor(DataType.FLOAT32, false, 2), (p, t) -> {
                calls.incrementAndGet(); return squared(p, t);
            }, new Sgd(.01));
            assertSame(nativeStore, owned.hostStorage().orElseThrow()); assertEquals(2, calls.get());
            model.train(stored(sequence.descriptor(), .1, .2, .3, .4, .5, .6), stored(initial.descriptor(), 0, 0));
        }
        Tensor bad = stored(weight.descriptor(), .5, Double.NaN);
        var badScan = pointwise(initial, sequence, bad, false);
        try (var model = Model.graph(List.of(sequence), badScan.finalCarry(),
                List.of(parameter("weight", bad), new Leaf("initial", Role.CONSTANT, initial)))) {
            var owned = model.stateDictionary().entries().getFirst().value(); var heap = owned.hostStorage().orElseThrow();
            assertThrows(IllegalArgumentException.class, () -> model.configureTraining(initial.descriptor(),
                    GeneralScanOwnedModelTest::squared, new Sgd(.01)));
            assertSame(heap, owned.hostStorage().orElseThrow()); assertNull(field(field(model, "execution"), "nativeArena"));
            assertThrows(IllegalStateException.class, () -> model.withEngine(null));
        }
    }

    @Test void scanLiveModeWrapperTensorAndStoreMutationInvalidateCachedOwners() throws Exception {
        for (String change : List.of("mode", "wrapper", "tensor", "storage")) {
            Tensor initial = stored(descriptor(DataType.FLOAT32, false, 2), .1, .2);
            Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 2));
            Tensor weight = stored(descriptor(DataType.FLOAT32, true, 2), .5, .25);
            var scan = pointwise(initial, sequence, weight, false);
            try (var model = Model.graph(List.of(sequence), scan.finalCarry(),
                    List.of(parameter("weight", weight), new Leaf("initial", Role.CONSTANT, initial)))) {
                Tensor batch = stored(sequence.descriptor(), .1, .2, .3, .4, .5, .6);
                model.compute(batch); var state = field(model, "execution");
                var prepared = (PreparedExecution) field(state, "inference");
                Tensor owned = model.stateDictionary().entries().getFirst().value();
                switch (change) {
                    case "mode" -> model.train();
                    case "tensor" -> model.parametersRecursively().get("weight").replace(stored(owned.descriptor(), .2, .3));
                    case "wrapper" -> {
                        var declarations = (java.util.Map<?, ?>) field(model, "parameterDeclarations");
                        Object slot = declarations.get("weight");
                        OwnedModelLifecycleTest.set(slot, "parameter", new Parameter("weight", owned));
                    }
                    default -> owned.replaceHostStorage(stored(owned.descriptor(), .2, .3).hostStorage().orElseThrow());
                }
                assertThrows(IllegalStateException.class, () -> model.compute(batch)); assertTrue(prepared.isClosed());
                assertThrows(IllegalStateException.class, () -> model.forward(List.of(sequence)));
                assertThrows(IllegalStateException.class, () -> model.configureTraining(initial.descriptor(),
                        GeneralScanOwnedModelTest::squared, new Sgd(.01)));
            }
        }
    }

    @Test void mixedFixedAndGeneralSourcesStillRejectEveryTrainingOverloadWithoutEffects() throws Exception {
        Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 1, 1));
        Tensor initial = TensorFactory.create(descriptor(DataType.FLOAT32, false, 1, 1));
        Tensor weight = stored(descriptor(DataType.FLOAT32, true, 1, 1), .5);
        Tensor lengths = TensorFactory.create(descriptor(DataType.INT64, false, 1));
        var fixed = RecurrentScan.rnn(sequence, lengths, initial, weight, weight,
                io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection.FORWARD);
        var general = pointwise(initial, sequence, weight, false);
        Tensor head = fixed.finalHidden().add(general.finalCarry().getFirst());
        try (var model = Model.graph(List.of(sequence, lengths, initial), List.of(head), List.of(parameter("weight", weight)))) {
            RuntimeLengthScanModelTest.assertTrainingRejected(model);
            Tensor batch = stored(sequence.descriptor(), .1, .2, .3);
            Tensor validLengths = TensorFactory.fromFlatArray(lengths.descriptor(), Optional.empty(), new long[]{3});
            Tensor start = stored(initial.descriptor(), 0);
            model.compute(List.of(batch, validLengths, start)); RuntimeLengthScanModelTest.assertTrainingRejected(model);
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void emptyStepStillReducesToNonemptyEmissionsAndTrainsNonemptyInvariant(DataType type) throws Exception {
        for (boolean reverse : List.of(false, true)) {
            Tensor initial = TensorFactory.create(descriptor(type, false, 0, 2));
            Tensor sequence = TensorFactory.create(descriptor(type, false, 3, 0, 2));
            Tensor weight = stored(descriptor(type, true, 2), .5, -.25);
            Tensor h = TensorFactory.create(initial.descriptor()), x = TensorFactory.create(initial.descriptor());
            Tensor w = TensorFactory.create(weight.descriptor());
            Tensor next = h.mul(w).add(x);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next.sum(0, false)));
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
            Tensor target = TensorFactory.create(descriptor(type, false, 3, 2));
            try (var model = Model.graph(List.of(sequence, initial), scan.outputs(), List.of(parameter("weight", weight)))) {
                var inputs = List.of(stored(sequence.descriptor()), stored(initial.descriptor()));
                assertArrayEquals(new double[6], numbers(model.compute(inputs).getFirst()));
                model.configureTraining(List.of(target), (p, t) -> squared(p.getFirst(), t.getFirst()), new Sgd(.01));
                var step = model.train(inputs, List.of(stored(target.descriptor(), .1, .2, .3, .4, .5, .6)));
                assertTrue(numbers(step.objective())[0] > 0);
                assertArrayEquals(new double[]{.5, -.25}, numbers(model.stateDictionary().entries().getFirst().value()));
                var session = field(field(model, "execution"), "session");
                var parameter = ((List<?>) field(session, "parameters")).getFirst();
                if (type == DataType.FLOAT64) assertArrayEquals(new double[2], (double[]) field(parameter, "gradient64"));
                else assertArrayEquals(new float[2], (float[]) field(parameter, "gradient32"));
            }
        }
    }

    @Test void scanBusyAndClosedPrecedenceAndCloseWaitForAdmittedSetupRemainUnchanged() throws Exception {
        Tensor initial = stored(descriptor(DataType.FLOAT32, false, 2), .1, .2);
        Tensor sequence = TensorFactory.create(descriptor(DataType.FLOAT32, false, 3, 2));
        Tensor weight = stored(descriptor(DataType.FLOAT32, true, 2), .5, .25);
        var scan = pointwise(initial, sequence, weight, false);
        var model = Model.graph(List.of(sequence), scan.finalCarry(),
                List.of(parameter("weight", weight), new Leaf("initial", Role.CONSTANT, initial)));
        var entered = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        try (Engine borrowed = Engine.standard(); var pool = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            model.withEngine(borrowed);
            var setup = pool.submit(() -> model.configureTraining(initial.descriptor(), (p, t) -> {
                entered.countDown();
                try { assertTrue(release.await(10, java.util.concurrent.TimeUnit.SECONDS)); }
                catch (InterruptedException failure) { throw new AssertionError(failure); }
                return squared(p, t);
            }, new Sgd(.01)));
            assertTrue(entered.await(10, java.util.concurrent.TimeUnit.SECONDS));
            assertTrue(assertThrows(IllegalStateException.class, () -> model.compute((Tensor) null)).getMessage().contains("busy"));
            assertTrue(assertThrows(IllegalStateException.class, () -> model.train((List<Tensor>) null, null)).getMessage().contains("busy"));
            var state = field(model, "execution"); var close = pool.submit(model::close);
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(10);
            synchronized (state) {
                while (!(Boolean) field(state, "closing") && System.nanoTime() < deadline) state.wait(10);
            }
            assertTrue((Boolean) field(state, "closing")); assertFalse(close.isDone());
            assertTrue(assertThrows(IllegalStateException.class, () -> model.configureTraining((TensorDescriptor) null, null, null))
                    .getMessage().contains("closing"));
            release.countDown(); setup.get(10, java.util.concurrent.TimeUnit.SECONDS); close.get(10, java.util.concurrent.TimeUnit.SECONDS);
            assertFalse(borrowed.isClosed());
            assertTrue(assertThrows(IllegalStateException.class, () -> model.compute((Tensor) null)).getMessage().contains("closed"));
            borrowed.close();
            assertTrue(assertThrows(IllegalStateException.class, () -> model.train((Tensor) null, null)).getMessage().contains("selected Engine"));
        } finally { release.countDown(); model.close(); }
    }

    /**
     * Reads the fixed source through the existing private-state test seam.
     *
     * @param model owned model under test
     * @return its fixed borrowed source, without reconstruction
     */
    private static GraphCheckpoint.Source source(Model<?, ?> model) throws Exception {
        return (GraphCheckpoint.Source) field(field(field(model, "execution"), "owned"), "source");
    }

    /**
     * Finds the scan occurrence beneath a tested unary head.
     *
     * @param root scan output or ordinary unary head
     * @return the canonical scan producer
     */
    private static TensorProducer scanProducer(Tensor root) {
        var producer = root.provenance().orElseThrow().producer();
        return producer.scanBody().isPresent() ? producer : scanProducer(producer.inputs().getFirst());
    }

    /**
     * @param initial non-gradient carry actual
     * @param sequence time-major input actual
     * @param weight trainable same-shape invariant
     * @param reverse whether traversal decreases original time
     * @return a pointwise scan emitting pre-step carry, including its conservative gradient flag
     */
    private static ScanResult pointwise(Tensor initial, Tensor sequence, Tensor weight, boolean reverse) {
        Tensor h = TensorFactory.create(initial.descriptor());
        Tensor x = TensorFactory.create(new TensorDescriptor(sequence.descriptor().dataType(), initial.descriptor().shape(),
                initial.descriptor().layout(), sequence.descriptor().requiresGrad()));
        Tensor w = TensorFactory.create(weight.descriptor());
        var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.mul(w).add(x)), List.of(h));
        return Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
    }

    /**
     * Builds one scalar objective consuming both public output roles.
     *
     * @param p final carry and emission predictions
     * @param t matching targets
     * @return scalar sum of both squared errors
     */
    private static Tensor loss(List<Tensor> p, List<Tensor> t) {
        return squared(p.get(0), t.get(0)).add(squared(p.get(1), t.get(1)));
    }

    /**
     * Builds a scalar squared-error objective.
     *
     * @param p prediction
     * @param t target
     * @return scalar squared-error sum
     */
    private static Tensor squared(Tensor p, Tensor t) { Tensor e = p.sub(t); return e.mul(e).sum(); }

    /**
     * Declares exact canonical geometry without allocating storage.
     *
     * @param type declared dtype
     * @param grad eligibility
     * @param dimensions static extents
     * @return canonical full descriptor
     */
    private static TensorDescriptor descriptor(DataType type, boolean grad, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    /**
     * Creates independent heap storage for a declared signature.
     *
     * @param descriptor exact geometry
     * @param lanes numeric values, narrowed to the exact original floating carrier
     * @return fresh stored heap value
     */
    private static Tensor stored(TensorDescriptor descriptor, double... lanes) {
        int count = Math.toIntExact(descriptor.shape().knownElementCount().orElseThrow()); assertEquals(count, lanes.length);
        MemorySegment memory;
        if (descriptor.dataType() == DataType.FLOAT64) memory = MemorySegment.ofArray(lanes.clone());
        else if (descriptor.dataType() == DataType.FLOAT32) {
            float[] values = new float[count]; for (int i = 0; i < count; i++) values[i] = (float) lanes[i];
            memory = MemorySegment.ofArray(values);
        } else {
            short[] values = new short[count];
            for (int i = 0; i < count; i++) values[i] = descriptor.dataType() == DataType.BFLOAT16
                    ? BFloat16Bits.fromFloat((float) lanes[i]) : Float16Bits.fromFloat((float) lanes[i]);
            memory = MemorySegment.ofArray(values);
        }
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(new MemorySegmentStorage(descriptor.dataType(), count, memory)));
    }

    /**
     * Snapshots represented storage bytes without numerical conversion.
     *
     * @param tensor live stored value
     * @return exact represented bytes
     */
    private static byte[] bits(Tensor tensor) { return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE); }

    /**
     * Reads represented floating values from live Tensor storage.
     *
     * @param tensor live F32/F64 stored value
     * @return represented values widened without arithmetic
     */
    private static double[] numbers(Tensor tensor) {
        return decode(ByteBuffer.wrap(bits(tensor)).order(java.nio.ByteOrder.nativeOrder()), tensor.descriptor().dataType());
    }

    /**
     * Reads detached floating publication values.
     *
     * @param value detached F32/F64 result
     * @return represented values widened without arithmetic
     */
    private static double[] numbers(HostTensorValue value) { return decode(value.bytes(), value.dataType()); }

    /**
     * Decodes floating lanes using the supplied byte order.
     *
     * @param bytes correctly ordered payload
     * @param type F32/F64
     * @return decoded lanes
     */
    private static double[] decode(ByteBuffer bytes, DataType type) {
        double[] values = new double[bytes.remaining() / type.byteWidth()];
        for (int i = 0; i < values.length; i++) values[i] = type == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat();
        return values;
    }

    /**
     * Copies a batch into a caller-controlled native lifetime.
     *
     * @param value stored source
     * @param arena caller-owned batch lifetime
     * @return copied exact-descriptor native batch
     */
    private static Tensor nativeValue(Tensor value, Arena arena) {
        var source = value.hostStorage().orElseThrow(); var memory = arena.allocate(source.byteSize(), value.descriptor().dataType().byteWidth());
        memory.copyFrom(source.segment());
        return TensorFactory.create(value.descriptor(), Optional.empty(), Optional.of(new MemorySegmentStorage(
                value.descriptor().dataType(), value.descriptor().shape().knownElementCount().orElseThrow(), memory)));
    }

    /**
     * Rejects any subsequent backend lookup through the existing Engine test seam.
     *
     * @param engine prepared borrowed Engine
     * @throws Exception if the existing no-lookup test seam fails
     */
    private static void poison(Engine engine) throws Exception {
        var method = Engine.class.getDeclaredMethod("poisonBackendLookupForTesting"); method.setAccessible(true); method.invoke(engine);
    }

    /**
     * @param type working arithmetic dtype
     * @param initial initial vector
     * @param sequence original-index time-major lanes
     * @param weights same-shape weights
     * @param time static iteration count
     * @param reverse decreasing traversal flag
     * @return final carry and pre-step emissions from represented primitive Java arithmetic
     */
    private static List<double[]> recurrence(DataType type, double[] initial, double[] sequence, double[] weights, int time, boolean reverse) {
        double[] state = initial.clone(), emissions = new double[sequence.length]; int n = initial.length;
        for (int k = 0; k < time; k++) {
            int t = reverse ? time - 1 - k : k;
            for (int i = 0; i < n; i++) {
                int p = t * n + i; emissions[p] = state[i];
                state[i] = type == DataType.FLOAT64 ? state[i] * weights[i] + sequence[p]
                        : (float) ((float) state[i] * (float) weights[i]) + (float) sequence[p];
            }
        }
        return List.of(state, emissions);
    }
}
