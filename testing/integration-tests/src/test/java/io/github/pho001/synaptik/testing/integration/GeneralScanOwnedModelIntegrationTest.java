package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Leaf;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Role;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingStep;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Public owned general-scan workflows compared with independent unrolled Tensor/Engine algebra. */
final class GeneralScanOwnedModelIntegrationTest {
    @TempDir Path directory;

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void definedSingleBoundaryComputesAndTrainsFreshBatchesWithoutReinvokingDefinition(DataType type) throws Exception {
        for (boolean reverse : List.of(false, true)) {
            Tensor external = value(type, true, Shape.of(1, 1), .5);
            Tensor initial = value(type, false, Shape.of(1, 1), .125);
            Tensor prototype = formal(type, false, Shape.of(3, 1, 1));
            var oldStore = external.hostStorage().orElseThrow(); byte[] oldBytes = bits(external);
            AtomicInteger definitions = new AtomicInteger(), forwards = new AtomicInteger(), objectives = new AtomicInteger();
            TrainingStep retained = null;
            Path before = directory.resolve("single-before"), after = directory.resolve("single-after");
            try (Engine borrowed = Engine.standard(); Engine oracle = Engine.standard();
                    var model = Model.defineTensor(prototype.descriptor(), topology -> {
                        definitions.incrementAndGet(); topology.addModule("linear", new Linear(external));
                        return sequence -> {
                            forwards.incrementAndGet();
                            Tensor h = formal(type, false, Shape.of(1, 1));
                            Tensor x = formal(type, false, Shape.of(1, 1));
                            Tensor w = formal(type, true, Shape.of(1, 1));
                            var body = new ScanBody(List.of(h), List.of(x), List.of(w),
                                    List.of(h.mul(w).add(x).tanh()), List.of());
                            return Scan.apply(List.of(initial), List.of(sequence), List.of(external), body, reverse).finalCarry().getFirst();
                        };
                    }, borrowed)) {
                assertEquals(List.of("linear.weight"), model.stateDictionary().entries().stream().map(e -> e.path()).toList());
                Tensor owned = model.stateDictionary().entries().getFirst().value(); assertNotSame(external, owned);
                model.save(before);
                Tensor batch = value(type, false, prototype.descriptor().shape(), .25, -.125, .375);
                Tensor target = value(type, false, Shape.of(1, 1), .2);
                Tensor direct = unroll(batch, initial, external, reverse, false).getFirst();
                assertArrayEquals(numbers(oracle.compute(direct)), numbers(model.compute(batch)), tolerance(type));
                model.configureTraining(target.descriptor(), (p, t) -> { objectives.incrementAndGet(); return squared(p, t); }, new Sgd(.01));
                for (int k = 0; k < 2; k++) {
                    Tensor x = value(type, false, prototype.descriptor().shape(), .125 + k * .125, -.25, .375);
                    Tensor y = value(type, false, target.descriptor().shape(), .2 - k * .1);
                    Tensor directW = value(type, true, owned.descriptor().shape(), numbers(owned));
                    Tensor prediction = unroll(x, initial, directW, reverse, false).getFirst();
                    var expected = oracle.backward(squared(prediction, y), List.of(directW), 10000);
                    retained = model.train(x, y);
                    assertArrayEquals(numbers(expected.objective()), numbers(retained.objective()), tolerance(type));
                    assertUpdate(type, directW, expected.gradients().getFirst(), owned);
                    assertArrayEquals(numbers(oracle.compute(unroll(x, initial, owned, reverse, false).getFirst())),
                            numbers(model.compute(x)), tolerance(type));
                }
                assertEquals(1, definitions.get()); assertEquals(1, forwards.get()); assertEquals(1, objectives.get());
                assertSame(owned, model.stateDictionary().entries().getFirst().value());
                model.save(after);
                assertNotNull(retained);
                model.close(); assertFalse(borrowed.isClosed());
                assertTrue(Double.isFinite(numbers(retained.objective())[0]));
            }
            assertSame(oldStore, external.hostStorage().orElseThrow()); assertArrayEquals(oldBytes, bits(external));
            for (Path saved : List.of(before, after)) loadedSingle(saved, type, prototype.descriptor(), initial, reverse);
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void graphListNonSquareMatmulTrainsFinalAndEmissionsAgainstUnrolledOracle(DataType type) throws Exception {
        for (boolean reverse : List.of(false, true)) {
            Tensor initial = formal(type, false, Shape.of(2, 3)), sequence = formal(type, false, Shape.of(3, 2, 2));
            Tensor weight = value(type, true, Shape.of(2, 3), .25, -.125, .5, -.25, .125, .375);
            Tensor h = formal(type, false, Shape.of(2, 3)), x = formal(type, false, Shape.of(2, 2));
            Tensor w = formal(type, true, Shape.of(2, 3));
            Tensor next = h.add(x.matmul(w)).tanh();
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(next));
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse);
            var roots = List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst());
            Tensor targetH = formal(type, false, Shape.of(2, 3)), targetY = formal(type, false, Shape.of(3, 2, 3));
            byte[] original = bits(weight); var originalStore = weight.hostStorage().orElseThrow();
            Path before = directory.resolve("list-before"), after = directory.resolve("list-after");
            try (Engine oracle = Engine.standard(); var model = Model.graph(List.of(sequence, initial), roots,
                    List.of(new Leaf("weight", Role.PARAMETER, weight)))) {
                Tensor owned = model.stateDictionary().entries().getFirst().value();
                model.save(before);
                model.configureTraining(List.of(targetH, targetY), (p, t) -> squared(p.get(0), t.get(0))
                        .add(squared(p.get(1), t.get(1))), new Sgd(.01));
                TrainingStep retained = null;
                for (int batch = 1; batch <= 2; batch++) {
                    Tensor input = value(type, false, sequence.descriptor().shape(),
                            .125 * batch, -.25, .375, -.125, .25, .125, -.25, .375, .125, -.375, .25, -.125);
                    Tensor start = value(type, false, initial.descriptor().shape(), .1, -.2, .3, -.1, .2, -.3);
                    Tensor th = value(type, false, targetH.descriptor().shape(), .1, .2, -.1, -.2, .3, .1);
                    double[] targets = new double[18]; java.util.Arrays.fill(targets, batch * .125);
                    Tensor ty = value(type, false, targetY.descriptor().shape(), targets);
                    Tensor directW = value(type, true, owned.descriptor().shape(), numbers(owned));
                    var direct = unroll(input, start, directW, reverse, true);
                    var beforeValues = model.compute(List.of(input, start));
                    for (int port = 0; port < 2; port++) assertArrayEquals(numbers(oracle.compute(direct.get(port))),
                            numbers(beforeValues.get(port)), tolerance(type));
                    var expected = oracle.backward(squared(direct.get(0), th).add(squared(direct.get(1), ty)), List.of(directW), 10000);
                    try (Arena arena = Arena.ofShared()) {
                        retained = model.train(List.of(nativeValue(input, arena), nativeValue(start, arena)),
                                List.of(nativeValue(th, arena), nativeValue(ty, arena)));
                    }
                    assertArrayEquals(numbers(expected.objective()), numbers(retained.objective()), tolerance(type));
                    assertUpdate(type, directW, expected.gradients().getFirst(), owned);
                    var updated = unroll(input, start, owned, reverse, true);
                    var afterValues = model.compute(List.of(input, start));
                    for (int port = 0; port < 2; port++) assertArrayEquals(numbers(oracle.compute(updated.get(port))),
                            numbers(afterValues.get(port)), tolerance(type));
                    assertSame(owned, model.stateDictionary().entries().getFirst().value());
                }
                assertNotNull(retained);
                model.save(after);
                model.close(); // detached values are readable after caller batches and Model close
                assertTrue(Double.isFinite(numbers(retained.objective())[0]));
            }
            assertSame(originalStore, weight.hostStorage().orElseThrow()); assertArrayEquals(original, bits(weight));
            for (Path saved : List.of(before, after)) loadedList(saved, type, sequence.descriptor(), initial.descriptor(), reverse);
        }
    }

    /**
     * Verifies cold single-boundary load, fresh execution/training and re-save after source closure.
     * @param path artifact saved before compute or after training, with no objective history
     * @param type F32/F64 precision
     * @param signature sequence boundary
     * @param initial independent oracle's original constant carry
     * @param reverse traversal flag
     * @throws Exception if persistence or public workflow validation fails
     */
    private void loadedSingle(Path path, DataType type, TensorDescriptor signature, Tensor initial, boolean reverse)
            throws Exception {
        Path again = directory.resolve("single-again");
        try (Engine oracle = Engine.standard(); var model = Model.load(path)) {
            Tensor owned = model.stateDictionary().entries().getFirst().value();
            assertFalse(owned.hostStorage().orElseThrow().segment().isNative());
            Tensor cold = model.forward(formal(type, false, signature.shape()).neg().contiguous());
            assertEquals(initial.descriptor().shape(), cold.descriptor().shape());
            model.save(again);
            model.configureTraining(initial.descriptor(), GeneralScanOwnedModelIntegrationTest::squared, new Sgd(.01));
            for (int batch = 0; batch < 2; batch++) {
                Tensor x = value(type, false, signature.shape(), .125 + batch * .125, -.25, .375);
                Tensor target = value(type, false, initial.descriptor().shape(), .1 * batch);
                Tensor w = value(type, true, owned.descriptor().shape(), numbers(owned));
                Tensor prediction = unroll(x, initial, w, reverse, false).getFirst();
                assertArrayEquals(numbers(oracle.compute(prediction)), numbers(model.compute(x)), tolerance(type));
                assertArrayEquals(numbers(oracle.compute(prediction)), numbers(model.compute(x)), tolerance(type));
                var expected = oracle.backward(squared(prediction, target), List.of(w), 10000);
                var step = model.train(x, target);
                assertEquals(batch + 1, step.executionNumber());
                assertEquals(batch + 1, step.optimizerStep().orElseThrow());
                assertArrayEquals(numbers(expected.objective()), numbers(step.objective()), tolerance(type));
                assertUpdate(type, w, expected.gradients().getFirst(), owned);
                assertSame(owned, model.stateDictionary().entries().getFirst().value());
            }
            model.save(again);
            byte[] trained = bits(owned); model.close();
            try (var restored = Model.load(again)) {
                assertArrayEquals(trained, bits(restored.stateDictionary().entries().getFirst().value()));
                Tensor x = value(type, false, signature.shape(), .125, .25, -.375);
                Tensor w = restored.stateDictionary().entries().getFirst().value();
                assertArrayEquals(numbers(oracle.compute(unroll(x, initial, w, reverse, false).getFirst())),
                        numbers(restored.compute(x)), tolerance(type));
            }
        }
    }

    /**
     * Verifies the public list load path against independent non-square MATMUL unrolling.
     * @param path artifact saved before compute or after training
     * @param type F32/F64 precision
     * @param sequence exact sequence boundary
     * @param initial exact initial-state boundary
     * @param reverse traversal flag
     * @throws Exception if cold composition, compute, training or persistence fails
     */
    private void loadedList(Path path, DataType type, TensorDescriptor sequence, TensorDescriptor initial, boolean reverse)
            throws Exception {
        Path again = directory.resolve("list-again");
        try (Engine oracle = Engine.standard(); var model = Model.loadGraph(path)) {
            Tensor owned = model.stateDictionary().entries().getFirst().value();
            assertFalse(owned.hostStorage().orElseThrow().segment().isNative());
            assertEquals(2, model.forward(List.of(formal(type, false, sequence.shape()).neg().contiguous(),
                    formal(type, false, initial.shape()).neg().contiguous())).size());
            model.save(again);
            Tensor th = formal(type, false, initial.shape()), ty = formal(type, false, Shape.of(3, 2, 3));
            model.configureTraining(List.of(th, ty), (p, t) -> squared(p.get(0), t.get(0)).add(squared(p.get(1), t.get(1))),
                    new Sgd(.01));
            for (int batch = 1; batch <= 2; batch++) {
                Tensor x = value(type, false, sequence.shape(), .125 * batch, -.25, .375, -.125, .25, .125,
                        -.25, .375, .125, -.375, .25, -.125);
                Tensor start = value(type, false, initial.shape(), .1, -.2, .3, -.1, .2, -.3);
                Tensor targetH = value(type, false, initial.shape(), .1, .2, -.1, -.2, .3, .1);
                Tensor targetY = value(type, false, ty.descriptor().shape(), new double[18]);
                Tensor w = value(type, true, owned.descriptor().shape(), numbers(owned));
                var direct = unroll(x, start, w, reverse, true);
                for (int repeat = 0; repeat < 2; repeat++) {
                    var actual = model.compute(List.of(x, start));
                    for (int port = 0; port < 2; port++) assertArrayEquals(numbers(oracle.compute(direct.get(port))),
                            numbers(actual.get(port)), tolerance(type));
                }
                var expected = oracle.backward(squared(direct.get(0), targetH).add(squared(direct.get(1), targetY)), List.of(w), 10000);
                var step = model.train(List.of(x, start), List.of(targetH, targetY));
                assertEquals(batch, step.executionNumber());
                assertArrayEquals(numbers(expected.objective()), numbers(step.objective()), tolerance(type));
                assertUpdate(type, w, expected.gradients().getFirst(), owned);
                assertSame(owned, model.stateDictionary().entries().getFirst().value());
            }
            model.save(again);
            byte[] trained = bits(owned); model.close();
            try (var restored = Model.loadGraph(again)) {
                assertArrayEquals(trained, bits(restored.stateDictionary().entries().getFirst().value()));
                Tensor x = value(type, false, sequence.shape(), new double[12]);
                Tensor start = value(type, false, initial.shape(), .1, -.2, .3, -.1, .2, -.3);
                var direct = unroll(x, start, restored.stateDictionary().entries().getFirst().value(), reverse, true);
                var actual = restored.compute(List.of(x, start));
                for (int port = 0; port < 2; port++) assertArrayEquals(numbers(oracle.compute(direct.get(port))),
                        numbers(actual.get(port)), tolerance(type));
            }
        }
    }

    /**
     * @param sequence three-step time-major input
     * @param initial initial non-gradient carry
     * @param weight independently stored trainable weight
     * @param reverse decreasing original-index traversal
     * @param matmul whether the update is non-square MATMUL rather than pointwise recurrence
     * @return ordinary final carry and emissions, constructed without Scan or cold reconstruction
     */
    private static List<Tensor> unroll(Tensor sequence, Tensor initial, Tensor weight, boolean reverse, boolean matmul) {
        Tensor state = initial; Tensor[] emissions = new Tensor[3];
        for (int k = 0; k < 3; k++) {
            int t = reverse ? 2 - k : k;
            state = (matmul ? state.add(sequence.select(0, t).matmul(weight)) : state.mul(weight).add(sequence.select(0, t))).tanh();
            emissions[t] = state;
        }
        return List.of(state, Tensor.stack(0, emissions));
    }

    /**
     * Checks each represented SGD weight delta against the independent gradient.
     *
     * @param type arithmetic precision
     * @param before pre-update weight
     * @param gradient oracle gradient
     * @param actual updated owned weight
     */
    private static void assertUpdate(DataType type, Tensor before, HostTensorValue gradient, Tensor actual) {
        double[] expected = numbers(before), g = numbers(gradient);
        for (int i = 0; i < expected.length; i++) expected[i] = type == DataType.FLOAT64
                ? expected[i] - .01 * g[i] : (float) expected[i] - .01f * (float) g[i];
        assertArrayEquals(expected, numbers(actual), tolerance(type));
    }

    /**
     * Builds a scalar squared-error objective.
     *
     * @param prediction output expression
     * @param target fresh target
     * @return scalar squared-error sum
     */
    private static Tensor squared(Tensor prediction, Tensor target) { Tensor e = prediction.sub(target); return e.mul(e).sum(); }

    /**
     * Declares a storage-free canonical input or body formal.
     *
     * @param type dtype
     * @param grad eligibility
     * @param shape static geometry
     * @return storage-free canonical prototype
     */
    private static Tensor formal(DataType type, boolean grad, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }

    /**
     * Creates independent stored values in the selected floating precision.
     *
     * @param type F32/F64
     * @param grad eligibility
     * @param shape exact geometry
     * @param lanes represented input values
     * @return stored canonical heap value
     */
    private static Tensor value(DataType type, boolean grad, Shape shape, double... lanes) {
        assertEquals(shape.knownElementCount().orElseThrow(), lanes.length);
        MemorySegment memory;
        if (type == DataType.FLOAT64) memory = MemorySegment.ofArray(lanes.clone());
        else { float[] data = new float[lanes.length]; for (int i = 0; i < data.length; i++) data[i] = (float) lanes[i]; memory = MemorySegment.ofArray(data); }
        return TensorFactory.create(formal(type, grad, shape).descriptor(), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, lanes.length, memory)));
    }

    /**
     * Copies a batch into a caller-controlled native lifetime.
     *
     * @param value stored tensor
     * @param arena caller-owned lifetime
     * @return exact-descriptor copied native batch
     */
    private static Tensor nativeValue(Tensor value, Arena arena) {
        var source = value.hostStorage().orElseThrow(); var memory = arena.allocate(source.byteSize(), value.descriptor().dataType().byteWidth());
        memory.copyFrom(source.segment());
        return TensorFactory.create(value.descriptor(), Optional.empty(), Optional.of(new MemorySegmentStorage(
                value.descriptor().dataType(), value.descriptor().shape().knownElementCount().orElseThrow(), memory)));
    }

    /**
     * Snapshots exact represented storage bytes.
     *
     * @param tensor live tensor
     * @return exact raw bytes
     */
    private static byte[] bits(Tensor tensor) { return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE); }
    /**
     * Reads represented floating values from live Tensor storage.
     *
     * @param tensor stored F32/F64
     * @return widened represented lanes
     */
    private static double[] numbers(Tensor tensor) { return decode(ByteBuffer.wrap(bits(tensor)).order(java.nio.ByteOrder.nativeOrder()), tensor.descriptor().dataType()); }
    /**
     * Reads detached floating publication values.
     *
     * @param value detached F32/F64
     * @return widened represented lanes
     */
    private static double[] numbers(HostTensorValue value) { return decode(value.bytes(), value.dataType()); }
    /**
     * Decodes represented floating lanes in the supplied byte order.
     *
     * @param bytes ordered payload
     * @param type F32/F64
     * @return decoded represented lanes
     */
    private static double[] decode(ByteBuffer bytes, DataType type) {
        double[] lanes = new double[bytes.remaining() / type.byteWidth()];
        for (int i = 0; i < lanes.length; i++) lanes[i] = type == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat(); return lanes;
    }
    /**
     * Selects test-only finite comparison bounds for the unrolled oracle.
     *
     * @param type arithmetic dtype
     * @return test-only unrolled comparison tolerance
     */
    private static double tolerance(DataType type) { return type == DataType.FLOAT64 ? 3e-12 : 3e-5; }
}
