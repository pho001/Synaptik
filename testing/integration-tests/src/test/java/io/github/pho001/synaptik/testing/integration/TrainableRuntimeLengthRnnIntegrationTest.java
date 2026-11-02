package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.nn.layers.RnnCell;
import io.github.pho001.synaptik.nn.layers.RnnSequence;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingStep;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Public RNN composition, actual AD and owned train/save/load against independent scalar BPTT. */
final class TrainableRuntimeLengthRnnIntegrationTest {
    @TempDir Path directory;

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void directForwardAndJointGradientsCoverEveryRequestedRoleAndInactiveNaNs(DataType type) throws Exception {
        for (boolean bias : List.of(false, true)) for (boolean reverse : List.of(false, true))
            for (int batch : List.of(1, 3)) for (int time : List.of(0, 3)) {
                int f = 3, h = 2;
                var parameters = parameters(type, bias);
                var sequence = new RnnSequence(bias ? new RnnCell(parameters.get(0), parameters.get(1), parameters.get(2))
                        : new RnnCell(parameters.get(0), parameters.get(1)));
                for (long[] lengths : lengthCases(time, batch)) {
                    Tensor x = value(type, true, Shape.of(time, batch, f), data(time, batch, f, lengths, .125));
                    double[] start = ramp(batch * h, -.125);
                    if (lengths[0] == 0) start[0] = -0.0;
                    Tensor initial = value(type, true, Shape.of(batch, h), start);
                    Tensor l = lengths(lengths);
                    var result = sequence.forwardGeneralScan(x, l, initial, direction(reverse));
                    var expected = oracle(numbers(x), numbers(initial), numbers(parameters.get(0)), numbers(parameters.get(1)),
                            bias ? numbers(parameters.get(2)) : null, time, batch, f, h, lengths, reverse);
                    try (Engine engine = Engine.standard()) {
                        var actual = engine.compute(List.of(result.finalHidden(), result.outputs()), 100000);
                        assertArrayEquals(expected.finalH, numbers(actual.get(0)), tolerance(type));
                        assertArrayEquals(expected.output, numbers(actual.get(1)), tolerance(type));
                        for (int b = 0; b < batch; b++) if (lengths[b] == 0) {
                            for (int k = 0; k < h; k++) assertEquals(raw(type, start[b * h + k]), raw(type, numbers(actual.get(0))[b * h + k]));
                        }
                        for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) if (t >= lengths[b])
                            for (int k = 0; k < h; k++) assertEquals(0L, raw(type, numbers(actual.get(1))[(t * batch + b) * h + k]));
                        var requested = new ArrayList<>(List.of(x, initial)); requested.addAll(parameters);
                        var inputs = new ArrayList<>(requested); inputs.add(l);
                        var gradients = backwardValues(engine,objective(List.of(result.finalHidden(),result.outputs())),
                                requested,inputs,time == 0,100000);
                        assertEquals(expected.loss, numbers(gradients.getFirst())[0], tolerance(type));
                        List<double[]> g = new ArrayList<>(List.of(expected.dx, expected.dInitial, expected.dw, expected.du));
                        if (bias) g.add(expected.db);
                        for (int i = 0; i < g.size(); i++) assertArrayEquals(g.get(i), numbers(gradients.get(i + 1)), tolerance(type), "requested role " + i);
                        if (Arrays.stream(lengths).allMatch(n -> n == 0)) for (int i = 2; i < g.size(); i++)
                            for (double lane : numbers(gradients.get(i + 1))) assertEquals(0.0, lane, 0.0);
                    }
                }
            }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void allZeroLengthsPreserveSpecialInitialBitsWithoutReadingPaddedInputs(DataType type) throws Exception {
        for (boolean reverse : List.of(false, true)) {
            Tensor initial = type == DataType.FLOAT64
                    ? TensorFactory.fromFlatArray(formal(type, false, Shape.of(2, 2)).descriptor(), Optional.empty(),
                            new double[]{-0.0, Double.longBitsToDouble(0x7ff8000000000123L), Double.NEGATIVE_INFINITY, .25})
                    : TensorFactory.fromFlatArray(formal(type, false, Shape.of(2, 2)).descriptor(), Optional.empty(),
                            new float[]{-0.0f, Float.intBitsToFloat(0x7fc00123), Float.NEGATIVE_INFINITY, .25f});
            var p = parameters(type, true);
            var sequence = new RnnSequence(new RnnCell(p.get(0), p.get(1), p.get(2)));
            var scan = sequence.forwardGeneralScan(value(type, false, Shape.of(3, 2, 3),
                    data(3, 2, 3, new long[]{0, 0}, 1)), lengths(0, 0), initial, direction(reverse));
            try (Engine engine = Engine.standard()) {
                var result = engine.compute(List.of(scan.finalHidden(), scan.outputs()), 10000);
                var original = initial.hostStorage().orElseThrow().segment();
                var actual = result.get(0).bytes();
                for (int i = 0; i < 4; i++) {
                    if (type == DataType.FLOAT64) assertEquals(original.get(ValueLayout.JAVA_LONG, i * 8L), actual.getLong(i * 8));
                    else assertEquals(original.get(ValueLayout.JAVA_INT, i * 4L), actual.getInt(i * 4));
                }
                var emissions = result.get(1).bytes();
                while (emissions.hasRemaining()) assertEquals(0, emissions.get());
            }
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void emptyBatchRetainsDirectGeometryAndRunsNoCellWork(DataType type) throws Exception {
        for (int time : List.of(0, 3)) for (boolean reverse : List.of(false, true)) {
            var p = parameters(type, true);
            var sequence = new RnnSequence(new RnnCell(p.get(0), p.get(1), p.get(2)));
            Tensor x = value(type,false,Shape.of(time,0,3)), initial = value(type,false,Shape.of(0,2)), l = lengths();
            var result = sequence.forwardGeneralScan(x,l,initial,direction(reverse));
            var inputs = new ArrayList<>(p); inputs.addAll(List.of(x,initial,l));
            assertEquals(Shape.of(0, 2), result.finalHidden().descriptor().shape());
            assertEquals(Shape.of(time, 0, 2), result.outputs().descriptor().shape());
            try (Engine engine = Engine.standard()) {
                var values = engine.compute(List.of(result.finalHidden(), result.outputs()), 10000);
                assertEquals(0, values.get(0).bytes().remaining()); assertEquals(0, values.get(1).bytes().remaining());
                var gradient = backwardValues(engine,objective(List.of(result.finalHidden(),result.outputs())),
                        p,inputs,time == 0,10000);
                assertEquals(p.size() + 1,gradient.size());
                for (int i = 1; i < gradient.size(); i++) for (double n : numbers(gradient.get(i))) assertEquals(0,n,0.0);
            }
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void graphModelsSaveColdTrainFreshBatchesSaveCloseLoadAndConfigureNewTraining(DataType type) throws Exception {
        for (boolean reverse : List.of(false, true)) {
            Tensor x = formal(type, false, Shape.of(3, 3, 3)), initial = formal(type, false, Shape.of(3, 2));
            Tensor l = TensorFactory.create(lengths(0, 0, 0).descriptor());
            var external = parameters(type, true);
            var stores = external.stream().map(t -> t.hostStorage().orElseThrow()).toList();
            var originalBytes = external.stream().map(TrainableRuntimeLengthRnnIntegrationTest::bits).toList();
            var sequence = new RnnSequence(new RnnCell(external.get(0), external.get(1), external.get(2)));
            var scan = sequence.forwardGeneralScan(x, l, initial, direction(reverse));
            List<GraphCheckpoint.Leaf> leaves = List.of(parameter("w", external.get(0)), parameter("u", external.get(1)), parameter("b", external.get(2)));
            Path before = directory.resolve("before"), after = directory.resolve("after");
            TrainingStep retained;
            List<byte[]> trainedBytes;
            try (Engine borrowed = Engine.standard(); var model = Model.graph(List.of(x, l, initial),
                    List.of(scan.finalHidden(), scan.outputs()), leaves, borrowed)) {
                model.save(before);
                retained = trainAndCompare(model, type, reverse, 3, 3, 3, 2);
                trainedBytes = model.stateDictionary().entries().stream().map(e -> bits(e.value())).toList();
                model.save(after); model.close(); assertFalse(borrowed.isClosed());
                assertTrue(Double.isFinite(numbers(retained.objective())[0]));
            }
            for (int i = 0; i < external.size(); i++) {
                assertSame(stores.get(i), external.get(i).hostStorage().orElseThrow());
                assertArrayEquals(originalBytes.get(i), bits(external.get(i)));
            }
            for (Path path : List.of(before, after)) {
                assertEquals("graph-model.v5", ArtifactFile.read(path,
                        new ArtifactLimits(1048576, 1048576, 100, 64, 1048576, 1048576)).kind());
                try (var loaded = Model.loadGraph(path)) {
                    var expectedBytes = path.equals(before) ? originalBytes : trainedBytes;
                    var restoredState = loaded.stateDictionary().entries();
                    for (int i = 0; i < restoredState.size(); i++) assertArrayEquals(expectedBytes.get(i), bits(restoredState.get(i).value()));
                    Tensor newX = formal(type, false, x.descriptor().shape()).neg().contiguous();
                    Tensor newH = formal(type, false, initial.descriptor().shape()).neg().contiguous();
                    var cold = loaded.forward(List.of(newX, l, newH));
                    assertSame(cold.getFirst(), cold.getLast().provenance().orElseThrow().producer().output(0));
                    trainAndCompare(loaded, type, reverse, 3, 3, 3, 2);
                    loaded.save(directory.resolve("resaved"));
                }
            }
        }
    }

    @ParameterizedTest @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void explicitBuilderDenseAndFinalRecipesTrainAndRoundtrip(DataType type) throws Exception {
        for (boolean finalHidden : List.of(false, true)) {
            var builder = Model.builder().input(type, 3, 1, 3).seed(19).rnnScan(2);
            if (finalHidden) builder.finalHidden();
            Path before = directory.resolve("builder-before"), after = directory.resolve("builder-after");
            try (var model = builder.buildWithLengths()) {
                model.save(before);
                trainBuilder(model, type, finalHidden);
                model.save(after);
            }
            for (Path path : List.of(before, after)) try (var restored = Model.loadGraph(path)) {
                trainBuilder(restored, type, finalHidden);
            }
        }
    }

    private static void trainBuilder(Model<List<Tensor>, List<Tensor>> model, DataType type, boolean finalHidden) throws Exception {
        model.configureTraining(List.of(), (p, t) -> p.getFirst().mul(p.getFirst()).sum(), new Sgd(.01));
        for (long[] length : List.of(new long[]{3}, new long[]{1}, new long[]{0})) {
            var p = model.stateDictionary().entries().stream().map(e -> e.value()).toList();
            double[][] before = p.stream().map(TrainableRuntimeLengthRnnIntegrationTest::numbers).toArray(double[][]::new);
            Tensor x = value(type, false, Shape.of(3, 1, 3), data(3, 1, 3, length, .1875));
            var expected = oracle(numbers(x), new double[2], before[0], before[1], before[2], 3, 1, 3, 2, length, false,
                    finalHidden ? 1 : 0, finalHidden ? 0 : 1);
            assertArrayEquals(finalHidden ? expected.finalH : expected.output, numbers(model.compute(x, lengths(length))), tolerance(type));
            TrainingStep step;
            try (Arena arena = Arena.ofShared()) {
                step = model.train(List.of(nativeValue(x, arena), nativeValue(lengths(length), arena)), List.of());
            }
            assertEquals(expected.loss, numbers(step.objective())[0], tolerance(type));
            assertUpdates(type, before, List.of(expected.dw, expected.du, expected.db), p);
        }
    }

    private static TrainingStep trainAndCompare(Model<List<Tensor>, List<Tensor>> model, DataType type,
            boolean reverse, int time, int batch, int f, int h) throws Exception {
        AtomicInteger objectives = new AtomicInteger();
        model.configureTraining(List.of(), (p, t) -> { objectives.incrementAndGet(); return objective(p); }, new Sgd(.01));
        TrainingStep retained = null;
        int execution = 0;
        for (long[] length : List.of(new long[]{3, 1, 0}, new long[]{1, 3, 2}, new long[]{0, 0, 0}, new long[]{3, 3, 3})) {
            var p = model.stateDictionary().entries().stream().map(e -> e.value()).toList();
            double[][] before = p.stream().map(TrainableRuntimeLengthRnnIntegrationTest::numbers).toArray(double[][]::new);
            Tensor x = value(type, false, Shape.of(time, batch, f), data(time, batch, f, length, .0625 * (execution + 1)));
            Tensor initial = value(type, false, Shape.of(batch, h), ramp(batch * h, -.125));
            Tensor l = lengths(length);
            var expected = oracle(numbers(x), numbers(initial), before[0], before[1], before[2], time, batch, f, h, length, reverse);
            var actual = model.compute(List.of(x, l, initial));
            assertArrayEquals(expected.finalH, numbers(actual.get(0)), tolerance(type));
            assertArrayEquals(expected.output, numbers(actual.get(1)), tolerance(type));
            try (Arena arena = Arena.ofShared()) {
                retained = model.train(List.of(nativeValue(x, arena), nativeValue(l, arena), nativeValue(initial, arena)), List.of());
            }
            assertEquals(++execution, retained.executionNumber()); assertEquals(execution, retained.optimizerStep().orElseThrow());
            assertEquals(expected.loss, numbers(retained.objective())[0], tolerance(type));
            assertUpdates(type, before, List.of(expected.dw, expected.du, expected.db), p);
            byte[][] unchanged = p.stream().map(TrainableRuntimeLengthRnnIntegrationTest::bits).toArray(byte[][]::new);
            try (Arena arena = Arena.ofShared()) {
                var invalid = List.of(nativeValue(x, arena), nativeValue(lengths(3, 1, 4), arena), nativeValue(initial, arena));
                assertThrows(IllegalArgumentException.class, () -> model.train(invalid, List.of()));
            }
            for (int i = 0; i < p.size(); i++) assertArrayEquals(unchanged[i], bits(p.get(i)));
        }
        assertEquals(1, objectives.get());
        return retained;
    }

    private static void assertUpdates(DataType type, double[][] before, List<double[]> gradient, List<Tensor> actual) {
        for (int i = 0; i < actual.size(); i++) {
            double[] expected = before[i].clone();
            for (int j = 0; j < expected.length; j++) expected[j] = type == DataType.FLOAT64
                    ? expected[j] - .01 * gradient.get(i)[j] : (float) expected[j] - .01f * (float) gradient.get(i)[j];
            assertArrayEquals(expected, numbers(actual.get(i)), tolerance(type));
        }
    }

    /** Scalar forward and hand-derived reverse; no Tensor equation, scan or engine oracle. */
    private static Oracle oracle(double[] x, double[] initial, double[] w, double[] u, double[] bias,
            int time, int batch, int f, int h, long[] lengths, boolean reverse) {
        return oracle(x, initial, w, u, bias, time, batch, f, h, lengths, reverse, 1, 1);
    }
    private static Oracle oracle(double[] x, double[] initial, double[] w, double[] u, double[] bias,
            int time, int batch, int f, int h, long[] lengths, boolean reverse, int finalSeed, int emissionSeed) {
        double[] state = initial.clone(), output = new double[time * batch * h], previous = new double[output.length];
        for (int b = 0; b < batch; b++) for (int k = 0; k < lengths[b]; k++) {
            int t = reverse ? (int) lengths[b] - 1 - k : k;
            double[] next = new double[h];
            System.arraycopy(state, b * h, previous, (t * batch + b) * h, h);
            for (int j = 0; j < h; j++) {
                double p = 0, q = 0;
                for (int i = 0; i < f; i++) p += x[(t * batch + b) * f + i] * w[j * f + i];
                if (bias != null) p += bias[j];
                for (int i = 0; i < h; i++) q += state[b * h + i] * u[j * h + i];
                next[j] = Math.tanh(p + q); output[(t * batch + b) * h + j] = next[j];
            }
            System.arraycopy(next, 0, state, b * h, h);
        }
        double loss = 0;
        for (double n : state) loss += finalSeed * n * n;
        for (double n : output) loss += emissionSeed * n * n;
        double[] dx = new double[x.length], dh = new double[initial.length], dw = new double[w.length], du = new double[u.length], db = new double[h];
        for (int b = 0; b < batch; b++) {
            double[] carry = new double[h];
            for (int j = 0; j < h; j++) carry[j] = 2 * finalSeed * state[b * h + j];
            for (int k = (int) lengths[b] - 1; k >= 0; k--) {
                int t = reverse ? (int) lengths[b] - 1 - k : k;
                double[] old = new double[h];
                for (int j = 0; j < h; j++) {
                    double y = output[(t * batch + b) * h + j];
                    double d = (carry[j] + 2 * emissionSeed * y) * (1 - y * y);
                    db[j] += d;
                    for (int i = 0; i < f; i++) {
                        dw[j * f + i] += d * x[(t * batch + b) * f + i];
                        dx[(t * batch + b) * f + i] += d * w[j * f + i];
                    }
                    for (int i = 0; i < h; i++) {
                        du[j * h + i] += d * previous[(t * batch + b) * h + i]; old[i] += d * u[j * h + i];
                    }
                }
                carry = old;
            }
            System.arraycopy(carry, 0, dh, b * h, h);
        }
        return new Oracle(state, output, loss, dx, dh, dw, du, db);
    }
    private record Oracle(double[] finalH, double[] output, double loss, double[] dx,
            double[] dInitial, double[] dw, double[] du, double[] db) {}

    private static List<long[]> lengthCases(int time, int batch) {
        if (time == 0) return List.of(new long[batch]);
        long[] full = new long[batch]; Arrays.fill(full, time);
        long[] partial = new long[batch]; for (int b = 0; b < batch; b++) partial[b] = b == 0 ? 1 : b == 1 ? time : 0;
        return List.of(full, partial, new long[batch]);
    }
    private static double[] data(int time, int batch, int f, long[] lengths, double scale) {
        double[] x = new double[time * batch * f];
        for (int t = 0; t < time; t++) for (int b = 0; b < batch; b++) for (int i = 0; i < f; i++)
            x[(t * batch + b) * f + i] = t < lengths[b] ? (i + 1 - b + t) * scale : Double.NaN;
        return x;
    }
    private static double[] ramp(int count, double scale) {
        double[] values = new double[count]; for (int i = 0; i < count; i++) values[i] = (i - 2) * scale; return values;
    }
    private static List<Tensor> parameters(DataType type, boolean bias) {
        Tensor w = value(type, true, Shape.of(2, 3), .25, -.125, .375, -.25, .125, .0625);
        Tensor u = value(type, true, Shape.of(2, 2), .125, -.1875, .25, .0625);
        return bias ? List.of(w, u, value(type, true, Shape.of(2), .03125, -.0625)) : List.of(w, u);
    }
    /** The convenience stays strict; T0's absent sequence/parameter routes need explicit ZERO. */
    private static List<HostTensorValue> backwardValues(Engine engine,Tensor objective,List<Tensor> targets,
            List<Tensor> inputs,boolean hasDisconnectedTarget,long maximumBytes) {
        if (!hasDisconnectedTarget) {
            var backward = engine.backward(objective,targets,maximumBytes);
            var values = new ArrayList<HostTensorValue>(targets.size() + 1);
            values.add(backward.objective()); values.addAll(backward.gradients());
            return List.copyOf(values);
        }
        Tensor seed = value(objective.descriptor().dataType(),false,Shape.of(),1);
        var compiled = EngineCpuGeneralScanIntegrationTest.compileScanGradients(
                engine,List.of(objective),List.of(seed),targets,true);
        var pool = new ArrayList<>(inputs); pool.add(seed);
        try (var prepared = engine.prepare(compiled);
                var result = engine.runValues(prepared,compiled.inputs().stream().map(input ->
                        pool.stream().filter(t -> t.id().equals(input.tensorId())).findFirst().orElseThrow()).toList())) {
            return result.publications().stream().map(p -> result.materialize(p,maximumBytes)).toList();
        }
    }

    private static Tensor objective(List<Tensor> p) { return p.get(0).mul(p.get(0)).sum().add(p.get(1).mul(p.get(1)).sum()); }
    private static RecurrentDirection direction(boolean reverse) { return reverse ? RecurrentDirection.REVERSE : RecurrentDirection.FORWARD; }
    private static GraphCheckpoint.Leaf parameter(String path, Tensor tensor) { return new GraphCheckpoint.Leaf(path, GraphCheckpoint.Role.PARAMETER, tensor); }
    private static Tensor formal(DataType type, boolean grad, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }
    private static Tensor lengths(long... values) {
        return TensorFactory.fromFlatArray(formal(DataType.INT64, false, Shape.of(values.length)).descriptor(), Optional.empty(), values);
    }
    private static Tensor value(DataType type, boolean grad, Shape shape, double... values) {
        assertEquals(shape.knownElementCount().orElseThrow(), values.length);
        if (type == DataType.FLOAT64) return TensorFactory.fromFlatArray(formal(type, grad, shape).descriptor(), Optional.empty(), values.clone());
        float[] lanes = new float[values.length]; for (int i = 0; i < lanes.length; i++) lanes[i] = (float) values[i];
        return TensorFactory.fromFlatArray(formal(type, grad, shape).descriptor(), Optional.empty(), lanes);
    }
    private static Tensor nativeValue(Tensor tensor, Arena arena) {
        var storage = tensor.hostStorage().orElseThrow(); var memory = arena.allocate(storage.byteSize(), tensor.descriptor().dataType().byteWidth());
        memory.copyFrom(storage.segment());
        return TensorFactory.create(tensor.descriptor(), Optional.empty(), Optional.of(new MemorySegmentStorage(
                tensor.descriptor().dataType(), tensor.descriptor().shape().knownElementCount().orElseThrow(), memory)));
    }
    private static byte[] bits(Tensor tensor) { return tensor.hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_BYTE); }
    private static double[] numbers(Tensor tensor) { return decode(ByteBuffer.wrap(bits(tensor)).order(ByteOrder.nativeOrder()), tensor.descriptor().dataType()); }
    private static double[] numbers(HostTensorValue value) { return decode(value.bytes(), value.dataType()); }
    private static double[] decode(ByteBuffer buffer, DataType type) {
        double[] values = new double[buffer.remaining() / type.byteWidth()];
        for (int i = 0; i < values.length; i++) values[i] = type == DataType.FLOAT64 ? buffer.getDouble() : buffer.getFloat(); return values;
    }
    private static long raw(DataType type, double value) { return type == DataType.FLOAT64 ? Double.doubleToRawLongBits(value) : Float.floatToRawIntBits((float) value); }
    private static double tolerance(DataType type) { return type == DataType.FLOAT64 ? 3e-11 : 8e-5; }
}
