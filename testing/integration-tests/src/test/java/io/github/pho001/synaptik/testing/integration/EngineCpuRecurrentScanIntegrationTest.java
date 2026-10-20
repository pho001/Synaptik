package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Ordinary public lifecycle, positional binding, publication, checkpoint and isolation evidence. */
final class EngineCpuRecurrentScanIntegrationTest {
    @TempDir Path directory;

    @Test void everySignatureAndDirectionRunsWithFlagsPortOrderingAndChangingLengths() {
        for (var kind : RecurrentScanKind.values()) for (boolean biased : List.of(false, true))
            for (var direction : RecurrentDirection.values()) try (Engine engine = Engine.standard()) {
                var p = parts(kind, biased, direction, true);
                var roots = new ArrayList<>(p.outputs()); Collections.reverse(roots);
                assertTrue(roots.stream().allMatch(t -> t.descriptor().requiresGrad()));
                var compiled = engine.compile(roots); // ordinary FORWARD_ONLY despite eligibility flags
                try (var prepared = engine.prepare(compiled)) {
                    var reverseInputs = new ArrayList<>(ordered(compiled, p.inputs())); Collections.reverse(reverseInputs);
                    try (var first = engine.run(prepared, reverseInputs)) {
                        assertEquals(roots.size(), first.resultCount());
                        var preserved = bytes(first, 0);
                        var fresh = freshValues(ordered(compiled, p.inputs()), p.lengths(), new long[] {0, 2});
                        try (var second = engine.runValues(prepared, fresh)) {
                            assertNotEquals(preserved, bytes(second, 0));
                            assertEquals(preserved, bytes(first, 0));
                        }
                        var invalid = freshValues(ordered(compiled, p.inputs()), p.lengths(), new long[] {1, 3});
                        assertThrows(IllegalArgumentException.class, () -> engine.runValues(prepared, invalid));
                        try (var repaired = engine.runValues(prepared, fresh)) { assertEquals(roots.size(), repaired.resultCount()); }
                    }
                    assertThrows(IllegalArgumentException.class, () -> engine.run(prepared,
                            Collections.nCopies(compiled.inputs().size(), p.inputs().getFirst())));
                }
                assertThrows(IllegalArgumentException.class, () -> engine.compile(List.of(p.outputs().get(0), p.outputs().get(0))));
                var objective = p.outputs().getLast().sum();
                assertThrows(IllegalArgumentException.class, () -> engine.backward(objective, List.of(p.weight()), 10000));
                assertThrows(IllegalArgumentException.class, () -> engine.compile(List.of(objective),
                        List.of(floats(Shape.scalar(), false, 1)), List.of(p.weight())));
            }
    }

    @Test void finalCellOnlyStillExecutesSiblingOutputsAndConcurrentRunsHaveIsolatedScratch() throws Exception {
        try (Engine engine = Engine.standard()) {
            var p = parts(RecurrentScanKind.LSTM, true, RecurrentDirection.REVERSE, false);
            var compiled = engine.compile(List.of(p.outputs().get(2)));
            try (var prepared = engine.prepare(compiled); var executor = Executors.newFixedThreadPool(2)) {
                var valuesA = freshValues(ordered(compiled, p.inputs()), p.lengths(), new long[] {0, 1});
                var valuesB = freshValues(ordered(compiled, p.inputs()), p.lengths(), new long[] {2, 2});
                List<Integer> expectedA, expectedB;
                try (var a = engine.runValues(prepared, valuesA); var b = engine.runValues(prepared, valuesB)) {
                    expectedA = bytes(a, 0); expectedB = bytes(b, 0); assertNotEquals(expectedA, expectedB);
                }
                var start = new CountDownLatch(1);
                var a = executor.submit(() -> { start.await(); try (var r = engine.runValues(prepared, valuesA)) { return bytes(r, 0); } });
                var b = executor.submit(() -> { start.await(); try (var r = engine.runValues(prepared, valuesB)) { return bytes(r, 0); } });
                start.countDown(); assertEquals(expectedA, a.get(10, TimeUnit.SECONDS)); assertEquals(expectedB, b.get(10, TimeUnit.SECONDS));
            }
        }
    }

    @Test void distinctEquivalentRootsAndFanoutPreservePublicationOccurrences() {
        try (Engine engine = Engine.standard()) {
            var p = parts(RecurrentScanKind.RNN_TANH, false, RecurrentDirection.FORWARD, false);
            var h = p.outputs().get(1); var alias = h.neg().neg();
            var compiled = engine.compile(List.of(alias, p.outputs().get(0), h, h.add(h)));
            try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, ordered(compiled, p.inputs()))) {
                assertEquals(4, result.resultCount()); assertNotSame(result.publications().get(0), result.publications().get(2));
                assertEquals(bytes(result, 0), bytes(result, 2));
                var hidden = floats(result, 2); var doubled = floats(result, 3);
                for (int i = 0; i < hidden.length; i++) assertEquals(hidden[i] + hidden[i], doubled[i]);
            }
        }
    }

    @Test void actualParameterWeightsSurviveGraphCheckpointSaveLoadAndOrdinaryRunValues() throws Exception {
        var p = parts(RecurrentScanKind.LSTM, true, RecurrentDirection.REVERSE, true);
        var external = p.inputs().subList(0, 4);
        var leaves = new ArrayList<GraphCheckpoint.Leaf>();
        for (int i = 4; i < p.inputs().size(); i++) leaves.add(new GraphCheckpoint.Leaf(
                "parameter" + i, GraphCheckpoint.Role.PARAMETER, p.inputs().get(i)));
        var limits = new ArtifactLimits(100000, 10000, 10, 32, 10000, 90000);
        var file = directory.resolve("scan.graph");
        GraphCheckpoint.save(file, new GraphCheckpoint.Source(external,
                List.of(p.outputs().get(2), p.outputs().get(0), p.outputs().get(1)), leaves), limits);
        var loaded = GraphCheckpoint.load(file, limits);
        assertTrue(loaded.leaves().stream().allMatch(l -> l.role() == GraphCheckpoint.Role.PARAMETER
                && l.tensor().descriptor().requiresGrad()));
        var rebound = GraphCheckpoint.rebind(loaded, external);
        var restoredInputs = new ArrayList<>(external); loaded.leaves().forEach(l -> restoredInputs.add(l.tensor()));
        try (Engine engine = Engine.standard()) {
            var original = engine.compile(List.of(p.outputs().get(2), p.outputs().get(0), p.outputs().get(1)));
            var restored = engine.compile(rebound);
            try (var a = engine.prepare(original); var b = engine.prepare(restored);
                    var x = engine.run(a, ordered(original, p.inputs())); var y = engine.run(b, ordered(restored, restoredInputs))) {
                for (int port = 0; port < 3; port++) assertEquals(bytes(x, port), bytes(y, port));
                var fresh = freshValues(ordered(restored, restoredInputs), p.lengths(), new long[] {0, 2});
                try (var changed = engine.runValues(b, fresh)) { assertNotEquals(bytes(y, 0), bytes(changed, 0)); }
            }
        }
    }

    @Test void castProducedLengthsAreOrdinaryRuntimeValuesAcrossPublicRuns() {
        try (Engine engine = Engine.standard()) {
            var x = floats(Shape.of(2, 2, 1), false, 1, 2, Float.NaN, 3);
            var source = floats(Shape.of(2), false, 1, 2);
            var h = floats(Shape.of(2, 1), false, -0f, 0);
            var w = floats(Shape.of(1, 1), false, 1); var u = floats(Shape.of(1, 1), false, 0);
            var scan = RecurrentScan.rnn(x, source.cast(DataType.INT64), h, w, u, RecurrentDirection.FORWARD);
            var pool = List.of(x, source, h, w, u);
            var compiled = engine.compile(List.of(scan.finalHidden(), scan.outputs()));
            try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, ordered(compiled, pool))) {
                assertEquals(0, Float.floatToRawIntBits(floats(result, 1)[2]));
                var invalid = new ArrayList<>(ordered(compiled, pool));
                invalid.set(invalid.indexOf(source), floats(Shape.of(2), false, 1, 3));
                assertThrows(IllegalArgumentException.class, () -> engine.runValues(prepared, invalid));
                try (var next = engine.runValues(prepared, ordered(compiled, pool))) { assertEquals(bytes(result, 0), bytes(next, 0)); }
            }
        }
    }

    private static List<Tensor> ordered(io.github.pho001.synaptik.engine.CompiledGraph compiled, List<Tensor> pool) {
        return compiled.inputs().stream().map(input -> pool.stream().filter(t -> t.id().equals(input.tensorId())).findFirst().orElseThrow()).toList();
    }

    private record Parts(List<Tensor> inputs, List<Tensor> outputs, Tensor lengths, Tensor weight) { }
    private static Parts parts(RecurrentScanKind kind, boolean biased, RecurrentDirection direction, boolean grad) {
        int gates = kind == RecurrentScanKind.LSTM ? 4 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 1;
        var x = floats(Shape.of(2, 2, 1), grad, .2f, .3f, .4f, .5f);
        var lengths = tensor(DataType.INT64, Shape.of(2), false, MemorySegment.ofArray(new long[] {1, 2}));
        var h = floats(Shape.of(2, 1), false, -.2f, .1f); var c = floats(Shape.of(2, 1), false, .3f, -.4f);
        float[] weightValues = new float[gates], biasValues = new float[gates];
        Arrays.fill(weightValues, .5f); Arrays.fill(biasValues, .07f);
        var w = floats(Shape.of(gates, 1), grad, weightValues); var u = floats(Shape.of(gates, 1), grad, weightValues);
        var bias = floats(Shape.of(gates), grad, biasValues);
        List<Tensor> outputs;
        if (kind == RecurrentScanKind.LSTM) {
            var s = biased ? RecurrentScan.lstm(x, lengths, h, c, w, u, bias, direction) : RecurrentScan.lstm(x, lengths, h, c, w, u, direction);
            outputs = List.of(s.outputs(), s.finalHidden(), s.finalCell());
        } else {
            var s = kind == RecurrentScanKind.RNN_TANH
                    ? biased ? RecurrentScan.rnn(x, lengths, h, w, u, bias, direction) : RecurrentScan.rnn(x, lengths, h, w, u, direction)
                    : biased ? RecurrentScan.gru(x, lengths, h, w, u, bias, direction) : RecurrentScan.gru(x, lengths, h, w, u, direction);
            outputs = List.of(s.outputs(), s.finalHidden());
        }
        var inputs = new ArrayList<>(List.of(x, lengths, h)); if (kind == RecurrentScanKind.LSTM) inputs.add(c);
        inputs.add(w); inputs.add(u); if (biased) inputs.add(bias);
        return new Parts(inputs, outputs, lengths, w);
    }
    private static List<Tensor> freshValues(List<Tensor> inputs, Tensor lengths, long[] values) {
        return inputs.stream().map(t -> t == lengths ? tensor(DataType.INT64, t.descriptor().shape(), false, MemorySegment.ofArray(values))
                : TensorFactory.create(t.descriptor(), Optional.empty(), Optional.of(new MemorySegmentStorage(t.descriptor().dataType(),
                    t.descriptor().shape().knownElementCount().orElseThrow(), t.hostStorage().orElseThrow().segment())))).toList();
    }
    private static Tensor floats(Shape shape, boolean grad, float... values) { return tensor(DataType.FLOAT32, shape, grad, MemorySegment.ofArray(values)); }
    private static Tensor tensor(DataType type, Shape shape, boolean grad, MemorySegment segment) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, shape.knownElementCount().orElseThrow(), segment)));
    }
    private static List<Integer> bytes(RunResult result, int port) {
        var bytes = result.materialize(result.publications().get(port), 100000).bytes();
        var values = new ArrayList<Integer>(); while (bytes.hasRemaining()) values.add(Byte.toUnsignedInt(bytes.get())); return values;
    }
    private static float[] floats(RunResult result, int port) {
        var bytes = result.materialize(result.publications().get(port), 100000).bytes();
        var values = new float[bytes.remaining() / 4]; for (int i = 0; i < values.length; i++) values[i] = bytes.getFloat(); return values;
    }
}
