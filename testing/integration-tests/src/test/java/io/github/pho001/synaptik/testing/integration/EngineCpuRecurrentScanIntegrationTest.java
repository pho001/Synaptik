package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
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
        for (var type : floatingTypes()) for (int mode = 0; mode < 3; mode++)
            for (var kind : RecurrentScanKind.values()) for (boolean biased : List.of(false, true))
            for (var direction : RecurrentDirection.values()) try (Engine engine = Engine.standard()) {
                var p = parts(kind, biased, direction, true, type, mode);
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
        for (var type : floatingTypes()) try (Engine engine = Engine.standard()) {
            var p = parts(RecurrentScanKind.LSTM, true, RecurrentDirection.REVERSE, false, type, 1);
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
        for (var type : floatingTypes()) try (Engine engine = Engine.standard()) {
            var p = parts(RecurrentScanKind.RNN_TANH, false, RecurrentDirection.FORWARD, false, type, 1);
            var h = p.outputs().get(1); var alias = h.neg().neg();
            var compiled = engine.compile(List.of(alias, p.outputs().get(0), h, h.add(h)));
            try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, ordered(compiled, p.inputs()))) {
                assertEquals(4, result.resultCount()); assertNotSame(result.publications().get(0), result.publications().get(2));
                assertEquals(bytes(result, 0), bytes(result, 2));
                var hidden = numbers(result, 2, type); var doubled = numbers(result, 3, type);
                for (int i = 0; i < hidden.length; i++) assertEquals(published(type, hidden[i] + hidden[i]), doubled[i]);
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
        for (var type : floatingTypes()) try (Engine engine = Engine.standard()) {
            var x = floating(type, Shape.of(2, 2, 1), false, 1, 1, 2, Double.NaN, 3);
            var source = floats(Shape.of(2), false, 1, 2);
            var h = floating(type, Shape.of(2, 1), false, 1, -0d, 0);
            var w = floating(type, Shape.of(1, 1), false, 1, 1); var u = floating(type, Shape.of(1, 1), false, 1, 0);
            var scan = RecurrentScan.rnn(x, source.cast(DataType.INT64), h, w, u, RecurrentDirection.FORWARD);
            var pool = List.of(x, source, h, w, u);
            var compiled = engine.compile(List.of(scan.finalHidden(), scan.outputs()));
            try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, ordered(compiled, pool))) {
                assertEquals(0d, numbers(result, 1, type)[2]);
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
        return parts(kind, biased, direction, grad, DataType.FLOAT32, 0);
    }
    private static Parts parts(RecurrentScanKind kind, boolean biased, RecurrentDirection direction, boolean grad, DataType type, int mode) {
        int gates = kind == RecurrentScanKind.LSTM ? 4 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 1;
        var x = floating(type, Shape.of(2, 2, 1), grad, mode, .2, .3, .4, .5);
        var lengthLayout = readLayout(Shape.of(2), mode);
        long[] lengthValues = new long[Math.toIntExact(lengthLayout.referencedElementSpan())]; Arrays.fill(lengthValues, Long.MAX_VALUE);
        lengthValues[Math.toIntExact(lengthLayout.storageOffset())] = 1;
        lengthValues[Math.toIntExact(lengthLayout.storageOffset() + lengthLayout.stride(0))] = mode == 2 ? 1 : 2;
        var lengths = TensorFactory.create(new TensorDescriptor(DataType.INT64, Shape.of(2), Optional.of(lengthLayout), false),
                Optional.empty(), Optional.of(new MemorySegmentStorage(DataType.INT64, lengthValues.length, MemorySegment.ofArray(lengthValues))));
        var h = floating(type, Shape.of(2, 1), false, mode, -.2, .1);
        var c = floating(type, Shape.of(2, 1), false, mode, .3, -.4);
        double[] weightValues = new double[gates], biasValues = new double[gates];
        Arrays.fill(weightValues, .5); Arrays.fill(biasValues, .07);
        var w = floating(type, Shape.of(gates, 1), grad, mode, weightValues);
        var u = floating(type, Shape.of(gates, 1), grad, mode, weightValues);
        var bias = floating(type, Shape.of(gates), grad, mode, biasValues);
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
        return inputs.stream().map(t -> {
            var d = t.descriptor(); var layout = d.layout().orElseThrow();
            long count = layout.referencedElementSpan();
            var old = t.hostStorage().orElseThrow().segment();
            // New identities and fresh dtype-correct heaps retain the entire referenced span,
            // including offset prefixes and poisoned gaps, not merely the logical Shape count.
            MemorySegment storage = switch (d.dataType()) {
                case FLOAT64 -> MemorySegment.ofArray(old.toArray(ValueLayout.JAVA_DOUBLE));
                case FLOAT32 -> MemorySegment.ofArray(old.toArray(ValueLayout.JAVA_FLOAT));
                case BFLOAT16, FLOAT16 -> MemorySegment.ofArray(old.toArray(ValueLayout.JAVA_SHORT));
                case INT64 -> MemorySegment.ofArray(old.toArray(ValueLayout.JAVA_LONG));
                default -> throw new AssertionError(d.dataType());
            };
            if (t == lengths) for (int b = 0; b < values.length; b++) storage.setAtIndex(ValueLayout.JAVA_LONG,
                    layout.storageOffset() + b * layout.stride(0), values[b]);
            return TensorFactory.create(d, Optional.empty(), Optional.of(new MemorySegmentStorage(d.dataType(), count, storage)));
        }).toList();
    }

    private static List<DataType> floatingTypes() { return List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16); }
    private static LayoutDescriptor readLayout(Shape shape, int mode) {
        if (mode == 0) return LayoutDescriptor.contiguous(shape);
        var strides = LayoutDescriptor.contiguous(shape).strides();
        for (int axis = 0; axis < strides.length; axis++) strides[axis] = mode == 1 ? strides[axis] * 2 : 0;
        return LayoutDescriptor.of(shape, strides, 3, true);
    }
    private static Tensor floating(DataType type, Shape shape, boolean grad, int mode, double... values) {
        var layout = readLayout(shape, mode); int count = Math.toIntExact(layout.referencedElementSpan());
        MemorySegment storage = switch (type) {
            case FLOAT64 -> MemorySegment.ofArray(new double[count]);
            case FLOAT32 -> MemorySegment.ofArray(new float[count]);
            case BFLOAT16, FLOAT16 -> MemorySegment.ofArray(new short[count]);
            default -> throw new AssertionError(type);
        };
        storage.fill((byte) 0x7f);
        long[] extents = shape.toLongArray();
        for (int p = 0; p < values.length; p++) {
            long address = layout.storageOffset(), ordinal = p;
            for (int axis = extents.length - 1; axis >= 0; axis--) { address += ordinal % extents[axis] * layout.stride(axis); ordinal /= extents[axis]; }
            switch (type) {
                case FLOAT64 -> storage.setAtIndex(ValueLayout.JAVA_DOUBLE, address, values[p]);
                case FLOAT32 -> storage.setAtIndex(ValueLayout.JAVA_FLOAT, address, (float) values[p]);
                case BFLOAT16 -> storage.setAtIndex(ValueLayout.JAVA_SHORT, address, BFloat16Bits.fromFloat((float) values[p]));
                case FLOAT16 -> storage.setAtIndex(ValueLayout.JAVA_SHORT, address, Float16Bits.fromFloat((float) values[p]));
                default -> throw new AssertionError(type);
            }
        }
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.of(layout), grad), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, count, storage)));
    }
    private static double published(DataType type, double value) {
        return type == DataType.FLOAT64 ? value : type == DataType.FLOAT32 ? (float) value
                : type == DataType.BFLOAT16 ? BFloat16Bits.toFloat(BFloat16Bits.fromFloat((float) value))
                : Float16Bits.toFloat(Float16Bits.fromFloat((float) value));
    }
    private static double[] numbers(RunResult result, int port, DataType type) {
        var data = result.materialize(result.publications().get(port), 100000).bytes();
        var values = new double[data.remaining() / type.byteWidth()];
        for (int p = 0; p < values.length; p++) values[p] = type == DataType.FLOAT64 ? data.getDouble()
                : type == DataType.FLOAT32 ? data.getFloat() : type == DataType.BFLOAT16 ? BFloat16Bits.toFloat(data.getShort())
                : Float16Bits.toFloat(data.getShort());
        return values;
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
