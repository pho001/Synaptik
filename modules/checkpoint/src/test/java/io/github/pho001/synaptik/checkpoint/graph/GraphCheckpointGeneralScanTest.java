package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Cold general-body normalization and ownership; no backend execution eligibility is inferred. */
final class GraphCheckpointGeneralScanTest {
    @Test void zeroCarryAndRowCountsCopySaveLoadAndRebindWithoutReadingExternalLengths(
            @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (long time : new long[] {0, 3}) for (long batch : new long[] {0, 1, 3})
                for (boolean reverse : List.of(false, true)) for (int topology = 0; topology < 4; topology++) {
                    boolean shared = topology >= 2, sharedSequence = topology == 3;
                    Tensor x = sharedSequence ? f(type, 2) : f(type, batch, 2), old = f(type, 2);
                    Tensor w = f(type, 2), state = tensor(type, Shape.of(2), false, true, raw(type, 2));
                    Tensor step = x.mul(w);
                    var body = new ScanBody(shared ? List.of(old) : List.of(), List.of(x), List.of(w),
                            shared ? List.of(old.add(sharedSequence ? step : step.sum(0, false))) : List.of(),
                            sharedSequence ? List.of() : List.of(step, step));
                    var initial = shared ? List.of(f(type, 2)) : List.<Tensor>of();
                    Tensor sequence = sharedSequence ? f(type, time, 2) : f(type, time, batch, 2);
                    Tensor lengths = f(DataType.INT64, batch);
                    MemorySegmentStorage control;
                    try (Arena arena = Arena.ofConfined()) {
                        control = new MemorySegmentStorage(DataType.INT64, batch, arena.allocate(batch * 8, 8));
                        lengths.replaceHostStorage(control);
                    }
                    var axes = List.of(sharedSequence ? java.util.OptionalInt.empty() : java.util.OptionalInt.of(0));
                    var result = topology == 0 ? Scan.apply(initial, List.of(sequence), List.of(state), body, reverse)
                            : Scan.apply(initial, List.of(sequence), List.of(state), lengths, 0, axes, body, reverse);
                    var inputs = new ArrayList<>(initial); inputs.add(sequence);
                    if (topology != 0) inputs.add(lengths);
                    var roots = new ArrayList<>(result.finalCarry()); roots.addAll(result.outputs()); roots.addAll(result.residuals());
                    var source = new GraphCheckpoint.Source(inputs, roots, List.of(
                            leaf("state", GraphCheckpoint.Role.BUFFER, state), leaf("alias", GraphCheckpoint.Role.BUFFER, state)));
                    var expected = GraphSource.capture(source).graph;
                    var path = directory.resolve("topology");
                    GraphCheckpoint.save(path, source, LIMITS);
                    for (var acquired : List.of(GraphCheckpoint.copy(source), GraphCheckpoint.load(path, LIMITS))) {
                        assertEquals(expected, GraphSource.capture(acquired).graph);
                        assertTrue(acquired.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                        Tensor owned = acquired.leaves().getFirst().tensor();
                        assertSame(owned, acquired.leaves().getLast().tensor()); assertNotSame(state, owned);
                        assertArrayEquals(raw(type, 2), GraphValueStorage.encode(
                                owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
                        var replacements = acquired.inputs().stream().map(t -> TensorFactory.create(t.descriptor())).toList();
                        var rebound = GraphCheckpoint.rebind(acquired, replacements);
                        assertEquals(expected, GraphSource.capture(new GraphCheckpoint.Source(
                                replacements, rebound, acquired.leaves())).graph);
                        var producer = rebound.getFirst().provenance().orElseThrow().producer();
                        for (int port = 0; port < roots.size(); port++) assertSame(producer.output(port), rebound.get(port));
                        if (topology != 0) assertSame(replacements.getLast(), producer.inputs().getLast());
                        if (!sharedSequence) {
                            int emission = initial.size();
                            assertNotSame(rebound.get(emission), rebound.get(emission + 1));
                            assertSame(producer.scanBody().orElseThrow().emissions().getFirst(),
                                    producer.scanBody().orElseThrow().emissions().getLast());
                        }
                        var wrong = new ArrayList<>(replacements);
                        wrong.set(0, f(type, 7));
                        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(acquired, wrong));
                    }
                    assertSame(control, lengths.hostStorage().orElseThrow()); assertFalse(control.isAlive());
                }
    }

    @Test void zeroRowScalarAccumulatorRetainsMixedRoleTypesAndRawLowState(
            @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        for (DataType stateType : List.of(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (long time : List.of(0L, 3L)) for (long batch : List.of(0L, 3L))
                for (boolean reverse : List.of(false, true)) {
                    Tensor old = f(DataType.FLOAT32), x = f(DataType.FLOAT64, batch, 2), w = f(stateType, 2);
                    Tensor product = x.mul(w.cast(DataType.FLOAT64));
                    var body = new ScanBody(List.of(old), List.of(x), List.of(w),
                            List.of(old.add(product.sum().cast(DataType.FLOAT32))), List.of(product));
                    Tensor initial = f(DataType.FLOAT32), sequence = f(DataType.FLOAT64, time, batch, 2);
                    Tensor lengths = f(DataType.INT64, batch);
                    Tensor state = tensor(stateType, Shape.of(2), false, true, raw(stateType, 2));
                    var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(state), lengths, 0,
                            List.of(java.util.OptionalInt.of(0)), body, reverse);
                    var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
                    var source = new GraphCheckpoint.Source(List.of(sequence, lengths, initial), roots,
                            List.of(leaf("state", GraphCheckpoint.Role.BUFFER, state)));
                    var expected = GraphSource.capture(source).graph;
                    var file = directory.resolve("mixed-zero-row");
                    GraphCheckpoint.save(file, source, LIMITS);
                    for (var acquired : List.of(GraphCheckpoint.copy(source), GraphCheckpoint.load(file, LIMITS))) {
                        assertEquals(expected, GraphSource.capture(acquired).graph);
                        Tensor owned = acquired.leaves().getFirst().tensor();
                        assertEquals(state.descriptor(), owned.descriptor());
                        assertArrayEquals(raw(stateType, 2), GraphValueStorage.encode(
                                owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
                        var replacements = acquired.inputs().stream().map(t -> TensorFactory.create(t.descriptor())).toList();
                        var rebound = GraphCheckpoint.rebind(acquired, replacements);
                        assertEquals(expected, GraphSource.capture(new GraphCheckpoint.Source(
                                replacements, rebound, acquired.leaves())).graph);
                        var fresh = rebound.getFirst().provenance().orElseThrow().producer().scanBody().orElseThrow();
                        assertEquals(DataType.FLOAT32, fresh.carry().getFirst().descriptor().dataType());
                        assertEquals(Shape.scalar(), fresh.carry().getFirst().descriptor().shape());
                        assertEquals(DataType.FLOAT64, fresh.step().getFirst().descriptor().dataType());
                        assertEquals(stateType, fresh.invariants().getFirst().descriptor().dataType());
                    }
                }
    }

    @Test void symbolicLengthSourceDoesNotOpenCopyRebindOrSave(@org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers");
        var width = new io.github.pho001.synaptik.model.shape.StaticDimension(2);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : new long[] {0, 3}) for (boolean reverse : List.of(false, true)) {
                Shape row = Shape.ofDimensions(batch, width);
                Tensor h = prototype(type, row, false, false), x = prototype(type, row, false, false);
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of(h));
                Tensor initial = prototype(type, row, false, false);
                Tensor sequence = prototype(type, Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(time), batch, width), false, false);
                Tensor lengths = prototype(DataType.INT64, Shape.ofDimensions(batch), false, false);
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                        List.of(java.util.OptionalInt.of(0)), body, reverse);
                var source = new GraphCheckpoint.Source(List.of(initial, sequence, lengths), scan.finalCarry(), List.of());
                Tensor probe = prototype(DataType.BOOL, Shape.scalar(), false, false);
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source));
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, source.inputs()));
                var path = directory.resolve("symbolic-length.syn");
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(path, source, LIMITS));
                assertFalse(java.nio.file.Files.exists(path));
                assertEquals(probe.id().value() + 1, prototype(DataType.BOOL, Shape.scalar(), false, false).id().value());
            }
    }

    @Test void staticLengthCopyPersistenceAndRebindPreservePrivateStateRawBitsAndUnreadExternalControls() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (long batch : List.of(0L, 1L, 3L)) for (long time : List.of(0L, 3L))
                for (boolean reverse : List.of(false, true)) {
                    Tensor h = f(type, batch, 2), x = f(type, batch, 2), w = f(type, 2);
                    Tensor initial = f(type, batch, 2), sequence = f(type, time, batch, 2);
                    Tensor lengths = f(DataType.INT64, batch);
                    // Even a closed external control association is never copied or inspected.
                    MemorySegmentStorage control;
                    try (Arena arena = Arena.ofConfined()) {
                        control = new MemorySegmentStorage(DataType.INT64, batch, arena.allocate(batch * 8, 8));
                        lengths.replaceHostStorage(control);
                    }
                    Tensor state = tensor(type, Shape.of(2), false, true, raw(type, 2));
                    var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x).add(w).tanh()), List.of(h));
                    var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(state), lengths, 1,
                            List.of(java.util.OptionalInt.of(0)), body, reverse);
                    var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
                    var source = new GraphCheckpoint.Source(List.of(lengths, sequence, initial), roots,
                            List.of(leaf("weight", GraphCheckpoint.Role.BUFFER, state), leaf("tied", GraphCheckpoint.Role.BUFFER, state)));
                    var copy = GraphCheckpoint.copy(source);
                    assertEquals(GraphSource.capture(source).graph, GraphSource.capture(copy).graph);
                    assertTrue(copy.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                    assertSame(copy.leaves().getFirst().tensor(), copy.leaves().getLast().tensor());
                    Tensor owned = copy.leaves().getFirst().tensor();
                    assertNotSame(state, owned);
                    assertArrayEquals(raw(type, 2), GraphValueStorage.encode(owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
                    var p = copy.outputs().getFirst().provenance().orElseThrow().producer();
                    assertSame(copy.inputs().getFirst(), p.inputs().getLast());
                    assertSame(owned, p.inputs().get(2));
                    for (int port = 0; port < p.outputCount(); port++) assertSame(p.output(port), copy.outputs().get(port));
                    var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                    assertEquals(type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2
                            ? "graph-model.v7" : "graph-model.v5", image.kind());
                    assertEquals(2, image.entries().size(), "only named weight bytes, never external lengths");
                    var loaded = GraphArtifactCodec.decode(image, LIMITS);
                    assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
                    assertTrue(loaded.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                    assertSame(loaded.inputs().getFirst(), loaded.outputs().getFirst().provenance().orElseThrow().inputs().getLast());
                    Tensor opaque = initial.abs().contiguous(), opaqueSequence = sequence.abs().contiguous();
                    Tensor freshLengths = f(DataType.INT64, batch);
                    var rebound = GraphCheckpoint.rebind(source, List.of(freshLengths, opaqueSequence, opaque));
                    var reboundProducer = rebound.getFirst().provenance().orElseThrow().producer();
                    assertSame(opaque, reboundProducer.inputs().getFirst());
                    assertSame(opaqueSequence, reboundProducer.inputs().get(1));
                    assertSame(state, reboundProducer.inputs().get(2));
                    assertSame(freshLengths, reboundProducer.inputs().getLast());
                    assertSame(control, lengths.hostStorage().orElseThrow()); assertFalse(control.isAlive());
                }
    }

    @Test void lowStateWithExplicitFloatWorkingSitesSurvivesSaveLoadCopyAndRebind(
            @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (boolean gradient : List.of(false, true)) for (boolean lengthAware : List.of(false, true))
                for (long batch : new long[]{0, 2}) for (long time : new long[]{0, 3})
                    for (boolean reverse : List.of(false, true)) {
                        Tensor h = prototype(type, Shape.of(batch, 2), gradient, true);
                        Tensor total = prototype(DataType.FLOAT32, Shape.of(batch, 2), gradient, true);
                        Tensor x = prototype(DataType.FLOAT32, Shape.of(batch, 2), gradient, true);
                        Tensor w = prototype(type, Shape.of(2, 2), gradient, true);
                        Tensor work = h.cast(DataType.FLOAT32).add(x.matmul(w.cast(DataType.FLOAT32)))
                                .add(total).tanh();
                        Tensor next = work.cast(type);
                        var body = new ScanBody(List.of(h, total), List.of(x), List.of(w),
                                List.of(next, work), List.of(next, work, next));
                        Tensor initial = TensorFactory.create(h.descriptor());
                        Tensor accumulator = TensorFactory.create(total.descriptor());
                        Tensor sequence = prototype(DataType.FLOAT32, Shape.of(time, batch, 2), gradient, true);
                        Tensor lengths = f(DataType.INT64, batch);
                        Tensor state = tensor(type, Shape.of(2, 2), gradient, true, raw(type, 4));
                        var scan = lengthAware
                                ? Scan.apply(List.of(initial, accumulator), List.of(sequence), List.of(state), lengths,
                                        2, List.of(java.util.OptionalInt.of(0)), body, reverse)
                                : Scan.apply(List.of(initial, accumulator), List.of(sequence), List.of(state), body, reverse);
                        var roots = new ArrayList<>(scan.finalCarry());
                        roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
                        var inputs = lengthAware ? List.of(sequence, accumulator, initial, lengths)
                                : List.of(sequence, accumulator, initial);
                        var role = gradient ? GraphCheckpoint.Role.PARAMETER : GraphCheckpoint.Role.BUFFER;
                        var source = new GraphCheckpoint.Source(inputs, roots,
                                List.of(leaf("weight", role, state), leaf("alias", role, state)));
                        var expected = GraphSource.capture(source).graph;
                        GraphCompilationPort.validateSource(expected);
                        var path = directory.resolve(type + "-" + gradient + "-" + lengthAware + "-"
                                + batch + "-" + time + "-" + reverse);
                        GraphCheckpoint.save(path, source, LIMITS);
                        for (var acquired : List.of(GraphCheckpoint.copy(source), GraphCheckpoint.load(path, LIMITS))) {
                            assertEquals(expected, GraphSource.capture(acquired).graph);
                            assertEquals(roots.stream().map(Tensor::descriptor).toList(),
                                    acquired.outputs().stream().map(Tensor::descriptor).toList());
                            Tensor owned = acquired.leaves().getFirst().tensor();
                            assertSame(owned, acquired.leaves().getLast().tensor());
                            assertNotSame(state, owned);
                            assertEquals(state.descriptor(), owned.descriptor());
                            assertArrayEquals(raw(type, 4), GraphValueStorage.encode(
                                    owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
                            assertTrue(acquired.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                            var p = acquired.outputs().getFirst().provenance().orElseThrow().producer();
                            for (int port = 0; port < p.outputCount(); port++)
                                assertSame(p.output(port), acquired.outputs().get(port));
                            assertSame(owned, p.inputs().get(3));
                            var fresh = p.scanBody().orElseThrow();
                            assertEquals(type, fresh.carry().getFirst().descriptor().dataType());
                            assertEquals(DataType.FLOAT32, fresh.carry().getLast().descriptor().dataType());
                            assertEquals(DataType.FLOAT32, fresh.step().getFirst().descriptor().dataType());
                            assertEquals(gradient, fresh.invariants().getFirst().descriptor().requiresGrad());
                            assertSame(fresh.nextCarry().getFirst(), fresh.emissions().getFirst());
                            assertSame(fresh.emissions().getFirst(), fresh.emissions().getLast());
                            var replacements = acquired.inputs().stream()
                                    .map(t -> TensorFactory.create(t.descriptor())).toList();
                            var rebound = GraphCheckpoint.rebind(acquired, replacements);
                            var rp = rebound.getFirst().provenance().orElseThrow().producer();
                            assertSame(replacements.get(2), rp.inputs().get(0));
                            assertSame(replacements.get(1), rp.inputs().get(1));
                            assertSame(replacements.get(0), rp.inputs().get(2));
                            assertSame(owned, rp.inputs().get(3));
                            if (lengthAware) assertSame(replacements.getLast(), rp.inputs().getLast());
                            assertEquals(expected, GraphSource.capture(new GraphCheckpoint.Source(
                                    replacements, rebound, acquired.leaves())).graph);
                        }
                        assertArrayEquals(raw(type, 4), GraphValueStorage.encode(
                                state.descriptor(), state.hostStorage().orElseThrow().segment()));
                    }
    }

    @Test void namedLengthsFollowOrdinaryNamedStateCopyAndPersistenceRules() throws Exception {
        Tensor h = f(DataType.FLOAT64, 2, 1), x = f(DataType.FLOAT64, 2, 1);
        Tensor initial = f(DataType.FLOAT64, 2, 1), sequence = f(DataType.FLOAT64, 0, 2, 1);
        // These invalid runtime values deliberately survive cold copying as raw named bytes.
        byte[] bytes = java.nio.ByteBuffer.allocate(16).putLong(-1).putLong(Long.MAX_VALUE).array();
        Tensor lengths = tensor(DataType.INT64, Shape.of(2), false, true, bytes);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of());
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                List.of(java.util.OptionalInt.of(0)), body, false);
        var source = new GraphCheckpoint.Source(List.of(initial, sequence), scan.finalCarry(),
                List.of(leaf("lengths", GraphCheckpoint.Role.BUFFER, lengths), leaf("alias", GraphCheckpoint.Role.BUFFER, lengths)));
        var copy = GraphCheckpoint.copy(source);
        Tensor owned = copy.leaves().getFirst().tensor();
        assertNotSame(lengths, owned); assertSame(owned, copy.leaves().getLast().tensor());
        assertArrayEquals(bytes, GraphValueStorage.encode(owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
        assertSame(owned, copy.outputs().getFirst().provenance().orElseThrow().inputs().getLast());
        var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        var loaded = GraphArtifactCodec.decode(image, LIMITS);
        Tensor restored = loaded.leaves().getFirst().tensor();
        assertNotSame(lengths, restored); assertSame(restored, loaded.leaves().getLast().tensor());
        assertArrayEquals(bytes, GraphValueStorage.encode(restored.descriptor(), restored.hostStorage().orElseThrow().segment()));
        assertSame(restored, loaded.outputs().getFirst().provenance().orElseThrow().inputs().getLast());
        assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
    }

    @Test void lengthRebindRejectsReplacementTiesToRetainedNamedStateBeforeAnyIds() {
        for (long batch : List.of(0L, 1L, 2L)) {
            Tensor h = f(DataType.FLOAT32, batch, 2), x = f(DataType.FLOAT32, batch, 2);
            Tensor w = f(DataType.FLOAT32, batch, 2), initial = f(DataType.FLOAT32, batch, 2);
            Tensor sequence = f(DataType.FLOAT32, 0, batch, 2), lengths = f(DataType.INT64, batch);
            Tensor retained = f(DataType.FLOAT32, batch, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x)), List.of());
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(retained), lengths, 1,
                    List.of(java.util.OptionalInt.of(0)), body, false);
            var source = new GraphCheckpoint.Source(List.of(initial, sequence, lengths),
                    List.of(initial.neg(), scan.finalCarry().getFirst()),
                    List.of(leaf("state", GraphCheckpoint.Role.BUFFER, retained)));
            Tensor before = f(DataType.BOOL);
            assertThrows(IllegalArgumentException.class,
                    () -> GraphCheckpoint.rebind(source, List.of(retained, sequence, lengths)));
            assertEquals(before.id().value() + 1, f(DataType.BOOL).id().value());
            assertTrue(retained.hostStorage().isEmpty());
        }
        // Newly repeated substitutions are valid when the outer roles agree, even with opaque DAGs.
        Tensor h = f(DataType.FLOAT64, 2, 2), h2 = f(DataType.FLOAT64, 2, 2);
        Tensor x = f(DataType.FLOAT64, 2, 2), x2 = f(DataType.FLOAT64, 2, 2);
        Tensor a = f(DataType.FLOAT64, 2, 2), b = f(DataType.FLOAT64, 2, 2);
        Tensor sa = f(DataType.FLOAT64, 3, 2, 2), sb = f(DataType.FLOAT64, 3, 2, 2), lengths = f(DataType.INT64, 2);
        var body = new ScanBody(List.of(h, h2), List.of(x, x2), List.of(), List.of(h.add(x), h2.add(x2)), List.of());
        var scan = Scan.apply(List.of(a, b), List.of(sa, sb), List.of(), lengths, 2,
                List.of(java.util.OptionalInt.of(0), java.util.OptionalInt.of(0)), body, false);
        var source = new GraphCheckpoint.Source(List.of(a, b, sa, sb, lengths), scan.finalCarry(), List.of());
        Tensor opaque = a.abs().contiguous(), opaqueSequence = sa.abs().contiguous();
        var rebound = GraphCheckpoint.rebind(source, List.of(opaque, opaque, opaqueSequence, opaqueSequence, lengths));
        var p = rebound.getFirst().provenance().orElseThrow().producer();
        assertSame(opaque, p.inputs().get(0)); assertSame(opaque, p.inputs().get(1));
        assertSame(opaqueSequence, p.inputs().get(2)); assertSame(opaqueSequence, p.inputs().get(3));
    }

    /** Independent outer IDs never rewrite local IDs, roles, unused formals or canonical ports. */
    @Test void copyAndRebindPreserveScopesAliasesOccurrencesAndEveryCanonicalPort() throws Exception {
        for (long time : List.of(0L, 3L, 1_000_000L)) {
            Tensor h = f(DataType.FLOAT32, 2), h2 = f(DataType.FLOAT32, 2);
            Tensor x = f(DataType.FLOAT32, 2), x2 = f(DataType.FLOAT32, 2);
            Tensor w = f(DataType.FLOAT32, 2), unused = f(DataType.FLOAT32, 2);
            Tensor next = h.add(x).add(w).tanh();
            var template = new ScanBody(List.of(h, h2), List.of(x, x2), List.of(w, unused),
                    List.of(next, next), List.of(next, h, next));
            Tensor initial = f(DataType.FLOAT32, 2), sequence = f(DataType.FLOAT32, time, 2);
            Tensor unusedInput = f(DataType.FLOAT32, 2);
            Tensor state = tensor(DataType.FLOAT32, Shape.of(2), false, true, raw(DataType.FLOAT32, 2));
            var a = Scan.apply(List.of(initial, initial), List.of(sequence, sequence), List.of(state, initial), template, false);
            var b = Scan.apply(List.of(initial, initial), List.of(sequence, sequence), List.of(state, initial), template, true);
            var roots = new ArrayList<Tensor>();
            roots.add(a.residuals().getLast().neg());
            roots.addAll(a.finalCarry()); roots.addAll(a.outputs()); roots.addAll(a.residuals());
            roots.addAll(b.finalCarry()); roots.addAll(b.outputs()); roots.addAll(b.residuals());
            roots.add(initial); roots.add(state);
            var source = new GraphCheckpoint.Source(List.of(sequence, unusedInput, initial), roots,
                    List.of(leaf("state", GraphCheckpoint.Role.BUFFER, state), leaf("alias", GraphCheckpoint.Role.BUFFER, state)));
            var normalized = GraphSource.capture(source).graph;
            var raw = GraphCompilationPort.captureSource(roots).graph();
            var rawScans = raw.nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
            var scans = normalized.nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
            assertEquals(2, scans.size());
            for (int i = 0; i < scans.size(); i++) {
                var local = scans.get(i).scanBody().orElseThrow();
                assertEquals(rawScans.get(i).scanBody().orElseThrow(), local);
                assertEquals(new ValueId(0), local.graph().inputs().getFirst());
                assertEquals(new ValueId(2), scans.get(i).inputs().getFirst());
                assertEquals(new ValueId(0), scans.get(i).inputs().get(2));
                assertEquals(new ValueId(3), scans.get(i).inputs().get(4));
                assertEquals(6, local.graph().inputs().size());
                assertTrue(local.demandedOutputIndices().isEmpty());
            }
            var originalStore = state.hostStorage().orElseThrow();
            var copy = GraphCheckpoint.copy(source);
            var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
            var loaded = GraphArtifactCodec.decode(image, LIMITS);
            assertEquals("graph-model.v4", image.kind());
            assertEquals(normalized, GraphSource.capture(loaded).graph);
            assertTrue(loaded.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
            assertSame(loaded.leaves().getFirst().tensor(), loaded.leaves().getLast().tensor());
            assertNotSame(state, loaded.leaves().getFirst().tensor());
            assertEquals(normalized, GraphSource.capture(copy).graph);
            assertEquals(source.inputs().stream().map(Tensor::descriptor).toList(), copy.inputs().stream().map(Tensor::descriptor).toList());
            assertTrue(copy.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
            Tensor owned = copy.leaves().getFirst().tensor();
            assertNotSame(state, owned); assertSame(owned, copy.leaves().getLast().tensor());
            assertSame(originalStore, state.hostStorage().orElseThrow());
            assertArrayEquals(raw(DataType.FLOAT32, 2), GraphValueStorage.encode(owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
            var pa = copy.outputs().get(1).provenance().orElseThrow().producer();
            int count = pa.outputCount();
            var pb = copy.outputs().get(count + 1).provenance().orElseThrow().producer();
            assertNotSame(pa, pb);
            assertSame(copy.outputs().get(count), copy.outputs().getFirst().provenance().orElseThrow().inputs().getFirst());
            for (int port = 0; port < count; port++) {
                assertSame(pa.output(port), copy.outputs().get(port + 1));
                assertSame(pb.output(port), copy.outputs().get(count + port + 1));
            }
            assertFalse(((ScanAttrs) pa.operation().attrs()).reverse());
            assertTrue(((ScanAttrs) pb.operation().attrs()).reverse());
            assertSame(pa.inputs().get(0), pa.inputs().get(1));
            assertSame(pa.inputs().get(2), pa.inputs().get(3));
            assertSame(pa.inputs().get(0), pa.inputs().get(5));
            assertSame(owned, pa.inputs().get(4));
            var ba = pa.scanBody().orElseThrow(); var bb = pb.scanBody().orElseThrow();
            assertNotSame(template, ba); assertNotSame(ba, bb);
            assertSame(ba.nextCarry().getFirst(), ba.nextCarry().getLast());
            assertSame(ba.nextCarry().getFirst(), ba.emissions().getFirst());
            assertSame(ba.emissions().getFirst(), ba.emissions().getLast());
            var fresh = formals(ba);
            assertEquals(6, fresh.stream().distinct().count());
            for (Tensor formal : fresh) {
                assertFalse(formals(template).contains(formal)); assertFalse(formals(bb).contains(formal));
                assertFalse(pa.inputs().contains(formal));
                assertTrue(formal.provenance().isEmpty()); assertTrue(formal.hostStorage().isEmpty());
            }
            // Repeated produced substitutions have cold-unsupported opaque upstream provenance.
            Tensor replacement = initial.abs().contiguous();
            Tensor replacementSequence = sequence.abs().contiguous();
            var replacements = List.of(replacementSequence, replacement, replacement);
            Tensor before = f(DataType.BOOL);
            var rebound = GraphCheckpoint.rebind(source, replacements);
            Tensor after = f(DataType.BOOL);
            long expected = normalized.nodes().size(); // one outer head plus general nodes below
            for (var node : scans) {
                var local = node.scanBody().orElseThrow().graph();
                expected += local.inputs().size() + local.nodes().stream().mapToLong(n -> n.outputs().size()).sum()
                        + node.outputs().size() - 1;
            }
            assertEquals(expected + 1, after.id().value() - before.id().value(), "O(body), not O(time), reconstruction");
            var reboundProducer = rebound.get(1).provenance().orElseThrow().producer();
            assertSame(replacement, reboundProducer.inputs().getFirst());
            assertSame(replacementSequence, reboundProducer.inputs().get(2));
            assertSame(state, reboundProducer.inputs().get(4));
            assertSame(replacement, rebound.get(rebound.size() - 2));
            assertSame(state, rebound.getLast());
            assertThrows(UnsupportedOperationException.class, rebound::clear);
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, List.of(sequence)));
            assertThrows(NullPointerException.class, () -> GraphCheckpoint.rebind(source, null));
            assertThrows(NullPointerException.class, () -> GraphCheckpoint.rebind(source, java.util.Arrays.asList(sequence, null, initial)));
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source,
                    List.of(sequence, unusedInput, prototype(DataType.FLOAT32, Shape.of(2), true, true))));
            assertEquals(normalized, GraphSource.capture(source).graph, "template remains independently reusable");
        }
    }

    /** Every applicable dtype is cold metadata; backend execution qualification is independent. */
    @Test void applicableModelDtypesAndExactBodyOnlySignaturesRoundTripWithoutBackendAdmission() throws Exception {
        for (DataType type : DataType.values()) {
            Tensor h = f(type, 2, 3), x = f(type, 2, 3), w = f(type, 2, 3);
            Tensor empty = f(type, 0, 3), scalar = f(type);
            var emissions = new ArrayList<>(List.of(h.contiguous(), h.cast(type), h.reshape(Shape.of(6)),
                    h.expandDims(0).squeeze(0), h.permute(1, 0), scalar.expand(Shape.of(2, 3)),
                    empty.reshape(Shape.of(3, 0))));
            if (type != DataType.BOOL) {
                ScalarValue value = scalar(type);
                emissions.addAll(List.of(h.add(w), h.sub(w), h.mul(w), h.add(value), h.sub(value), h.mul(value),
                        h.sum(), h.sum(0, false), h.sum(new int[]{1, 0}, true), h.sum(new int[]{}, false),
                        h.sumToShape(Shape.of(1, 3)), empty.sum(0)));
                if (type.isFloating()) emissions.addAll(List.of(h.div(w), h.div(value), h.neg(), h.relu(), h.tanh(), h.sigmoid()));
            }
            var body = new ScanBody(List.of(h), List.of(x), List.of(w, empty, scalar), List.of(h), emissions);
            for (boolean reverse : List.of(false, true)) for (long time : List.of(0L, 3L)) {
                Tensor initial = f(type, 2, 3), sequence = f(type, time, 2, 3);
                var states = List.of(tensor(type, Shape.of(2, 3), false, true, raw(type, 6)),
                        tensor(type, Shape.of(0, 3), false, true, raw(type, 0)),
                        tensor(type, Shape.scalar(), false, true, raw(type, 1)));
                var scan = Scan.apply(List.of(initial), List.of(sequence), states, body, reverse);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
                var leaves = List.of(leaf("buffer", GraphCheckpoint.Role.BUFFER, states.get(0)),
                        leaf("empty", GraphCheckpoint.Role.CONSTANT, states.get(1)),
                        leaf("scalar", GraphCheckpoint.Role.CONSTANT, states.get(2)));
                var source = new GraphCheckpoint.Source(List.of(sequence, initial), roots, leaves);
                var copy = GraphCheckpoint.copy(source);
                var loaded = GraphArtifactCodec.decode(GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS), LIMITS);
                assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
                for (int i = 0; i < states.size(); i++) assertArrayEquals(
                        GraphValueStorage.encode(states.get(i).descriptor(), states.get(i).hostStorage().orElseThrow().segment()),
                        GraphValueStorage.encode(loaded.leaves().get(i).tensor().descriptor(),
                                loaded.leaves().get(i).tensor().hostStorage().orElseThrow().segment()));
                assertEquals(GraphSource.capture(source).graph, GraphSource.capture(copy).graph);
                assertEquals(roots.stream().map(Tensor::descriptor).toList(), copy.outputs().stream().map(Tensor::descriptor).toList());
                for (int i = 0; i < states.size(); i++) {
                    Tensor owned = copy.leaves().get(i).tensor();
                    assertEquals(states.get(i).descriptor(), owned.descriptor());
                    assertArrayEquals(GraphValueStorage.encode(states.get(i).descriptor(), states.get(i).hostStorage().orElseThrow().segment()),
                            GraphValueStorage.encode(owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
                }
            }
        }
    }

    /** Acquisition detaches all roles and raw bits from overlapping native storage exactly once. */
    @Test void copiedStateOutlivesNativeOwnerPreservingRolesAliasesAndMixedGradientFlags() throws Exception {
        GraphCheckpoint.Source copy;
        io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope saved;
        try (Arena arena = Arena.ofConfined()) {
            var segment = arena.allocate(8, 4);
            segment.copyFrom(MemorySegment.ofArray(new int[]{0xffc00123, 0x80000000}));
            Tensor parameter = prototype(DataType.FLOAT32, Shape.of(2), true, true);
            Tensor buffer = prototype(DataType.FLOAT32, Shape.of(2), true, true);
            Tensor constant = prototype(DataType.FLOAT32, Shape.of(2), false, true);
            for (Tensor state : List.of(parameter, buffer, constant))
                state.replaceHostStorage(new MemorySegmentStorage(DataType.FLOAT32, 2, segment));
            Tensor h = f(DataType.FLOAT32, 2), x = prototype(DataType.FLOAT32, Shape.of(2), true, true);
            Tensor p = prototype(DataType.FLOAT32, Shape.of(2), true, true);
            Tensor b = prototype(DataType.FLOAT32, Shape.of(2), true, true), c = f(DataType.FLOAT32, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(p, b, c), List.of(h.add(x).add(p)), List.of(h, b, c));
            Tensor initial = f(DataType.FLOAT32, 2), sequence = prototype(DataType.FLOAT32, Shape.of(0, 2), true, true);
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(parameter, buffer, constant), body, false);
            var source = new GraphCheckpoint.Source(List.of(sequence, initial), scan.finalCarry(), List.of(
                    leaf("p", GraphCheckpoint.Role.PARAMETER, parameter), leaf("tied", GraphCheckpoint.Role.PARAMETER, parameter),
                    leaf("b", GraphCheckpoint.Role.BUFFER, buffer), leaf("c", GraphCheckpoint.Role.CONSTANT, constant)));
            copy = GraphCheckpoint.copy(source);
            saved = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
            assertSame(segment, parameter.hostStorage().orElseThrow().segment());
            assertEquals(source.leaves().stream().map(GraphCheckpoint.Leaf::role).toList(), copy.leaves().stream().map(GraphCheckpoint.Leaf::role).toList());
            assertSame(copy.leaves().get(0).tensor(), copy.leaves().get(1).tensor());
            assertNotSame(copy.leaves().get(0).tensor(), copy.leaves().get(2).tensor());
            assertNotSame(copy.leaves().get(2).tensor().hostStorage().orElseThrow(), copy.leaves().get(3).tensor().hostStorage().orElseThrow());
            var rebuiltBody = copy.outputs().getFirst().provenance().orElseThrow().producer().scanBody().orElseThrow();
            assertFalse(rebuiltBody.carry().getFirst().descriptor().requiresGrad());
            assertTrue(rebuiltBody.step().getFirst().descriptor().requiresGrad());
            assertTrue(copy.outputs().getFirst().descriptor().requiresGrad());
        }
        for (var leaf : copy.leaves()) {
            var store = leaf.tensor().hostStorage().orElseThrow();
            assertTrue(store.isAlive()); assertFalse(store.segment().isNative()); assertFalse(store.isReadOnly());
            assertArrayEquals(raw(DataType.FLOAT32, 2), GraphValueStorage.encode(leaf.tensor().descriptor(), store.segment()));
        }
        assertDoesNotThrow(() -> GraphCheckpoint.rebind(copy, copy.inputs()));
        var loaded = GraphArtifactCodec.decode(saved, LIMITS);
        assertEquals(GraphSource.capture(copy).graph, GraphSource.capture(loaded).graph);
        assertSame(loaded.leaves().get(0).tensor(), loaded.leaves().get(1).tensor());
        for (var leaf : loaded.leaves()) {
            var store = leaf.tensor().hostStorage().orElseThrow();
            assertFalse(store.segment().isNative()); assertFalse(store.isReadOnly());
            assertArrayEquals(raw(DataType.FLOAT32, 2), GraphValueStorage.encode(leaf.tensor().descriptor(), store.segment()));
        }
    }

    /** Unsupported exact body signatures reject even behind an unused emission at zero time. */
    @Test void unsupportedPureBodiesFailBeforeCopyAllocationOrReplacementAdmission() {
        Tensor h = f(DataType.FLOAT32, 2, 2), x = f(DataType.FLOAT32, 2, 2);
        for (Tensor unsupported : List.of(h.abs(), h.mean(), h.cast(DataType.INT32), h.cast(DataType.BOOL),
                h.permute(0, 1), h.pow(x))) {
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of(unsupported));
            Tensor initial = f(DataType.FLOAT32, 2, 2), sequence = f(DataType.FLOAT32, 0, 2, 2);
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
            var source = new GraphCheckpoint.Source(List.of(sequence, initial), List.of(initial.neg(), scan.finalCarry().getFirst()), List.of());
            Tensor before = f(DataType.BOOL);
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source));
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, null));
            Tensor after = f(DataType.BOOL);
            assertEquals(1, after.id().value() - before.id().value());
        }
    }

    /**
     * Creates exact canonical storage-free non-gradient metadata of any Model dtype.
     * @param type logical dtype, independently of execution capability
     * @param extents static nonnegative dimensions, empty for scalar
     * @return fresh canonical provenance-free leaf
     */
    private static Tensor f(DataType type, long... extents) { return prototype(type, Shape.of(extents), false, true); }

    /**
     * Lists formals in the public role order without a reconstruction map.
     * @param body immutable source template
     * @return complete ordered declarations, including unused ones
     */
    private static List<Tensor> formals(ScanBody body) {
        var result = new ArrayList<>(body.carry()); result.addAll(body.step()); result.addAll(body.invariants()); return result;
    }

    /**
     * Supplies a represented two in each numeric dtype, without cross-type scalar promotion.
     * @param type numeric Model dtype, never BOOL
     * @return exact typed scalar attribute
     */
    private static ScalarValue scalar(DataType type) {
        return switch (type) {
            case FLOAT64 -> ScalarValue.float64(2); case FLOAT32 -> ScalarValue.float32(2);
            case BFLOAT16 -> ScalarValue.bfloat16(2); case FLOAT16 -> ScalarValue.float16(2);
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3(2); case FLOAT8_E5M2 -> ScalarValue.float8E5m2(2);
            case INT32 -> ScalarValue.int32(2); case INT64 -> ScalarValue.int64(2);
            default -> throw new AssertionError(type);
        };
    }
}
