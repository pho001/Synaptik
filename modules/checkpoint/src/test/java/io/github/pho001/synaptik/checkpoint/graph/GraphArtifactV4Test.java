package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Independently authored v4 wire, scoped corruption and preflight/allocation boundary checks. */
final class GraphArtifactV4Test {
    @TempDir Path directory;

    @Test void carryFreeV4HeaderRejectsBeforeAnyBodyFieldsOrCandidates() throws Exception {
        for (long time : new long[] {0, 1}) {
            var f = fixture(0, new byte[0], true, time);
            byte[] header = Arrays.copyOf(f.bytes(), f.offsets().get("reverse") + 1);
            ByteBuffer.wrap(header).putInt(f.offsets().get("carryCount"), 0);
            Tensor before = f();
            var failure = assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(header), LIMITS));
            assertInstanceOf(IllegalArgumentException.class, failure.getCause());
            assertEquals(before.id().value() + 1, f().id().value());
        }
    }

    @Test
    void independentTypedLengthEncodingPreservesFloatingDeclarationsAndRetainsDiscreteAndSymbolicRejections() throws Exception {
        for (var batch : List.of(new io.github.pho001.synaptik.model.shape.StaticDimension(0),
                new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers")))
            for (long time : new long[] {0, 3}) for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16,
                    DataType.FLOAT64, DataType.INT32, DataType.INT64, DataType.BOOL)) for (int role = -1; role < 3; role++) {
                var path = directory.resolve("typed-length-" + batch.isStatic() + "-" + type + "-" + time + "-" + role);
                var raw = typedLengthCodecFixture(type, batch, time, role);
                var source = raw.source;
                if (type.isFloating() && batch.isStatic()) {
                    var expected = GraphSource.capture(source).graph;
                    assertEquals(expected, raw.graph);
                    var encoded = GraphArtifactCodec.encode(raw, LIMITS);
                    assertEquals("graph-model.v5", encoded.kind());
                    var decoded = GraphArtifactCodec.decode(encoded, LIMITS);
                    var copy = GraphCheckpoint.copy(source);
                    assertEquals(expected, GraphSource.capture(decoded).graph);
                    assertEquals(expected, GraphSource.capture(copy).graph);
                    var rebound = GraphCheckpoint.rebind(source, source.inputs());
                    assertEquals(expected, GraphSource.capture(new GraphCheckpoint.Source(
                            source.inputs(), rebound, source.leaves())).graph);
                    GraphCheckpoint.save(path, source, LIMITS);
                    assertEquals(expected, GraphSource.capture(GraphCheckpoint.load(path, LIMITS)).graph);
                    assertTrue(decoded.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                    continue;
                }
                Tensor probe = f();
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source));
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, source.inputs()));
                assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(raw, LIMITS));
                assertThrows(IllegalArgumentException.class,
                        () -> GraphCheckpoint.save(path, source, LIMITS));
                assertFalse(Files.exists(path));
                assertEquals(probe.id().value() + 1, f().id().value(), "cold rejection allocates no Tensors");
                assertTrue(source.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
            }
    }

    @Test void broaderSharedCoordinatesCannotBypassRawLengthEncoderColdPreflight() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : new long[] {0, 1, 3}) for (long time : new long[] {0, 2}) {
                Tensor h = prototype(type, Shape.of(batch, 2), false, true);
                Tensor old = prototype(type, Shape.of(2), false, true);
                Tensor x = prototype(type, Shape.of(batch, 2), false, true);
                Tensor w = prototype(type, Shape.of(2), false, true);
                Tensor relu = w.relu(), next = old.add(relu);
                var body = new ScanBody(List.of(h, old), List.of(x), List.of(w), List.of(h, next), List.of());
                var actuals = List.of(prototype(type, Shape.of(batch, 2), false, true),
                        prototype(type, Shape.of(2), false, true), prototype(type, Shape.of(time, batch, 2), false, true),
                        prototype(type, Shape.of(2), false, true), prototype(DataType.INT64, Shape.of(batch), false, true));
                var result = Scan.apply(actuals.subList(0, 2), List.of(actuals.get(2)), List.of(actuals.get(3)),
                        actuals.get(4), 1, List.of(OptionalInt.of(0)), body, false);
                // Independently author the two ordinary occurrences and canonical retained sites.
                var localValues = List.of(new GraphValue(new ValueId(0), h.descriptor()),
                        new GraphValue(new ValueId(1), old.descriptor()), new GraphValue(new ValueId(2), x.descriptor()),
                        new GraphValue(new ValueId(3), w.descriptor()), new GraphValue(new ValueId(4), relu.descriptor()),
                        new GraphValue(new ValueId(5), next.descriptor()));
                var reluNode = new CompiledNode(new NodeId(0), relu.provenance().orElseThrow().producer().operation(),
                        List.of(new ValueId(3)), List.of(new ValueId(4)));
                var addNode = new CompiledNode(new NodeId(1), next.provenance().orElseThrow().producer().operation(),
                        List.of(new ValueId(1), new ValueId(4)), List.of(new ValueId(5)));
                var locals = new CompiledGraphModel(localValues, List.of(reluNode, addNode),
                        List.of(new ValueId(0), new ValueId(1), new ValueId(2), new ValueId(3)),
                        List.of(new ValueId(0), new ValueId(5), new ValueId(1), new ValueId(4)),
                        Map.of(new NodeId(0), GraphPhase.FORWARD, new NodeId(1), GraphPhase.FORWARD));
                var compiledBody = new CompiledScanBody(locals, List.of(new ValueId(0), new ValueId(5)), List.of(),
                        List.of(new ValueId(0), new ValueId(1), new ValueId(4), new ValueId(5)), List.of());
                var producer = result.finalCarry().getFirst().provenance().orElseThrow().producer();
                var outer = new ArrayList<GraphValue>();
                for (Tensor actual : actuals) outer.add(new GraphValue(new ValueId(outer.size()), actual.descriptor()));
                var inputIds = outer.stream().map(GraphValue::id).toList();
                var outputIds = new ArrayList<ValueId>();
                for (var descriptor : producer.outputDescriptors()) {
                    var id = new ValueId(outer.size()); outputIds.add(id); outer.add(new GraphValue(id, descriptor));
                }
                var node = new CompiledNode(new NodeId(0), producer.operation(), inputIds, outputIds, Optional.of(compiledBody));
                var graph = new CompiledGraphModel(outer, List.of(node), inputIds, outputIds.subList(0, 2),
                        Map.of(new NodeId(0), GraphPhase.FORWARD));
                var source = new GraphCheckpoint.Source(actuals, result.finalCarry(), List.of());
                var constructor = GraphSource.class.getDeclaredConstructor(GraphCheckpoint.Source.class,
                        CompiledGraphModel.class, List.class, List.class, List.class);
                constructor.setAccessible(true);
                var raw = constructor.newInstance(source, graph, List.of(), List.of(), List.of());
                Tensor marker = f();
                assertThrows(IllegalArgumentException.class, () -> GraphArtifactCodec.encode(raw, LIMITS));
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source));
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, source.inputs()));
                assertEquals(marker.id().value() + 1, f().id().value());
            }
    }

    /**
     * Authors Model-valid typed source and raw graph metadata without Compiler cold capture.
     * @param type cast-interior or unused declaration dtype
     * @param batch exact static or named row Dimension
     * @param time fixed T
     * @param role -1 for cast interior, otherwise unused carry/step/invariant (0/1/2)
     * @return raw association at the encoder seam, independent of Compiler cold capture
     * @throws Exception if reflective construction of the raw association fails
     */
    private static GraphSource typedLengthCodecFixture(DataType type, io.github.pho001.synaptik.model.shape.Dimension batch,
            long time, int role) throws Exception {
        Shape row = Shape.ofDimensions(batch, new io.github.pho001.synaptik.model.shape.StaticDimension(2));
        Tensor h = prototype(DataType.FLOAT32, row, false, row.isFullyStatic());
        Tensor x = prototype(DataType.FLOAT32, row, false, row.isFullyStatic());
        Shape unusedShape = role == 2 ? Shape.of(2) : row;
        Tensor unused = prototype(type, unusedShape, false, unusedShape.isFullyStatic());
        var carry = role == 0 ? List.of(h, unused) : List.of(h);
        var step = role == 1 ? List.of(x, unused) : List.of(x);
        var inv = role == 2 ? List.of(unused) : List.<Tensor>of();
        Tensor next = role < 0 ? h.cast(type).cast(DataType.FLOAT32) : h;
        var body = new ScanBody(carry, step, inv, role == 0 ? List.of(next, unused) : List.of(next), List.of());
        Shape sequence = Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(time), batch,
                new io.github.pho001.synaptik.model.shape.StaticDimension(2));
        var actualCarry = carry.stream().map(t -> prototype(t.descriptor().dataType(), row, false, row.isFullyStatic())).toList();
        var actualStep = step.stream().map(t -> prototype(t.descriptor().dataType(), sequence, false, sequence.isFullyStatic())).toList();
        var actualInv = inv.stream().map(t -> prototype(t.descriptor().dataType(), t.descriptor().shape(), false, true)).toList();
        Shape lengthShape = Shape.ofDimensions(batch);
        var scan = Scan.apply(actualCarry, actualStep, actualInv, prototype(DataType.INT64, lengthShape, false, lengthShape.isFullyStatic()),
                carry.size(), Collections.nCopies(step.size(), OptionalInt.of(0)), body, false);
        var p = scan.finalCarry().getFirst().provenance().orElseThrow().producer();
        var locals = new ArrayList<GraphValue>(); var localIds = new IdentityHashMap<Tensor, ValueId>();
        var formals = new ArrayList<Tensor>(carry); formals.addAll(step); formals.addAll(inv);
        for (Tensor t : formals) { var id = new ValueId(locals.size()); localIds.put(t, id); locals.add(new GraphValue(id, t.descriptor())); }
        var localNodes = new ArrayList<CompiledNode>();
        for (Tensor t : body.retainedValues()) if (!localIds.containsKey(t)) {
            var cast = t.provenance().orElseThrow().producer();
            assertEquals(io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST, cast.operation().kind());
            var id = new ValueId(locals.size()); localIds.put(t, id); locals.add(new GraphValue(id, t.descriptor()));
            localNodes.add(new CompiledNode(new NodeId(localNodes.size()), cast.operation(),
                    List.of(localIds.get(cast.inputs().getFirst())), List.of(id)));
        }
        var nextIds = body.nextCarry().stream().map(localIds::get).toList();
        var residuals = body.retainedValues().stream().map(localIds::get).toList();
        var roots = new LinkedHashSet<>(nextIds); roots.addAll(residuals);
        var local = new CompiledGraphModel(locals, localNodes, formals.stream().map(localIds::get).toList(), List.copyOf(roots),
                localNodes.stream().collect(java.util.stream.Collectors.toMap(CompiledNode::id, n -> GraphPhase.FORWARD)));
        var compiledBody = new CompiledScanBody(local, nextIds, List.of(), residuals, List.of());
        var outer = new ArrayList<GraphValue>();
        for (Tensor t : p.inputs()) outer.add(new GraphValue(new ValueId(outer.size()), t.descriptor()));
        var inputs = outer.stream().map(GraphValue::id).toList(); var outputs = new ArrayList<ValueId>();
        for (var d : p.outputDescriptors()) { var id = new ValueId(outer.size()); outputs.add(id); outer.add(new GraphValue(id, d)); }
        var node = new CompiledNode(new NodeId(0), p.operation(), inputs, outputs, Optional.of(compiledBody));
        var graph = new CompiledGraphModel(outer, List.of(node), inputs, outputs.subList(0, carry.size()), Map.of(node.id(), GraphPhase.FORWARD));
        var source = new GraphCheckpoint.Source(p.inputs(), scan.finalCarry(), List.of());
        var constructor = GraphSource.class.getDeclaredConstructor(GraphCheckpoint.Source.class, CompiledGraphModel.class, List.class, List.class, List.class);
        constructor.setAccessible(true);
        return constructor.newInstance(source, graph, List.of(), List.of(), List.of());
    }

    @Test
    void lengthSaveAndDirectEncoderAdmitWireBeforeRejectingMissingStateWithoutTargetEffects() throws Exception {
        for (long time : List.of(0L, 3L)) {
            Tensor initial = f(2, 2), sequence = f(time, 2, 2), h = f(2, 2), x = f(2, 2);
            Tensor lengths = prototype(DataType.INT64, Shape.of(2), false, true);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of());
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                    List.of(OptionalInt.of(0)), body, false);
            Tensor early = initial.neg();
            // V5 admits the cold source but still requires named state storage before publication.
            var source = new GraphCheckpoint.Source(List.of(sequence, lengths), List.of(early, scan.finalCarry().getFirst()),
                    List.of(new GraphCheckpoint.Leaf("state", GraphCheckpoint.Role.BUFFER, initial)));
            Path target = directory.resolve("length-" + time);
            Files.write(target, new byte[] {4, 3, 2, 1});
            assertDoesNotThrow(() -> GraphCheckpoint.rebind(source, source.inputs()));
            assertThrows(IllegalStateException.class, () -> GraphCheckpoint.copy(source));
            var admitted = GraphSource.capture(source);
            Tensor probe = f();
            assertThrows(IllegalStateException.class, () -> GraphArtifactCodec.encode(admitted, LIMITS));
            assertThrows(IllegalStateException.class, () -> GraphCheckpoint.save(target, source, LIMITS));
            assertEquals(probe.id().value() + 1, f().id().value());
            assertArrayEquals(new byte[] {4, 3, 2, 1}, Files.readAllBytes(target));
            Path absent = directory.resolve("absent-length-" + time);
            assertThrows(IllegalStateException.class, () -> GraphCheckpoint.save(absent, source, LIMITS));
            assertFalse(Files.exists(absent));
            assertTrue(initial.hostStorage().isEmpty());
        }
    }

    /**
     * Test-owned bytes and named field offsets; no production encoder authors them.
     * @param bytes independently authored mutable fixture bytes
     * @param offsets test-owned named field offsets for targeted corruption
     */
    record Fixture(byte[] bytes, Map<String, Integer> offsets) { }

    /**
     * Writes canonical non-gradient F32 metadata independently of production descriptor encoding.
     * @param out test-owned stream
     * @param canonical whether a contiguous layout is present
     * @param extents static dimensions
     * @throws Exception if fixture writing fails
     */
    private static void descriptor(DataOutputStream out, boolean canonical, long... extents) throws Exception {
        out.writeByte(2); out.writeByte(0); out.writeInt(extents.length);
        for (long n : extents) out.writeLong(n);
        out.writeByte(canonical ? 1 : 0);
        if (canonical) {
            out.writeLong(0); out.writeByte(0);
            long stride = 1; long[] strides = new long[extents.length];
            for (int i = extents.length - 1; i >= 0; i--) { strides[i] = stride; stride *= extents[i]; }
            for (long n : strides) out.writeLong(n);
        }
    }

    /**
     * Authors a scalar-carry scan with one scalar emission and all latent residuals.
     * @param op zero for a body with no producer, otherwise a one-output scalar body opcode
     * @param attrs exact independent attribute bytes
     * @param late whether a valid outer NEG precedes the scan
     * @param time nonnegative static time count
     * @return independently authored wire and corruption offsets
     * @throws Exception if fixture writing fails
     */
    static Fixture fixture(int op, byte[] attrs, boolean late, long time) throws Exception {
        return fixture(op, attrs, late, time, false);
    }

    /**
     * Authors the same scan with optional named carry state for host-import probes.
     * @param op local producer opcode, or zero
     * @param attrs independently authored attribute bytes
     * @param late whether a valid outer producer precedes the scan
     * @param time static step count
     * @param namedState whether the carry is stored state rather than a declared input
     * @return independent bytes and exact field offsets
     * @throws Exception if fixture writing fails
     */
    static Fixture fixture(int op, byte[] attrs, boolean late, long time, boolean namedState) throws Exception {
        var bytes = new ByteArrayOutputStream(); var out = new DataOutputStream(bytes);
        var at = new HashMap<String, Integer>();
        out.writeBytes("SYNGRF04"); out.writeInt(4);
        if (namedState) {
            out.writeInt(1); descriptor(out, true, time);
            out.writeInt(1); descriptor(out, true); out.writeByte(2); out.writeLong(4);
            out.writeInt(1); out.writeInt(5); out.writeBytes("state"); out.writeInt(0);
        } else {
            out.writeInt(2); descriptor(out, true); descriptor(out, true, time);
            out.writeInt(0); out.writeInt(0);
        }
        out.writeInt(late ? 2 : 1);
        if (late) {
            out.writeByte(1); out.writeByte(3); out.writeInt(namedState ? 1 : 0); out.writeInt(1); descriptor(out, false);
        }
        at.put("phase", bytes.size()); out.writeByte(1);
        at.put("opcode", bytes.size()); out.writeByte(23);
        at.put("actualCount", bytes.size()); out.writeInt(2);
        at.put("actual", bytes.size()); out.writeInt(namedState ? 1 : 0); out.writeInt(namedState ? 0 : 1);
        at.put("time", bytes.size()); out.writeLong(time);
        at.put("carryCount", bytes.size()); out.writeInt(1);
        at.put("sequenceCount", bytes.size()); out.writeInt(1);
        at.put("invariantCount", bytes.size()); out.writeInt(0);
        at.put("emissionCount", bytes.size()); out.writeInt(1);
        at.put("residualCount", bytes.size()); out.writeInt(op == 0 ? 1 : 2);
        at.put("reverse", bytes.size()); out.writeByte(0);
        at.put("formalCount", bytes.size()); out.writeInt(2);
        at.put("formal", bytes.size()); descriptor(out, true); descriptor(out, true);
        at.put("nodeCount", bytes.size()); out.writeInt(op == 0 ? 0 : 1);
        if (op != 0) {
            at.put("bodyPhase", bytes.size()); out.writeByte(1);
            at.put("bodyOpcode", bytes.size()); out.writeByte(op);
            at.put("bodyInput", bytes.size()); out.writeInt(0);
            if (op == 1 || op == 9 || op == 24) out.writeInt(1);
            at.put("attrs", bytes.size()); out.write(attrs);
            at.put("bodyQ", bytes.size()); out.writeInt(1);
            at.put("bodyResult", bytes.size());
            if (op == 30) {
                out.writeByte(2); out.writeByte(0); out.writeInt(0); out.writeByte(1);
                out.writeLong(0); out.writeByte(1);
            } else descriptor(out, op == 31);
        }
        at.put("rootCount", bytes.size()); out.writeInt(op == 0 ? 1 : 2);
        at.put("root", bytes.size()); out.writeInt(0);
        if (op != 0) out.writeInt(2);
        at.put("nextCount", bytes.size()); out.writeInt(1);
        at.put("next", bytes.size()); out.writeInt(0);
        at.put("emitCount", bytes.size()); out.writeInt(1);
        at.put("emit", bytes.size()); out.writeInt(op == 0 ? 0 : 2);
        at.put("residualListCount", bytes.size()); out.writeInt(op == 0 ? 1 : 2);
        at.put("residual", bytes.size()); out.writeInt(0);
        if (op != 0) out.writeInt(2);
        at.put("demand", bytes.size()); out.writeInt(0);
        at.put("q", bytes.size()); out.writeInt(op == 0 ? 3 : 4);
        at.put("outerResult", bytes.size()); descriptor(out, true);
        descriptor(out, true, time); descriptor(out, true, time);
        if (op != 0) descriptor(out, true, time);
        out.writeInt(late ? 2 : 1);
        if (late) out.writeInt(2);
        out.writeInt(late ? 3 : 2);
        return new Fixture(bytes.toByteArray(), at);
    }

    /**
     * Wraps independent bytes in an exact v4 tuple.
     * @param bytes schema fixture
     * @return zero-state envelope
     */
    static ArtifactEnvelope image(byte[] bytes) {
        return new ArtifactEnvelope("graph-model.v4", List.of(new ArtifactEnvelope.Entry("schema", bytes)));
    }

    @Test void independentlyAuthoredFloatingCastsRetainTargetAndEveryCanonicalPortWithoutVersionChange() throws Exception {
        for (DataType target : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16,
                DataType.INT32, DataType.INT64, DataType.BOOL)) for (long time : new long[]{0, 3}) {
            int wireType = switch (target) {
                case FLOAT64 -> 1; case FLOAT32 -> 2; case BFLOAT16 -> 3;
                case INT32 -> 4; case INT64 -> 5; case BOOL -> 6; case FLOAT16 -> 7;
            };
            var fixture = fixture(32, new byte[]{(byte) wireType}, false, time);
            byte[] golden = fixture.bytes().clone();
            golden[fixture.offsets().get("bodyResult")] = (byte) wireType;
            var cursor = ByteBuffer.wrap(golden);
            cursor.position(fixture.offsets().get("outerResult"));
            skipDescriptor(cursor); // final carry remains F32
            int emission = cursor.position(); skipDescriptor(cursor);
            skipDescriptor(cursor); // carry residual remains F32
            golden[emission] = (byte) wireType;
            golden[cursor.position()] = (byte) wireType; // cast residual
            if (!target.isFloating()) {
                assertRejected(golden, "nonfloating cross-type body CAST remains outside the cold inventory");
                continue;
            }
            Tensor h = f(), x = f(), initial = f(), sequence = f(time);
            Tensor cast = h.cast(target);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of(cast));
            var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
            var source = new GraphCheckpoint.Source(List.of(initial, sequence), scan.finalCarry(), List.of());
            var expected = GraphSource.capture(source).graph;
            assertArrayEquals(golden, schema(GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS)));
            var loaded = GraphArtifactCodec.decode(image(golden), LIMITS);
            assertEquals(expected, GraphSource.capture(loaded).graph);
            var producer = loaded.outputs().getFirst().provenance().orElseThrow().producer();
            assertEquals(target, producer.output(1).descriptor().dataType());
            assertEquals(target, producer.output(3).descriptor().dataType());
            var fresh = producer.scanBody().orElseThrow();
            assertSame(fresh.emissions().getFirst(), fresh.retainedValues().getLast());
            assertEquals(target, fresh.emissions().getFirst().descriptor().dataType());
            var encoded = GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS);
            assertEquals("graph-model.v4", encoded.kind());
            assertArrayEquals(golden, schema(encoded));
        }
    }

    @Test void independentLowStateAndFloatCastMetadataPreflightBeforeNamedImport() throws Exception {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) for (long time : new long[]{0, 3}) {
            int lowType = type == DataType.BFLOAT16 ? 3 : 7;
            var fixture = fixture(32, new byte[]{2}, false, time, true);
            byte[] golden = fixture.bytes().clone();
            var cursor = ByteBuffer.wrap(golden);
            cursor.position(16); // input count was one
            int sequence = cursor.position(); skipDescriptor(cursor);
            assertEquals(1, cursor.getInt());
            int state = cursor.position(); skipDescriptor(cursor);
            assertEquals(2, cursor.get()); // BUFFER role
            int stateSize = cursor.position();
            assertEquals(4, cursor.getLong());
            cursor.putLong(stateSize, 2);
            var lowDescriptors = new ArrayList<>(List.of(sequence, state));
            cursor.position(fixture.offsets().get("formal"));
            lowDescriptors.add(cursor.position()); skipDescriptor(cursor);
            lowDescriptors.add(cursor.position()); skipDescriptor(cursor);
            cursor.position(fixture.offsets().get("outerResult"));
            lowDescriptors.add(cursor.position()); skipDescriptor(cursor); // low final carry
            skipDescriptor(cursor); // F32 emission
            lowDescriptors.add(cursor.position()); // low carry residual
            for (int offset : lowDescriptors) golden[offset] = (byte) lowType;
            var valid = new ArtifactEnvelope("graph-model.v4", List.of(
                    new ArtifactEnvelope.Entry("schema", golden),
                    new ArtifactEnvelope.Entry("state/00000000", raw(type, 1))));
            var loaded = GraphArtifactCodec.decode(valid, LIMITS);
            Tensor restored = loaded.leaves().getFirst().tensor();
            assertEquals(type, restored.descriptor().dataType());
            assertArrayEquals(raw(type, 1), GraphValueStorage.encode(
                    restored.descriptor(), restored.hostStorage().orElseThrow().segment()));
            var producer = loaded.outputs().getFirst().provenance().orElseThrow().producer();
            assertSame(restored, producer.inputs().getFirst());
            assertEquals(type, producer.output(0).descriptor().dataType());
            assertEquals(DataType.FLOAT32, producer.output(1).descriptor().dataType());
            assertArrayEquals(golden, schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
            byte[] bad = golden.clone();
            bad[fixture.offsets().get("attrs")] = 1; // F64 target with stored F32 working result
            var malformed = new ArtifactEnvelope("graph-model.v4", List.of(
                    new ArtifactEnvelope.Entry("schema", bad),
                    new ArtifactEnvelope.Entry("state/00000000", raw(type, 1))));
            Tensor before = f();
            var failure = assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(malformed, LIMITS));
            assertInstanceOf(IllegalArgumentException.class, failure.getCause());
            assertEquals(before.id().value() + 1, f().id().value(),
                    "invalid working metadata rejects before candidate leaves, state import or body identities");
        }
    }

    /**
     * Copies detached schema bytes for inspection/mutation.
     * @param image encoded envelope
     * @return independent schema array
     */
    private static byte[] schema(ArtifactEnvelope image) {
        byte[] bytes = new byte[image.entries().getFirst().size()];
        image.entries().getFirst().payload().get(bytes); return bytes;
    }

    /**
     * Supplies exact canonical scalar/sequence F32 descriptors.
     * @param extents static dimensions, empty for scalar
     * @return fresh storage-free non-gradient leaf
     */
    private static Tensor f(long... extents) { return prototype(DataType.FLOAT32, Shape.of(extents), false, true); }

    @Test void independentGoldenBodyFreeAndProducerBodiesPreserveExactFields() throws Exception {
        var sizes = new HashMap<Integer, Integer>();
        for (long time : List.of(0L, 3L, 1_000_000L)) {
            Tensor initial = f(), sequence = f(time), h = f(), x = f();
            for (int op : new int[]{0, 1, 9, 24, 25, 26, 27, 28, 29, 30, 31, 32}) {
                Tensor emission = switch (op) {
                    case 0 -> h; case 1 -> h.add(x); case 9 -> h.sub(x); case 24 -> h.div(x);
                    case 25 -> h.add(ScalarValue.float32(Float.intBitsToFloat(0xffc00123)));
                    case 26 -> h.sub(ScalarValue.float32(Float.intBitsToFloat(0xffc00123)));
                    case 27 -> h.mul(ScalarValue.float32(Float.intBitsToFloat(0xffc00123)));
                    case 28 -> h.div(ScalarValue.float32(Float.intBitsToFloat(0xffc00123)));
                    case 29 -> h.sum(); case 30 -> h.expand(Shape.scalar());
                    case 31 -> h.contiguous(); case 32 -> h.cast(DataType.FLOAT32);
                    default -> throw new AssertionError();
                };
                byte[] attrs = op >= 25 && op <= 28 ? new byte[]{2, (byte) 0xff, (byte) 0xc0, 1, 0x23}
                        : op == 29 ? new byte[]{0} : op == 30 ? new byte[4] : op == 32 ? new byte[]{2} : new byte[0];
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of(emission));
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
                var source = new GraphCheckpoint.Source(List.of(initial, sequence), scan.finalCarry(), List.of());
                var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                var fixture = fixture(op, attrs, false, time);
                assertArrayEquals(fixture.bytes(), schema(encoded), "opcode " + op);
                assertEquals(sizes.computeIfAbsent(op, ignored -> fixture.bytes().length), fixture.bytes().length);
                assertEquals(op == 0 ? 0 : 1, GraphSource.capture(source).graph.nodes().getFirst()
                        .scanBody().orElseThrow().graph().nodes().size());
                var loaded = GraphArtifactCodec.decode(image(fixture.bytes()), LIMITS);
                assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
                assertArrayEquals(fixture.bytes(), schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
            }
        }
    }

    @Test void countIdRoleDemandPhaseAndOpcodeCorruptionsFailBeforeAnyCandidate() throws Exception {
        var f = fixture(1, new byte[0], true, 0);
        var mutations = new LinkedHashMap<String, Integer>();
        mutations.put("actualCount", Integer.MAX_VALUE); mutations.put("actual", 3);
        mutations.put("carryCount", Integer.MAX_VALUE); mutations.put("sequenceCount", -1);
        mutations.put("invariantCount", Integer.MAX_VALUE); mutations.put("emissionCount", -1);
        mutations.put("residualCount", Integer.MAX_VALUE); mutations.put("formalCount", Integer.MAX_VALUE);
        mutations.put("nodeCount", Integer.MAX_VALUE); mutations.put("bodyInput", 2);
        mutations.put("bodyQ", Integer.MAX_VALUE); mutations.put("rootCount", 3);
        mutations.put("root", 2); mutations.put("nextCount", 0); mutations.put("next", 3);
        mutations.put("emitCount", 0); mutations.put("emit", 3); mutations.put("residualListCount", 1);
        mutations.put("residual", 2); mutations.put("demand", 1); mutations.put("q", Integer.MAX_VALUE);
        for (var mutation : mutations.entrySet()) {
            byte[] bad = f.bytes().clone(); ByteBuffer.wrap(bad).putInt(f.offsets().get(mutation.getKey()), mutation.getValue());
            assertRejected(bad, mutation.getKey());
        }
        for (String field : List.of("phase", "reverse", "bodyPhase")) {
            byte[] bad = f.bytes().clone(); bad[f.offsets().get(field)] = 2; assertRejected(bad, field);
        }
        for (int op : new int[]{0, 24, 25, 29, 32, 33, 255}) {
            byte[] bad = f.bytes().clone(); bad[f.offsets().get("opcode")] = (byte) op;
            assertRejected(bad, "outer opcode " + op);
        }
        for (int op : new int[]{0, 20, 21, 22, 23, 33, 255}) {
            byte[] bad = f.bytes().clone(); bad[f.offsets().get("bodyOpcode")] = (byte) op;
            assertRejected(bad, "body opcode " + op);
        }
        for (int length = 0; length < f.bytes().length; length++)
            assertRejected(Arrays.copyOf(f.bytes(), length), "truncation " + length);
        assertRejected(Arrays.copyOf(f.bytes(), f.bytes().length + 1), "trailing");
        // Header-valid metadata with a wrong canonical formal flag still allocates no Tensor.
        byte[] semantic = f.bytes().clone(); semantic[f.offsets().get("formal") + 1] = 1;
        assertRejected(semantic, "late semantic formal mismatch");
    }

    /**
     * Probes complete metadata validation before any candidate, body formal or producer Tensor.
     * @param bytes malformed independent schema
     * @param label diagnostic mutation label
     */
    private static void assertRejected(byte[] bytes, String label) {
        Tensor before = f();
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bytes), LIMITS), label);
        Tensor after = f();
        assertEquals(1, after.id().value() - before.id().value(), label);
    }

    @Test void exactTuplesOldOpcodeClosureAndBodyFreeV4DowngradeAreSelective() throws Exception {
        var good = fixture(0, new byte[0], false, 3);
        for (int version : new int[]{1, 2, 3}) {
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(new ArtifactEnvelope(
                    "graph-model.v" + version, image(good.bytes()).entries()), LIMITS));
            byte[] old = good.bytes().clone(); old[7] = (byte) ('0' + version); ByteBuffer.wrap(old).putInt(8, version);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(new ArtifactEnvelope(
                    "graph-model.v" + version, List.of(new ArtifactEnvelope.Entry("schema", old))), LIMITS));
        }
        Tensor x = f(), y = f();
        for (Tensor root : List.of(x.add(y), x.tanh(), x.sub(y))) {
            var inputs = root.provenance().orElseThrow().inputs().size() == 1 ? List.of(x) : List.of(x, y);
            var original = GraphArtifactCodec.encode(GraphSource.capture(new GraphCheckpoint.Source(inputs, List.of(root), List.of())), LIMITS);
            byte[] v4 = schema(original);
            if (!original.kind().equals("graph-model.v3")) {
                // Older records have implicit Q; insert its v3-style field before the descriptor.
                int descriptor = v4.length - 8 - 7;
                byte[] expanded = new byte[v4.length + 4];
                System.arraycopy(v4, 0, expanded, 0, descriptor);
                ByteBuffer.wrap(expanded).putInt(descriptor, 1);
                System.arraycopy(v4, descriptor, expanded, descriptor + 4, v4.length - descriptor); v4 = expanded;
            }
            v4[7] = '4'; ByteBuffer.wrap(v4).putInt(8, 4);
            var loaded = GraphArtifactCodec.decode(image(v4), LIMITS);
            var saved = GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS);
            assertEquals(original.kind(), saved.kind()); assertArrayEquals(schema(original), schema(saved));
        }
        var floatingSub = GraphArtifactCodec.encode(GraphSource.capture(
                new GraphCheckpoint.Source(List.of(x, y), List.of(x.sub(y)), List.of())), LIMITS);
        byte[] integralSub = schema(floatingSub);
        integralSub[7] = '4'; ByteBuffer.wrap(integralSub).putInt(8, 4);
        var cursor = ByteBuffer.wrap(integralSub); cursor.position(16);
        for (int i = 0; i < 2; i++) {
            int dtype = cursor.position(); skipDescriptor(cursor); integralSub[dtype] = 4;
        }
        assertEquals(0, cursor.getInt()); assertEquals(0, cursor.getInt()); assertEquals(1, cursor.getInt());
        cursor.position(cursor.position() + 14); integralSub[cursor.position()] = 4;
        assertRejected(integralSub, "numeric body SUB must not widen outer floating SUB");
    }

    @Test void bodyAttributesRanksGeometryAndWrongDerivedResidualsFailClosed() throws Exception {
        var scalar = fixture(25, new byte[]{2, 0, 0, 0, 0}, true, 0);
        for (int dtype : new int[]{0, 8, 255}) {
            byte[] bad = scalar.bytes().clone(); bad[scalar.offsets().get("attrs")] = (byte) dtype;
            assertRejected(bad, "scalar dtype " + dtype);
        }
        var sum = fixture(29, new byte[]{0}, true, 0);
        byte[] tag = sum.bytes().clone(); tag[sum.offsets().get("attrs")] = 4;
        assertRejected(tag, "SUM tag");
        // Scalar input has no legal reduction axis; complete metadata is parsed before preflight.
        var axis = fixture(29, new byte[]{1, 0, 0, 0, 0, 0}, true, 0);
        assertRejected(axis.bytes(), "SUM scalar axis");
        byte[] flag = axis.bytes().clone(); flag[axis.offsets().get("attrs") + 5] = 2;
        assertRejected(flag, "SUM keep flag");
        var axes = fixture(29, new byte[]{2, 127, -1, -1, -1, 0}, true, 0);
        assertRejected(axes.bytes(), "SUM count before axes allocation");
        var reshape = fixture(10, new byte[]{127, -1, -1, -1}, true, 0);
        assertRejected(reshape.bytes(), "target rank");
        var cast = fixture(32, new byte[]{1}, true, 0);
        assertRejected(cast.bytes(), "CAST target disagrees with its stored result descriptor");
        var good = fixture(1, new byte[0], true, 0);
        for (int rank : new int[]{-1, Integer.MAX_VALUE}) {
            byte[] bad = good.bytes().clone(); ByteBuffer.wrap(bad).putInt(good.offsets().get("formal") + 2, rank);
            assertRejected(bad, "formal rank " + rank);
        }
        byte[] layout = good.bytes().clone(); ByteBuffer.wrap(layout).putLong(good.offsets().get("formal") + 7, Long.MAX_VALUE);
        assertRejected(layout, "overflowing formal geometry");
        byte[] residual = good.bytes().clone();
        ByteBuffer.wrap(residual).putInt(good.offsets().get("residual") + 4, 0);
        assertRejected(residual, "duplicate derived residual");
        // The body producer remains in the wire but no root can reach it.
        byte[] dead = good.bytes().clone();
        ByteBuffer.wrap(dead).putInt(good.offsets().get("root") + 4, 1)
                .putInt(good.offsets().get("emit"), 1).putInt(good.offsets().get("residual") + 4, 1);
        assertRejected(dead, "dead contained producer");
    }

    @Test void scalarAttributesPreserveEveryApplicableRawDtypeWithoutCpuQualification() throws Exception {
        for (DataType type : DataType.values()) {
            if (type == DataType.BOOL) continue;
            for (boolean negativeZero : List.of(false, true)) {
                ScalarValue scalar = switch (type) {
                    case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(negativeZero ? Long.MIN_VALUE : 0xfff8000000000123L));
                    case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(negativeZero ? Integer.MIN_VALUE : 0xffc00123));
                    case BFLOAT16 -> ScalarValue.bfloat16Bits((short) (negativeZero ? 0x8000 : 0xffc3));
                    case FLOAT16 -> ScalarValue.float16Bits((short) (negativeZero ? 0x8000 : 0xfe23));
                    case INT64 -> ScalarValue.int64(negativeZero ? Long.MIN_VALUE : 0x123456789abcdefL);
                    case INT32 -> ScalarValue.int32(negativeZero ? Integer.MIN_VALUE : 0x12345678);
                    default -> throw new AssertionError(type);
                };
                Tensor h = prototype(type, Shape.of(2), false, true), x = prototype(type, Shape.of(2), false, true);
                var emissions = new ArrayList<>(List.of(h.add(scalar), h.sub(scalar), h.mul(scalar)));
                if (type.isFloating()) emissions.add(h.div(scalar));
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), emissions);
                Tensor initial = prototype(type, Shape.of(2), false, true), sequence = prototype(type, Shape.of(0, 2), false, true);
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
                var source = new GraphCheckpoint.Source(List.of(sequence, initial), scan.finalCarry(), List.of());
                var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                var loaded = GraphArtifactCodec.decode(image, LIMITS);
                assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
                assertArrayEquals(schema(image), schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
            }
        }
    }

    @Test void sumAttributeTagsAxesOrderEmptyAxesAndTargetShapeHaveIndependentBytes() throws Exception {
        Tensor h = f(2, 3), x = f(2, 3), initial = f(2, 3), sequence = f(0, 2, 3);
        var emissions = List.of(h.sum(), h.sum(1, true), h.sum(new int[]{1, 0}, false),
                h.sum(new int[]{}, true), h.sumToShape(Shape.of(1, 3)));
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), emissions);
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
        var source = new GraphCheckpoint.Source(List.of(initial, sequence), scan.finalCarry(), List.of());
        byte[] schema = schema(GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS));
        var in = ByteBuffer.wrap(schema); in.position(12);
        assertEquals(2, in.getInt()); skipDescriptor(in); skipDescriptor(in);
        assertEquals(0, in.getInt()); assertEquals(0, in.getInt()); assertEquals(1, in.getInt());
        assertEquals(1, in.get()); assertEquals(23, in.get()); assertEquals(2, in.getInt());
        assertEquals(0, in.getInt()); assertEquals(1, in.getInt());
        assertEquals(0, in.getLong());
        for (int n : new int[]{1, 1, 0, 5, 6}) assertEquals(n, in.getInt());
        assertEquals(0, in.get()); assertEquals(2, in.getInt()); skipDescriptor(in); skipDescriptor(in);
        assertEquals(5, in.getInt());
        var target = ByteBuffer.allocate(21).put((byte) 3).putInt(2).putLong(1).putLong(3).array();
        var expected = List.of(new byte[]{0}, new byte[]{1, 0, 0, 0, 1, 1},
                new byte[]{2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                new byte[]{2, 0, 0, 0, 0, 1}, target);
        for (byte[] attrs : expected) {
            assertEquals(1, in.get()); assertEquals(29, in.get()); assertEquals(0, in.getInt());
            byte[] actual = new byte[attrs.length]; in.get(actual); assertArrayEquals(attrs, actual);
            assertEquals(1, in.getInt()); skipDescriptor(in);
        }
        var loaded = GraphArtifactCodec.decode(image(schema), LIMITS);
        assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
    }

    /**
     * Walks descriptor fields independently so attribute tests cannot share the codec's parser.
     * @param in test-owned big-endian schema cursor
     */
    private static void skipDescriptor(ByteBuffer in) {
        assertEquals(2, in.get()); assertEquals(0, in.get());
        int rank = in.getInt(); in.position(in.position() + 8 * rank);
        int layout = in.get();
        assertTrue(layout == 0 || layout == 1);
        if (layout == 1) in.position(in.position() + 9 + 8 * rank);
    }

    @Test void allSixExactBoundsIncludeBodiesBeforeHostAdmissionAndPreserveTarget() throws Exception {
        Tensor initial = f(2), sequence = f(3, 2), h = f(2), x = f(2), w = f(2);
        Tensor state = tensor(DataType.FLOAT32, Shape.of(2), false, true, raw(DataType.FLOAT32, 2));
        var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(x).mul(w).tanh()), List.of(h));
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(state), body, false);
        var source = new GraphCheckpoint.Source(List.of(sequence, initial), scan.finalCarry(),
                List.of(leaf("a", GraphCheckpoint.Role.BUFFER, state), leaf("b", GraphCheckpoint.Role.BUFFER, state)));
        var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        int d = schema(encoded).length, b = 8, m = 130, file = 48 + m + d + b;
        var exact = new ArtifactLimits(file, m, 2, 14, d, d + b);
        Path target = directory.resolve("bounds"); GraphCheckpoint.save(target, source, exact);
        byte[] prior = Files.readAllBytes(target);
        assertEquals(file, prior.length); assertEquals(GraphSource.capture(source).graph, GraphSource.capture(GraphCheckpoint.load(target, exact)).graph);
        state.clearHostStorage();
        for (var below : List.of(new ArtifactLimits(file - 1, m, 2, 14, d, d + b),
                new ArtifactLimits(file, m - 1, 2, 14, d, d + b), new ArtifactLimits(file, m, 1, 14, d, d + b),
                new ArtifactLimits(file, m, 2, 13, d, d + b), new ArtifactLimits(file, m, 2, 14, d - 1, d + b),
                new ArtifactLimits(file, m, 2, 14, d, d + b - 1))) {
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(target, source, below));
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(encoded, below));
            assertArrayEquals(prior, Files.readAllBytes(target));
        }
    }

    @Test void unchangedFlatConstructorsAndBroadMatmulGuardsRemainAvailableInsideBodies() throws Exception {
        Tensor h = f(), x = f(), a = f(2, 3), b = f(3, 4), vector = f(3);
        Tensor batched = f(2, 2, 3), matrices = f(2, 3, 4), sequence = f(2, 2, 3);
        Tensor indices = prototype(DataType.INT64, Shape.of(1), false, true);
        Tensor image = f(1, 1, 3, 3), kernel = f(2, 1, 2, 2), bias = f(2);
        Tensor volume = f(1, 1, 3, 3, 3), volumeKernel = f(2, 1, 2, 2, 2);
        var invariants = List.of(a, b, vector, batched, matrices, sequence, indices, image, kernel, bias, volume, volumeKernel);
        var emissions = List.of(a.matmul(b), vector.matmul(vector), batched.matmul(matrices), a.gather(indices, 0),
                a.sliceByLength(new long[]{1}, new long[]{2}, new int[]{1}, new long[]{1}),
                Tensor.concat(0, sequence, sequence), sequence.select(0, 1),
                image.conv2d(kernel, bias, io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs.defaults()),
                volume.conv3d(volumeKernel, bias, io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs.defaults()),
                image.maxPool2d(new io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false)));
        var body = new ScanBody(List.of(h), List.of(x), invariants, List.of(h), emissions);
        for (long time : List.of(0L, 3L)) {
            Tensor initial = f(), input = f(time);
            var actuals = invariants.stream().map(t -> TensorFactory.create(t.descriptor())).toList();
            var scan = Scan.apply(List.of(initial), List.of(input), actuals, body, false);
            var inputs = new ArrayList<>(List.of(initial, input)); inputs.addAll(actuals);
            var source = new GraphCheckpoint.Source(inputs, scan.finalCarry(), List.of());
            var imageBytes = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
            assertEquals(GraphSource.capture(source).graph, GraphSource.capture(GraphArtifactCodec.decode(imageBytes, LIMITS)).graph);
        }
        long[] extents = new long[20]; Arrays.fill(extents, 1);
        Tensor high = f(extents), step = f(extents), initial = f(extents);
        long[] timed = new long[21]; Arrays.fill(timed, 1); timed[0] = 0;
        Tensor input = f(timed);
        var highBody = new ScanBody(List.of(high), List.of(step), List.of(), List.of(high), List.of(high.contiguous()));
        var highScan = Scan.apply(List.of(initial), List.of(input), List.of(), highBody, false);
        var highSource = new GraphCheckpoint.Source(List.of(initial, input), highScan.finalCarry(), List.of());
        var encoded = GraphArtifactCodec.encode(GraphSource.capture(highSource), LIMITS);
        assertEquals(GraphSource.capture(highSource).graph, GraphSource.capture(GraphArtifactCodec.decode(encoded, LIMITS)).graph);
    }
}
