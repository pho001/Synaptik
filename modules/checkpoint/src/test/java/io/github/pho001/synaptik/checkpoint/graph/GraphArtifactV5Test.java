package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
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

/** V5 role bytes, independent malformed images and cold boundaries, without execution dependencies. */
final class GraphArtifactV5Test {
    @TempDir Path directory;

    @Test void invalidCountsRejectBeforeRoleAndBodyAllocationEvenAtZeroDomains() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : new long[] {0, 1}) for (long batch : new long[] {0, 1})
                for (boolean reverse : List.of(false, true)) {
                    var f = fixture(type, time, batch, 0, reverse, false, true, false);
                    byte[] noCarry = Arrays.copyOf(f.bytes(), f.offsets().get("reverse") + 1);
                    ByteBuffer.wrap(noCarry).putInt(f.offsets().get("carryCount"), -1);
                    assertInstanceOf(IllegalArgumentException.class, rejected(image(noCarry)).getCause());
                    byte[] noRows = Arrays.copyOf(f.bytes(), f.offsets().get("rowCarryCount") + 4);
                    ByteBuffer.wrap(noRows).putInt(f.offsets().get("rowCarryCount"), -1);
                    assertInstanceOf(IllegalArgumentException.class, rejected(image(noRows)).getCause());
                }
    }

    /** Test-owned bytes and offsets, independent of the production writer. */
    private record Fixture(byte[] bytes, Map<String, Integer> offsets) { }

    /**
     * Writes exact static descriptor bytes using the public wire definition only.
     * @param out test stream
     * @param dtype explicit wire dtype ID
     * @param canonical whether canonical layout fields are present
     * @param shape static extents
     * @throws Exception if fixture writing fails
     */
    private static void descriptor(DataOutputStream out, int dtype, boolean canonical, long... shape) throws Exception {
        out.writeByte(dtype); out.writeByte(0); out.writeInt(shape.length);
        for (long extent : shape) out.writeLong(extent);
        out.writeByte(canonical ? 1 : 0);
        if (canonical) {
            out.writeLong(0); out.writeByte(0);
            long stride = 1; long[] strides = new long[shape.length];
            for (int i = shape.length - 1; i >= 0; i--) { strides[i] = stride; stride *= shape[i]; }
            for (long value : strides) out.writeLong(value);
        }
    }

    /**
     * Authors an identity-carry length scan with optional scalar emission and preceding NEG.
     * @param type homogeneous floating dtype
     * @param time static T
     * @param batch static B
     * @param axis -1 for SHARED, otherwise step ROW axis zero or one
     * @param reverse traversal direction
     * @param producer whether emission is a raw-NaN scalar ADD instead of the carry
     * @param late whether a valid NEG precedes the scan
     * @param state whether the carry is named state instead of an input
     * @return independent complete schema and field offsets
     * @throws Exception if fixture writing fails
     */
    private static Fixture fixture(DataType type, long time, long batch, int axis, boolean reverse,
            boolean producer, boolean late, boolean state) throws Exception {
        int dtype = switch (type) {
            case FLOAT64 -> 1; case FLOAT32 -> 2; case BFLOAT16 -> 3; case FLOAT16 -> 7;
            default -> throw new AssertionError(type);
        };
        var bytes = new ByteArrayOutputStream(); var out = new DataOutputStream(bytes);
        var at = new HashMap<String, Integer>();
        out.writeBytes("SYNGRF05"); out.writeInt(5);
        out.writeInt(state ? 2 : 3);
        if (!state) descriptor(out, dtype, true, batch, 2);
        descriptor(out, dtype, true, sequence(time, batch, axis));
        at.put("lengthDescriptor", bytes.size()); descriptor(out, 5, true, batch);
        out.writeInt(state ? 1 : 0);
        if (state) {
            descriptor(out, dtype, true, batch, 2); out.writeByte(2);
            int width = switch (type) {
                case FLOAT32 -> 4; case FLOAT64 -> 8; case BFLOAT16, FLOAT16 -> 2;
                default -> throw new AssertionError(type);
            };
            out.writeLong(batch * 2 * width);
        }
        out.writeInt(state ? 1 : 0);
        if (state) { out.writeInt(5); out.writeBytes("state"); out.writeInt(0); }
        out.writeInt(late ? 2 : 1);
        int carry = state ? 2 : 0;
        if (late) {
            out.writeByte(1); out.writeByte(3); out.writeInt(carry); out.writeInt(1);
            descriptor(out, dtype, false, batch, 2);
        }
        at.put("phase", bytes.size()); out.writeByte(1);
        at.put("opcode", bytes.size()); out.writeByte(33);
        at.put("actualCount", bytes.size()); out.writeInt(3);
        at.put("actual", bytes.size()); out.writeInt(carry); out.writeInt(state ? 0 : 1);
        at.put("lengthReference", bytes.size()); out.writeInt(state ? 1 : 2);
        at.put("time", bytes.size()); out.writeLong(time);
        at.put("carryCount", bytes.size()); out.writeInt(1);
        at.put("sequenceCount", bytes.size()); out.writeInt(1);
        at.put("invariantCount", bytes.size()); out.writeInt(0);
        at.put("emissionCount", bytes.size()); out.writeInt(1);
        at.put("residualCount", bytes.size()); out.writeInt(producer ? 2 : 1);
        at.put("reverse", bytes.size()); out.writeByte(reverse ? 1 : 0);
        at.put("rowCarryCount", bytes.size()); out.writeInt(1);
        at.put("axisCount", bytes.size()); out.writeInt(1);
        at.put("presence", bytes.size()); out.writeByte(axis < 0 ? 0 : 1);
        if (axis >= 0) { at.put("axis", bytes.size()); out.writeInt(axis); }
        at.put("formalCount", bytes.size()); out.writeInt(2);
        at.put("formal", bytes.size()); descriptor(out, dtype, true, batch, 2);
        descriptor(out, dtype, true, step(batch, axis));
        at.put("nodeCount", bytes.size()); out.writeInt(producer ? 1 : 0);
        if (producer) {
            out.writeByte(1); at.put("bodyOpcode", bytes.size()); out.writeByte(25); out.writeInt(0);
            out.writeByte(dtype);
            switch (type) {
                case FLOAT32 -> out.writeInt(0xffc00123);
                case FLOAT64 -> out.writeLong(0xfff8000000000123L);
                case BFLOAT16 -> out.writeShort(0xffc3);
                case FLOAT16 -> out.writeShort(0xfe23);
                default -> throw new AssertionError(type);
            }
            out.writeInt(1); descriptor(out, dtype, false, batch, 2);
        }
        out.writeInt(producer ? 2 : 1); out.writeInt(0); if (producer) out.writeInt(2);
        out.writeInt(1); out.writeInt(0);
        out.writeInt(1); out.writeInt(producer ? 2 : 0);
        at.put("residualListCount", bytes.size()); out.writeInt(producer ? 2 : 1);
        at.put("residual", bytes.size()); out.writeInt(0); if (producer) out.writeInt(2);
        at.put("demand", bytes.size()); out.writeInt(0);
        at.put("q", bytes.size()); out.writeInt(producer ? 4 : 3);
        at.put("outerResult", bytes.size()); descriptor(out, dtype, true, batch, 2);
        descriptor(out, dtype, true, time, batch, 2); descriptor(out, dtype, true, time, batch, 2);
        if (producer) descriptor(out, dtype, true, time, batch, 2);
        out.writeInt(late ? 2 : 1); if (late) out.writeInt(3); out.writeInt(late ? 4 : 3);
        return new Fixture(bytes.toByteArray(), at);
    }

    /**
     * Supplies a step declaration for an explicit role.
     * @param batch static B
     * @param axis -1 for SHARED, otherwise step ROW axis
     * @return step extents for the selected explicit role
     */
    private static long[] step(long batch, int axis) {
        return axis < 0 ? new long[]{2} : axis == 0 ? new long[]{batch, 2} : new long[]{2, batch};
    }

    /**
     * Prepends the fixed time dimension to a step declaration.
     * @param time static T
     * @param batch static B
     * @param axis -1 for SHARED, otherwise step ROW axis
     * @return full sequence extents with leading time
     */
    private static long[] sequence(long time, long batch, int axis) {
        return axis < 0 ? new long[]{time, 2} : axis == 0 ? new long[]{time, batch, 2} : new long[]{time, 2, batch};
    }

    /**
     * Creates a descriptor-only canonical leaf.
     * @param type leaf dtype
     * @param shape static extents
     * @return fresh canonical storage-free leaf
     */
    private static Tensor f(DataType type, long... shape) {
        return prototype(type, Shape.of(shape), false, true);
    }

    /**
     * Copies schema bytes for independent comparison.
     * @param image detached encoded envelope
     * @return detached schema bytes
     */
    private static byte[] schema(ArtifactEnvelope image) {
        byte[] bytes = new byte[image.entries().getFirst().size()];
        image.entries().getFirst().payload().get(bytes); return bytes;
    }

    /**
     * Wraps independently authored schema without named state.
     * @param bytes independent schema
     * @return exact zero-state v5 envelope
     */
    private static ArtifactEnvelope image(byte[] bytes) {
        return new ArtifactEnvelope("graph-model.v5", List.of(new ArtifactEnvelope.Entry("schema", bytes)));
    }

    @Test void independentlyAuthoredGoldenBytesCoverDtypesTraversalEmptyDomainsAndRoleAxes() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16))
            for (long time : new long[]{0, 3, 1_000_000}) for (long batch : new long[]{0, 1, 3})
                for (int axis : new int[]{-1, 0, 1}) for (boolean reverse : List.of(false, true))
                    for (boolean producer : List.of(false, true)) {
                        Tensor h = f(type, batch, 2), x = f(type, step(batch, axis));
                        Tensor emission = producer ? h.add(scalar(type)) : h;
                        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of(emission));
                        Tensor initial = f(type, batch, 2), input = f(type, sequence(time, batch, axis));
                        Tensor lengths = f(DataType.INT64, batch);
                        var scan = Scan.apply(List.of(initial), List.of(input), List.of(), lengths, 1,
                                List.of(axis < 0 ? OptionalInt.empty() : OptionalInt.of(axis)), body, reverse);
                        var source = new GraphCheckpoint.Source(List.of(initial, input, lengths), scan.finalCarry(), List.of());
                        var golden = fixture(type, time, batch, axis, reverse, producer, false, false);
                        var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                        assertEquals("graph-model.v5", encoded.kind()); assertArrayEquals(golden.bytes(), schema(encoded));
                        var loaded = GraphArtifactCodec.decode(image(golden.bytes()), LIMITS);
                        assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
                        assertTrue(loaded.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                        assertArrayEquals(golden.bytes(), schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
                    }
    }

    /**
     * Supplies the exact NaN payload used by independent scalar fixtures.
     * @param type homogeneous floating dtype
     * @return typed raw-NaN scalar
     */
    private static ScalarValue scalar(DataType type) {
        return switch (type) {
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0xffc00123));
            case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(0xfff8000000000123L));
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xffc3);
            case FLOAT16 -> ScalarValue.float16Bits((short) 0xfe23);
            default -> throw new AssertionError(type);
        };
    }

    @Test void malformedRolesReferencesTuplesAndNestedScansFailBeforeProducerIds() throws Exception {
        var fixture = fixture(DataType.FLOAT32, 0, 3, 0, false, true, true, false);
        var mutations = new LinkedHashMap<String, Integer>();
        mutations.put("actualCount", Integer.MAX_VALUE); mutations.put("actual", -1);
        mutations.put("lengthReference", 4); mutations.put("carryCount", Integer.MAX_VALUE);
        mutations.put("sequenceCount", -1); mutations.put("invariantCount", Integer.MAX_VALUE);
        mutations.put("emissionCount", -1); mutations.put("residualCount", Integer.MAX_VALUE);
        mutations.put("rowCarryCount", 0); mutations.put("axisCount", Integer.MAX_VALUE);
        mutations.put("axis", 2); mutations.put("formalCount", 3); mutations.put("nodeCount", Integer.MAX_VALUE);
        mutations.put("residualListCount", 0); mutations.put("demand", 1); mutations.put("q", Integer.MAX_VALUE);
        for (var mutation : mutations.entrySet()) {
            byte[] bad = fixture.bytes().clone();
            ByteBuffer.wrap(bad).putInt(fixture.offsets().get(mutation.getKey()), mutation.getValue());
            rejected(image(bad));
        }
        for (int count : new int[]{-1, 0, 2}) {
            byte[] bad = fixture.bytes().clone(); ByteBuffer.wrap(bad).putInt(fixture.offsets().get("axisCount"), count);
            rejected(image(bad));
        }
        for (int row : new int[]{-1, 2, Integer.MAX_VALUE}) {
            byte[] bad = fixture.bytes().clone(); ByteBuffer.wrap(bad).putInt(fixture.offsets().get("rowCarryCount"), row);
            rejected(image(bad));
        }
        byte[] negativeAxis = fixture.bytes().clone(); ByteBuffer.wrap(negativeAxis).putInt(fixture.offsets().get("axis"), -1);
        rejected(image(negativeAxis));
        for (String flag : List.of("phase", "reverse", "presence")) {
            byte[] bad = fixture.bytes().clone(); bad[fixture.offsets().get(flag)] = 2; rejected(image(bad));
        }
        for (int op : new int[]{20, 21, 22, 23, 33, 255}) {
            byte[] bad = fixture.bytes().clone(); bad[fixture.offsets().get("bodyOpcode")] = (byte) op;
            rejected(image(bad));
        }
        for (int length = 0; length < fixture.bytes().length; length++)
            rejected(image(Arrays.copyOf(fixture.bytes(), length)));
        rejected(image(Arrays.copyOf(fixture.bytes(), fixture.bytes().length + 1)));
        for (int version = 1; version <= 4; version++) {
            rejected(new ArtifactEnvelope("graph-model.v" + version, image(fixture.bytes()).entries()));
            for (int opcode : new int[]{23, 33}) {
                byte[] old = fixture.bytes().clone(); old[7] = (byte) ('0' + version);
                ByteBuffer.wrap(old).putInt(8, version); old[fixture.offsets().get("opcode")] = (byte) opcode;
                rejected(new ArtifactEnvelope("graph-model.v" + version,
                        List.of(new ArtifactEnvelope.Entry("schema", old))));
            }
        }
        for (int opcode : new int[]{23, 24, 32, 34}) {
            byte[] bad = fixture.bytes().clone(); bad[fixture.offsets().get("opcode")] = (byte) opcode;
            rejected(image(bad));
        }
        rejected(new ArtifactEnvelope("graph-model.v6", image(fixture.bytes()).entries()));
        for (int offset : new int[]{fixture.offsets().get("lengthDescriptor"), fixture.offsets().get("lengthDescriptor") + 1}) {
            byte[] bad = fixture.bytes().clone(); bad[offset] = (byte) (offset == fixture.offsets().get("lengthDescriptor") ? 4 : 1);
            rejected(image(bad));
        }
        for (String field : List.of("time", "outerResult")) {
            byte[] bad = fixture.bytes().clone();
            if (field.equals("time")) ByteBuffer.wrap(bad).putLong(fixture.offsets().get(field), -1);
            else ByteBuffer.wrap(bad).putLong(fixture.offsets().get(field) + 6, 4);
            rejected(image(bad));
        }
        byte[] wrongBatch = fixture.bytes().clone();
        ByteBuffer.wrap(wrongBatch).putLong(fixture.offsets().get("lengthDescriptor") + 6, 4);
        rejected(image(wrongBatch));
        byte[] wrongRole = fixture.bytes().clone(); ByteBuffer.wrap(wrongRole).putInt(fixture.offsets().get("axis"), 1);
        rejected(image(wrongRole));
        byte[] wrongLengths = fixture.bytes().clone(); ByteBuffer.wrap(wrongLengths).putInt(fixture.offsets().get("lengthReference"), 0);
        rejected(image(wrongLengths));
    }

    @Test void scalarDomainsAndActualCardinalityRejectBeforeReadingOrAllocatingRoles() throws Exception {
        var fixture = fixture(DataType.FLOAT32, 0, 3, 0, false, true, true, false);
        // No row count, axis count or role bytes exist: header failure must win over underflow.
        byte[] header = Arrays.copyOf(fixture.bytes(), fixture.offsets().get("reverse") + 1);
        for (var mutation : List.of(Map.of("carryCount", -1), Map.of("sequenceCount", 0),
                Map.of("invariantCount", -1), Map.of("emissionCount", -1), Map.of("residualCount", -1),
                Map.of("carryCount", 2), Map.of("sequenceCount", 1_000_000))) {
            byte[] bad = header.clone();
            for (var entry : mutation.entrySet())
                ByteBuffer.wrap(bad).putInt(fixture.offsets().get(entry.getKey()), entry.getValue());
            assertInstanceOf(IllegalArgumentException.class, rejected(image(bad)).getCause());
        }
        byte[] negativeTime = header.clone();
        ByteBuffer.wrap(negativeTime).putLong(fixture.offsets().get("time"), -1);
        assertInstanceOf(IllegalArgumentException.class, rejected(image(negativeTime)).getCause());
        for (var mutation : List.of(Map.of("carryCount", Integer.MAX_VALUE, "sequenceCount", 1_000_000),
                Map.of("emissionCount", Integer.MAX_VALUE),
                Map.of("carryCount", Integer.MAX_VALUE - 1, "emissionCount", 0, "residualCount", 0))) {
            byte[] bad = header.clone();
            for (var entry : mutation.entrySet())
                ByteBuffer.wrap(bad).putInt(fixture.offsets().get(entry.getKey()), entry.getValue());
            assertInstanceOf(ArithmeticException.class, rejected(image(bad)).getCause());
        }
    }

    /**
     * Requires complete invalid-metadata rejection before any candidate, formal or producer Tensor.
     * @param image invalid independent envelope
     * @return retained format failure for optional cause-based ordering assertions
     */
    private static ArtifactFormatException rejected(ArtifactEnvelope image) {
        Tensor before = f(DataType.FLOAT32);
        var failure = assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image, LIMITS));
        assertEquals(1, f(DataType.FLOAT32).id().value() - before.id().value());
        return failure;
    }

    @Test void lateInvalidBodyWithNamedStateFailsBeforeImportAndAnyFormalOrProducer() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16))
            for (long time : new long[]{0, 3}) {
                var fixture = fixture(type, time, 2, 0, false, true, true, true);
                var valid = new ArtifactEnvelope("graph-model.v5", List.of(new ArtifactEnvelope.Entry("schema", fixture.bytes()),
                        new ArtifactEnvelope.Entry("state/00000000", raw(type, 4))));
                var loaded = GraphArtifactCodec.decode(valid, LIMITS);
                assertArrayEquals(fixture.bytes(), schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
                var state = loaded.leaves().getFirst().tensor();
                assertArrayEquals(raw(type, 4), GraphValueStorage.encode(state.descriptor(),
                        state.hostStorage().orElseThrow().segment()));
                byte[] bad = fixture.bytes().clone(); bad[fixture.offsets().get("formal") + 1] = 1;
                var image = new ArtifactEnvelope("graph-model.v5", List.of(new ArtifactEnvelope.Entry("schema", bad),
                        new ArtifactEnvelope.Entry("state/00000000", raw(type, 4))));
                rejected(image);
            }
    }

    @Test void rowRolesTerminalAccumulatorsAliasesUnusedFormalsAndAllPortsRoundTrip() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean reverse : List.of(false, true))
            for (long time : new long[]{0, 3}) for (long batch : new long[]{0, 1, 3}) {
                Tensor h = f(type, batch, 2), unusedCarry = f(type, batch, 2), accumulator = f(type, 2);
                Tensor row0 = f(type, batch, 2), duplicate = f(type, batch, 2), row1 = f(type, 2, batch), shared = f(type, 2);
                Tensor w = f(type, 2), unused = f(type, 2);
                Tensor next = h.add(row0).add(row1.permute(1, 0)).add(shared).mul(w).add(scalar(type));
                var body = new ScanBody(List.of(h, unusedCarry, accumulator), List.of(row0, duplicate, row1, shared),
                        List.of(w, unused), List.of(next, next, accumulator.add(next.sum(0, false))), List.of(next, next, h));
                Tensor initial = f(type, batch, 2), acc = f(type, 2), seq0 = f(type, time, batch, 2);
                Tensor seq1 = f(type, time, 2, batch), seqShared = f(type, time, 2), lengths = f(DataType.INT64, batch);
                Tensor state = tensor(type, Shape.of(2), false, true, raw(type, 2));
                var carry = List.of(initial, initial, acc); var sequences = List.of(seq0, seq0, seq1, seqShared);
                var roles = List.of(OptionalInt.of(0), OptionalInt.of(0), OptionalInt.of(1), OptionalInt.empty());
                var scan = Scan.apply(carry, sequences, List.of(state, state), lengths, 2, roles, body, reverse);
                var other = Scan.apply(carry, sequences, List.of(state, state), lengths, 2, roles, body, !reverse);
                var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs()); roots.addAll(scan.residuals());
                roots.add(other.finalCarry().getFirst());
                var source = new GraphCheckpoint.Source(List.of(lengths, seqShared, seq1, seq0, acc, initial), roots,
                        List.of(leaf("weight", GraphCheckpoint.Role.BUFFER, state), leaf("alias", GraphCheckpoint.Role.BUFFER, state)));
                var graph = GraphSource.capture(source).graph;
                var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                var loaded = GraphArtifactCodec.decode(encoded, LIMITS);
                assertEquals(graph, GraphSource.capture(loaded).graph);
                assertSame(loaded.leaves().getFirst().tensor(), loaded.leaves().getLast().tensor());
                assertArrayEquals(raw(type, 2), GraphValueStorage.encode(loaded.leaves().getFirst().tensor().descriptor(),
                        loaded.leaves().getFirst().tensor().hostStorage().orElseThrow().segment()));
                var scans = graph.nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
                assertEquals(2, scans.size());
                for (var node : scans) {
                    assertEquals(9, node.scanBody().orElseThrow().graph().inputs().size());
                    assertEquals(10, node.inputs().size());
                    assertEquals(roles, ((ScanAttrs) node.operation().attrs()).sequenceRowAxes());
                    assertEquals(node.inputs().get(0), node.inputs().get(1));
                    assertEquals(node.inputs().get(3), node.inputs().get(4));
                }
                var fresh = loaded.inputs().stream().map(t -> TensorFactory.create(t.descriptor())).toList();
                var rebound = GraphCheckpoint.rebind(loaded, fresh);
                assertEquals(graph, GraphSource.capture(new GraphCheckpoint.Source(fresh, rebound, loaded.leaves())).graph);
                assertSame(fresh.getFirst(), rebound.getFirst().provenance().orElseThrow().inputs().getLast());
                assertArrayEquals(schema(encoded), schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
            }
    }

    @Test void v5ContainsOldAndNewScansAndOldOnlyImagesResaveAtSmallestVersion() throws Exception {
        var old = GraphArtifactV4Test.fixture(0, new byte[0], false, 3);
        byte[] promoted = old.bytes().clone(); promoted[7] = '5'; ByteBuffer.wrap(promoted).putInt(8, 5);
        var loaded = GraphArtifactCodec.decode(image(promoted), LIMITS);
        var saved = GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS);
        assertEquals("graph-model.v4", saved.kind()); assertArrayEquals(old.bytes(), schema(saved));
        Tensor h = f(DataType.FLOAT32, 2, 2), x = f(DataType.FLOAT32, 2, 2);
        Tensor initial = f(DataType.FLOAT32, 2, 2), sequence = f(DataType.FLOAT32, 3, 2, 2), lengths = f(DataType.INT64, 2);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of());
        var all = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
        var length = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                List.of(OptionalInt.of(0)), body, true);
        var source = new GraphCheckpoint.Source(List.of(sequence, lengths, initial),
                List.of(all.finalCarry().getFirst(), length.finalCarry().getFirst()), List.of());
        var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        assertEquals("graph-model.v5", encoded.kind());
        assertEquals(GraphSource.capture(source).graph, GraphSource.capture(GraphArtifactCodec.decode(encoded, LIMITS)).graph);
        // V1/V2/V3 old records use exactly their original bytes after a v5 read as well.
        for (Tensor output : List.of(initial.neg(), initial.tanh(), initial.sub(initial))) {
            var flat = new GraphCheckpoint.Source(List.of(initial), List.of(output), List.of());
            var original = GraphArtifactCodec.encode(GraphSource.capture(flat), LIMITS);
            byte[] bytes = schema(original);
            if (!original.kind().equals("graph-model.v3")) {
                int descriptor = 16 + (2 + 4 + 16 + 1 + 9 + 16) + 12 + 6;
                byte[] expanded = new byte[bytes.length + 4];
                System.arraycopy(bytes, 0, expanded, 0, descriptor); ByteBuffer.wrap(expanded).putInt(descriptor, 1);
                System.arraycopy(bytes, descriptor, expanded, descriptor + 4, bytes.length - descriptor); bytes = expanded;
            }
            bytes[7] = '5'; ByteBuffer.wrap(bytes).putInt(8, 5);
            var resaved = GraphArtifactCodec.encode(GraphSource.capture(GraphArtifactCodec.decode(image(bytes), LIMITS)), LIMITS);
            assertEquals(original.kind(), resaved.kind()); assertArrayEquals(schema(original), schema(resaved));
        }
    }

    @Test void allSixExactAndOneBelowBoundsIncludeOptionalAxesBeforeStateAdmission() throws Exception {
        for (int axis : new int[]{-1, 0, 1}) {
            Tensor h = f(DataType.FLOAT64, 2, 2), x = f(DataType.FLOAT64, step(2, axis)), w = f(DataType.FLOAT64, 2);
            Tensor state = tensor(DataType.FLOAT64, Shape.of(2), false, true, raw(DataType.FLOAT64, 2));
            Tensor initial = f(DataType.FLOAT64, 2, 2), input = f(DataType.FLOAT64, sequence(3, 2, axis)), lengths = f(DataType.INT64, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(w)), List.of(h));
            var scan = Scan.apply(List.of(initial), List.of(input), List.of(state), lengths, 1,
                    List.of(axis < 0 ? OptionalInt.empty() : OptionalInt.of(axis)), body, false);
            var source = new GraphCheckpoint.Source(List.of(lengths, input, initial), scan.finalCarry(),
                    List.of(leaf("weight", GraphCheckpoint.Role.BUFFER, state), leaf("alias", GraphCheckpoint.Role.BUFFER, state)));
            var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
            int d = schema(encoded).length, b = 16, m = 130, file = 48 + m + d + b;
            var exact = new ArtifactLimits(file, m, 2, 14, d, d + b);
            Path target = directory.resolve("limits-" + axis); GraphCheckpoint.save(target, source, exact);
            byte[] prior = Files.readAllBytes(target); assertEquals(file, prior.length);
            assertEquals(GraphSource.capture(source).graph, GraphSource.capture(GraphCheckpoint.load(target, exact)).graph);
            state.clearHostStorage();
            for (var below : List.of(new ArtifactLimits(file - 1, m, 2, 14, d, d + b),
                    new ArtifactLimits(file, m - 1, 2, 14, d, d + b), new ArtifactLimits(file, m, 1, 14, d, d + b),
                    new ArtifactLimits(file, m, 2, 13, d, d + b), new ArtifactLimits(file, m, 2, 14, d - 1, d + b),
                    new ArtifactLimits(file, m, 2, 14, d, d + b - 1))) {
                Tensor before = f(DataType.FLOAT32);
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(target, source, below));
                assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(encoded, below));
                assertEquals(before.id().value() + 1, f(DataType.FLOAT32).id().value());
                assertArrayEquals(prior, Files.readAllBytes(target));
            }
        }
    }
}
