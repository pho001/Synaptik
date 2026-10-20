package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.recurrent.*;
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

/** Independently authored v3 fields, multi-port identities, closed decoding and size-before-host gates. */
final class GraphArtifactV3Test {
    @TempDir Path directory;

    /**
     * Supplies unresolved floating metadata without host storage.
     * @param shape nonnegative static extents
     * @return fresh gradient-eligible FLOAT32 leaf
     */
    private static Tensor f(long... shape) { return prototype(DataType.FLOAT32, Shape.of(shape), true, false); }
    /**
     * Supplies runtime-length metadata without any saved length values.
     * @param batch nonnegative vector extent
     * @return fresh non-gradient INT64 vector with unresolved layout and no storage
     */
    private static Tensor lengths(long batch) { return prototype(DataType.INT64, Shape.of(batch), false, false); }

    /**
     * Detaches schema bytes so malformed-fixture mutations cannot modify the envelope.
     * @param image envelope with schema as its first entry
     * @return independent exact schema bytes
     */
    private static byte[] schema(ArtifactEnvelope image) {
        byte[] bytes = new byte[image.entries().getFirst().size()];
        image.entries().getFirst().payload().get(bytes);
        return bytes;
    }

    /**
     * Wraps independently authored zero-state v3 bytes, including deliberately invalid bytes.
     * @param schema schema fixture, snapshotted by the envelope
     * @return detached zero-state v3 envelope
     */
    private static ArtifactEnvelope image(byte[] schema) {
        return new ArtifactEnvelope("graph-model.v3", List.of(new ArtifactEnvelope.Entry("schema", schema)));
    }

    /**
     * Writes an unresolved descriptor independently of production wire helpers or enum ordinals.
     * @param out test-owned big-endian destination
     * @param type explicit wire dtype ID
     * @param grad gradient flag
     * @param shape static logical extents
     * @throws Exception if test stream writing fails
     */
    private static void descriptor(DataOutputStream out, int type, boolean grad, long... shape) throws Exception {
        out.writeByte(type); out.writeByte(grad ? 1 : 0); out.writeInt(shape.length);
        for (long extent : shape) out.writeLong(extent);
        out.writeByte(0);
    }

    /**
     * Authors the common zero-state header independently of the encoder.
     * @param inputs ordered test signature, including unused slots
     * @param nodes declared producer count
     * @return exact header bytes through the node-count field
     * @throws Exception if test stream writing fails
     */
    private static byte[] header(List<Tensor> inputs, int nodes) throws Exception {
        var bytes = new ByteArrayOutputStream(); var out = new DataOutputStream(bytes);
        out.writeBytes("SYNGRF03"); out.writeInt(3); out.writeInt(inputs.size());
        for (Tensor input : inputs) descriptor(out, input.descriptor().dataType() == DataType.INT64 ? 5 : 2,
                input.descriptor().requiresGrad(), input.descriptor().shape().toLongArray());
        out.writeInt(0); out.writeInt(0); out.writeInt(nodes);
        return bytes.toByteArray();
    }

    /**
     * Authors a one-output, one-producer fixture without consulting production encoding.
     * @param inputs unique ordered prototypes, used once each by this fixture's producer
     * @param op explicit stable wire opcode
     * @param attrs exact independently authored attribute bytes
     * @param result expected logical output extents, with unresolved layout
     * @return complete v3 schema including Q and one root
     * @throws Exception if test stream writing fails
     */
    private static byte[] golden(List<Tensor> inputs, int op, byte[] attrs, long[] result) throws Exception {
        var bytes = new ByteArrayOutputStream(); bytes.write(header(inputs, 1));
        var out = new DataOutputStream(bytes); out.writeByte(1); out.writeByte(op);
        if (op == 16) out.writeInt(inputs.size());
        for (int i = 0; i < inputs.size(); i++) out.writeInt(i);
        out.write(attrs); out.writeInt(1); descriptor(out, 2, true, result);
        out.writeInt(1); out.writeInt(inputs.size());
        return bytes.toByteArray();
    }

    /**
     * Authors consecutive i32 fields for golden fixtures.
     * @param values exact signed field values in wire order
     * @return independent big-endian bytes
     */
    private static byte[] ints(int... values) {
        var buffer = ByteBuffer.allocate(values.length * 4);
        for (int value : values) buffer.putInt(value);
        return buffer.array();
    }

    /**
     * Authors consecutive i64 fields for golden fixtures.
     * @param values exact signed field values in wire order
     * @return independent big-endian bytes
     */
    private static byte[] longs(long... values) {
        var buffer = ByteBuffer.allocate(values.length * 8);
        for (long value : values) buffer.putLong(value);
        return buffer.array();
    }

    /**
     * Combines independently specified fields without mutating them.
     * @param arrays ordered field byte sequences
     * @return new concatenated bytes
     */
    private static byte[] join(byte[]... arrays) {
        var out = new ByteArrayOutputStream();
        for (byte[] bytes : arrays) out.writeBytes(bytes);
        return out.toByteArray();
    }

    /**
     * Checks golden bytes, closed Q/attributes/version admission and truncation rejection.
     * @param inputs unique ordered storage-free source prototypes
     * @param output supported single-producer expression over those prototypes
     * @param op independently selected wire opcode
     * @param attrs independently selected exact attribute bytes
     * @param result independently calculated output extents
     * @throws Exception if valid golden decoding or fixture writing fails
     */
    private static void check(List<Tensor> inputs, Tensor output, int op, byte[] attrs, long... result) throws Exception {
        var source = new GraphCheckpoint.Source(inputs, List.of(output), List.of());
        var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        byte[] expected = golden(inputs, op, attrs, result);
        assertEquals("graph-model.v3", encoded.kind());
        assertArrayEquals(expected, schema(encoded), "opcode " + op);
        var loaded = GraphArtifactCodec.decode(image(expected), LIMITS);
        assertArrayEquals(expected, schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
        for (Tensor input : loaded.inputs()) assertTrue(input.hostStorage().isEmpty());
        int attributeOffset = header(inputs, 1).length + 2 + inputs.size() * 4 + (op == 16 ? 4 : 0);
        int q = attributeOffset + attrs.length;
        for (int value : new int[]{-1, 0, 2, Integer.MAX_VALUE}) {
            byte[] bad = expected.clone(); ByteBuffer.wrap(bad).putInt(q, value);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bad), LIMITS));
        }
        if (attrs.length >= 4) {
            byte[] bad = expected.clone(); ByteBuffer.wrap(bad).putInt(attributeOffset, -1);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bad), LIMITS));
        }
        for (int length = 0; length < expected.length; length++) {
            byte[] truncated = Arrays.copyOf(expected, length);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(truncated), LIMITS));
        }
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(Arrays.copyOf(expected, expected.length + 1)), LIMITS));
        for (int version : new int[]{1, 2}) {
            byte[] bad = expected.clone(); bad[7] = (byte) ('0' + version); ByteBuffer.wrap(bad).putInt(8, version);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(new ArtifactEnvelope(
                    "graph-model.v" + version, List.of(new ArtifactEnvelope.Entry("schema", bad))), LIMITS));
        }
    }

    /**
     * Covers opcodes 9..19 with independent descriptor, attribute and cardinality fields.
     * @throws Exception if a supported fixture cannot roundtrip
     */
    @Test void everyAddedSingleOutputWireHasIndependentGoldenAttributesAndQ() throws Exception {
        Tensor a = f(2, 3), b = f(2, 3);
        check(List.of(a, b), a.sub(b), 9, new byte[0], 2, 3);
        check(List.of(a), a.reshape(3, 2), 10, join(ints(2), longs(3, 2)), 3, 2);
        check(List.of(a), a.expandDims(2), 11, ints(2), 2, 3, 1);
        Tensor singleton = f(2, 3, 1);
        check(List.of(singleton), singleton.squeeze(2), 12, ints(2), 2, 3);
        Tensor sequence = f(2, 2, 3);
        check(List.of(sequence), sequence.select(0, 1), 13, join(ints(0), longs(1)), 2, 3);
        Tensor indices = lengths(1);
        check(List.of(a, indices), a.gather(indices, 0), 14, ints(0), 1, 3);
        check(List.of(a), a.sliceByLength(new long[]{1}, new long[]{2}, new int[]{1}, new long[]{1}),
                15, join(ints(1), longs(1, 2), ints(1), longs(1)), 2, 2);
        Tensor other = f(1, 2, 3);
        check(List.of(sequence, other), Tensor.concat(0, sequence, other), 16, ints(0), 3, 2, 3);
        Tensor x = f(1, 1, 3, 3), w = f(2, 1, 2, 2), bias = f(2);
        check(List.of(x, w, bias), x.conv2d(w, bias, Conv2dAttrs.defaults()), 17,
                longs(1, 1, 0, 0, 1, 1, 1), 1, 2, 2, 2);
        Tensor volume = f(1, 1, 3, 3, 3), kernel = f(2, 1, 2, 2, 2);
        check(List.of(volume, kernel, bias), volume.conv3d(kernel, bias, Conv3dAttrs.defaults()), 18,
                longs(1, 1, 1, 0, 0, 0, 1, 1, 1, 1), 1, 2, 2, 2, 2);
        check(List.of(x), x.maxPool2d(new MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false)), 19,
                join(longs(2, 2, 2, 2, 0, 0, 1, 1), new byte[]{0}), 1, 1, 1, 1);
    }

    /**
     * Builds one canonical scan occurrence, not separate occurrences for its sibling ports.
     * @param in exact signature-ordered prototypes, optionally ending in bias
     * @param op explicit scan opcode 20, 21 or 22
     * @param direction one supported traversal direction
     * @return ordered retained canonical output ports
     */
    private static List<Tensor> scan(List<Tensor> in, int op, RecurrentDirection direction) {
        if (op == 22) {
            var r = in.size() == 6 ? RecurrentScan.lstm(in.get(0), in.get(1), in.get(2), in.get(3), in.get(4), in.get(5), direction)
                    : RecurrentScan.lstm(in.get(0), in.get(1), in.get(2), in.get(3), in.get(4), in.get(5), in.get(6), direction);
            return List.of(r.outputs(), r.finalHidden(), r.finalCell());
        }
        var r = op == 20 ? (in.size() == 5 ? RecurrentScan.rnn(in.get(0), in.get(1), in.get(2), in.get(3), in.get(4), direction)
                : RecurrentScan.rnn(in.get(0), in.get(1), in.get(2), in.get(3), in.get(4), in.get(5), direction))
                : (in.size() == 5 ? RecurrentScan.gru(in.get(0), in.get(1), in.get(2), in.get(3), in.get(4), direction)
                : RecurrentScan.gru(in.get(0), in.get(1), in.get(2), in.get(3), in.get(4), in.get(5), direction));
        return List.of(r.outputs(), r.finalHidden());
    }

    /**
     * Supplies signature metadata for a one-feature, one-hidden-unit scan.
     * @param op explicit scan opcode
     * @param bias whether to append packed bias
     * @param time nonnegative static time extent
     * @param batch nonnegative static batch extent
     * @return ordered unique provenance-free storage-free prototypes
     */
    private static List<Tensor> scanInputs(int op, boolean bias, long time, long batch) {
        int gates = op == 22 ? 4 : op == 21 ? 3 : 1;
        var inputs = new ArrayList<>(List.of(f(time, batch, 1), lengths(batch), f(batch, 1)));
        if (op == 22) inputs.add(f(batch, 1));
        inputs.add(f(gates, 1)); inputs.add(f(gates, 1));
        if (bias) inputs.add(f(gates));
        return inputs;
    }

    /**
     * Checks all scan signatures/directions, zero T/B and prefix-sum references after a multi-output node.
     * @throws Exception if golden encoding, decoding or file roundtrip fails
     */
    @Test void scanGoldenFieldsPrefixIdsAllOverloadsAndUnusedSiblingsRoundtrip() throws Exception {
        for (int op = 20; op <= 22; op++) for (boolean bias : List.of(false, true))
            for (var direction : RecurrentDirection.values()) for (long[] dims : List.of(
                    new long[]{2, 1}, new long[]{0, 1}, new long[]{2, 0})) {
                var inputs = scanInputs(op, bias, dims[0], dims[1]);
                var ports = scan(inputs, op, direction);
                var bytes = new ByteArrayOutputStream(); bytes.write(header(inputs, 2));
                var out = new DataOutputStream(bytes);
                out.writeByte(1); out.writeByte(op); out.writeInt(inputs.size());
                for (int i = 0; i < inputs.size(); i++) out.writeInt(i);
                out.writeByte(direction == RecurrentDirection.FORWARD ? 1 : 2); out.writeInt(ports.size());
                descriptor(out, 2, true, dims[0], dims[1], 1);
                for (int j = 1; j < ports.size(); j++) descriptor(out, 2, true, dims[1], 1);
                // A following producer consumes the same nonzero port twice. Its output ID uses sum(Q).
                out.writeByte(1); out.writeByte(1); out.writeInt(inputs.size() + 1); out.writeInt(inputs.size() + 1);
                out.writeInt(1); descriptor(out, 2, true, dims[1], 1);
                out.writeInt(3); out.writeInt(inputs.size() + ports.size() - 1);
                out.writeInt(inputs.size() + ports.size()); out.writeInt(inputs.size());
                byte[] expected = bytes.toByteArray();
                var source = new GraphCheckpoint.Source(inputs,
                        List.of(ports.getLast(), ports.get(1).add(ports.get(1)), ports.getFirst()), List.of());
                var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                assertArrayEquals(expected, schema(encoded));
                var loaded = GraphArtifactCodec.decode(image(expected), LIMITS);
                assertArrayEquals(expected, schema(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS)));
                var p = loaded.outputs().getFirst().provenance().orElseThrow().producer();
                assertSame(loaded.outputs().get(2), p.output(0));
                assertSame(p.output(1), loaded.outputs().get(1).provenance().orElseThrow().inputs().getFirst());
                var onlyLast = new GraphCheckpoint.Source(inputs, List.of(ports.getLast()), List.of());
                Path file = directory.resolve("scan"); GraphCheckpoint.save(file, onlyLast, LIMITS);
                var cold = GraphCheckpoint.load(file, LIMITS);
                assertEquals(ports.size(), cold.outputs().getFirst().provenance().orElseThrow().producer().outputCount());
                assertEquals(1, GraphSource.capture(cold).graph.nodes().size());
                for (Tensor input : cold.inputs()) assertTrue(input.hostStorage().isEmpty());
            }
    }

    /**
     * Rejects malformed scan records and dead producers while allowing unused sibling descriptors.
     * @throws Exception if fixture construction fails
     */
    @Test void malformedScanCountsDirectionsQDescriptorsReferencesTuplesAndEveryTruncationReject() throws Exception {
        var inputs = scanInputs(22, false, 2, 1);
        var ports = scan(inputs, 22, RecurrentDirection.REVERSE);
        byte[] good = schema(GraphArtifactCodec.encode(GraphSource.capture(new GraphCheckpoint.Source(
                inputs, List.of(ports.getLast()), List.of())), LIMITS));
        int node = header(inputs, 1).length;
        int count = node + 2, direction = count + 4 + inputs.size() * 4, q = direction + 1;
        for (int position : new int[]{12, count, q, q + 6, good.length - 8, good.length - 4})
            for (int value : new int[]{-1, 0, Integer.MAX_VALUE}) {
                byte[] bad = good.clone(); ByteBuffer.wrap(bad).putInt(position, value);
                assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bad), LIMITS));
            }
        for (int value : new int[]{0, 3, 255}) {
            byte[] bad = good.clone(); bad[direction] = (byte) value;
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bad), LIMITS));
        }
        for (int value : new int[]{0, 23, 255}) {
            byte[] bad = good.clone(); bad[node + 1] = (byte) value;
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bad), LIMITS));
        }
        for (int ref : new int[]{-1, inputs.size(), inputs.size() + 2, Integer.MAX_VALUE}) {
            byte[] bad = good.clone(); ByteBuffer.wrap(bad).putInt(count + 4, ref);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bad), LIMITS));
        }
        for (int length = 0; length < good.length; length++) {
            byte[] truncated = Arrays.copyOf(good, length);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(truncated), LIMITS));
        }
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(Arrays.copyOf(good, good.length + 1)), LIMITS));
        byte[] dead = good.clone(); ByteBuffer.wrap(dead).putInt(dead.length - 4, 0);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(dead), LIMITS));
        byte[] duplicate = Arrays.copyOf(good, good.length + 4);
        ByteBuffer.wrap(duplicate).putInt(good.length - 8, 2).putInt(good.length, inputs.size() + 2);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(duplicate), LIMITS));
        for (String kind : List.of("graph-model.v1", "graph-model.v2", "graph-model.v4"))
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(new ArtifactEnvelope(kind, image(good).entries()), LIMITS));
        for (int version : new int[]{1, 2, 4}) {
            byte[] bad = good.clone(); ByteBuffer.wrap(bad).putInt(8, version);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(bad), LIMITS));
        }
    }

    /**
     * Verifies smallest-version selection and all six exact ceilings before absent-state admission.
     * @throws Exception if a valid exact-limit fixture cannot be published or loaded
     */
    @Test void olderKindsInV3DowngradeAndAllSixBoundsPrecedeHostAdmissionPreservingTarget() throws Exception {
        Tensor x = f(2, 3), y = f(2, 3);
        byte[] sub = golden(List.of(x, y), 9, new byte[0], new long[]{2, 3});
        int op = header(List.of(x, y), 1).length + 1;
        sub[op] = 1;
        assertEquals("graph-model.v1", GraphArtifactCodec.encode(GraphSource.capture(GraphArtifactCodec.decode(image(sub), LIMITS)), LIMITS).kind());
        byte[] relu = golden(List.of(x), 6, new byte[0], new long[]{2, 3});
        assertEquals("graph-model.v2", GraphArtifactCodec.encode(GraphSource.capture(GraphArtifactCodec.decode(image(relu), LIMITS)), LIMITS).kind());
        Tensor state = tensor(DataType.FLOAT32, Shape.of(2, 3), true, false, raw(DataType.FLOAT32, 6));
        var source = new GraphCheckpoint.Source(List.of(x), List.of(x.sub(state)), List.of(
                leaf("a", GraphCheckpoint.Role.PARAMETER, state), leaf("b", GraphCheckpoint.Role.PARAMETER, state)));
        var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        int d = schema(encoded).length, b = 24, m = 130, fileSize = 48 + m + d + b;
        var exact = new ArtifactLimits(fileSize, m, 2, 14, d, d + b);
        Path file = directory.resolve("bounds"); GraphCheckpoint.save(file, source, exact);
        byte[] before = Files.readAllBytes(file);
        var loaded = GraphCheckpoint.load(file, exact);
        assertSame(loaded.leaves().getFirst().tensor(), loaded.leaves().getLast().tensor());
        assertArrayEquals(raw(DataType.FLOAT32, 6), GraphValueStorage.encode(loaded.leaves().getFirst().tensor().descriptor(),
                loaded.leaves().getFirst().tensor().hostStorage().orElseThrow().segment()));
        state.clearHostStorage();
        for (var limit : List.of(new ArtifactLimits(fileSize - 1, m, 2, 14, d, d + b),
                new ArtifactLimits(fileSize, m - 1, 2, 14, d, d + b), new ArtifactLimits(fileSize, m, 1, 14, d, d + b),
                new ArtifactLimits(fileSize, m, 2, 13, d, d + b), new ArtifactLimits(fileSize, m, 2, 14, d - 1, d + b),
                new ArtifactLimits(fileSize, m, 2, 14, d, d + b - 1))) {
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(file, source, limit));
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(encoded, limit));
            assertArrayEquals(before, Files.readAllBytes(file));
            assertTrue(state.hostStorage().isEmpty());
        }
    }

    /**
     * Checks multi-producer identity, aliases and opaque rebinds through both acquisition paths.
     * @throws Exception if copy or file roundtrip fails
     */
    @Test void copyRebindAndSaveKeepDistinctProducersStateAliasesRepeatedPortsAndPassThroughs() throws Exception {
        Tensor x = f(2, 1, 1), n = lengths(1), h = f(1, 1);
        Tensor state = tensor(DataType.FLOAT32, Shape.of(1, 1), true, false, raw(DataType.FLOAT32, 1));
        var a = RecurrentScan.rnn(x, n, h, state, state, RecurrentDirection.FORWARD);
        var b = RecurrentScan.rnn(x, n, h, state, state, RecurrentDirection.FORWARD);
        var source = new GraphCheckpoint.Source(List.of(x, n, h), List.of(a.finalHidden(), b.outputs(),
                a.finalHidden().add(a.finalHidden()), x, h, state), List.of(
                leaf("a", GraphCheckpoint.Role.PARAMETER, state), leaf("b", GraphCheckpoint.Role.PARAMETER, state)));
        Path file = directory.resolve("identities"); GraphCheckpoint.save(file, source, LIMITS);
        byte[] original = Files.readAllBytes(file);
        for (var acquired : List.of(GraphCheckpoint.copy(source), GraphCheckpoint.load(file, LIMITS))) {
            Tensor owned = acquired.leaves().getFirst().tensor();
            assertNotSame(state, owned); assertSame(owned, acquired.leaves().getLast().tensor());
            assertSame(owned, acquired.outputs().getLast());
            var first = acquired.outputs().getFirst().provenance().orElseThrow().producer();
            var second = acquired.outputs().get(1).provenance().orElseThrow().producer();
            assertNotSame(first, second);
            assertSame(first.inputs().get(3), first.inputs().get(4));
            assertSame(owned, first.inputs().get(3));
            assertSame(acquired.outputs().getFirst(), acquired.outputs().get(2).provenance().orElseThrow().inputs().getFirst());
            Tensor freshX = f(2, 1, 1).abs(), freshH = f(1, 1).abs(), freshN = lengths(1);
            var rebound = GraphCheckpoint.rebind(acquired, List.of(freshX, freshN, freshH));
            assertSame(freshX, rebound.get(3)); assertSame(freshH, rebound.get(4));
            assertSame(freshN, rebound.getFirst().provenance().orElseThrow().inputs().get(1));
            GraphCheckpoint.save(file, acquired, LIMITS); assertArrayEquals(original, Files.readAllBytes(file));
        }
        Tensor sequence = f(1, 1, 1);
        var repeated = new GraphCheckpoint.Source(List.of(sequence), List.of(Tensor.concat(0, sequence, sequence)), List.of());
        var copied = GraphCheckpoint.copy(repeated);
        var operands = copied.outputs().getFirst().provenance().orElseThrow().inputs();
        assertSame(operands.get(0), operands.get(1));
    }

    /**
     * Exercises producer postorder beyond practical recursive traversal depth after a nonzero root port.
     * @throws Exception if the deep source cannot encode or decode within its explicit limits
     */
    @Test void producerTopologyRemainsIterativeAtLargeDepthAfterNonzeroScanPort() throws Exception {
        var inputs = scanInputs(20, false, 2, 1);
        Tensor output = scan(inputs, 20, RecurrentDirection.FORWARD).get(1);
        for (int j = 0; j < 2000; j++) output = output.add(inputs.get(2));
        var source = new GraphCheckpoint.Source(inputs, List.of(output), List.of());
        var encoded = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        var decoded = GraphArtifactCodec.decode(encoded, LIMITS);
        assertEquals(2001, GraphSource.capture(decoded).graph.nodes().size());
        assertArrayEquals(schema(encoded), schema(GraphArtifactCodec.encode(GraphSource.capture(decoded), LIMITS)));
    }
}
