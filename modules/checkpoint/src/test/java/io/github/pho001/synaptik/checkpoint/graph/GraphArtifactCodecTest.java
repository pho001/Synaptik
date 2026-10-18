package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Fixed wire IDs, preallocation bounds and malformed topology tests with explicit encoded fixtures. */
final class GraphArtifactCodecTest {
    @TempDir Path directory;

    @Test
    void scalarWireHasExactFixedFieldsAndIndependentGoldenBytes() throws Exception {
        ArtifactEnvelope image = encoded();
        assertEquals("graph-model.v1", image.kind());
        byte[] golden = java.util.HexFormat.of().parseHex(
                "53594e4752463031" + "00000001" + "00000001" + "02010000000000"
                + "00000001" + "02010000000000" + "01" + "0000000000000004"
                + "00000001" + "00000001" + "78" + "00000000"
                + "00000001" + "0101" + "00000000" + "00000001" + "02010000000000"
                + "00000001" + "00000002");
        assertArrayEquals(golden, schema(image));
        assertEquals(85, golden.length);
        assertEquals("state/00000000", image.entries().get(1).key());
        assertArrayEquals(new byte[] {0x3f, (byte) 0x80, 0, 0}, payload(image));
        var source = GraphArtifactCodec.decode(image, LIMITS);
        assertEquals(GraphCheckpoint.Role.PARAMETER, source.leaves().getFirst().role());
        assertArrayEquals(golden, schema(GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS)));
    }

    @Test
    void allSixLimitsAcceptExactAndRejectOneBelowBeforeHostAdmissionOrPublication() throws Exception {
        var exact = new ArtifactLimits(267, 130, 2, 14, 85, 89);
        ArtifactEnvelope image = GraphArtifactCodec.encode(GraphSource.capture(scalar()), exact);
        Path file = directory.resolve("exact");
        GraphCheckpoint.save(file, scalar(), exact);
        assertEquals(267, Files.size(file));
        assertEquals(1, GraphCheckpoint.load(file, exact).leaves().size());
        var below = List.of(new ArtifactLimits(266, 130, 2, 14, 85, 89),
                new ArtifactLimits(267, 129, 2, 14, 85, 89),
                new ArtifactLimits(267, 130, 1, 14, 85, 89),
                new ArtifactLimits(267, 130, 2, 13, 85, 89),
                new ArtifactLimits(267, 130, 2, 14, 84, 89),
                new ArtifactLimits(267, 130, 2, 14, 85, 88));
        for (ArtifactLimits limit : below) {
            var source = scalar(); source.leaves().getFirst().tensor().clearHostStorage();
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(file, source, limit));
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image, limit));
            assertThrows(ArtifactFormatException.class, () -> GraphCheckpoint.load(file, limit));
            assertEquals(267, Files.size(file));
        }
    }

    @Test
    void malformedIdsFlagsCountsRanksLengthsReferencesAndUtf8FailWithCauses() throws Exception {
        byte[] original = schema(encoded());
        for (int position : List.of(16, 27, 70, 34, 60, 61)) {
            byte[] bytes = original.clone(); bytes[position] = (byte) 0xff;
            invalid(bytes);
        }
        for (int position : List.of(17, 22, 28, 33, 71, 76)) {
            byte[] bytes = original.clone(); bytes[position] = 2;
            invalid(bytes);
        }
        for (int position : List.of(12, 18, 23, 29, 43, 47, 52, 56, 62, 66, 72, 77, 81)) {
            for (int value : List.of(-1, Integer.MAX_VALUE)) {
                byte[] bytes = original.clone(); ByteBuffer.wrap(bytes).putInt(position, value);
                invalid(bytes);
            }
        }
        for (int value : List.of(0, 2, 100)) {
            byte[] bytes = original.clone(); ByteBuffer.wrap(bytes).putInt(8, value); invalid(bytes);
        }
        byte[] badMagic = original.clone(); badMagic[0] = 0; invalid(badMagic);
        byte[] badText = original.clone(); badText[51] = (byte) 0x80; invalid(badText);
        for (char bad : new char[] {'.', ' '}) {
            byte[] badName = original.clone(); badName[51] = (byte) bad; invalid(badName);
        }
        for (long length : new long[] {-1, 0, 3, 5, Long.MAX_VALUE}) {
            byte[] bytes = original.clone(); ByteBuffer.wrap(bytes).putLong(35, length); invalid(bytes);
        }
        byte[] forward = original.clone(); ByteBuffer.wrap(forward).putInt(62, 2); invalid(forward);
        byte[] dead = original.clone(); ByteBuffer.wrap(dead).putInt(81, 0); invalid(dead);
        byte[] incompatible = original.clone(); incompatible[70] = 1; invalid(incompatible);
        byte[] untrainable = original.clone(); untrainable[28] = 0; invalid(untrainable);
    }

    @Test
    void everyTruncationAndTrailingSchemaByteRejectsWithoutPartialResults() throws Exception {
        byte[] original = schema(encoded());
        for (int length = 0; length < original.length; length++) invalid(Arrays.copyOf(original, length));
        invalid(Arrays.copyOf(original, original.length + 1));
        byte[] duplicateOutput = Arrays.copyOf(original, original.length + 4);
        ByteBuffer.wrap(duplicateOutput).putInt(77, 2).putInt(85, 2);
        invalid(duplicateOutput);
        byte[] emptyOutput = Arrays.copyOf(original, 81); ByteBuffer.wrap(emptyOutput).putInt(77, 0);
        invalid(emptyOutput);
    }

    @Test
    void completeKeysPayloadLengthsAndBoolBytesAreValidatedIndependently() throws Exception {
        ArtifactEnvelope good = encoded();
        for (ArtifactEnvelope image : List.of(
                new ArtifactEnvelope("model.v1", good.entries()),
                new ArtifactEnvelope("graph-model.v1", List.of()),
                new ArtifactEnvelope("graph-model.v1", List.of(new ArtifactEnvelope.Entry("other", schema(good)))),
                new ArtifactEnvelope("graph-model.v1", List.of(good.entries().getFirst())),
                image(schema(good), new byte[3]), image(schema(good), new byte[5]),
                new ArtifactEnvelope("graph-model.v1", List.of(good.entries().getFirst(),
                        new ArtifactEnvelope.Entry("state/00000001", payload(good)))),
                new ArtifactEnvelope("graph-model.v1", List.of(good.entries().getFirst(), good.entries().get(1),
                        new ArtifactEnvelope.Entry("unused", new byte[0])))))
            assertNotNull(assertThrows(ArtifactFormatException.class,
                    () -> GraphArtifactCodec.decode(image, LIMITS)).getCause());
        Tensor bool = tensor(DataType.BOOL, Shape.of(1), false, false, new byte[] {1});
        var boolImage = GraphArtifactCodec.encode(GraphSource.capture(new GraphCheckpoint.Source(List.of(),
                List.of(bool), List.of(leaf("flag", GraphCheckpoint.Role.CONSTANT, bool)))), LIMITS);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image(schema(boolImage), new byte[] {2}), LIMITS));
    }

    @Test
    void aliasesMustBeCanonicalReferencedAndPreserveEqualIndependentIdentities() throws Exception {
        var source = scalar(); Tensor state = source.leaves().getFirst().tensor();
        var aliased = new GraphCheckpoint.Source(source.inputs(), source.outputs(), List.of(
                leaf("x", GraphCheckpoint.Role.PARAMETER, state), leaf("y", GraphCheckpoint.Role.PARAMETER, state)));
        var good = GraphArtifactCodec.encode(GraphSource.capture(aliased), LIMITS);
        var loaded = GraphArtifactCodec.decode(good, LIMITS);
        assertSame(loaded.leaves().get(0).tensor(), loaded.leaves().get(1).tensor());
        byte[] bytes = schema(good);
        // First registration must introduce state zero; second registration cannot introduce state one.
        byte[] first = bytes.clone(); ByteBuffer.wrap(first).putInt(52, 1); invalid(first);
        byte[] second = bytes.clone(); ByteBuffer.wrap(second).putInt(61, 1); invalid(second);
        byte[] duplicateName = bytes.clone(); duplicateName[60] = 'x'; invalid(duplicateName);
        Tensor other = tensor(DataType.FLOAT32, Shape.of(), true, false, payload(encoded()));
        var independent = new GraphCheckpoint.Source(List.of(), List.of(state.add(other)), List.of(
                leaf("x", GraphCheckpoint.Role.PARAMETER, state), leaf("y", GraphCheckpoint.Role.PARAMETER, other)));
        var two = GraphArtifactCodec.encode(GraphSource.capture(independent), LIMITS);
        var twoLoaded = GraphArtifactCodec.decode(two, LIMITS);
        assertNotSame(twoLoaded.leaves().get(0).tensor(), twoLoaded.leaves().get(1).tensor());
        byte[] swapped = schema(two);
        // Second state's descriptor/role/length is 16 bytes; first registration index shifts by 16.
        ByteBuffer.wrap(swapped).putInt(68, 1);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(
                new ArtifactEnvelope(two.kind(), List.of(new ArtifactEnvelope.Entry("schema", swapped),
                        two.entries().get(1), two.entries().get(2))), LIMITS));
    }

    @Test
    void topologyOrderDeadNodesAndPermuteAttributesFailBeforePublication() throws Exception {
        Tensor input = prototype(DataType.FLOAT32, Shape.of(), true, false);
        // Two independent occurrences; reversing roots without changing stored node order is noncanonical.
        var source = new GraphCheckpoint.Source(List.of(input), List.of(input.neg(), input.mul(input)), List.of());
        var good = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        byte[] swapped = schema(good);
        int tail = swapped.length - 8;
        ByteBuffer buffer = ByteBuffer.wrap(swapped);
        int first = buffer.getInt(tail), second = buffer.getInt(tail + 4);
        buffer.putInt(tail, second).putInt(tail + 4, first);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(imageNoState(swapped), LIMITS));
        Tensor matrix = prototype(DataType.FLOAT32, Shape.of(2, 3), true, false);
        var transpose = new GraphCheckpoint.Source(List.of(matrix), List.of(matrix.permute(1, 0)), List.of());
        var transposed = GraphArtifactCodec.encode(GraphSource.capture(transpose), LIMITS);
        byte[] schema = schema(transposed);
        // Header + input descriptor (23 bytes) + three collection counts = node phase at 51.
        assertEquals(5, Byte.toUnsignedInt(schema[52]));
        for (int offset : List.of(57, 61, 65)) {
            byte[] bad = schema.clone(); ByteBuffer.wrap(bad).putInt(offset, 9);
            assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(imageNoState(bad), LIMITS));
        }
    }

    @Test
    void ranksBeyondBackendCeilingsAndGeometryOverflowHaveCorrectOwnership() throws Exception {
        long[] extents = new long[20]; Arrays.fill(extents, 1);
        Tensor input = prototype(DataType.FLOAT32, Shape.of(extents), true, false);
        var source = new GraphCheckpoint.Source(List.of(input), List.of(input.neg()), List.of());
        var good = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        assertEquals(20, GraphArtifactCodec.decode(good, LIMITS).inputs().getFirst().descriptor().shape().rank());
        byte[] bad = schema(good); ByteBuffer.wrap(bad).putLong(22, Long.MAX_VALUE).putLong(30, 2);
        var failure = assertThrows(ArtifactFormatException.class,
                () -> GraphArtifactCodec.decode(imageNoState(bad), LIMITS));
        assertInstanceOf(ArithmeticException.class, failure.getCause());
    }

    /**
     * Creates the scalar golden fixture with unresolved layouts and one FLOAT32 parameter.
     * @return ordinary input-plus-weight ADD source
     */
    private static GraphCheckpoint.Source scalar() {
        Tensor input = prototype(DataType.FLOAT32, Shape.of(), true, false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(), true, false, new byte[] {0x3f, (byte) 0x80, 0, 0});
        return new GraphCheckpoint.Source(List.of(input), List.of(input.add(weight)),
                List.of(leaf("x", GraphCheckpoint.Role.PARAMETER, weight)));
    }

    /** @return independently validated encoded scalar fixture */
    private static ArtifactEnvelope encoded() { return GraphArtifactCodec.encode(GraphSource.capture(scalar()), LIMITS); }

    /**
     * @param image immutable test image
     * @return detached schema bytes for mutation tests
     */
    private static byte[] schema(ArtifactEnvelope image) {
        byte[] result = new byte[image.entries().getFirst().size()]; image.entries().getFirst().payload().get(result); return result;
    }

    /**
     * @param image immutable one-state fixture
     * @return detached state bytes
     */
    private static byte[] payload(ArtifactEnvelope image) {
        byte[] result = new byte[image.entries().get(1).size()]; image.entries().get(1).payload().get(result); return result;
    }

    /**
     * @param schema mutated scalar schema
     * @throws Exception if fixture construction unexpectedly fails
     */
    private static void invalid(byte[] schema) throws Exception {
        var failure = assertThrows(ArtifactFormatException.class,
                () -> GraphArtifactCodec.decode(image(schema, payload(encoded())), LIMITS));
        assertNotNull(failure.getCause());
    }

    /**
     * @param schema caller-owned raw schema fixture
     * @param payload caller-owned exact state bits
     * @return immutable image with one fixed state key
     */
    private static ArtifactEnvelope image(byte[] schema, byte[] payload) {
        return new ArtifactEnvelope("graph-model.v1", List.of(new ArtifactEnvelope.Entry("schema", schema),
                new ArtifactEnvelope.Entry("state/00000000", payload)));
    }

    /**
     * @param schema caller-owned raw zero-state schema
     * @return immutable image without state payloads
     */
    private static ArtifactEnvelope imageNoState(byte[] schema) {
        return new ArtifactEnvelope("graph-model.v1", List.of(new ArtifactEnvelope.Entry("schema", schema)));
    }
}
