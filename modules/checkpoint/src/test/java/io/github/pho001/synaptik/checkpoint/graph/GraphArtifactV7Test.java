package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Independent FP8 wire admission and exact raw ownership, without execution or numeric conversion. */
final class GraphArtifactV7Test {
    @Test void everyFp8EncodingAndParameterAliasSurvivesColdImportCopyAndResave() throws Exception {
        byte[] raw = new byte[256];
        for (int i = 0; i < raw.length; i++) raw[i] = (byte) i;
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            int id = type == DataType.FLOAT8_E4M3 ? 8 : 9;
            var image = image(7, schema(7, id, true, 256, 256, 0), raw);
            var loaded = GraphArtifactCodec.decode(image, LIMITS);
            var state = loaded.leaves().getFirst().tensor();
            assertEquals(type, state.descriptor().dataType());
            assertTrue(state.descriptor().requiresGrad());
            assertSame(state, loaded.leaves().getLast().tensor());
            assertSame(state, loaded.outputs().getFirst());
            assertEquals(byte[].class, state.hostStorage().orElseThrow().segment().heapBase().orElseThrow().getClass());
            for (var acquired : List.of(loaded, GraphCheckpoint.copy(loaded))) {
                Tensor owned = acquired.leaves().getFirst().tensor();
                assertArrayEquals(raw, GraphValueStorage.encode(owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
                var resaved = GraphArtifactCodec.encode(GraphSource.capture(acquired), LIMITS);
                assertEquals("graph-model.v7", resaved.kind());
                assertArrayEquals(bytes(image.entries().getFirst()), bytes(resaved.entries().getFirst()));
                assertArrayEquals(raw, bytes(resaved.entries().get(1)));
            }
        }
    }

    @Test void ordinaryFp8SourcesRetainDeclaredOperationsAndNamedValues() throws Exception {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            Tensor state = tensor(type, Shape.of(3), true, false, raw(type, 3));
            Tensor input = prototype(type, Shape.of(3), true, false);
            for (Tensor output : List.of(state.add(input), state.mul(input), state.neg(), state.tanh())) {
                var source = new GraphCheckpoint.Source(List.of(input), List.of(output),
                        List.of(leaf("weight", GraphCheckpoint.Role.PARAMETER, state)));
                var graph = GraphSource.capture(source).graph;
                var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                var loaded = GraphArtifactCodec.decode(image, LIMITS);
                assertEquals(graph, GraphSource.capture(loaded).graph);
                assertEquals(type, loaded.outputs().getFirst().descriptor().dataType());
                assertArrayEquals(raw(type, 3), bytes(image.entries().get(1)));
                assertArrayEquals(bytes(image.entries().getFirst()),
                        bytes(GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS).entries().getFirst()));
            }
        }
    }

    @Test void fp8IdsCannotEnterOldDomainsAndInvalidLateMetadataAllocatesNoCandidateTensor() throws Exception {
        for (int id : new int[]{8, 9}) {
            for (int old = 1; old <= 6; old++)
                rejected(image(old, schema(old, id, true, 1, 1, 0), new byte[]{(byte) 0xff}));
            rejected(image(7, schema(7, id, false, 1, 1, 0), new byte[]{(byte) 0xff}));
            rejected(image(7, schema(7, id, true, 1, 1, 1), new byte[]{(byte) 0xff}));
            rejected(image(7, schema(7, id, true, 1, 2, 0), new byte[]{(byte) 0xff}));
        }
        rejected(image(7, schema(7, 10, true, 1, 1, 0), new byte[]{0}));
    }

    @Test void v7WithOnlyOldDtypesResavesToTheByteIdenticalSmallestOldDomain() throws Exception {
        byte[] raw = { (byte) 0xff, (byte) 0xc0, 0x12, 0x34 };
        var loaded = GraphArtifactCodec.decode(image(7, schema(7, 2, true, 1, 4, 0), raw), LIMITS);
        var saved = GraphArtifactCodec.encode(GraphSource.capture(loaded), LIMITS);
        assertEquals("graph-model.v1", saved.kind());
        assertArrayEquals(schema(1, 2, true, 1, 4, 0), bytes(saved.entries().getFirst()));
        assertArrayEquals(raw, bytes(saved.entries().get(1)));
    }

    @Test void containedFp8ScalarAndCastsSelectV7WithFloat32FormalsAndPublishedOutputs() throws Exception {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (byte bits : new byte[]{(byte) 0x80, (byte) 0xff}) for (long time : new long[]{0, 3}) {
                ScalarValue scalar = type == DataType.FLOAT8_E4M3
                        ? ScalarValue.float8E4m3Bits(bits) : ScalarValue.float8E5m2Bits(bits);
                Tensor h = prototype(DataType.FLOAT32, Shape.of(2), false, false);
                Tensor x = prototype(DataType.FLOAT32, Shape.of(2), false, false);
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h),
                        List.of(h.cast(type).add(scalar).cast(DataType.FLOAT32)));
                Tensor initial = prototype(DataType.FLOAT32, Shape.of(2), false, false);
                Tensor sequence = prototype(DataType.FLOAT32, Shape.of(time, 2), false, false);
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
                var source = new GraphCheckpoint.Source(List.of(initial, sequence), scan.outputs(), List.of());
                var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
                assertEquals("graph-model.v7", image.kind());
                var loaded = GraphArtifactCodec.decode(image, LIMITS);
                assertEquals(GraphSource.capture(source).graph, GraphSource.capture(loaded).graph);
                assertEquals(DataType.FLOAT32, loaded.outputs().getFirst().descriptor().dataType());
                byte[] invalid = bytes(image.entries().getFirst());
                ByteBuffer.wrap(invalid).putInt(invalid.length - 4, Integer.MAX_VALUE);
                rejected(new ArtifactEnvelope("graph-model.v7", List.of(new ArtifactEnvelope.Entry("schema", invalid))));
            }
    }

    /** Writes one named parameter with two aliases, using only the published field definition. */
    private static byte[] schema(int version, int type, boolean grad, long count, long length, int output) throws Exception {
        var bytes = new ByteArrayOutputStream();
        var out = new DataOutputStream(bytes);
        out.writeBytes("SYNGRF0" + version); out.writeInt(version);
        out.writeInt(0); out.writeInt(1);
        out.writeByte(type); out.writeByte(grad ? 1 : 0); out.writeInt(1); out.writeLong(count); out.writeByte(0);
        out.writeByte(1); out.writeLong(length);
        out.writeInt(2);
        for (String name : List.of("weight", "alias")) {
            out.writeInt(name.length()); out.writeBytes(name); out.writeInt(0);
        }
        out.writeInt(0); out.writeInt(1); out.writeInt(output);
        return bytes.toByteArray();
    }

    private static ArtifactEnvelope image(int version, byte[] schema, byte[] raw) {
        return new ArtifactEnvelope("graph-model.v" + version, List.of(new ArtifactEnvelope.Entry("schema", schema),
                new ArtifactEnvelope.Entry("state/00000000", raw)));
    }

    private static byte[] bytes(ArtifactEnvelope.Entry entry) {
        byte[] result = new byte[entry.size()]; entry.payload().get(result); return result;
    }

    private static void rejected(ArtifactEnvelope image) {
        Tensor before = prototype(DataType.FLOAT32, Shape.scalar(), false, true);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image, LIMITS));
        assertEquals(1, prototype(DataType.FLOAT32, Shape.scalar(), false, true).id().value() - before.id().value());
    }
}
