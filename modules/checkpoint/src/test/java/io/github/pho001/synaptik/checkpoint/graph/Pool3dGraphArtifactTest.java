package io.github.pho001.synaptik.checkpoint.graph;

import static io.github.pho001.synaptik.checkpoint.graph.GraphCheckpointTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Append-only Pool3d wire, exact geometry, scoped roles and fail-before-allocation regressions. */
final class Pool3dGraphArtifactTest {
    private static final MaxPool3dAttrs MAX = new MaxPool3dAttrs(
            2, 3, 2, 1, 2, 1, 1, 0, 2, 2, 1, 2, true);
    private static final AveragePool3dAttrs AVG = new AveragePool3dAttrs(
            3, 2, 2, 2, 1, 3, 0, 2, 1, 1, 2, 1, false);
    @TempDir Path directory;

    @Test void exactGeometryRolesAliasesAndRawStateSurviveCopyRebindAndFileRoundTrip() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor input = prototype(type, Shape.of(2, 1, 5, 6, 7), false, true);
            int width = type == DataType.FLOAT64 ? 8 : type == DataType.FLOAT32 ? 4 : 2;
            byte[] raw = new byte[width]; raw[0] = (byte) 0x80; // Exact negative zero in wire order.
            Tensor state = tensor(type, Shape.scalar(), true, true, raw);
            Tensor buffer = tensor(type, Shape.scalar(), false, true, raw);
            Tensor constant = tensor(type, Shape.scalar(), true, true, raw);
            Tensor base = input.add(state);
            var source = new GraphCheckpoint.Source(List.of(input),
                    List.of(base.maxPool3d(MAX), base.add(buffer).add(constant).averagePool3d(AVG)),
                    List.of(leaf("weight", GraphCheckpoint.Role.PARAMETER, state),
                            leaf("tied", GraphCheckpoint.Role.PARAMETER, state),
                            leaf("buffer", GraphCheckpoint.Role.BUFFER, buffer),
                            leaf("constant", GraphCheckpoint.Role.CONSTANT, constant)));
            var captured = GraphSource.capture(source);
            var image = GraphArtifactCodec.encode(captured, LIMITS);
            assertEquals("graph-model.v6", image.kind());
            Path file = directory.resolve(type.name());
            GraphCheckpoint.save(file, source, LIMITS);
            for (var restored : List.of(GraphCheckpoint.copy(source), GraphCheckpoint.load(file, LIMITS))) {
                assertEquals(captured.graph, GraphSource.capture(restored).graph);
                assertEquals(source.leaves().stream().map(GraphCheckpoint.Leaf::role).toList(),
                        restored.leaves().stream().map(GraphCheckpoint.Leaf::role).toList());
                assertSame(restored.leaves().getFirst().tensor(), restored.leaves().get(1).tensor());
                assertNotSame(state, restored.leaves().getFirst().tensor());
                var encoded = GraphArtifactCodec.encode(GraphSource.capture(restored), LIMITS);
                for (int ordinal = 1; ordinal <= 3; ordinal++) assertArrayEquals(raw, payload(encoded, ordinal));
                Tensor replacement = prototype(type, input.descriptor().shape(), false, true).abs().contiguous();
                var rebound = GraphCheckpoint.rebind(restored, List.of(replacement));
                assertEquals(MAX, rebound.getFirst().provenance().orElseThrow().operation().attrs());
                assertEquals(AVG, rebound.getLast().provenance().orElseThrow().operation().attrs());
                assertSame(replacement, rebound.getFirst().provenance().orElseThrow().inputs().getFirst()
                        .provenance().orElseThrow().inputs().getFirst());
            }
            sameImage(image, ArtifactFile.read(file, LIMITS));
        }
    }

    @Test void bodyOnlyPool3dSelectsV6AndPreservesAllValidAndLengthProtocols() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean length : List.of(false, true)) for (boolean reverse : List.of(false, true))
                for (long time : List.of(0L, 2L)) {
                    Tensor h = prototype(type, Shape.of(2, 1, 1, 1, 1), false, true);
                    Tensor x = prototype(type, Shape.of(2, 1, 2, 2, 2), false, true);
                    var max = new MaxPool3dAttrs(2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, false);
                    var avg = new AveragePool3dAttrs(2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, false);
                    Tensor maximum = x.maxPool3d(max), average = x.averagePool3d(avg);
                    var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(average)),
                            List.of(maximum, average, maximum));
                    Tensor initial = prototype(type, h.descriptor().shape(), false, true);
                    Tensor sequence = prototype(type, Shape.of(time, 2, 1, 2, 2, 2), false, true);
                    Tensor lengths = prototype(DataType.INT64, Shape.of(2), false, true);
                    var result = length ? Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths,
                            1, List.of(OptionalInt.of(0)), body, reverse)
                            : Scan.apply(List.of(initial), List.of(sequence), List.of(), body, reverse);
                    var inputs = new ArrayList<>(List.of(initial, sequence));
                    if (length) inputs.add(lengths);
                    var outputs = new ArrayList<>(result.finalCarry()); outputs.addAll(result.outputs());
                    var source = new GraphCheckpoint.Source(inputs, outputs, List.of());
                    var captured = GraphSource.capture(source);
                    var image = GraphArtifactCodec.encode(captured, LIMITS);
                    assertEquals("graph-model.v6", image.kind());
                    var loaded = GraphArtifactCodec.decode(image, LIMITS);
                    assertEquals(captured.graph, GraphSource.capture(loaded).graph);
                    var local = GraphSource.capture(loaded).graph.nodes().getFirst().scanBody().orElseThrow();
                    assertEquals(local.emissions().getFirst(), local.emissions().getLast());
                    // Downgrading the complete tuple must not smuggle a new opcode into old bodies.
                    byte[] old = payload(image, 0); old[7] = (byte) (length ? '5' : '4');
                    ByteBuffer.wrap(old).putInt(8, length ? 5 : 4);
                    rejectsWithoutTensorAllocation(versioned(length ? "graph-model.v5" : "graph-model.v4", old));
                }
    }

    @Test void closedVersionOpcodeGeometryAndTruncationRejectBeforeAnyTensorAllocation() throws Exception {
        for (boolean average : List.of(false, true)) {
            var source = flat(average);
            var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
            byte[] wire = payload(image, 0);
            // Header=12, counts=16, unresolved rank-five input descriptor=47: phase at 75.
            assertEquals(average ? 35 : 34, Byte.toUnsignedInt(wire[76]));
            ByteBuffer fields = ByteBuffer.wrap(wire);
            long[] geometry = average ? new long[]{3, 2, 2, 2, 1, 3, 0, 2, 1, 1, 2, 1}
                    : new long[]{2, 3, 2, 1, 2, 1, 1, 0, 2, 2, 1, 2};
            for (int i = 0; i < geometry.length; i++) assertEquals(geometry[i], fields.getLong(81 + 8 * i));
            assertEquals(average ? 0 : 1, Byte.toUnsignedInt(wire[177]));
            for (int old = 1; old <= 5; old++) {
                byte[] downgraded = wire.clone(); downgraded[7] = (byte) ('0' + old);
                ByteBuffer.wrap(downgraded).putInt(8, old);
                rejectsWithoutTensorAllocation(versioned("graph-model.v" + old, downgraded));
                rejectsWithoutTensorAllocation(versioned("graph-model.v" + old, wire));
            }
            for (int opcode : List.of(0, 24, 32, 36, 255)) {
                byte[] bad = wire.clone(); bad[76] = (byte) opcode;
                rejectsWithoutTensorAllocation(versioned(image.kind(), bad));
            }
            for (int field : List.of(0, 3, 6, 9)) {
                byte[] bad = wire.clone(); ByteBuffer.wrap(bad).putLong(81 + field * 8, field == 6 ? -1 : 0);
                rejectsWithoutTensorAllocation(versioned(image.kind(), bad));
            }
            byte[] flag = wire.clone(); flag[177] = 2;
            rejectsWithoutTensorAllocation(versioned(image.kind(), flag));
            byte[] overflow = wire.clone(); ByteBuffer.wrap(overflow).putLong(81, Long.MAX_VALUE);
            rejectsWithoutTensorAllocation(versioned(image.kind(), overflow));
            byte[] wrongDescriptor = wire.clone(); ByteBuffer.wrap(wrongDescriptor).putLong(188, 3);
            rejectsWithoutTensorAllocation(versioned(image.kind(), wrongDescriptor));
            for (int length = 0; length < wire.length; length++)
                rejectsWithoutTensorAllocation(versioned(image.kind(), Arrays.copyOf(wire, length)));
            rejectsWithoutTensorAllocation(versioned(image.kind(), Arrays.copyOf(wire, wire.length + 1)));
        }
    }

    @Test void v6MarkerDoesNotForceOldSourcesToStayV6AndBoundsPrecedeStorageAndFiles() throws Exception {
        Tensor x = prototype(DataType.FLOAT32, Shape.of(1), false, false);
        var original = GraphArtifactCodec.encode(GraphSource.capture(
                new GraphCheckpoint.Source(List.of(x), List.of(x.neg()), List.of())), LIMITS);
        // V3+ stores explicit result counts; retain old bytes only when re-saving old sources.
        Tensor volume = prototype(DataType.FLOAT32, Shape.of(1, 1, 5, 6, 7), false, true);
        var reshape = new GraphCheckpoint.Source(List.of(volume), List.of(volume.reshape(1, 210)), List.of());
        var v3 = GraphArtifactCodec.encode(GraphSource.capture(reshape), LIMITS);
        byte[] upgraded = payload(v3, 0); upgraded[7] = '6'; ByteBuffer.wrap(upgraded).putInt(8, 6);
        var restored = GraphArtifactCodec.decode(versioned("graph-model.v6", upgraded), LIMITS);
        sameImage(v3, GraphArtifactCodec.encode(GraphSource.capture(restored), LIMITS));
        assertEquals("graph-model.v1", original.kind());

        var source = flat(false);
        var image = GraphArtifactCodec.encode(GraphSource.capture(source), LIMITS);
        int schema = Math.toIntExact(image.entries().getFirst().size());
        var exact = new ArtifactLimits(48 + 72 + schema, 72, 1, 14, schema, schema);
        sameImage(image, GraphArtifactCodec.encode(GraphSource.capture(source), exact));
        var below = new ArtifactLimits(48 + 72 + schema, 72, 1, 14, schema - 1, schema);
        Path target = directory.resolve("sentinel"); Files.writeString(target, "unchanged");
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(target, source, below));
        assertEquals("unchanged", Files.readString(target));
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image, below));
        Tensor state = tensor(DataType.FLOAT32, Shape.scalar(), true, true, new byte[4]);
        var named = new GraphCheckpoint.Source(List.of(volume),
                List.of(volume.add(state).averagePool3d(AVG)),
                List.of(leaf("weight", GraphCheckpoint.Role.PARAMETER, state)));
        var namedImage = GraphArtifactCodec.encode(GraphSource.capture(named), LIMITS);
        int namedSchema = Math.toIntExact(namedImage.entries().getFirst().size());
        state.clearHostStorage();
        var namedBelow = new ArtifactLimits(1_000_000, 100_000, 1000, 64, namedSchema - 1, 900_000);
        // Missing storage would throw IllegalStateException if bounds were checked too late.
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(target, named, namedBelow));
        assertEquals("unchanged", Files.readString(target));
    }

    private static void sameImage(ArtifactEnvelope expected, ArtifactEnvelope actual) {
        assertEquals(expected.kind(), actual.kind());
        assertEquals(expected.entries().stream().map(ArtifactEnvelope.Entry::key).toList(),
                actual.entries().stream().map(ArtifactEnvelope.Entry::key).toList());
        for (int i = 0; i < expected.entries().size(); i++)
            assertArrayEquals(payload(expected, i), payload(actual, i));
    }

    private static GraphCheckpoint.Source flat(boolean average) {
        Tensor volume = prototype(DataType.FLOAT32, Shape.of(1, 1, 5, 6, 7), true, false);
        return new GraphCheckpoint.Source(List.of(volume),
                List.of(average ? volume.averagePool3d(AVG) : volume.maxPool3d(MAX)), List.of());
    }

    private static byte[] payload(ArtifactEnvelope image, int index) {
        byte[] bytes = new byte[image.entries().get(index).size()];
        image.entries().get(index).payload().get(bytes); return bytes;
    }

    private static ArtifactEnvelope versioned(String kind, byte[] wire) {
        return new ArtifactEnvelope(kind, List.of(new ArtifactEnvelope.Entry("schema", wire)));
    }

    private static void rejectsWithoutTensorAllocation(ArtifactEnvelope image) {
        Tensor marker = prototype(DataType.BOOL, Shape.scalar(), false, true);
        assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(image, LIMITS));
        assertEquals(marker.id().value() + 1, prototype(DataType.BOOL, Shape.scalar(), false, true).id().value());
    }
}
