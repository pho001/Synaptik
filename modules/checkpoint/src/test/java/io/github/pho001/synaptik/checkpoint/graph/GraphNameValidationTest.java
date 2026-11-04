package io.github.pho001.synaptik.checkpoint.graph;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exact name grammar, dot-prefix semantics and encounter order through both admission paths. */
final class GraphNameValidationTest {
    private static final ArtifactLimits LIMITS =
            new ArtifactLimits(1_000_000, 100_000, 100, 64, 200_000, 900_000);

    @Test void unicodeAndLexicallyInterleavedNamesRetainExactOrderAndAliases() throws Exception {
        var names = List.of("z.🦉", "é", "e\u0301", "a!between", "a.b", "a\u0000tail", "ab");
        var admitted = GraphCheckpoint.copy(source(names));
        var decoded = GraphArtifactCodec.decode(wire(names), LIMITS);
        for (var result : List.of(admitted, decoded)) {
            assertEquals(names, result.leaves().stream().map(GraphCheckpoint.Leaf::name).toList());
            for (var leaf : result.leaves()) assertSame(result.outputs().getFirst(), leaf.tensor());
        }
        var original = wire(names);
        var encoded = GraphArtifactCodec.encode(GraphSource.capture(decoded), LIMITS);
        assertEquals(original.kind(), encoded.kind());
        assertEquals(original.entries().stream().map(ArtifactEnvelope.Entry::payload).toList(),
                encoded.entries().stream().map(ArtifactEnvelope.Entry::payload).toList());
    }

    @Test void collisionsRejectInEitherOrderEvenWithLexicallyInterveningNames() {
        for (var names : List.of(List.of("a", "a!between", "a.b"),
                List.of("層.🦉.leaf", "層.🦉"), List.of("a", "a\u0000tail", "a.b"),
                List.of("exact", "exact"))) {
            var reversed = new ArrayList<>(names);
            Collections.reverse(reversed);
            for (var order : List.of(names, reversed)) {
                assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source(order)));
                assertThrows(ArtifactFormatException.class, () -> GraphArtifactCodec.decode(wire(order), LIMITS));
            }
        }
    }

    @Test void oneDeepLegalNameHasNoNamespaceDepthRestriction() throws Exception {
        var names = List.of("a.".repeat(20_000) + "終🦉");
        var copied = GraphCheckpoint.copy(source(names));
        var decoded = GraphArtifactCodec.decode(wire(names), LIMITS);
        assertEquals(names.getFirst(), copied.leaves().getFirst().name());
        assertEquals(names.getFirst(), decoded.leaves().getFirst().name());
        assertSame(copied.outputs().getFirst(), copied.leaves().getFirst().tensor());
        assertSame(decoded.outputs().getFirst(), decoded.leaves().getFirst().tensor());
    }

    @Test void nonPrefixTextIsNotNormalizedAndBlankSegmentsStillReject() {
        for (String name : List.of("", ".a", "a.", "a..b", "a. \t.b", "bad\ud800"))
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(source(List.of(name))));
    }

    private static GraphCheckpoint.Source source(List<String> names) {
        var value = TensorFactory.scalar(0.0f, Optional.empty(), false);
        return new GraphCheckpoint.Source(List.of(), List.of(value), names.stream()
                .map(name -> new GraphCheckpoint.Leaf(name, GraphCheckpoint.Role.BUFFER, value)).toList());
    }

    /** Independently authors the frozen v1 zero-node scalar wire, including possibly invalid names. */
    private static ArtifactEnvelope wire(List<String> names) {
        int size = 52;
        for (String name : names) size += 8 + name.getBytes(StandardCharsets.UTF_8).length;
        var schema = ByteBuffer.allocate(size);
        schema.putLong(0x53594e4752463031L).putInt(1).putInt(0).putInt(1);
        schema.put((byte) 2).put((byte) 0).putInt(0).put((byte) 0); // FLOAT32 scalar, no layout
        schema.put((byte) 2).putLong(4).putInt(names.size());
        for (String name : names) {
            byte[] text = name.getBytes(StandardCharsets.UTF_8);
            schema.putInt(text.length).put(text).putInt(0);
        }
        schema.putInt(0).putInt(1).putInt(0);
        return new ArtifactEnvelope("graph-model.v1", List.of(
                new ArtifactEnvelope.Entry("schema", schema.array()),
                new ArtifactEnvelope.Entry("state/00000000", new byte[4])));
    }
}
