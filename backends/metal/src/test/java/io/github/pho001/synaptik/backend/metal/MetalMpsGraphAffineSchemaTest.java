package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphAffineSchemaTest {
    @Test
    void schemaVersionTwoEncodesTypedDiscriminantsAndRequiredUnusedSentinels() {
        var reshape = MetalMpsGraphProgram.Node.targetShape(
                MetalMpsGraphProgram.NodeKind.RESHAPE, 0, 1, new long[] {3, 2});
        byte[] encoded = new MetalMpsGraphProgram(List.of(reshape)).encodedNodeRecords();
        assertEquals(2, MetalMpsGraphProgram.SCHEMA_VERSION);
        assertEquals(MetalMpsGraphProgram.NODE_RECORD_BYTES, encoded.length);

        ByteBuffer record = ByteBuffer.wrap(encoded).order(ByteOrder.BIG_ENDIAN);
        assertEquals(6, record.getInt());
        assertEquals(1, record.getInt());
        assertEquals(0, record.getInt());
        assertEquals(-1, record.getInt());
        assertEquals(1, record.getInt());
        assertEquals(2, record.getInt());
        assertEquals(-1, record.getInt());
        assertEquals(0, record.getInt());
        assertEquals(3L, record.getLong());
        assertEquals(2L, record.getLong());
        while (record.hasRemaining()) {
            assertEquals(0L, record.getLong());
        }
    }

    @Test
    void JavaPreflightAcceptsEachTypedAffineMapping() {
        validate(
                new long[][] {{2, 3}, {3, 2}},
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0, 1, new long[] {3, 2}));
        validate(
                new long[][] {{1, 3}, {2, 3}},
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.EXPAND,
                        0, 1, new long[] {2, 3}));
        validate(
                new long[][] {{2, 3}, {3, 2}},
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0)));
        validate(
                new long[][] {{2, 3}, {2, 1, 3}},
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 0, 1, 1));
        validate(
                new long[][] {{2, 1, 3}, {2, 3}},
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.SQUEEZE, 0, 1, 1));
    }

    @Test
    void typedNodeConstructionRejectsMalformedBoundsAndPairings() {
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE, 0, 1, new long[0]));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.EXPAND, 0, 1, new long[] {2, 0}));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE, 0, 1, new long[17]));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.NEG, 0, 1, new long[] {1}));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of()));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(0, 0)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(0, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 0, 1, -1));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.SQUEEZE, 0, 1, 16));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.ADD, 0, 1, 0));
    }

    @Test
    void JavaPreflightRejectsMismatchedGeometryAndAffineResultInputs() {
        assertInvalid(
                new long[][] {{2, 3}, {2, 4}},
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0, 1, new long[] {2, 4}));
        assertInvalid(
                new long[][] {{2, 3}, {2, 4}},
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.EXPAND,
                        0, 1, new long[] {2, 4}));
        assertInvalid(
                new long[][] {{2, 3}, {2, 3}},
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0)));
        assertInvalid(
                new long[][] {{2, 3}, {1, 2, 3}},
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 0, 1, 1));
        assertInvalid(
                new long[][] {{2, 2, 3}, {2, 3}},
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.SQUEEZE, 0, 1, 1));

        int[] ranks = {1, 2, 1};
        long[] dimensions = dimensions(new long[][] {{6}, {2, 3}, {6}});
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0, 1, new long[] {2, 3}),
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        1, 2, new long[] {6})));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        ranks, dimensions, program, new int[] {0}, new int[] {2}));
    }

    private static void validate(long[][] shapes, MetalMpsGraphProgram.Node node) {
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                ranks(shapes), dimensions(shapes),
                new MetalMpsGraphProgram(List.of(node)),
                new int[] {0}, new int[] {1});
    }

    private static void assertInvalid(long[][] shapes, MetalMpsGraphProgram.Node node) {
        assertThrows(IllegalArgumentException.class, () -> validate(shapes, node));
    }

    private static int[] ranks(long[][] shapes) {
        int[] result = new int[shapes.length];
        for (int index = 0; index < shapes.length; index++) {
            result[index] = shapes[index].length;
        }
        return result;
    }

    private static long[] dimensions(long[][] shapes) {
        long[] result = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, result, value * 16, shapes[value].length);
        }
        return result;
    }
}
