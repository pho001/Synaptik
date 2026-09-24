package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphAffineSchemaTest {
    @Test
    void schemaVersionSixEncodesTypedDiscriminantsAndRequiredUnusedSentinels() {
        var reshape = MetalMpsGraphProgram.Node.targetShape(
                MetalMpsGraphProgram.NodeKind.RESHAPE, 0, 1, new long[] {3, 2});
        byte[] encoded = new MetalMpsGraphProgram(List.of(reshape)).encodedNodeRecords();
        assertEquals(6, MetalMpsGraphProgram.SCHEMA_VERSION);
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
    void contiguousWireElevenHasNoAttributesAndAFullZeroPayload() {
        byte[] encoded = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.contiguous(3, 4))).encodedNodeRecords();
        ByteBuffer record = ByteBuffer.wrap(encoded).order(ByteOrder.BIG_ENDIAN);
        assertEquals(11, record.getInt());
        assertEquals(0, record.getInt());
        assertEquals(3, record.getInt());
        assertEquals(-1, record.getInt());
        assertEquals(4, record.getInt());
        assertEquals(0, record.getInt());
        assertEquals(-1, record.getInt());
        assertEquals(0, record.getInt());
        while (record.hasRemaining()) {
            assertEquals(0L, record.getLong());
        }
    }

    @Test
    void JavaPreflightAcceptsAffineCompositionAndExplicitContiguousStateTransition() {
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0, 1, new long[] {2, 1, 3}),
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.EXPAND,
                        1, 2, new long[] {2, 4, 3}),
                MetalMpsGraphProgram.Node.permutation(2, 3, List.of(1, 0, 2)),
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 3, 4, 2),
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.SQUEEZE, 4, 5, 2),
                MetalMpsGraphProgram.Node.contiguous(5, 6),
                MetalMpsGraphProgram.Node.neg(6, 7),
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        7, 8, new long[] {4, 6})));
        long[][] shapes = {
            {6}, {2, 1, 3}, {2, 4, 3}, {4, 2, 3}, {4, 2, 1, 3},
            {4, 2, 3}, {4, 2, 3}, {4, 2, 3}, {4, 6}
        };
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
                ranks(shapes),
                dimensions(shapes),
                program,
                new int[] {0},
                new int[] {1, 2, 3, 4, 5, 6, 7, 8});
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
                        MetalMpsGraphProgram.NodeKind.NEG, 0, 1, 0));
    }

    @Test
    void JavaPreflightRejectsMismatchedGeometryUnavailableStateAndViewToNeg() {
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

        long[][] shapes = {{6}, {2, 3}, {2, 3}};
        var viewToNeg = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0, 1, new long[] {2, 3}),
                MetalMpsGraphProgram.Node.neg(1, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        ranks(shapes),
                        dimensions(shapes),
                        viewToNeg,
                        new int[] {0},
                        new int[] {2}));

        var unavailableInput = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.contiguous(1, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        ranks(shapes),
                        dimensions(shapes),
                        unavailableInput,
                        new int[] {0},
                        new int[] {2}));
    }

    private static void validate(long[][] shapes, MetalMpsGraphProgram.Node node) {
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
                ranks(shapes),
                dimensions(shapes),
                new MetalMpsGraphProgram(List.of(node)),
                new int[] {0},
                new int[] {1});
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
