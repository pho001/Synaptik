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
    void schemaVersionTwelveRetainsTypedDiscriminantsAndRequiredUnusedSentinels() {
        var reshape = MetalMpsGraphProgram.Node.targetShape(
                MetalMpsGraphProgram.NodeKind.RESHAPE, 0, 1, new long[] {3, 2});
        byte[] encoded = new MetalMpsGraphProgram(List.of(reshape)).encodedNodeRecords();
        assertEquals(12, MetalMpsGraphProgram.SCHEMA_VERSION);
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
    void JavaPreflightAcceptsAcceleratorAffineBridgeIntoBinaryAndGeneralViewTarget() {
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
                MetalMpsGraphProgram.Node.binary(
                        MetalMpsGraphProgram.NodeKind.ADD, 7, 8, 9),
                MetalMpsGraphProgram.Node.abs(9, 10)));
        long[][] shapes = {
            {6}, {2, 1, 3}, {2, 4, 3}, {4, 2, 3}, {4, 2, 1, 3},
            {4, 2, 3}, {4, 2, 3}, {4, 2, 3}, {4, 2, 3}, {4, 2, 3},
            {4, 2, 3}
        };
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                ranks(shapes),
                dimensions(shapes),
                program,
                new int[] {0, 8},
                new int[] {3, 10});
    }

    @Test
    void schemaTwelveRetainsMatmulWireAndAuthenticatesOnlyLocalRankTwoTransposes() {
        var direct = MetalMpsGraphProgram.Node.matmul(0, 1, 2);
        ByteBuffer record = ByteBuffer.wrap(
                new MetalMpsGraphProgram(List.of(direct)).encodedNodeRecords())
                .order(ByteOrder.BIG_ENDIAN);
        assertEquals(15, record.getInt());
        assertEquals(0, record.getInt());
        assertEquals(0, record.getInt());
        assertEquals(1, record.getInt());
        assertEquals(2, record.getInt());
        assertEquals(0, record.getInt());
        assertEquals(-1, record.getInt());
        assertEquals(0, record.getInt());
        while (record.hasRemaining()) assertEquals(0L, record.getLong());

        long[][] directShapes = {{2, 3}, {3, 4}, {2, 4}};
        MetalMpsGraphProgram directProgram = new MetalMpsGraphProgram(List.of(direct));
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                ranks(directShapes),
                dimensions(directShapes),
                directProgram,
                new int[] {0, 1},
                new int[] {2});
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        ranks(directShapes),
                        dimensions(directShapes),
                        directProgram,
                        new int[] {0, 1},
                        new int[] {2}));

        long[][] transposedShapes = {
            {3, 2}, {4, 3}, {2, 3}, {3, 4}, {2, 4}
        };
        MetalMpsGraphProgram transposed = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 2, List.of(1, 0)),
                MetalMpsGraphProgram.Node.permutation(1, 3, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(2, 3, 4)));
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                ranks(transposedShapes),
                dimensions(transposedShapes),
                transposed,
                new int[] {0, 1},
                new int[] {4});

        long[][] malformedShapes = {
            {6}, {3, 4}, {2, 3}, {2, 4}
        };
        MetalMpsGraphProgram materializedView = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0,
                        2,
                        new long[] {2, 3}),
                MetalMpsGraphProgram.Node.matmul(2, 1, 3)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        ranks(malformedShapes),
                        dimensions(malformedShapes),
                        materializedView,
                        new int[] {0, 1},
                        new int[] {3}));

        long[][] wrongOutputShapes = {{2, 3}, {3, 4}, {2, 5}};
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        ranks(wrongOutputShapes),
                        dimensions(wrongOutputShapes),
                        directProgram,
                        new int[] {0, 1},
                        new int[] {2}));
    }

    @Test
    void schemaTwelveRetainsGatherAndOneHotWithExactTypedAttributes() {
        var gather = MetalMpsGraphProgram.Node.gather(0, 1, 2, 1);
        var oneHot = MetalMpsGraphProgram.Node.oneHot(3, 4, 5);
        ByteBuffer records = ByteBuffer.wrap(
                new MetalMpsGraphProgram(List.of(gather, oneHot)).encodedNodeRecords())
                .order(ByteOrder.BIG_ENDIAN);

        assertEquals(16, records.getInt());
        assertEquals(3, records.getInt());
        assertEquals(0, records.getInt());
        assertEquals(1, records.getInt());
        assertEquals(2, records.getInt());
        assertEquals(1, records.getInt());
        assertEquals(1, records.getInt());
        assertEquals(0, records.getInt());
        for (int cell = 0; cell < 16; cell++) assertEquals(0L, records.getLong());

        assertEquals(17, records.getInt());
        assertEquals(5, records.getInt());
        assertEquals(3, records.getInt());
        assertEquals(-1, records.getInt());
        assertEquals(4, records.getInt());
        assertEquals(1, records.getInt());
        assertEquals(-1, records.getInt());
        assertEquals(0, records.getInt());
        assertEquals(5L, records.getLong());
        for (int cell = 1; cell < 16; cell++) assertEquals(0L, records.getLong());

        long[][] gatherShapes = {{2, 3, 4}, {5, 6}, {2, 5, 6, 4}};
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
                ranks(gatherShapes),
                dimensions(gatherShapes),
                new MetalMpsGraphProgram(List.of(gather)),
                new int[] {0, 1},
                new int[] {2});
        long[][] oneHotShapes = {{2, 3}, {2, 3, 5}};
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                ranks(oneHotShapes),
                dimensions(oneHotShapes),
                new MetalMpsGraphProgram(List.of(
                        MetalMpsGraphProgram.Node.oneHot(0, 1, 5))),
                new int[] {0},
                new int[] {1});

        long[][] boolConsumerShapes = {{2, 3}, {2, 3, 5}, {2, 3, 5}};
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        ranks(boolConsumerShapes),
                        dimensions(boolConsumerShapes),
                        new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.oneHot(0, 1, 5),
                                MetalMpsGraphProgram.Node.neg(1, 2))),
                        new int[] {0},
                        new int[] {2}));
    }

    @Test
    void schemaTwelveAppendsTask0052WiresAndRetainsExactScalarAndScanWords() {
        List<MetalMpsGraphProgram.NodeKind> kinds = List.of(
                MetalMpsGraphProgram.NodeKind.GT,
                MetalMpsGraphProgram.NodeKind.GE,
                MetalMpsGraphProgram.NodeKind.LT,
                MetalMpsGraphProgram.NodeKind.LE,
                MetalMpsGraphProgram.NodeKind.EQ,
                MetalMpsGraphProgram.NodeKind.NE,
                MetalMpsGraphProgram.NodeKind.TENSOR_MIN,
                MetalMpsGraphProgram.NodeKind.TENSOR_MAX,
                MetalMpsGraphProgram.NodeKind.SCALAR_MIN,
                MetalMpsGraphProgram.NodeKind.SCALAR_MAX,
                MetalMpsGraphProgram.NodeKind.CLAMP,
                MetalMpsGraphProgram.NodeKind.REDUCTION_MIN,
                MetalMpsGraphProgram.NodeKind.REDUCTION_MAX,
                MetalMpsGraphProgram.NodeKind.CUM_SUM,
                MetalMpsGraphProgram.NodeKind.CUM_PROD);
        for (int index = 0; index < kinds.size(); index++) {
            assertEquals(20 + index, kinds.get(index).wireIdentity());
        }
        var scalar = MetalMpsGraphProgram.Node.scalarExtreme(
                MetalMpsGraphProgram.NodeKind.SCALAR_MIN,
                0,
                1,
                0xffc1_2345);
        ByteBuffer scalarRecord = ByteBuffer.wrap(
                new MetalMpsGraphProgram(List.of(scalar)).encodedNodeRecords())
                .order(ByteOrder.BIG_ENDIAN);
        assertEquals(28, scalarRecord.getInt());
        assertEquals(7, scalarRecord.getInt());
        scalarRecord.position(32);
        assertEquals(0x0000_0000_ffc1_2345L, scalarRecord.getLong());

        var scan = MetalMpsGraphProgram.Node.scan(
                MetalMpsGraphProgram.NodeKind.CUM_PROD,
                3,
                4,
                2,
                true,
                false);
        ByteBuffer scanRecord = ByteBuffer.wrap(
                new MetalMpsGraphProgram(List.of(scan)).encodedNodeRecords())
                .order(ByteOrder.BIG_ENDIAN);
        assertEquals(34, scanRecord.getInt());
        assertEquals(9, scanRecord.getInt());
        assertEquals(3, scanRecord.getInt());
        assertEquals(-1, scanRecord.getInt());
        assertEquals(4, scanRecord.getInt());
        assertEquals(2, scanRecord.getInt());
        assertEquals(2, scanRecord.getInt());
        assertEquals(0, scanRecord.getInt());
        assertEquals(1L, scanRecord.getLong());
        assertEquals(0L, scanRecord.getLong());
        while (scanRecord.hasRemaining()) assertEquals(0L, scanRecord.getLong());

        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.scalarExtreme(
                        MetalMpsGraphProgram.NodeKind.CLAMP, 0, 1, 0));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.scan(
                        MetalMpsGraphProgram.NodeKind.NEG, 0, 1, 0, false, false));
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
