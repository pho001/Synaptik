package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalMpsGraphAffineSchemaTest {

    @Test
    void schemaRegistryIsCompleteWithExactStructuralExecutionSet() {
        MetalMpsGraphProgram.NodeKind[] operations = MetalMpsGraphProgram.NodeKind.values();
        assertEquals(115, operations.length);
        int executable = 0;
        for (int index = 0; index < operations.length; index++) {
            assertEquals(index + 1, operations[index].wireIdentity());
            if (operations[index].executable()) executable++;
        }
        assertEquals(103, executable);
        assertEquals(12, operations.length - executable);

        MetalMpsGraphProgram.AttributeKind[] attributes =
                MetalMpsGraphProgram.AttributeKind.values();
        assertEquals(42, attributes.length);
        for (int index = 0; index < attributes.length; index++) {
            assertEquals(index, attributes[index].wireIdentity());
        }
    }

    @Test
    void JavaPreflightAcceptsAcceleratorAffineBridgeOnlyOnCustomRoute() {
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
        MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), program), program, new int[] {0, 8}, new int[] {3, 10}, MetalPreparedRoute.CUSTOM_PROGRAM);
    }

    @Test
    void JavaPreflightAuthenticatesSelectedLocalRankTwoTransposesOnlyOnCustomRoute() {
        var direct = MetalMpsGraphProgram.Node.matmul(0, 1, 2);

        long[][] directShapes = {{2, 3}, {3, 4}, {2, 4}};
        MetalMpsGraphProgram directProgram = new MetalMpsGraphProgram(List.of(direct));
        MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(directShapes), dimensions(directShapes), directProgram), directProgram, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH);
        long[][] transposedShapes = {
            {3, 2}, {4, 3}, {2, 3}, {3, 4}, {2, 4}
        };
        MetalMpsGraphProgram transposed = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 2, List.of(1, 0)),
                MetalMpsGraphProgram.Node.permutation(1, 3, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(2, 3, 4)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(transposedShapes), dimensions(transposedShapes), transposed), transposed, new int[] {0, 1}, new int[] {4}, MetalPreparedRoute.MPSGRAPH));
    MetalNativeApi.ProgramExecutableAbi.validateCreate(
        MetalTestProgram.descriptors(
            ranks(transposedShapes), dimensions(transposedShapes), transposed),
        transposed,
        new int[] {0, 1},
        new int[] {4},
        MetalPreparedRoute.CUSTOM_PROGRAM);

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
                MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(malformedShapes), dimensions(malformedShapes), materializedView), materializedView, new int[] {0, 1}, new int[] {3}, MetalPreparedRoute.MPSGRAPH));

        long[][] wrongOutputShapes = {{2, 3}, {3, 4}, {2, 5}};
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(wrongOutputShapes), dimensions(wrongOutputShapes), directProgram), directProgram, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH));
    }

    @Test
    void JavaPreflightValidatesGatherAndOneHotTypesAndShapes() {
        var gather = MetalMpsGraphProgram.Node.gather(0, 1, 2, 1);
        long[][] gatherShapes = {{2, 3, 4}, {5, 6}, {2, 5, 6, 4}};
        MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(gatherShapes), dimensions(gatherShapes), new MetalMpsGraphProgram(List.of(gather))), new MetalMpsGraphProgram(List.of(gather)), new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.CUSTOM_PROGRAM);
        long[][] oneHotShapes = {{2, 3}, {2, 3, 5}};
        MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(oneHotShapes), dimensions(oneHotShapes), new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.oneHot(0, 1, 5)))), new MetalMpsGraphProgram(List.of(
        MetalMpsGraphProgram.Node.oneHot(0, 1, 5))), new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM);

        long[][] boolConsumerShapes = {{2, 3}, {2, 3, 5}, {2, 3, 5}};
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(boolConsumerShapes), dimensions(boolConsumerShapes), new MetalMpsGraphProgram(List.of(
                        MetalMpsGraphProgram.Node.oneHot(0, 1, 5),
                        MetalMpsGraphProgram.Node.neg(1, 2)))), new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.oneHot(0, 1, 5),
                MetalMpsGraphProgram.Node.neg(1, 2))), new int[] {0}, new int[] {2}, MetalPreparedRoute.CUSTOM_PROGRAM));
    }

    @Test
    void Task0052WireIdentitiesStayStableAndFactoriesRejectWrongKinds() {
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
        var scalar = MetalMpsGraphProgram.Node.scalarValue(
                MetalMpsGraphProgram.NodeKind.SCALAR_MIN,
                0,
                1,
                0xffc1_2345);
        assertEquals(List.of(1L, 0x0000_0000_ffc1_2345L),
                java.util.Arrays.stream(scalar.attributeWords()).boxed().toList());

        var scan = MetalMpsGraphProgram.Node.scan(
                MetalMpsGraphProgram.NodeKind.CUM_PROD,
                3,
                4,
                2,
                true,
                false);
        assertEquals(List.of(2L, 1L, 0L),
                java.util.Arrays.stream(scan.attributeWords()).boxed().toList());

        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.CLAMP, 0, 1, 0));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.scan(
                        MetalMpsGraphProgram.NodeKind.NEG, 0, 1, 0, false, false));
    }

    @Test
    void JavaPreflightRejectsLegacyCustomOnlyKindsOnDirectMpsGraph() {
        var comparison = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.GT,
                new int[] {0, 1},
                new int[] {2},
                MetalMpsGraphProgram.AttributeKind.NONE,
                new long[0]);
        var program = new MetalMpsGraphProgram(List.of(comparison));
        long[][] shapes = {{4}, {4}, {4}};
        List<MetalMpsGraphProgram.ValueDescriptor> values = MetalTestProgram.descriptors(
                ranks(shapes), dimensions(shapes), program);
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values,
                program,
                new int[] {0, 1},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        values,
                        program,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.MPSGRAPH));
    }

    @Test
    void typedNodeConstructionRejectsMalformedBoundsAndPairings() {
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
                MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), viewToNeg), viewToNeg, new int[] {0}, new int[] {2}, MetalPreparedRoute.MPSGRAPH));

        var unavailableInput = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.contiguous(1, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), unavailableInput), unavailableInput, new int[] {0}, new int[] {2}, MetalPreparedRoute.MPSGRAPH));
    }

    @Test
    void schemaTwentyProgramImageCarriesRouteAndAllSevenDataTypeWires() {
        var program = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CONCAT,
                new int[] {0, 1, 2, 3, 4, 5, 6},
                new int[] {7},
                MetalMpsGraphProgram.AttributeKind.AXIS,
                new long[] {0L})));
        var float16 = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT16, new long[0], false);
        var values = List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 3}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.INT32, new long[0], false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.BOOL, new long[0], false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT64, new long[0], false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.BFLOAT16, new long[0], false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.INT64, new long[0], false),
                float16,
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 3}, true));
        byte[] actual = program.encodedProgramImage(
                values, new int[] {0, 1, 2, 3, 4, 5, 6}, new int[] {7});

        assertEquals(1L, float16.elementCount());
        assertEquals(Short.BYTES, float16.byteCount());
        ByteBuffer expected = ByteBuffer.allocate(612).order(ByteOrder.LITTLE_ENDIAN);
        expected.putInt(MetalMpsGraphProgram.MAGIC);
        expected.putInt(MetalMpsGraphProgram.SCHEMA_VERSION);
        expected.putInt(MetalMpsGraphProgram.HEADER_BYTES);
        expected.putInt(612);
        expected.putInt(MetalPreparedRoute.MPSGRAPH.wireIdentity());
        expected.putInt(0);
        expected.putInt(0);
        expected.putInt(8);
        expected.putInt(1);
        expected.putInt(7);
        expected.putInt(1);
        expected.putInt(4);
        expected.putInt(4);
        expected.putInt(16);
        expected.putInt(1);
        for (int extensionField = 0; extensionField < 11; extensionField++) expected.putInt(0);
        expected.putInt(-1);
        for (int capField = 0; capField < 4; capField++) expected.putInt(0);
        putValue(expected, 1, 2, 0, 0, 2, 1, 0L, 6L);
        putValue(expected, 2, 0, 2, 2, 2, 1, 0L, 1L);
        putValue(expected, 3, 0, 2, 2, 2, 1, 0L, 1L);
        putValue(expected, 4, 0, 2, 2, 2, 1, 0L, 1L);
        putValue(expected, 5, 0, 2, 2, 2, 1, 0L, 1L);
        putValue(expected, 6, 0, 2, 2, 2, 1, 0L, 1L);
        putValue(expected, 7, 0, 2, 2, 2, 1, 0L, 1L);
        putValue(expected, 1, 2, 2, 2, 3, 1, 0L, 6L);
        expected.putInt(77);
        expected.putInt(3);
        expected.putInt(8);
        expected.putInt(7);
        expected.putInt(15);
        expected.putInt(1);
        expected.putInt(0);
        expected.putInt(1);
        expected.putLong(2);
        expected.putLong(3);
        expected.putLong(2);
        expected.putLong(3);
        expected.putLong(3);
        expected.putLong(1);
        expected.putLong(3);
        expected.putLong(1);
        for (int reference :
                new int[] {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7}) {
            expected.putInt(reference);
        }
        expected.putLong(0L);
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected.array(), actual);
    }

    @Test
    void task0064SingletonHeightViewRequiresTheExactLocalExpandDimsProducer() {
        Shape viewShape = Shape.of(2, 3, 1, 4);
        var view = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                viewShape.toLongArray(),
                Optional.of(LayoutDescriptor.of(
                        viewShape, new long[] {12, 4, 4, 1}, 0, true)),
                false,
                true);
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 3, 4}, false),
                view,
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 3, 1, 3}, false));
        var pool = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                new int[] {1},
                new int[] {2},
                MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                new long[] {1, 2, 1, 1, 0, 0, 1, 1, 0});
        var localProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 0, 1, 2),
                pool));
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values,
                localProgram,
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);

        var externalProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                        new long[] {1, 2, 1, 1, 0, 0, 1, 1, 0})));
        assertThrows(IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        List.of(view, values.get(2)),
                        externalProgram,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
    }

    private static void putValue(
            ByteBuffer buffer, int type, int rank, int dimensionOffset, int strideOffset,
            int flags, int kind, long storageOffset, long referencedSpan) {
        buffer.putInt(type);
        buffer.putInt(rank);
        buffer.putInt(dimensionOffset);
        buffer.putInt(strideOffset);
        buffer.putInt(flags);
        buffer.putInt(kind);
        buffer.putLong(storageOffset);
        buffer.putLong(referencedSpan);
    }

    private static void validate(long[][] shapes, MetalMpsGraphProgram.Node node) {
        MetalNativeApi.ProgramExecutableAbi.validateCreate(MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), new MetalMpsGraphProgram(List.of(node))), new MetalMpsGraphProgram(List.of(node)), new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
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
