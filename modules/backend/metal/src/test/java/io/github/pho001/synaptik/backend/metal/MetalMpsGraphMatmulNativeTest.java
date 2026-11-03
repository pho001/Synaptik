package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class MetalMpsGraphMatmulNativeTest {
    @Test
    void typedMatmulKeepsScalarResultsAndBatchBroadcastWithGradientFlags() {
        Path library = configuredLibrary();
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                    MetalMpsGraphProgram.Node.matmul(3, 4, 5)));
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {3}, true),
                    typed(type, 3),
                    new MetalMpsGraphProgram.ValueDescriptor(type, new long[0], true),
                    new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2, 2, 3}, true),
                    typed(type, 1, 3, 2),
                    new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2, 2, 2}, true));
            List<byte[]> actual = executeCustom(library, program, values,
                    new int[] {0, 1, 3, 4}, new int[] {2, 5},
                    List.of(numericBytes(type, 1, 2, 3), numericBytes(type, 4, 5, 6),
                            numericBytes(type, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                            numericBytes(type, 1, 0, 0, 1, 1, 1)));
            assertArrayEquals(numericBytes(type, 32), actual.get(0), type.toString());
            assertArrayEquals(numericBytes(type, 4, 5, 10, 11, 16, 17, 22, 23), actual.get(1));
        }
    }

    private static byte[] numericBytes(DataType type, float... values) {
        if (type == DataType.FLOAT32) return floatBytes(values);
        if (type == DataType.BFLOAT16) return bfloatBytes(values);
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES).order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putShort(
                io.github.pho001.synaptik.model.datatype.Float16Bits.fromFloat(value));
        return bytes.array();
    }

    @Test
    void rankTwoSingletonPublishesBroadcastProductsInDirectAndNestedGraphs() {
        Path library = configuredLibrary();
        float[] left = {-0.0f, 0.0f, 2.0f};
        float[] right = {1.0f, -1.0f, -0.0f, 2.0f};
        float[] expected = new float[left.length * right.length];
        for (int row = 0; row < left.length; row++) {
            for (int column = 0; column < right.length; column++) {
                expected[row * right.length + column] = left[row] * right[column];
            }
        }
        for (MetalPreparedRoute route : List.of(
                MetalPreparedRoute.MPSGRAPH, MetalPreparedRoute.CUSTOM_PROGRAM)) {
            var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
            nodes.add(MetalMpsGraphProgram.Node.matmul(0, 1, 2));
            var values = new ArrayList<>(List.of(
                    typed(DataType.FLOAT32, 3, 1), typed(DataType.FLOAT32, 1, 4),
                    typed(DataType.FLOAT32, 3, 4)));
            int[] targets = {2};
            if (route == MetalPreparedRoute.CUSTOM_PROGRAM) {
                nodes.add(MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FLOOR, new int[] {2}, new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0]));
                values.add(typed(DataType.FLOAT32, 3, 4));
                targets = new int[] {2, 3};
            }
            List<byte[]> actual = execute(library, new MetalMpsGraphProgram(List.copyOf(nodes)),
                    List.copyOf(values), new int[] {0, 1}, targets,
                    List.of(floatBytes(left), floatBytes(right)), route);
            assertArrayEquals(floatBytes(expected), actual.getFirst(), route.toString());
        }
    }

    @Test
    void customSingletonKeepsProductSignsForEveryCarrierInVectorBatchAndTransposeForms() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.matmul(3, 4, 5),
                MetalMpsGraphProgram.Node.matmul(6, 7, 8),
                MetalMpsGraphProgram.Node.permutation(9, 10, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(10, 11, 12)));
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            Shape transposed = Shape.of(2, 1);
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    typed(type, 1), typed(type, 1), typed(type),
                    typed(type, 1), typed(type, 1, 3), typed(type, 3),
                    typed(type, 2, 2, 1), typed(type, 1, 1, 3), typed(type, 2, 2, 3),
                    typed(type, 1, 2),
                    new MetalMpsGraphProgram.ValueDescriptor(type, transposed.toLongArray(),
                            Optional.of(LayoutDescriptor.of(
                                    transposed, new long[] {1, 2}, 0L, true)), false, true),
                    typed(type, 1, 3), typed(type, 2, 3));
            List<byte[]> actual = executeCustom(library, program, values,
                    new int[] {0, 1, 3, 4, 6, 7, 9, 11}, new int[] {2, 5, 8, 12},
                    List.of(numericBytes(type, -0.0f), numericBytes(type, 1),
                            numericBytes(type, 0.0f), numericBytes(type, -1, 1, -2),
                            numericBytes(type, -0.0f, 0.0f, 0.0f, -0.0f),
                            numericBytes(type, 1, -1, 2),
                            numericBytes(type, -0.0f, 0.0f), numericBytes(type, -1, 1, -2)));
            assertArrayEquals(numericBytes(type, -0.0f), actual.get(0), type + " scalar");
            assertArrayEquals(numericBytes(type, -0.0f, 0.0f, -0.0f),
                    actual.get(1), type + " vector");
            assertArrayEquals(numericBytes(type,
                            -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f,
                            0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f),
                    actual.get(2), type + " batch broadcast");
            assertArrayEquals(numericBytes(type, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f),
                    actual.get(3), type + " authenticated transpose");
        }
    }

    @Test
    void rankTwoMatmulKeepsCanonicalDirectAndRoutesAuthenticatedTransposeFormsCustomAcrossReuse() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        try {
            context = api.createContext();
            runEveryForm(api, context);
            runK1NumericalSentinels(api, context);
        } finally {
            if (context != null) {
                api.releaseContext(context);
            }
            api.close();
        }
    }

    @Test
    void customGeneralFloatingDomainRunsVectorBatchBroadcastAndBothMixedOrders() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.matmul(3, 4, 5),
                MetalMpsGraphProgram.Node.matmul(6, 7, 8),
                MetalMpsGraphProgram.Node.matmul(9, 10, 11)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 3),
                typed(DataType.FLOAT32, 3),
                typed(DataType.FLOAT32),
                typed(DataType.FLOAT32, 2, 2, 3),
                typed(DataType.FLOAT32, 1, 3, 2),
                typed(DataType.FLOAT32, 2, 2, 2),
                typed(DataType.BFLOAT16, 2, 2),
                typed(DataType.FLOAT32, 2, 1),
                typed(DataType.FLOAT32, 2, 1),
                typed(DataType.FLOAT32, 2, 2),
                typed(DataType.BFLOAT16, 2, 1),
                typed(DataType.FLOAT32, 2, 1));
        List<byte[]> actual = executeCustom(
                library,
                program,
                values,
                new int[] {0, 1, 3, 4, 6, 7, 9, 10},
                new int[] {2, 5, 8, 11},
                List.of(
                        floatBytes(1, 2, 3),
                        floatBytes(4, 5, 6),
                        floatBytes(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                        floatBytes(1, 0, 0, 1, 1, 1),
                        bfloatBytes(1.5f, -2.25f, 0.5f, 4.0f),
                        floatBytes(2.0f, -1.0f),
                        floatBytes(1.5f, -2.25f, 0.5f, 4.0f),
                        bfloatBytes(2.0f, -1.0f)));
        assertArrayEquals(floatBytes(32), actual.get(0));
        assertArrayEquals(floatBytes(4, 5, 10, 11, 16, 17, 22, 23), actual.get(1));
        assertArrayEquals(floatBytes(5.25f, -3.0f), actual.get(2));
        assertArrayEquals(floatBytes(5.25f, -3.0f), actual.get(3));
    }

    @Test
    void customFloatMatmulRunsAllRankPairingsRankSixteenAndConcurrentSessions()
            throws Exception {
        Path library = configuredLibrary();
        long[] rankSixteenLeft = new long[16];
        java.util.Arrays.fill(rankSixteenLeft, 1L);
        rankSixteenLeft[14] = 2L;
        rankSixteenLeft[15] = 2L;
        long[] rankSixteenOutput = rankSixteenLeft.clone();
        rankSixteenOutput[15] = 1L;
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.matmul(3, 4, 5),
                MetalMpsGraphProgram.Node.matmul(6, 7, 8)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, 2, 2),
                typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, 2, 2),
                typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, 2),
                typed(DataType.FLOAT32, rankSixteenLeft),
                typed(DataType.FLOAT32, 2, 1),
                typed(DataType.FLOAT32, rankSixteenOutput));
        int[] feeds = {0, 1, 3, 4, 6, 7};
        int[] targets = {2, 5, 8};
        List<byte[]> feedBytes = List.of(
                floatBytes(2, 3),
                floatBytes(4, 5, 6, 7),
                floatBytes(1, 2, 3, 4),
                floatBytes(5, 6),
                floatBytes(1, 2, 3, 4),
                floatBytes(5, 6));
        List<byte[]> expected = List.of(
                floatBytes(26, 31),
                floatBytes(17, 39),
                floatBytes(17, 39));

        var executor = Executors.newFixedThreadPool(4);
        try {
            var calls = new ArrayList<java.util.concurrent.Callable<List<byte[]>>>();
            for (int call = 0; call < 8; call++) {
                calls.add(() -> executeCustom(
                        library,
                        program,
                        values,
                        feeds,
                        targets,
                        feedBytes));
            }
            for (var future : executor.invokeAll(calls)) {
                List<byte[]> actual = future.get();
                for (int target = 0; target < expected.size(); target++) {
                    assertArrayEquals(expected.get(target), actual.get(target));
                }
            }
        } finally {
            executor.shutdown();
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
        }
    }

    @Test
    void customFloatMatmulPublishesOnlyPermittedSpecialClasses() {
        Path library = configuredLibrary();
        var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        var feeds = new int[14];
        var targets = new int[7];
        List<byte[]> feedBytes = List.of(
                floatBytes(Float.POSITIVE_INFINITY), floatBytes(1),
                floatBytes(Float.NEGATIVE_INFINITY), floatBytes(1),
                floatBytes(Float.intBitsToFloat(0x7fa12345)), floatBytes(1),
                floatBytes(0.0f), floatBytes(-1),
                floatBytes(Float.intBitsToFloat(0x00000001)), floatBytes(1),
                floatBytes(Float.MAX_VALUE), floatBytes(2),
                floatBytes(Float.intBitsToFloat(0x00000001)), floatBytes(-0.5f));
        for (int node = 0; node < 7; node++) {
            int value = node * 3;
            nodes.add(MetalMpsGraphProgram.Node.matmul(value, value + 1, value + 2));
            values.add(typed(DataType.FLOAT32, 1));
            values.add(typed(DataType.FLOAT32, 1));
            values.add(typed(DataType.FLOAT32));
            feeds[node * 2] = value;
            feeds[node * 2 + 1] = value + 1;
            targets[node] = value + 2;
        }
        List<byte[]> actual = executeCustom(
                library,
                new MetalMpsGraphProgram(List.copyOf(nodes)),
                List.copyOf(values),
                feeds,
                targets,
                feedBytes);
        assertEquals(Float.floatToRawIntBits(Float.POSITIVE_INFINITY), rawInt(actual.get(0)));
        assertEquals(Float.floatToRawIntBits(Float.NEGATIVE_INFINITY), rawInt(actual.get(1)));
        assertTrue(Float.isNaN(Float.intBitsToFloat(rawInt(actual.get(2)))));
        assertEquals(0x80000000, rawInt(actual.get(3)));
        assertTrue(
                rawInt(actual.get(4)) == 0x00000001 || rawInt(actual.get(4)) == 0x00000000);
        assertEquals(Float.floatToRawIntBits(Float.POSITIVE_INFINITY), rawInt(actual.get(5)));
        assertEquals(0x80000000, rawInt(actual.get(6)),
                "singleton negative underflow retains its product sign");
    }

    @Test
    void customProgramKeepsRankTwoFloatMatmulAsNestedMpsGraphStep() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FLOOR,
                        new int[] {2},
                        new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.NONE,
                        new long[0])));
        List<byte[]> actual = executeCustom(
                library,
                program,
                List.of(
                        typed(DataType.FLOAT32, 2, 2),
                        typed(DataType.FLOAT32, 2, 2),
                        typed(DataType.FLOAT32, 2, 2),
                        typed(DataType.FLOAT32, 2, 2)),
                new int[] {0, 1},
                new int[] {3},
                List.of(
                        floatBytes(1.25f, 2.0f, 3.0f, 4.0f),
                        floatBytes(1.0f, 0.0f, 0.0f, 1.0f)));
        assertArrayEquals(floatBytes(1.0f, 2.0f, 3.0f, 4.0f), actual.getFirst());
    }

    @Test
    void customProgramConsumesTransposedRankTwoFloatMatmulWithoutMpsGraphViewFallback() {
        Path library = configuredLibrary();
        Shape leftShape = Shape.of(2, 3);
        Shape rightShape = Shape.of(3, 2);
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0)),
                MetalMpsGraphProgram.Node.permutation(2, 3, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(1, 3, 4),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FLOOR,
                        new int[] {4},
                        new int[] {5},
                        MetalMpsGraphProgram.AttributeKind.NONE,
                        new long[0])));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 3, 2),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32,
                        leftShape.toLongArray(),
                        Optional.of(LayoutDescriptor.of(
                                leftShape, new long[] {1, 2}, 0L, true)),
                        false,
                        true),
                typed(DataType.FLOAT32, 2, 3),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32,
                        rightShape.toLongArray(),
                        Optional.of(LayoutDescriptor.of(
                                rightShape, new long[] {1, 3}, 0L, true)),
                        false,
                        true),
                typed(DataType.FLOAT32, 2, 2),
                typed(DataType.FLOAT32, 2, 2));
        List<byte[]> actual = executeCustom(
                library,
                program,
                values,
                new int[] {0, 2},
                new int[] {5},
                List.of(
                        floatBytes(1, 4, 2, 5, 3, 6),
                        floatBytes(1, 0, 1, 0, 1, 1)));
        assertArrayEquals(floatBytes(4, 5, 10, 11), actual.getFirst());
    }

    @Test
    void customIntegralDomainRunsAllPromotionsWithModularOverflow() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.matmul(3, 4, 5),
                MetalMpsGraphProgram.Node.matmul(6, 7, 8),
                MetalMpsGraphProgram.Node.matmul(9, 10, 11)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.INT32, 1, 2), typed(DataType.INT32, 2, 1),
                typed(DataType.INT32, 1, 1),
                typed(DataType.INT32, 1, 2), typed(DataType.INT64, 2, 1),
                typed(DataType.INT64, 1, 1),
                typed(DataType.INT64, 1, 2), typed(DataType.INT32, 2, 1),
                typed(DataType.INT64, 1, 1),
                typed(DataType.INT64, 1, 2), typed(DataType.INT64, 2, 1),
                typed(DataType.INT64, 1, 1));
        List<byte[]> actual = executeCustom(
                library,
                program,
                values,
                new int[] {0, 1, 3, 4, 6, 7, 9, 10},
                new int[] {2, 5, 8, 11},
                List.of(
                        intBytes(Integer.MAX_VALUE, 2), intBytes(2, 3),
                        intBytes(-2, 3), longBytes(Long.MAX_VALUE, 3),
                        longBytes(-2, 3), intBytes(Integer.MAX_VALUE, 3),
                        longBytes(Long.MAX_VALUE, 2), longBytes(2, 3)));
        assertArrayEquals(intBytes(4), actual.get(0));
        assertArrayEquals(longBytes(11), actual.get(1));
        assertArrayEquals(longBytes(-4_294_967_285L), actual.get(2));
        assertArrayEquals(longBytes(4), actual.get(3));
    }

    @Test
    void customMatmulConsumesAuthenticatedIntegralAndBfloatTransposeCarriers() {
        Path library = configuredLibrary();
        Shape intTransposeShape = Shape.of(2, 3);
        Shape bfloatTransposeShape = Shape.of(2, 2);
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(1, 2, 3),
                MetalMpsGraphProgram.Node.permutation(4, 5, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(5, 6, 7)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.INT32, 3, 2),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.INT32,
                        intTransposeShape.toLongArray(),
                        Optional.of(LayoutDescriptor.of(
                                intTransposeShape, new long[] {1, 2}, 0L, true)),
                        false,
                        true),
                typed(DataType.INT64, 3, 1),
                typed(DataType.INT64, 2, 1),
                typed(DataType.BFLOAT16, 2, 2),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.BFLOAT16,
                        bfloatTransposeShape.toLongArray(),
                        Optional.of(LayoutDescriptor.of(
                                bfloatTransposeShape, new long[] {1, 2}, 0L, true)),
                        false,
                        true),
                typed(DataType.FLOAT32, 2, 1),
                typed(DataType.FLOAT32, 2, 1));
        List<byte[]> actual = executeCustom(
                library,
                program,
                values,
                new int[] {0, 2, 4, 6},
                new int[] {3, 7},
                List.of(
                        intBytes(1, 4, 2, 5, 3, 6),
                        longBytes(1, 10, 100),
                        bfloatBytes(1, 3, 2, 4),
                        floatBytes(5, 6)));
        assertArrayEquals(longBytes(321, 654), actual.get(0));
        assertArrayEquals(floatBytes(17, 39), actual.get(1));
    }

    @Test
    void customRankThreeMatmulLoadsAuthenticatedTransposeFromPhysicalSource() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 2, List.of(0, 2, 1)),
                MetalMpsGraphProgram.Node.matmul(2, 1, 3)));
        Shape transposeShape = Shape.of(2, 2, 3);
        var affine = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                transposeShape.toLongArray(),
                Optional.of(LayoutDescriptor.of(
                        transposeShape, new long[] {6, 1, 2}, 0L, true)),
                false,
                true);
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 2, 3, 2),
                typed(DataType.FLOAT32, 3, 1),
                affine,
                typed(DataType.FLOAT32, 2, 2, 1));
        List<byte[]> actual = executeCustom(
                library,
                program,
                values,
                new int[] {0, 1},
                new int[] {3},
                List.of(
                        floatBytes(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                        floatBytes(1, 10, 100)));
        assertArrayEquals(floatBytes(531, 642, 1197, 1308), actual.getFirst());
    }

    @Test
    void directFloat32MatmulKeepsDenseBoundaryViewsAcrossReuse() {
        Path library = configuredLibrary();
        for (int views = 1; views <= 3; views++) {
            var left = (views & 1) != 0 ? denseView(DataType.FLOAT32, Shape.of(2, 3), false)
                    : new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 3}, true);
            var right = (views & 2) != 0 ? denseView(DataType.FLOAT32, Shape.of(3, 4), false)
                    : new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {3, 4}, true);
            List<byte[]> actual = execute(
                    library, new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.matmul(0, 1, 2))),
                    List.of(left, right, new MetalMpsGraphProgram.ValueDescriptor(
                            DataType.FLOAT32, new long[] {2, 4}, true)),
                    new int[] {0, 1}, new int[] {2},
                    List.of(floatBytes(1, 2, 3, 4, 5, 6),
                            floatBytes(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)),
                    MetalPreparedRoute.MPSGRAPH);
            assertArrayEquals(floatBytes(74, 80, 86, 92, 173, 188, 203, 218), actual.getFirst());
        }
    }

    @Test
    void denseViewsAndTheirTransposesUseTheRepresentedBufferForEveryOriginalFloatingType() {
        Path library = configuredLibrary();
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean local : List.of(false, true)) {
                for (int views = 1; views <= 3; views++) {
                    for (int transposes = 0; transposes <= 3; transposes++) {
                        runDenseViewForm(library, type, views, local, transposes, false);
                    }
                }
                // Publishing a transpose and a second PERMUTE prevents its existing elision.
                runDenseViewForm(library, type, 3, local, 3, true);
            }
        }
    }

    @Test
    void transposeOfLogicallyDenseSelectUsesItsDivergentRepresentedBufferBeforeMatmul() {
        Path library = configuredLibrary();
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean left : List.of(false, true)) {
                Shape permuted = Shape.of(3, 2, 4);
                Shape transposed = Shape.of(4, 3);
                var program = new MetalMpsGraphProgram(List.of(
                        MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0, 2)),
                        MetalMpsGraphProgram.Node.generic(
                                MetalMpsGraphProgram.NodeKind.SELECT,
                                new int[] {1}, new int[] {2},
                                MetalMpsGraphProgram.AttributeKind.SELECT, new long[] {1, 0}),
                        MetalMpsGraphProgram.Node.permutation(2, 3, List.of(1, 0)),
                        MetalMpsGraphProgram.Node.matmul(left ? 3 : 4, left ? 4 : 3, 5)));
                var values = List.of(
                        new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2, 3, 4}, true),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                type, permuted.toLongArray(),
                                Optional.of(LayoutDescriptor.of(
                                        permuted, new long[] {4, 12, 1}, 0L, true)), true, true),
                        denseView(type, Shape.of(3, 4), false),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                type, transposed.toLongArray(),
                                Optional.of(LayoutDescriptor.of(
                                        transposed, new long[] {1, 4}, 0L, true)), true, true),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                type, left ? new long[] {3, 2} : new long[] {2, 4}, true),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                type, left ? new long[] {4, 2} : new long[] {2, 3}, true));
                long width = type == DataType.FLOAT32 ? Float.BYTES : Short.BYTES;
                // SELECT retains represented strides [8,1], span 20, despite logical [4,1].
                long[] representedBytes = {24 * width, 24 * width, 20 * width,
                        12 * width, (left ? 6 : 8) * width, (left ? 8 : 6) * width};
                List<byte[]> actual = execute(
                        library, program, values, new int[] {0, 4}, new int[] {5},
                        List.of(floatingBytes(type,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
                                left ? floatingBytes(type, 1, 2, 3, 4, 5, 6)
                                        : floatingBytes(type, 1, 2, 3, 4, 5, 6, 7, 8)),
                        MetalPreparedRoute.CUSTOM_PROGRAM, representedBytes);
                assertArrayEquals(left ? floatingBytes(type, 61, 76, 70, 88, 79, 100, 88, 112)
                                : floatingBytes(type, 30, 70, 110, 70, 174, 278),
                        actual.getFirst(), type + " transpose operand=" + (left ? "left" : "right"));
            }
        }
    }

    private static void runDenseViewForm(
            Path library, DataType type, int views, boolean local,
            int transposes, boolean publishViews) {
        var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        var feeds = new ArrayList<Integer>();
        var feedBytes = new ArrayList<byte[]>();
        var published = new ArrayList<Integer>();
        var publishedBytes = new ArrayList<byte[]>();
        int[] operands = new int[2];
        float[][] canonical = {
            {1, 2, 3, 4, 5, 6},
            {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
        };
        float[][] transposed = {
            {1, 4, 2, 5, 3, 6},
            {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18}
        };
        for (int operand = 0; operand < 2; operand++) {
            boolean view = (views & (1 << operand)) != 0;
            boolean transpose = (transposes & (1 << operand)) != 0;
            long rows = operand == 0 ? 2 : 3;
            long columns = operand == 0 ? 3 : 4;
            Shape sourceShape = transpose ? Shape.of(columns, rows) : Shape.of(rows, columns);
            int source = values.size();
            feeds.add(source);
            feedBytes.add(floatingBytes(type, transpose ? transposed[operand] : canonical[operand]));
            if (local && view) {
                values.add(new MetalMpsGraphProgram.ValueDescriptor(
                        type, new long[] {1, transpose ? columns : rows, transpose ? rows : columns},
                        true));
                nodes.add(MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE, source, source + 1,
                        sourceShape.toLongArray()));
                values.add(denseView(type, sourceShape, true));
                source++;
            } else {
                values.add(view ? denseView(type, sourceShape, false)
                        : new MetalMpsGraphProgram.ValueDescriptor(type, sourceShape.toLongArray(), true));
            }
            if (transpose) {
                Shape shape = Shape.of(rows, columns);
                int result = values.size();
                nodes.add(MetalMpsGraphProgram.Node.permutation(source, result, List.of(1, 0)));
                values.add(new MetalMpsGraphProgram.ValueDescriptor(
                        type, shape.toLongArray(),
                        Optional.of(LayoutDescriptor.of(shape, new long[] {1, rows}, 0L, true)),
                        true, true));
                operands[operand] = result;
                if (publishViews) {
                    published.add(result);
                    publishedBytes.add(floatingBytes(type, canonical[operand]));
                    int back = values.size();
                    nodes.add(MetalMpsGraphProgram.Node.permutation(result, back, List.of(1, 0)));
                    values.add(denseView(type, sourceShape, true));
                    published.add(back);
                    publishedBytes.add(floatingBytes(type, transposed[operand]));
                }
            } else {
                operands[operand] = source;
            }
        }
        int product = values.size();
        nodes.add(MetalMpsGraphProgram.Node.matmul(operands[0], operands[1], product));
        values.add(new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2, 4}, true));
        int output = values.size();
        nodes.add(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.FLOOR, new int[] {product}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]));
        values.add(new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2, 4}, true));
        published.addFirst(output);
        publishedBytes.addFirst(floatingBytes(type, 74, 80, 86, 92, 173, 188, 203, 218));
        List<byte[]> actual = executeCustom(
                library, new MetalMpsGraphProgram(List.copyOf(nodes)), List.copyOf(values),
                feeds.stream().mapToInt(Integer::intValue).toArray(),
                published.stream().mapToInt(Integer::intValue).toArray(), List.copyOf(feedBytes));
        for (int target = 0; target < published.size(); target++) {
            assertArrayEquals(publishedBytes.get(target), actual.get(target),
                    type + " views=" + views + " local=" + local + " transposes=" + transposes
                            + " publication=" + publishViews + " target=" + target);
        }
    }

    private static MetalMpsGraphProgram.ValueDescriptor denseView(
            DataType type, Shape shape, boolean affine) {
        return new MetalMpsGraphProgram.ValueDescriptor(
                type, shape.toLongArray(),
                Optional.of(LayoutDescriptor.of(
                        shape, new long[] {shape.toLongArray()[1], 1L}, 0L, true)),
                true, affine);
    }

    private static byte[] floatingBytes(DataType type, float... values) {
        if (type == DataType.FLOAT32) return floatBytes(values);
        if (type == DataType.BFLOAT16) return bfloatBytes(values);
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putShort(Float.floatToFloat16(value));
        return bytes.array();
    }

    private static void runEveryForm(
            MetalNativeApi api, MetalNativeApi.Handle context) {
        int[][] inputBits = {
            bits(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f),
            bits(
                    7.0f, 8.0f, 9.0f, 10.0f,
                    11.0f, 12.0f, 13.0f, 14.0f,
                    15.0f, 16.0f, 17.0f, 18.0f),
            bits(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f),
            bits(
                    7.0f, 11.0f, 15.0f,
                    8.0f, 12.0f, 16.0f,
                    9.0f, 13.0f, 17.0f,
                    10.0f, 14.0f, 18.0f)
        };
        int[] expected = bits(
                74.0f, 80.0f, 86.0f, 92.0f,
                173.0f, 188.0f, 203.0f, 218.0f);
        int[] ranks = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
        long[] dimensions = dimensions(new long[][] {
            {2, 3}, {3, 4}, {3, 2}, {4, 3},
            {2, 4}, {2, 4}, {2, 4}, {2, 4},
            {2, 3}, {3, 4}
        });
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(2, 8, List.of(1, 0)),
                MetalMpsGraphProgram.Node.permutation(3, 9, List.of(1, 0)),
                MetalMpsGraphProgram.Node.matmul(0, 1, 4),
                MetalMpsGraphProgram.Node.matmul(8, 1, 5),
                MetalMpsGraphProgram.Node.matmul(0, 9, 6),
                MetalMpsGraphProgram.Node.matmul(8, 9, 7)));
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
    var internals = new ArrayList<MetalNativeApi.Handle>();
        try {
            executable = api.createProgramExecutable(context, MetalTestProgram.descriptors(ranks, dimensions, program), program, new int[] {0, 1, 2, 3}, new int[] {4, 5, 6, 7}, MetalPreparedRoute.CUSTOM_PROGRAM);
            for (int[] bits : inputBits) {
                MetalNativeApi.Handle buffer = api.createBuffer(
                        context, Math.multiplyExact((long) bits.length, Integer.BYTES));
                inputs.add(buffer);
                upload(api, buffer, bits);
            }
            for (int output = 0; output < 4; output++) {
                outputs.add(api.createBuffer(
                        context, Math.multiplyExact((long) expected.length, Integer.BYTES)));
            }
      internals.add(
          api.createBuffer(context, Math.multiplyExact((long) inputBits[2].length, Integer.BYTES)));
      internals.add(
          api.createBuffer(context, Math.multiplyExact((long) inputBits[3].length, Integer.BYTES)));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size() + outputs.size() + internals.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
        }
        for (int index = 0; index < outputs.size(); index++) {
          inputAddresses.setAtIndex(ADDRESS, inputs.size() + index, outputs.get(index).carrier());
        }
        for (int index = 0; index < internals.size(); index++) {
          inputAddresses.setAtIndex(
              ADDRESS, inputs.size() + outputs.size() + index, internals.get(index).carrier());
                }
                for (int index = 0; index < outputs.size(); index++) {
                    outputAddresses.setAtIndex(ADDRESS, index, outputs.get(index).carrier());
                }
                for (int repetition = 0; repetition < 2; repetition++) {
                    for (MetalNativeApi.Handle output : outputs) {
                        upload(api, output, new int[] {
                            0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
                            0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef
                        });
                    }
                    api.runExecutable(
                            executable,
                            inputs.size() + outputs.size() + internals.size(),
                            inputAddresses,
                            outputs.size(),
                            outputAddresses);
                    for (int output = 0; output < outputs.size(); output++) {
                        assertArrayEquals(
                                expected,
                                download(api, outputs.get(output), expected.length),
                                "output=" + output + " repetition=" + repetition);
                    }
                    for (int input = 0; input < inputs.size(); input++) {
                        assertArrayEquals(
                                inputBits[input],
                                download(api, inputs.get(input), inputBits[input].length),
                                "input=" + input + " repetition=" + repetition);
                    }
                }
            }
        } finally {
            for (int index = outputs.size(); index-- > 0;) {
                api.releaseBuffer(outputs.get(index));
      }
      for (int index = internals.size(); index-- > 0; ) {
        api.releaseBuffer(internals.get(index));
            }
            for (int index = inputs.size(); index-- > 0;) {
                api.releaseBuffer(inputs.get(index));
            }
            if (executable != null) {
                api.releaseExecutable(executable);
            }
        }
    }

    private static void runK1NumericalSentinels(
            MetalNativeApi api, MetalNativeApi.Handle context) {
        float precisionInput = 1.0009765625f;
        int[][] inputBits = {
            bits(precisionInput, 0.0f, Float.intBitsToFloat(0x00000001)),
            bits(precisionInput, -1.0f)
        };
        int precisionProduct = Float.floatToRawIntBits(precisionInput * precisionInput);
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2)));
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        MetalNativeApi.Handle output = null;
        try {
            executable = api.createProgramExecutable(context, MetalTestProgram.descriptors(new int[] {2, 2, 2}, dimensions(new long[][] {{3, 1}, {1, 2}, {3, 2}}), program), program, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH);
            for (int[] bits : inputBits) {
                MetalNativeApi.Handle buffer = api.createBuffer(
                        context, Math.multiplyExact((long) bits.length, Integer.BYTES));
                inputs.add(buffer);
                upload(api, buffer, bits);
            }
            output = api.createBuffer(context, 6L * Integer.BYTES);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, 1);
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                outputAddresses.setAtIndex(ADDRESS, 0, output.carrier());
                for (int repetition = 0; repetition < 2; repetition++) {
                    upload(api, output, new int[] {
                        0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
                        0xdeadbeef, 0xdeadbeef, 0xdeadbeef
                    });
                    api.runExecutable(
                            executable, 2, inputAddresses, 1, outputAddresses);
                    int[] actual = download(api, output, 6);
                    assertEquals(precisionProduct, actual[0],
                            "reduced-precision sentinel repetition=" + repetition);
                    assertEquals(Float.floatToRawIntBits(-precisionInput), actual[1]);
                    assertEquals(0x00000000, actual[2]);
                    assertEquals(0x80000000, actual[3]);
                    assertTrue(
                            actual[4] == 0x00000001 || actual[4] == 0x00000000,
                            "positive DAZ/FTZ result=0x"
                                    + Integer.toHexString(actual[4]));
                    assertTrue(
                            actual[5] == 0x80000001 || actual[5] == 0x80000000,
                            "negative DAZ/FTZ product sign result=0x"
                                    + Integer.toHexString(actual[5]));
                    for (int input = 0; input < inputs.size(); input++) {
                        assertArrayEquals(
                                inputBits[input],
                                download(api, inputs.get(input), inputBits[input].length));
                    }
                }
            }
        } finally {
            if (output != null) {
                api.releaseBuffer(output);
            }
            for (int index = inputs.size(); index-- > 0;) {
                api.releaseBuffer(inputs.get(index));
            }
            if (executable != null) {
                api.releaseExecutable(executable);
            }
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static List<byte[]> executeCustom(
            Path library,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> feedBytes) {
        return execute(library, program, values, feeds, targets, feedBytes, MetalPreparedRoute.CUSTOM_PROGRAM);
    }

    private static List<byte[]> execute(
            Path library,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> feedBytes,
            MetalPreparedRoute route) {
        return execute(library, program, values, feeds, targets, feedBytes, route,
                values.stream().mapToLong(MetalMpsGraphProgram.ValueDescriptor::byteCount).toArray());
    }

    private static List<byte[]> execute(
            Path library,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> feedBytes,
            MetalPreparedRoute route,
            long[] representedBytes) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context, values, program, feeds, targets, route);
            for (int value = 0; value < values.size(); value++) {
                MetalNativeApi.Handle buffer = api.createBuffer(context, representedBytes[value]);
                buffers.add(buffer);
                byte[] poison = new byte[Math.toIntExact(representedBytes[value])];
                java.util.Arrays.fill(poison, (byte) 0xa5);
                uploadBytes(api, buffer, poison);
            }
            for (int index = 0; index < feeds.length; index++) {
                assertEquals(values.get(feeds[index]).byteCount(), feedBytes.get(index).length);
                uploadBytes(api, buffers.get(feeds[index]), feedBytes.get(index));
            }
            List<byte[]> firstResult = null;
            try (Arena arena = Arena.ofConfined()) {
                int inputCount = route == MetalPreparedRoute.CUSTOM_PROGRAM ? buffers.size() : feeds.length;
                MemorySegment addresses = arena.allocate(ADDRESS, inputCount);
                for (int index = 0; index < inputCount; index++) {
                    int value = route == MetalPreparedRoute.CUSTOM_PROGRAM ? index : feeds[index];
                    addresses.setAtIndex(ADDRESS, index, buffers.get(value).carrier());
                }
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++) {
                    outputs.setAtIndex(
                            ADDRESS, index, buffers.get(targets[index]).carrier());
                }
                for (int repetition = 0; repetition < 2; repetition++) {
                    for (int target : targets) {
                        byte[] poison = new byte[Math.toIntExact(values.get(target).byteCount())];
                        java.util.Arrays.fill(poison, (byte) 0xa5);
                        uploadBytes(api, buffers.get(target), poison);
                    }
                    api.runExecutable(
                            executable, inputCount, addresses, targets.length, outputs);
                    var current = new ArrayList<byte[]>(targets.length);
                    for (int target : targets) {
                        current.add(downloadBytes(
                                api, buffers.get(target), values.get(target).byteCount()));
                    }
                    if (firstResult == null) {
                        firstResult = List.copyOf(current);
                    } else {
                        for (int target = 0; target < targets.length; target++) {
                            assertArrayEquals(firstResult.get(target), current.get(target));
                        }
                    }
                }
            }
            for (int index = 0; index < feeds.length; index++) {
                assertArrayEquals(
                        feedBytes.get(index),
                        downloadBytes(
                                api,
                                buffers.get(feeds[index]),
                                values.get(feeds[index]).byteCount()));
            }
            return firstResult;
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void uploadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1);
            source.copyFrom(MemorySegment.ofArray(bytes));
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static byte[] downloadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long byteCount) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(byteCount, 1);
            api.download(buffer, 0L, target, target.byteSize());
            return target.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        }
    }

    private static byte[] floatBytes(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putInt(Float.floatToRawIntBits(value));
        return bytes.array();
    }

    private static byte[] bfloatBytes(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : values) {
            bytes.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
        }
        return bytes.array();
    }

    private static int rawInt(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).getInt();
    }

    private static byte[] intBytes(int... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int value : values) bytes.putInt(value);
        return bytes.array();
    }

    private static byte[] longBytes(long... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        for (long value : values) bytes.putLong(value);
        return bytes.array();
    }

    private static int[] bits(float... values) {
        int[] bits = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            bits[index] = Float.floatToRawIntBits(values[index]);
        }
        return bits;
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] bits) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(
                    Math.multiplyExact((long) bits.length, Integer.BYTES), Integer.BYTES);
            for (int index = 0; index < bits.length; index++) {
                source.setAtIndex(JAVA_INT, index, bits[index]);
            }
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static int[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(
                    Math.multiplyExact((long) count, Integer.BYTES), Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            int[] result = new int[count];
            for (int index = 0; index < count; index++) {
                result[index] = target.getAtIndex(JAVA_INT, index);
            }
            return result;
        }
    }

    private static long[] dimensions(long[][] shapes) {
        long[] dimensions = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, dimensions, value * 16, shapes[value].length);
        }
        return dimensions;
    }
}
