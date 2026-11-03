package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
import org.junit.jupiter.api.Test;

/** Device regressions for ordinary, masked, affine and stored-value reductions. */
class MetalAuditReductionNativeTest {
    @Test
    void actualBroadcastReductionRequiresAnInjectiveRepresentedSubspace() {
        Shape shape = Shape.of(2, 2, 2);
        var node = reduction(MetalMpsGraphProgram.NodeKind.SUM, 0, 1, List.of(2), false);
        var program = new MetalMpsGraphProgram(List.of(node));
        var output = value(DataType.FLOAT32, 2, 2);
        LayoutDescriptor aliased = LayoutDescriptor.of(
                shape, new long[] {0, 1, 1}, 0, true);
        var aliasedInput = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                shape.toLongArray(), Optional.of(aliased), false, false);
        assertThrows(IllegalArgumentException.class, () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                List.of(aliasedInput, output), program, new int[] {0}, new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM));
        var input = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, shape.toLongArray(),
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 2, 1}, 0, true)), false, false);
        assertArrayEquals(new float[] {3, 7, 3, 7}, floats(execute(List.of(input, output),
                List.of(node), new int[] {0}, new byte[][] {f32(1, 2, 3, 4)}, 1)));
    }

    @Test
    void scalarSumPreservesStoredPointAndAffineSumUsesLogicalContributors() {
        var scalar = reduction(MetalMpsGraphProgram.NodeKind.SUM, 0, 1, List.of(), false);
        int signalingNan = 0x7f800123;
        byte[] point = bytes(DataType.FLOAT32, signalingNan);
        assertArrayEquals(point, execute(List.of(value(DataType.FLOAT32), value(DataType.FLOAT32)),
                List.of(scalar), new int[] {0}, new byte[][] {point}, 1));

        Shape shape = Shape.of(2, 3);
        var affine = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                new long[] {2, 3}, Optional.of(LayoutDescriptor.of(
                        shape, new long[] {8, 2}, 1L, true)), false, false);
        float[] storage = new float[14];
        storage[1] = 1; storage[3] = 2; storage[5] = 3;
        storage[9] = 4; storage[11] = 5; storage[13] = 6;
        float[] actual = floats(execute(List.of(affine, value(DataType.FLOAT32, 2)),
                List.of(reduction(MetalMpsGraphProgram.NodeKind.SUM, 0, 1, List.of(1), false)),
                new int[] {0}, new byte[][] {f32(storage)}, 1));
        assertArrayEquals(new float[] {6, 15}, actual);
    }

    @Test
    void generatedMeanCountShapeReadsExpandedScalarInsteadOfPhysicalBufferLength() {
        var expanded = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                new long[] {2, 3}, Optional.of(LayoutDescriptor.of(
                        Shape.of(2, 3), new long[] {0, 0}, 0, true)), false, true);
        var values = List.of(value(DataType.FLOAT32), expanded,
                value(DataType.FLOAT32, 2, 1), value(DataType.FLOAT32, 2, 1));
        var nodes = List.of(MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.EXPAND, 0, 1, new long[] {2, 3}),
                reduction(MetalMpsGraphProgram.NodeKind.SUM, 1, 2, List.of(1), true),
                reduction(MetalMpsGraphProgram.NodeKind.MEAN, 1, 3, List.of(1), true));
        assertArrayEquals(new float[] {3, 3}, floats(execute(values, nodes,
                new int[] {0}, new byte[][] {f32(1)}, 2, 3)));
        assertArrayEquals(new float[] {1, 1}, floats(execute(values, nodes,
                new int[] {0}, new byte[][] {f32(1)}, 3, 2)));
    }

    @Test
    void maskedReducersExcludeNonfiniteValuesBeforeLoadsAndUsePerOutputTrueCount() {
        var values = List.of(value(DataType.FLOAT32, 2, 3), value(DataType.BOOL, 2, 3),
                value(DataType.FLOAT32, 2), value(DataType.FLOAT32, 2));
        var nodes = List.of(masked(MetalMpsGraphProgram.NodeKind.SUM, 0, 1, 2, 1),
                masked(MetalMpsGraphProgram.NodeKind.MEAN, 0, 1, 3, 1));
        byte[][] inputs = {f32(Float.NaN, 4, Float.POSITIVE_INFINITY, 2, 8, Float.NEGATIVE_INFINITY),
                new byte[] {0, 1, 0, 1, 1, 0}};
        assertArrayEquals(new float[] {4, 10}, floats(execute(values, nodes, new int[] {0, 1}, inputs, 2, 3)));
        assertArrayEquals(new float[] {4, 5}, floats(execute(values, nodes, new int[] {0, 1}, inputs, 3, 2)));
        inputs[1] = new byte[6];
        assertArrayEquals(new byte[8], execute(values, nodes, new int[] {0, 1}, inputs, 2, 3));
        for (float result : floats(execute(values, nodes, new int[] {0, 1}, inputs, 3, 2)))
            assertTrue(Float.isNaN(result));

        var broadcastValues = List.of(value(DataType.FLOAT32, 2, 3), value(DataType.BOOL, 3),
                value(DataType.FLOAT32, 2));
        assertArrayEquals(new float[] {2, 5}, floats(execute(broadcastValues,
                List.of(masked(MetalMpsGraphProgram.NodeKind.MEAN, 0, 1, 2, 1)),
                new int[] {0, 1}, new byte[][] {f32(1, 2, 3, 4, 5, 6), new byte[] {1, 0, 1}}, 2)));
    }

    @Test
    void selectedStoredInfinityClassesTakePrecedenceOverFiniteIntermediateOverflow() {
        byte[] input = f32(
                Float.MAX_VALUE, Float.MAX_VALUE, Float.NEGATIVE_INFINITY,
                -Float.MAX_VALUE, -Float.MAX_VALUE, Float.POSITIVE_INFINITY,
                Float.MAX_VALUE, Float.MAX_VALUE, Float.POSITIVE_INFINITY,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0,
                Float.NaN, Float.MAX_VALUE, Float.POSITIVE_INFINITY);
        byte[] allSelected = new byte[15];
        java.util.Arrays.fill(allSelected, (byte) 1);
        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.SUM, MetalMpsGraphProgram.NodeKind.MEAN)) {
            for (boolean useMask : List.of(false, true)) {
                var values = useMask
                        ? List.of(value(DataType.FLOAT32, 5, 3), value(DataType.BOOL, 5, 3),
                                value(DataType.FLOAT32, 5))
                        : List.of(value(DataType.FLOAT32, 5, 3), value(DataType.FLOAT32, 5));
                var node = useMask ? masked(kind, 0, 1, 2, 1)
                        : reduction(kind, 0, 1, List.of(1), false);
                float[] actual = floats(execute(values, List.of(node),
                        useMask ? new int[] {0, 1} : new int[] {0},
                        useMask ? new byte[][] {input, allSelected} : new byte[][] {input},
                        useMask ? 2 : 1));
                assertEquals(Float.NEGATIVE_INFINITY, actual[0]);
                assertEquals(Float.POSITIVE_INFINITY, actual[1]);
                assertEquals(Float.POSITIVE_INFINITY, actual[2]);
                assertTrue(Float.isNaN(actual[3]));
                assertTrue(Float.isNaN(actual[4]));
            }
        }
    }

    @Test
    void floatingProductsRetainZerosParityAndInvalidZeroTimesInfinity() {
        var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {3, 3}, true),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {3}, true));
        var node = reduction(MetalMpsGraphProgram.NodeKind.PROD, 0, 1, List.of(1), false);
        float[] actual = floats(execute(values, List.of(node), new int[] {0},
                new byte[][] {f32(0, -2, 3, -0.0f, -2, 3, 0, Float.POSITIVE_INFINITY, 1)}, 1));
        assertEquals(0x80000000, Float.floatToRawIntBits(actual[0]));
        assertEquals(0, Float.floatToRawIntBits(actual[1]));
        assertTrue(Float.isNaN(actual[2]));
    }

    @Test
    void axisFoldAccumulatesEveryOverlapAndZerosTheUncoveredTail() {
        var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 2, 3}, true),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 6}, true));
        var node = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.FOLD_AXIS,
                new int[] {0}, new int[] {1}, MetalMpsGraphProgram.AttributeKind.WINDOW_AXIS,
                new long[] {1, 3, 2});
        assertArrayEquals(new float[] {1, 2, 7, 5, 6, 0, 7, 8, 19, 11, 12, 0},
                floats(execute(values, List.of(node), new int[] {0},
                        new byte[][] {f32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)}, 1)));
    }

    @Test
    void signedSliceMaterializesRawAdjointSitesWithoutInventingNegativeStorageStrides() {
        var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {4}, true),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2}, true));
        var node = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SLICE,
                new int[] {0}, new int[] {1}, MetalMpsGraphProgram.AttributeKind.SLICE,
                new long[] {1, 3, 2, 0, -2});
        assertArrayEquals(bytes(DataType.FLOAT32, 0x7f812345L, 0x80000000L),
                execute(values, List.of(node), new int[] {0}, new byte[][] {
                        bytes(DataType.FLOAT32, 0x3f800000L, 0x80000000L, 1, 0x7f812345L)}, 1));
    }

    @Test
    void correctedMultiaxisStatisticsAndGeneralNormsHaveNumericResults() {
        var values = List.of(value(DataType.FLOAT32, 2, 2, 2), value(DataType.FLOAT32, 2));
        byte[][] inputs = {f32(1, 2, 3, 4, 5, 6, 7, 8)};
        var variance = MetalMpsGraphProgram.Node.statisticalReduction(
                MetalMpsGraphProgram.NodeKind.VARIANCE, 0, 1, List.of(0, 2), false, 1);
        var std = MetalMpsGraphProgram.Node.statisticalReduction(
                MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION, 0, 1, List.of(0, 2), false, 1);
        assertArrayEquals(new float[] {17.0f / 3, 17.0f / 3}, floats(execute(values,
                List.of(variance), new int[] {0}, inputs, 1)), 1e-6f);
        assertArrayEquals(new float[] {(float) Math.sqrt(17.0 / 3), (float) Math.sqrt(17.0 / 3)},
                floats(execute(values, List.of(std), new int[] {0}, inputs, 1)), 1e-6f);
        assertArrayEquals(new float[] {14, 22}, floats(execute(values,
                List.of(reduction(MetalMpsGraphProgram.NodeKind.L1_NORM, 0, 1, List.of(0, 2), false)),
                new int[] {0}, inputs, 1)));
        assertArrayEquals(new float[] {(float) Math.sqrt(66), (float) Math.sqrt(138)}, floats(execute(values,
                List.of(reduction(MetalMpsGraphProgram.NodeKind.L2_NORM, 0, 1, List.of(0, 2), false)),
                new int[] {0}, inputs, 1)), 1e-5f);
    }

    @Test
    void pointNormsAndConstantStatisticsDoNotCopyNegativeInputOrOverflow() {
        var values = List.of(value(DataType.FLOAT32, 3), value(DataType.FLOAT32, 3));
        byte[] points = bytes(DataType.FLOAT32, 0xc0400000L, 0x80000001L, 0x80000000L);
        assertArrayEquals(bytes(DataType.FLOAT32, 0x40400000L, 1, 0), execute(values,
                List.of(reduction(MetalMpsGraphProgram.NodeKind.L2_NORM, 0, 1, List.of(), false)),
                new int[] {0}, new byte[][] {points}, 1));
        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.VARIANCE,
                MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION)) {
            assertArrayEquals(new byte[12], execute(values,
                    List.of(MetalMpsGraphProgram.Node.statisticalReduction(kind, 0, 1, List.of(), false, 0)),
                    new int[] {0}, new byte[][] {points}, 1));
            assertArrayEquals(new byte[4], execute(List.of(value(DataType.FLOAT32, 3), value(DataType.FLOAT32)),
                    List.of(MetalMpsGraphProgram.Node.statisticalReduction(kind, 0, 1, List.of(0), false, 1)),
                    new int[] {0}, new byte[][] {f32(Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE)}, 1));
        }
    }

    @Test
    void logSumExpHandlesBothInfinitySignsNanAndSignedZeroPoint() {
        var values = List.of(value(DataType.FLOAT32, 5, 2), value(DataType.FLOAT32, 5));
        var node = reduction(MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP, 0, 1, List.of(1), false);
        float[] actual = floats(execute(values, List.of(node), new int[] {0}, new byte[][] {f32(
                Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.POSITIVE_INFINITY, Float.NaN, 1000, 1000, 0, 0)}, 1));
        assertEquals(Float.NEGATIVE_INFINITY, actual[0]);
        assertEquals(Float.POSITIVE_INFINITY, actual[1]);
        assertTrue(Float.isNaN(actual[2]));
        assertEquals((float) (1000 + Math.log(2)), actual[3], 1e-4f);
        assertEquals((float) Math.log(2), actual[4], 1e-6f);
        assertArrayEquals(bytes(DataType.FLOAT32, 0x80000000L), execute(
                List.of(value(DataType.FLOAT32), value(DataType.FLOAT32)),
                List.of(reduction(MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP, 0, 1, List.of(), false)),
                new int[] {0}, new byte[][] {f32(-0.0f)}, 1));
    }

    @Test
    void exactReducersKeepIntegralWidthAndFloat64StoredOrdering() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            long max = type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
            long min = type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
            var values = List.of(value(type, 2, 2), value(type, 2));
            byte[][] input = {bytes(type, max, 1, min, -1)};
            assertArrayEquals(bytes(type, min, max), execute(values,
                    List.of(reduction(MetalMpsGraphProgram.NodeKind.SUM, 0, 1, List.of(1), false)),
                    new int[] {0}, input, 1));
            assertArrayEquals(bytes(type, 1, min), execute(values,
                    List.of(reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MIN, 0, 1, List.of(1), false)),
                    new int[] {0}, input, 1));
            assertArrayEquals(bytes(type, max, -1), execute(values,
                    List.of(reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MAX, 0, 1, List.of(1), false)),
                    new int[] {0}, input, 1));
        }
        var values = List.of(value(DataType.FLOAT64, 2, 2), value(DataType.FLOAT64, 2));
        byte[][] input = {bytes(DataType.FLOAT64, 0, 0x8000000000000000L,
                0x0000000000000002L, 0x0000000000000001L)};
        assertArrayEquals(bytes(DataType.FLOAT64, 0x8000000000000000L, 1), execute(values,
                List.of(reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MIN, 0, 1, List.of(1), false)),
                new int[] {0}, input, 1));
        assertArrayEquals(bytes(DataType.FLOAT64, 0, 2), execute(values,
                List.of(reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MAX, 0, 1, List.of(1), false)),
                new int[] {0}, input, 1));
    }

    @Test
    void fullAndMultiaxisExactExtremaRetainLargeIntegersAndPropagateNan() {
        long large = 9_007_199_254_740_993L;
        var integerValues = List.of(value(DataType.INT64, 2, 1, 2), value(DataType.INT64, 1));
        byte[][] integers = {bytes(DataType.INT64, large, large + 1, -large, -large - 1)};
        assertArrayEquals(bytes(DataType.INT64, large + 1), execute(integerValues,
                List.of(reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MAX, 0, 1, List.of(0, 2), false)),
                new int[] {0}, integers, 1));
        var fullValues = List.of(value(DataType.INT64, 2, 1, 2), value(DataType.INT64));
        var full = MetalMpsGraphProgram.Node.reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MIN,
                0, 1, MetalMpsGraphProgram.ReductionForm.FULL, List.of(), false);
        assertArrayEquals(bytes(DataType.INT64, -large - 1), execute(fullValues,
                List.of(full), new int[] {0}, integers, 1));
        var floatValues = List.of(value(DataType.FLOAT64, 2), value(DataType.FLOAT64));
        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.REDUCTION_MIN,
                MetalMpsGraphProgram.NodeKind.REDUCTION_MAX)) {
            byte[] result = execute(floatValues, List.of(reduction(kind, 0, 1, List.of(0), false)),
                    new int[] {0}, new byte[][] {bytes(DataType.FLOAT64,
                            0x7ff0000000000000L, 0x7ff0000000000123L)}, 1);
            assertTrue(Double.isNaN(ByteBuffer.wrap(result).order(ByteOrder.nativeOrder()).getDouble()));
        }
    }

    @Test
    void gradientBearingSumUsesDeterministicStoredContributorOrder() {
        var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {3}, true),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[0], true));
        var node = reduction(MetalMpsGraphProgram.NodeKind.SUM, 0, 1, List.of(0), false);
        for (int repetition = 0; repetition < 2; repetition++) {
            assertArrayEquals(f32(0), execute(values, List.of(node), new int[] {0},
                    new byte[][] {f32(16_777_216, 1, -16_777_216)}, 1));
        }
    }

    @Test
    void correctionAtContributorCountIsRejectedBeforeNativeDispatch() {
        var values = List.of(value(DataType.FLOAT32, 2, 3), value(DataType.FLOAT32, 2));
        for (long correction : new long[] {3, 4}) {
            var program = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.statisticalReduction(
                    MetalMpsGraphProgram.NodeKind.VARIANCE, 0, 1, List.of(1), false, correction)));
            assertThrows(IllegalArgumentException.class, () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                    values, program, new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM));
        }
    }

    @Test
    void originalNumericCarriersExecuteAllNineteenUnariesIncludingRankZero() {
        var kinds = List.of(MetalMpsGraphProgram.NodeKind.NEG, MetalMpsGraphProgram.NodeKind.ABS,
                MetalMpsGraphProgram.NodeKind.FLOOR, MetalMpsGraphProgram.NodeKind.CEIL,
                MetalMpsGraphProgram.NodeKind.SIGN, MetalMpsGraphProgram.NodeKind.RELU,
                MetalMpsGraphProgram.NodeKind.RECIPROCAL, MetalMpsGraphProgram.NodeKind.EXP,
                MetalMpsGraphProgram.NodeKind.LOG, MetalMpsGraphProgram.NodeKind.LOG1P,
                MetalMpsGraphProgram.NodeKind.EXPM1, MetalMpsGraphProgram.NodeKind.ERF,
                MetalMpsGraphProgram.NodeKind.SQRT, MetalMpsGraphProgram.NodeKind.RSQRT,
                MetalMpsGraphProgram.NodeKind.SIGMOID, MetalMpsGraphProgram.NodeKind.TANH,
                MetalMpsGraphProgram.NodeKind.GELU, MetalMpsGraphProgram.NodeKind.GELU_TANH_APPROXIMATION,
                MetalMpsGraphProgram.NodeKind.SILU);
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (var kind : kinds) {
                for (boolean scalar : List.of(false, true)) {
                    float[] input = scalar ? new float[] {2} : new float[] {0.125f, 1, 2};
                    long[] dimensions = scalar ? new long[0] : new long[] {input.length};
                    var node = MetalMpsGraphProgram.Node.generic(kind, new int[] {0}, new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
                    float[] actual = numericFloats(type, execute(List.of(value(type, dimensions),
                            value(type, dimensions)), List.of(node), new int[] {0},
                            new byte[][] {numeric(type, input)}, 1));
                    for (int index = 0; index < input.length; index++) {
                        float x = input[index];
                        float expected = switch (kind) {
                            case NEG -> -x; case ABS, RELU -> x;
                            case FLOOR -> (float) Math.floor(x); case CEIL -> (float) Math.ceil(x);
                            case SIGN -> 1f; case RECIPROCAL -> 1f / x;
                            case EXP -> (float) StrictMath.exp(x);
                            case SIGMOID -> 1f / (1f + (float) StrictMath.exp(-x));
                            default -> MetalFloat32MathNativeTest.reference(kind, x);
                        };
                        assertNumeric(type, expected, actual[index], type + " " + kind + " scalar=" + scalar);
                    }
                }
            }
        }
    }

    @Test
    void typedScalarTensorAndPowerOperationsReadOriginalBitsAndAffineOperands() {
        var scalarKinds = List.of(MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                MetalMpsGraphProgram.NodeKind.SCALAR_SUB, MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                MetalMpsGraphProgram.NodeKind.SCALAR_DIV, MetalMpsGraphProgram.NodeKind.SCALAR_POW,
                MetalMpsGraphProgram.NodeKind.SCALAR_MIN, MetalMpsGraphProgram.NodeKind.SCALAR_MAX);
        var tensorKinds = List.of(MetalMpsGraphProgram.NodeKind.ADD, MetalMpsGraphProgram.NodeKind.SUB,
                MetalMpsGraphProgram.NodeKind.MUL, MetalMpsGraphProgram.NodeKind.DIV,
                MetalMpsGraphProgram.NodeKind.TENSOR_POW, MetalMpsGraphProgram.NodeKind.TENSOR_MIN,
                MetalMpsGraphProgram.NodeKind.TENSOR_MAX);
        float[] expected = {5, 1, 6, 1.5f, 9, 2, 3};
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            for (int operation = 0; operation < scalarKinds.size(); operation++) {
                var scalar = MetalMpsGraphProgram.Node.scalarValue(scalarKinds.get(operation), 0, 1,
                        type, numericWord(type, 2));
                assertNumeric(type, expected[operation], numericFloats(type, execute(
                        List.of(value(type), value(type)), List.of(scalar), new int[] {0},
                        new byte[][] {numeric(type, 3)}, 1))[0], type + " scalar " + scalarKinds.get(operation));
                var affine = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2}, Optional.of(
                        LayoutDescriptor.of(Shape.of(2), new long[] {2}, 1, true)), true, false);
                var affineOutput = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2}, true);
                float[] affineResults = numericFloats(type, execute(List.of(affine, affineOutput),
                        List.of(scalar), new int[] {0},
                        new byte[][] {numeric(type, Float.NaN, 3, Float.NaN, 3)}, 1));
                for (float result : affineResults)
                    assertNumeric(type, expected[operation], result, type + " affine " + scalarKinds.get(operation));
                var right = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2}, Optional.of(
                        LayoutDescriptor.of(Shape.of(2), new long[] {2}, 1, true)), false, false);
                var tensor = MetalMpsGraphProgram.Node.binary(tensorKinds.get(operation), 0, 1, 2);
                float[] actual = numericFloats(type, execute(List.of(value(type), right, value(type, 2)),
                        List.of(tensor), new int[] {0, 1}, new byte[][] {numeric(type, 3),
                                numeric(type, Float.NaN, 2, Float.NaN, 2)}, 2));
                for (float result : actual) assertNumeric(type, expected[operation], result, type + " tensor");
            }
            var affine = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2}, Optional.of(
                    LayoutDescriptor.of(Shape.of(2), new long[] {2}, 1, true)), true, false);
            var clamp = MetalMpsGraphProgram.Node.clamp(0, 1, type, numericWord(type, -1), numericWord(type, 1));
            assertArrayEquals(numeric(type, -1, 1), execute(List.of(affine,
                            new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2}, true)),
                    List.of(clamp), new int[] {0},
                    new byte[][] {numeric(type, Float.NaN, -2, Float.NaN, 2)}, 1));
        }
    }

    @Test
    void typedReductionsPreserveMaskedStatisticalAndExpandedOneGeometry() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var expanded = new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2, 3}, Optional.of(
                    LayoutDescriptor.of(Shape.of(2, 3), new long[] {0, 0}, 0, true)), true, true);
            var countValues = List.of(new MetalMpsGraphProgram.ValueDescriptor(type, new long[0], true),
                    expanded, new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2, 1}, true));
            var countNodes = List.of(MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.EXPAND, 0, 1, new long[] {2, 3}),
                    reduction(MetalMpsGraphProgram.NodeKind.SUM, 1, 2, List.of(1), true));
            assertArrayEquals(numeric(type, 3, 3), execute(countValues, countNodes, new int[] {0},
                    new byte[][] {numeric(type, 1)}, 2));
            var kinds = List.of(MetalMpsGraphProgram.NodeKind.SUM, MetalMpsGraphProgram.NodeKind.MEAN,
                    MetalMpsGraphProgram.NodeKind.PROD, MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP,
                    MetalMpsGraphProgram.NodeKind.L1_NORM, MetalMpsGraphProgram.NodeKind.L2_NORM,
                    MetalMpsGraphProgram.NodeKind.REDUCTION_MIN, MetalMpsGraphProgram.NodeKind.REDUCTION_MAX);
            float[] expected = {6, 2, 6, (float) Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
                    6, (float) Math.sqrt(14), 1, 3};
            var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {3}, true),
                    new MetalMpsGraphProgram.ValueDescriptor(type, new long[0], true));
            for (int index = 0; index < kinds.size(); index++) {
                float actual = numericFloats(type, execute(values, List.of(reduction(kinds.get(index),
                        0, 1, List.of(0), false)), new int[] {0}, new byte[][] {numeric(type, 1, 2, 3)}, 1))[0];
                assertNumeric(type, expected[index], actual, type + " " + kinds.get(index));
            }
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.VARIANCE,
                    MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION)) {
                var node = MetalMpsGraphProgram.Node.statisticalReduction(kind, 0, 1, List.of(0), false, 1);
                assertArrayEquals(numeric(type, 1), execute(values, List.of(node), new int[] {0},
                        new byte[][] {numeric(type, 1, 2, 3)}, 1));
            }
            var maskValues = List.of(value(type, 3), value(DataType.BOOL, 3), value(type));
            for (var kind : List.of(MetalMpsGraphProgram.NodeKind.SUM, MetalMpsGraphProgram.NodeKind.MEAN)) {
                assertArrayEquals(numeric(type, 2), execute(maskValues,
                        List.of(masked(kind, 0, 1, 2, 0)), new int[] {0, 1},
                        new byte[][] {numeric(type, Float.NaN, 2, Float.POSITIVE_INFINITY), new byte[] {0, 1, 0}}, 2));
            }
            var tile = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.TILE,
                    new int[] {0}, new int[] {1}, MetalMpsGraphProgram.AttributeKind.TILE, new long[] {1, 2});
            var tileValues = List.of(new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2}, true),
                    new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {4}, true),
                    new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {4}, true));
            assertArrayEquals(numeric(type, 3, 4, 3, 4), execute(tileValues,
                    List.of(tile, MetalMpsGraphProgram.Node.scalarValue(
                            MetalMpsGraphProgram.NodeKind.SCALAR_ADD, 1, 2, type, numericWord(type, 2))),
                    new int[] {0}, new byte[][] {numeric(type, 1, 2)}, 2));
        }
    }

    @Test
    void cumulativeIntegerCarriersWrapInEveryDirectionAndExclusiveMode() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            long max = type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
            long[] input = {max, 2, -3};
            for (boolean reverse : List.of(false, true)) for (boolean exclusive : List.of(false, true)) {
                for (boolean product : List.of(false, true)) {
                    long[] expected = new long[3]; long accumulator = product ? 1 : 0;
                    for (int step = 0; step < input.length; step++) {
                        int index = reverse ? input.length - step - 1 : step;
                        if (exclusive) expected[index] = accumulator;
                        accumulator = product ? accumulator * input[index] : accumulator + input[index];
                        if (type == DataType.INT32) accumulator = (int) accumulator;
                        if (!exclusive) expected[index] = accumulator;
                    }
                    var node = MetalMpsGraphProgram.Node.scan(product ? MetalMpsGraphProgram.NodeKind.CUM_PROD
                            : MetalMpsGraphProgram.NodeKind.CUM_SUM, 0, 1, 0, exclusive, reverse);
                    assertArrayEquals(bytes(type, expected), execute(List.of(value(type, 3), value(type, 3)),
                            List.of(node), new int[] {0}, new byte[][] {bytes(type, input)}, 1));
                }
            }
        }
    }

    @Test
    void lowStoredExtremaPreserveNanPayloadSignedZerosAndSubnormals() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            long nan = type == DataType.BFLOAT16 ? 0x7f81 : 0x7c01;
            var values = List.of(value(type, 3, 2), value(type, 3));
            byte[][] input = {bytes(type, 0x8000, 0, 1, 2, nan, 0)};
            assertArrayEquals(bytes(type, 0x8000, 1, nan), execute(values,
                    List.of(reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MIN, 0, 1, List.of(1), false)),
                    new int[] {0}, input, 1));
            assertArrayEquals(bytes(type, 0, 2, nan), execute(values,
                    List.of(reduction(MetalMpsGraphProgram.NodeKind.REDUCTION_MAX, 0, 1, List.of(1), false)),
                    new int[] {0}, input, 1));
        }
    }

    @Test
    void lowCumulativeKeepsFloat32AccumulatorUntilEachPrefixPublication() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            float halfUlp = type == DataType.BFLOAT16 ? 0x1p-8f : 0x1p-11f;
            var values = List.of(value(type, 3), value(type, 3));
            var node = MetalMpsGraphProgram.Node.scan(MetalMpsGraphProgram.NodeKind.CUM_SUM,
                    0, 1, 0, false, false);
            assertArrayEquals(numeric(type, 1, 1, 1 + 2 * halfUlp), execute(values, List.of(node),
                    new int[] {0}, new byte[][] {numeric(type, 1, halfUlp, halfUlp)}, 1));
        }
    }

    private static long numericWord(DataType type, float value) {
        int word = Float.floatToRawIntBits(value);
        if (type == DataType.FLOAT32) return Integer.toUnsignedLong(word);
        if (type == DataType.FLOAT16) return Short.toUnsignedLong(Float.floatToFloat16(value));
        if (Float.isNaN(value)) return 0x7fc0;
        return Integer.toUnsignedLong(word + 0x7fff + ((word >>> 16) & 1)) >>> 16;
    }

    private static byte[] numeric(DataType type, float... input) {
        long[] words = new long[input.length];
        for (int index = 0; index < input.length; index++) words[index] = numericWord(type, input[index]);
        return bytes(type, words);
    }

    private static float[] numericFloats(DataType type, byte[] bytes) {
        if (type == DataType.FLOAT32) return floats(bytes);
        var buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        float[] result = new float[bytes.length / Short.BYTES];
        for (int index = 0; index < result.length; index++) {
            short word = buffer.getShort();
            result[index] = type == DataType.FLOAT16 ? Float.float16ToFloat(word)
                    : Float.intBitsToFloat(Short.toUnsignedInt(word) << 16);
        }
        return result;
    }

    private static void assertNumeric(DataType type, float expected, float actual, String context) {
        float rounded = numericFloats(type, numeric(type, expected))[0];
        float tolerance = type == DataType.FLOAT32 ? Math.max(1e-6f, Math.abs(rounded) * 2e-6f)
                : Math.max(type == DataType.BFLOAT16 ? 0x1p-126f : 0x1p-24f,
                        Math.abs(rounded) * (type == DataType.BFLOAT16 ? 0x1p-6f : 0x1p-9f));
        assertEquals(rounded, actual, tolerance, context);
    }

    private static MetalMpsGraphProgram.Node reduction(MetalMpsGraphProgram.NodeKind kind,
            int input, int output, List<Integer> axes, boolean keep) {
        return MetalMpsGraphProgram.Node.reduction(kind, input, output,
                MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, axes, keep);
    }

    private static MetalMpsGraphProgram.Node masked(MetalMpsGraphProgram.NodeKind kind,
            int input, int mask, int output, int axis) {
        return MetalMpsGraphProgram.Node.generic(kind, new int[] {input, mask}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.MASKED_REDUCTION, new long[] {axis});
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static byte[] execute(List<MetalMpsGraphProgram.ValueDescriptor> values,
            List<MetalMpsGraphProgram.Node> nodes, int[] feeds, byte[][] initial,
            int target, int... siblingTargets) {
        String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(library != null && !library.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(Path.of(library).toAbsolutePath().normalize());
        MetalNativeApi.Handle context = null, executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try (Arena arena = Arena.ofConfined()) {
            context = api.createContext();
            int[] targetIndices = new int[siblingTargets.length + 1];
            targetIndices[0] = target;
            System.arraycopy(siblingTargets, 0, targetIndices, 1, siblingTargets.length);
            executable = api.createProgramExecutable(context, values, new MetalMpsGraphProgram(nodes),
                    feeds, targetIndices, MetalPreparedRoute.CUSTOM_PROGRAM);
            MemorySegment all = arena.allocate(ADDRESS, values.size());
            for (int index = 0; index < values.size(); index++) {
                var buffer = api.createBuffer(context, values.get(index).byteCount());
                buffers.add(buffer); all.setAtIndex(ADDRESS, index, buffer.carrier());
            }
            for (int index = 0; index < feeds.length; index++) {
                MemorySegment source = arena.allocate(initial[index].length, 1);
                MemorySegment.copy(initial[index], 0, source, JAVA_BYTE, 0, initial[index].length);
                api.upload(buffers.get(feeds[index]), 0, source, source.byteSize());
            }
            MemorySegment targets = arena.allocate(ADDRESS, targetIndices.length);
            for (int index = 0; index < targetIndices.length; index++) {
                targets.setAtIndex(ADDRESS, index, buffers.get(targetIndices[index]).carrier());
            }
            api.runExecutable(executable, buffers.size(), all, targetIndices.length, targets);
            MemorySegment result = arena.allocate(values.get(target).byteCount(), 1);
            api.download(buffers.get(target), 0, result, result.byteSize());
            for (int index = 0; index < feeds.length; index++) {
                MemorySegment source = arena.allocate(initial[index].length, 1);
                api.download(buffers.get(feeds[index]), 0, source, source.byteSize());
                assertArrayEquals(initial[index], source.toArray(JAVA_BYTE));
            }
            return result.toArray(JAVA_BYTE);
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static byte[] bytes(DataType type, long... words) {
        var result = ByteBuffer.allocate(Math.multiplyExact(words.length, type.byteWidth()))
                .order(ByteOrder.nativeOrder());
        for (long word : words) {
            if (type.byteWidth() == 2) result.putShort((short) word);
            else if (type.byteWidth() == 4) result.putInt((int) word); else result.putLong(word);
        }
        return result.array();
    }

    private static byte[] f32(float... values) {
        long[] words = new long[values.length];
        for (int index = 0; index < values.length; index++) words[index] = Float.floatToRawIntBits(values[index]);
        return bytes(DataType.FLOAT32, words);
    }

    private static float[] floats(byte[] bytes) {
        var buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
        float[] result = new float[bytes.length / Float.BYTES];
        for (int index = 0; index < result.length; index++) result[index] = buffer.getFloat();
        return result;
    }
}
