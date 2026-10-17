package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.NumericalForm;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class CpuMatmulReferenceTest {
    private final CpuMatmulLowering lowering = new CpuMatmulLowering();

    @Test void evaluatesFloatingMixedBfloatAndEmptyK() {
        var mixed = lowering.lower(d(DataType.BFLOAT16, 2, 3), d(DataType.FLOAT32, 3, 2),
                d(DataType.FLOAT32, 2, 2));
        short[] left = {bf(1), bf(2), bf(3), bf(4), bf(5), bf(6)};
        float[] right = {7, 8, 9, 10, 11, 12}; float[] output = new float[4];
        CpuMatmulReferenceKernel.evaluate(mixed, NumericalForm.SEQUENTIAL, left, right, output);
        assertArrayEquals(new float[] {58, 64, 139, 154}, output);
        var empty = lowering.lower(d(DataType.FLOAT64, 2, 0), d(DataType.FLOAT64, 0, 3),
                d(DataType.FLOAT64, 2, 3));
        double[] zeros = new double[6];
        CpuMatmulReferenceKernel.evaluate(empty, NumericalForm.SEQUENTIAL,
                new double[0], new double[0], zeros);
        for (double value : zeros) assertEquals(0L, Double.doubleToRawLongBits(value));
    }

    @Test void evaluatesAllNineOrderedFloatingPromotionPairs() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType leftType : floating) for (DataType rightType : floating) {
            DataType resultType = io.github.pho001.synaptik.model.datatype.DataTypePromotion
                    .promoteNumeric(leftType, rightType);
            var geometry = lowering.lower(d(leftType, 1, 2), d(rightType, 2, 1),
                    d(resultType, 1, 1));
            Object result = carrier(resultType, 1.0f, 0.0f);
            CpuMatmulReferenceKernel.evaluate(geometry, NumericalForm.SEQUENTIAL,
                    carrier(leftType, 2.0f, 3.0f), carrier(rightType, 4.0f, 5.0f), result);
            assertEquals(23.0, value(resultType, result), 0.0,
                    leftType + " x " + rightType);
        }
    }

    @Test void evaluatesAllIntegralPromotionsWithModularArithmetic() {
        var i32 = lowering.lower(d(DataType.INT32, 1, 2), d(DataType.INT32, 2, 1),
                d(DataType.INT32, 1, 1));
        int[] out32 = new int[1];
        CpuMatmulReferenceKernel.evaluate(i32, NumericalForm.SEQUENTIAL,
                new int[] {Integer.MAX_VALUE, 2}, new int[] {2, 3}, out32);
        assertEquals(4, out32[0]);
        for (DataType leftType : new DataType[] {DataType.INT32, DataType.INT64})
            for (DataType rightType : new DataType[] {DataType.INT32, DataType.INT64}) {
                DataType resultType = leftType == DataType.INT64 || rightType == DataType.INT64
                        ? DataType.INT64 : DataType.INT32;
                var geometry = lowering.lower(d(leftType, 1, 1), d(rightType, 1, 1),
                        d(resultType, 1, 1));
                Object left = leftType == DataType.INT32 ? new int[] {3} : new long[] {3};
                Object right = rightType == DataType.INT32 ? new int[] {4} : new long[] {4};
                Object result = resultType == DataType.INT32 ? new int[1] : new long[1];
                CpuMatmulReferenceKernel.evaluate(geometry, NumericalForm.SEQUENTIAL, left, right, result);
                assertEquals(12L, result instanceof int[] values ? values[0] : ((long[]) result)[0]);
            }
    }

    private static TensorDescriptor d(DataType type, long... extents) {
        Shape shape = Shape.of(extents);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
    private static short bf(float value) { return BFloat16Bits.fromFloat(value); }

    private static Object carrier(DataType type, float first, float second) {
        return switch (type) {
            case BFLOAT16 -> new short[] {bf(first), bf(second)};
            case FLOAT32 -> new float[] {first, second};
            case FLOAT64 -> new double[] {first, second};
            default -> throw new AssertionError(type);
        };
    }

    private static double value(DataType type, Object carrier) {
        return switch (type) {
            case BFLOAT16 -> BFloat16Bits.toFloat(((short[]) carrier)[0]);
            case FLOAT32 -> ((float[]) carrier)[0];
            case FLOAT64 -> ((double[]) carrier)[0];
            default -> throw new AssertionError(type);
        };
    }
}
