package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.NumericalForm;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float8E4M3Bits;
import io.github.pho001.synaptik.model.datatype.Float8E5M2Bits;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class CpuMatmulReferenceTest {
    private final CpuMatmulLowering lowering = new CpuMatmulLowering();

    /** Qualifies singleton typed products independently of the permitted multi-term FMA form. */
    @Test void singletonProductsNeverAddAnIdentityOrFuseWithZero() {
        for (var type : java.util.List.of(DataType.FLOAT32,DataType.FLOAT64))
            for (var form : NumericalForm.values()) for (boolean underflow : java.util.List.of(false,true)) {
                var geometry = lowering.lower(d(type,1,1),d(type,1,1),d(type,1,1));
                Object left = type == DataType.FLOAT64 ? new double[]{underflow ? -Double.MIN_VALUE : -0d}
                        : new float[]{underflow ? -Float.MIN_VALUE : -0f};
                Object right = type == DataType.FLOAT64 ? new double[]{underflow ? .25d : 1d}
                        : new float[]{underflow ? .25f : 1f};
                Object output = type == DataType.FLOAT64 ? new double[1] : new float[1];
                CpuMatmulReferenceKernel.evaluate(geometry,form,left,right,output);
                assertEquals(Long.MIN_VALUE,Double.doubleToRawLongBits(value(type,output)));
                var empty = lowering.lower(d(type,1,0),d(type,0,1),d(type,1,1));
                CpuMatmulReferenceKernel.evaluate(empty,form,left,right,output);
                assertEquals(0L,Double.doubleToRawLongBits(value(type,output)));
            }
    }

    /** Mixed low operands promote before arithmetic; empty contractions publish positive zero. */
    @Test void evaluatesFloatingMixedLowAndEmptyK() {
        for (var type : java.util.List.of(DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            var mixed = lowering.lower(d(type, 1, 2), d(DataType.FLOAT32, 2, 1),
                    d(DataType.FLOAT32, 1, 1));
            float[] result = {Float.NaN};
            CpuMatmulReferenceKernel.evaluate(mixed, NumericalForm.SEQUENTIAL,
                    carrier(type, 2, 3), new float[]{4, 5}, result);
            assertEquals(23f, result[0], type.toString());
        }
        for (var type : java.util.List.of(DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2, DataType.FLOAT32, DataType.FLOAT64)) {
            var empty = lowering.lower(d(type, 2, 0), d(type, 0, 3), d(type, 2, 3));
            Object zeros = java.lang.reflect.Array.newInstance(
                    carrier(type, 0, 0).getClass().componentType(), 6);
            for (int i = 0; i < 6; i++) java.lang.reflect.Array.set(zeros, i,
                    java.lang.reflect.Array.get(carrier(type, 1, 1), 0));
            CpuMatmulReferenceKernel.evaluate(empty, NumericalForm.SEQUENTIAL,
                    carrier(type, 0, 0), carrier(type, 0, 0), zeros);
            for (int i = 0; i < 6; i++)
                assertEquals(0L, Double.doubleToRawLongBits(
                        ((Number) java.lang.reflect.Array.get(zeros, i)).doubleValue()), type + ":" + i);
        }
    }

    /** Checks all floating promotion pairs without bypassing low carriers or final narrowing. */
    @Test void evaluatesAllOrderedFloatingPromotionPairs() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT16,
                DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType leftType : floating) for (DataType rightType : floating) {
            DataType resultType = io.github.pho001.synaptik.model.datatype.DataTypePromotion
                    .promoteNumeric(leftType, rightType);
            var geometry = lowering.lower(d(leftType, 1, 2), d(rightType, 2, 1),
                    d(resultType, 1, 1));
            Object result = carrier(resultType, 1.0f, 0.0f);
            for (var form : NumericalForm.values()) {
                CpuMatmulReferenceKernel.evaluate(geometry, form,
                        carrier(leftType, 2.0f, 3.0f), carrier(rightType, 4.0f, 5.0f), result);
                assertEquals(value(resultType, carrier(resultType, 23.0f, 0.0f)),
                        value(resultType, result), 0.0, leftType + " x " + rightType + "/" + form);
            }
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
            case FLOAT16 -> new short[] {Float.floatToFloat16(first), Float.floatToFloat16(second)};
            case FLOAT8_E4M3 -> new byte[] {Float8E4M3Bits.fromFloat(first), Float8E4M3Bits.fromFloat(second)};
            case FLOAT8_E5M2 -> new byte[] {Float8E5M2Bits.fromFloat(first), Float8E5M2Bits.fromFloat(second)};
            case FLOAT32 -> new float[] {first, second};
            case FLOAT64 -> new double[] {first, second};
            default -> throw new AssertionError(type);
        };
    }

    private static double value(DataType type, Object carrier) {
        return switch (type) {
            case BFLOAT16 -> BFloat16Bits.toFloat(((short[]) carrier)[0]);
            case FLOAT16 -> Float.float16ToFloat(((short[]) carrier)[0]);
            case FLOAT8_E4M3 -> Float8E4M3Bits.toFloat(((byte[]) carrier)[0]);
            case FLOAT8_E5M2 -> Float8E5M2Bits.toFloat(((byte[]) carrier)[0]);
            case FLOAT32 -> ((float[]) carrier)[0];
            case FLOAT64 -> ((double[]) carrier)[0];
            default -> throw new AssertionError(type);
        };
    }
}
