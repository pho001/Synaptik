package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import java.util.Arrays;
import java.util.Objects;

/**
 * Independent clean-Java oracle for the admitted stable softmax algorithm.
 *
 * <p>The oracle mirrors the useful primitive work of each generated kind without calling
 * production lowering or numerical helpers: softmax does not evaluate a logarithm, while
 * log-softmax evaluates one logarithm per complete normalization slice. FLOAT64 works in double;
 * FLOAT32 and both low types work in float, with one final low narrowing. The all-finite
 * algorithm retains compensated accumulation. NaN, infinity and empty slices follow
 * Model's separate result-class rules; negative-infinity slices use literal unshifted sites.</p>
 */
public final class CpuSoftmaxReferenceKernel {
    private CpuSoftmaxReferenceKernel() { }

    /**
     * Evaluates one represented static softmax input into dense logical row-major order.
     * @param kind non-null exact softmax or log-softmax meaning
     * @param type non-null represented floating input/output type
     * @param input non-null represented input values
     * @param extents non-null rank-positive static Shape extents
     * @param inputOffset non-negative input element offset
     * @param inputStrides non-null non-negative input element strides
     * @param axis normalized selected axis, including an empty selected extent
     * @return a new dense logical result rounded to {@code type}
     * @throws NullPointerException if a reference argument is {@code null}
     * @throws IllegalArgumentException if geometry or the represented input span is invalid
     * @throws ArithmeticException if count or address arithmetic overflows
     */
    public static double[] evaluate(SoftmaxKind kind, DataType type, double[] input,
            long[] extents, long inputOffset, long[] inputStrides, int axis) {
        Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(type, "type");
        Objects.requireNonNull(input, "input"); Objects.requireNonNull(extents, "extents");
        Objects.requireNonNull(inputStrides, "inputStrides");
        if ((type != DataType.FLOAT64 && type != DataType.FLOAT32
                && type != DataType.BFLOAT16 && type != DataType.FLOAT16 && type != DataType.FLOAT8_E4M3 && type != DataType.FLOAT8_E5M2)
                || extents.length == 0 || extents.length != inputStrides.length
                || axis < 0 || axis >= extents.length || inputOffset < 0
                || Arrays.stream(extents).anyMatch(v -> v < 0)
                || Arrays.stream(inputStrides).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("softmax reference geometry is invalid");
        long count = count(extents);
        if (count > Integer.MAX_VALUE) throw new IllegalArgumentException("reference output is too large");
        validateSpan(input.length, inputOffset, extents, inputStrides);
        double[] output = new double[(int) count];
        if (count == 0) return output;
        long slices = count / extents[axis];
        for (long slice = 0; slice < slices; slice++) {
            long[] coordinates = sliceCoordinates(slice, extents, axis);
            if (type == DataType.FLOAT64) evaluateDouble(kind, input, inputOffset, inputStrides,
                    extents, axis, coordinates, output);
            else evaluateFloat(kind, type, input, inputOffset, inputStrides, extents, axis,
                    coordinates, output);
        }
        return output;
    }

    private static void evaluateDouble(SoftmaxKind kind, double[] input, long offset,
            long[] strides, long[] extents, int axis, long[] coordinates, double[] output) {
        double maximum = Double.NEGATIVE_INFINITY;
        boolean unshifted = false;
        for (long c = 0; c < extents[axis]; c++) {
            coordinates[axis] = c;
            double value = value(input, offset, strides, coordinates);
            maximum = Math.max(maximum, value);
            unshifted |= value == Double.NEGATIVE_INFINITY;
        }
        if (maximum == Double.POSITIVE_INFINITY) {
            for (long c = 0; c < extents[axis]; c++) {
                coordinates[axis] = c;
                double value = value(input, offset, strides, coordinates);
                output[Math.toIntExact(ordinal(coordinates, extents))] =
                        value == Double.POSITIVE_INFINITY ? Double.NaN
                                : kind == SoftmaxKind.SOFTMAX ? 0.0d : Double.NEGATIVE_INFINITY;
            }
            return;
        }
        if (unshifted && !Double.isNaN(maximum)) maximum = 0.0d;
        double sum = 0, compensation = 0;
        for (long c = 0; c < extents[axis]; c++) {
            coordinates[axis] = c;
            double addend = StrictMath.exp(value(input, offset, strides, coordinates) - maximum);
            if (unshifted) sum += addend;
            else {
                double adjusted = addend - compensation, temporary = sum + adjusted;
                compensation = (temporary - sum) - adjusted;
                sum = temporary;
            }
        }
        double logarithm = kind == SoftmaxKind.LOG_SOFTMAX ? StrictMath.log(sum) : 0.0d;
        for (long c = 0; c < extents[axis]; c++) {
            coordinates[axis] = c;
            double shift = value(input, offset, strides, coordinates) - maximum;
            output[Math.toIntExact(ordinal(coordinates, extents))] =
                    kind == SoftmaxKind.SOFTMAX ? StrictMath.exp(shift) / sum : shift - logarithm;
        }
    }

    private static void evaluateFloat(SoftmaxKind kind, DataType type, double[] input, long offset,
            long[] strides, long[] extents, int axis, long[] coordinates, double[] output) {
        float maximum = Float.NEGATIVE_INFINITY;
        boolean unshifted = false;
        for (long c = 0; c < extents[axis]; c++) {
            coordinates[axis] = c;
            float value = (float) represented(type, value(input, offset, strides, coordinates));
            maximum = Math.max(maximum, value);
            unshifted |= value == Float.NEGATIVE_INFINITY;
        }
        if (maximum == Float.POSITIVE_INFINITY) {
            for (long c = 0; c < extents[axis]; c++) {
                coordinates[axis] = c;
                float value = (float) represented(type, value(input, offset, strides, coordinates));
                output[Math.toIntExact(ordinal(coordinates, extents))] =
                        value == Float.POSITIVE_INFINITY ? Double.NaN
                                : kind == SoftmaxKind.SOFTMAX ? 0.0d : Double.NEGATIVE_INFINITY;
            }
            return;
        }
        if (unshifted && !Float.isNaN(maximum)) maximum = 0.0f;
        float sum = 0, compensation = 0;
        for (long c = 0; c < extents[axis]; c++) {
            coordinates[axis] = c;
            float value = (float) represented(type, value(input, offset, strides, coordinates));
            float addend = (float) StrictMath.exp(value - maximum);
            if (unshifted) sum += addend;
            else {
                float adjusted = addend - compensation, temporary = sum + adjusted;
                compensation = (temporary - sum) - adjusted;
                sum = temporary;
            }
        }
        float logarithm = kind == SoftmaxKind.LOG_SOFTMAX ? (float) StrictMath.log(sum) : 0.0f;
        for (long c = 0; c < extents[axis]; c++) {
            coordinates[axis] = c;
            float value = (float) represented(type, value(input, offset, strides, coordinates));
            float shift = value - maximum;
            float result = kind == SoftmaxKind.SOFTMAX ? (float) StrictMath.exp(shift) / sum
                    : shift - logarithm;
            output[Math.toIntExact(ordinal(coordinates, extents))] = represented(type, result);
        }
    }

    private static long[] sliceCoordinates(long slice, long[] extents, int axis) {
        long[] coordinates = new long[extents.length];
        for (int dimension = extents.length - 1; dimension >= 0; dimension--) {
            if (dimension == axis) continue;
            coordinates[dimension] = slice % extents[dimension]; slice /= extents[dimension];
        }
        return coordinates;
    }
    private static double value(double[] input, long offset, long[] strides, long[] coordinates) {
        long address = offset;
        for (int axis = 0; axis < coordinates.length; axis++)
            address = Math.addExact(address, Math.multiplyExact(coordinates[axis], strides[axis]));
        return input[Math.toIntExact(address)];
    }
    private static long ordinal(long[] coordinates, long[] extents) {
        long result = 0; for (int axis = 0; axis < extents.length; axis++)
            result = Math.addExact(Math.multiplyExact(result, extents[axis]), coordinates[axis]);
        return result;
    }
    private static long count(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1; for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }
    private static void validateSpan(int length, long offset, long[] extents, long[] strides) {
        if (count(extents) == 0) return; long maximum = offset;
        for (int axis = 0; axis < extents.length; axis++) maximum = Math.addExact(maximum,
                Math.multiplyExact(extents[axis] - 1, strides[axis]));
        if (maximum >= length) throw new IllegalArgumentException("reference input span is too small");
    }
    private static double represented(DataType type, double value) {
        if (type == DataType.FLOAT64) return value;
        float narrowed = (float) value;
        if (type == DataType.FLOAT32) return narrowed;
        if (type == DataType.FLOAT8_E4M3) return io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.toFloat(io.github.pho001.synaptik.model.datatype.Float8E4M3Bits.fromFloat(narrowed));
        if (type == DataType.FLOAT8_E5M2) return io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.toFloat(io.github.pho001.synaptik.model.datatype.Float8E5M2Bits.fromFloat(narrowed));
        if (Float.isNaN(narrowed)) return Double.NaN;
        if (type == DataType.FLOAT16)
            return Float.float16ToFloat(Float.floatToFloat16(narrowed));
        int bits = Float.floatToRawIntBits(narrowed);
        int upper = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
        return Float.intBitsToFloat(upper << 16);
    }
}
