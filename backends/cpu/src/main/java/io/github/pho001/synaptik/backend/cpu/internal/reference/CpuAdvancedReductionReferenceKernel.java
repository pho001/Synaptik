package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Objects;

/**
 * Independent mathematical oracle for one static advanced floating reduction.
 *
 * <p>The oracle derives output and selected-domain coordinates directly from Shapes and axes. It
 * deliberately does not consume CPU lowering geometry, packed invocation data, generated-code
 * helpers, or exact-state workspace. Exact represented L1 sums and statistical means use decimal
 * images of the binary values before the result-format rounding seam; transcendental finishing
 * uses {@link StrictMath}.</p>
 */
public final class CpuAdvancedReductionReferenceKernel {
    private static final long CANONICAL_NAN = 0x7ff8000000000000L;
    private static final MathContext DIVISION = new MathContext(110, RoundingMode.HALF_EVEN);

    private CpuAdvancedReductionReferenceKernel() { }

    /**
     * Evaluates one reduction into a dense row-major binary64 projection of represented results.
     *
     * @param kind non-null exact reduction meaning
     * @param dataType non-null represented input/output type
     * @param input represented input values addressed by {@code inputOffset} and
     *     {@code inputStrides}; not {@code null}
     * @param inputExtents non-null static non-negative input extents
     * @param inputOffset non-negative input element offset
     * @param inputStrides non-null non-negative input element strides, one per input axis
     * @param orderedAxes non-null normalized ordered distinct selected axes
     * @param keepDimensions whether selected axes remain as extent-one output axes
     * @param correction non-negative statistical denominator correction, otherwise zero
     * @return a newly allocated dense result whose values have been rounded to {@code dataType}
     * @throws NullPointerException if a reference argument is {@code null}
     * @throws IllegalArgumentException if rank, axes, type, correction, or addressed span is invalid
     * @throws ArithmeticException if count or address arithmetic overflows
     */
    public static double[] evaluate(CpuAdvancedReductionIr.Kind kind, DataType dataType,
            double[] input, long[] inputExtents, long inputOffset, long[] inputStrides,
            int[] orderedAxes, boolean keepDimensions, long correction) {
        Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(input, "input"); Objects.requireNonNull(inputExtents, "inputExtents");
        Objects.requireNonNull(inputStrides, "inputStrides");
        Objects.requireNonNull(orderedAxes, "orderedAxes");
        if (inputExtents.length != inputStrides.length || inputOffset < 0
                || !floating(dataType) || correction < 0) throw new IllegalArgumentException(
                        "advanced-reference structural facts disagree");
        boolean statistical = kind == CpuAdvancedReductionIr.Kind.VARIANCE
                || kind == CpuAdvancedReductionIr.Kind.STANDARD_DEVIATION;
        if (!statistical && correction != 0) throw new IllegalArgumentException(
                "correction belongs only to statistics");
        boolean[] selected = new boolean[inputExtents.length];
        for (int axis : orderedAxes) {
            if (axis < 0 || axis >= selected.length || selected[axis])
                throw new IllegalArgumentException("axes must be normalized and distinct");
            selected[axis] = true;
        }
        long domain = 1;
        for (int axis = 0; axis < selected.length; axis++) {
            if (inputExtents[axis] < 0 || inputStrides[axis] < 0)
                throw new IllegalArgumentException("negative Shape or layout fact");
            if (selected[axis]) domain = Math.multiplyExact(domain, inputExtents[axis]);
        }
        if (statistical && domain <= correction) throw new IllegalArgumentException(
                "selected-domain count must exceed correction");
        long[] outputExtents = reduced(inputExtents, selected, keepDimensions);
        long outputCount = count(outputExtents);
        if (outputCount > Integer.MAX_VALUE) throw new IllegalArgumentException(
                "reference output is too large for a heap oracle");
        validateSpan(input.length, inputOffset, inputExtents, inputStrides);
        double[] output = new double[(int) outputCount];
        long[] outputCoordinates = new long[outputExtents.length];
        long[] inputCoordinates = new long[inputExtents.length];
        for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
            Arrays.fill(inputCoordinates, 0);
            for (int inputAxis = 0, outputAxis = 0; inputAxis < selected.length; inputAxis++) {
                if (selected[inputAxis]) { if (keepDimensions) outputAxis++; }
                else inputCoordinates[inputAxis] = outputCoordinates[outputAxis++];
            }
            output[outputIndex] = represented(dataType, reduce(kind, dataType, input, inputOffset,
                    inputStrides, inputExtents, selected, inputCoordinates, domain, correction));
            increment(outputCoordinates, outputExtents);
        }
        return output;
    }

    private static double reduce(CpuAdvancedReductionIr.Kind kind, DataType type, double[] input,
            long offset, long[] strides, long[] extents, boolean[] selected, long[] coordinates,
            long domain, long correction) {
        if (domain == 0) return kind == CpuAdvancedReductionIr.Kind.LOG_SUM_EXP
                ? Double.NEGATIVE_INFINITY : 0.0;
        if (kind == CpuAdvancedReductionIr.Kind.LOG_SUM_EXP && domain == 1) {
            double point = value(input, offset, strides, coordinates);
            return Double.isNaN(point) ? canonicalNaN() : point;
        }
        BigDecimal exact = BigDecimal.ZERO;
        double maximum = Double.NEGATIVE_INFINITY, scale = 0.0, scaledSquares = 0.0;
        boolean nan = false, positiveInfinity = false, infinity = false;
        long[] domainCoordinates = coordinates.clone();
        for (long index = 0; index < domain; index++) {
            double value = value(input, offset, strides, domainCoordinates);
            nan |= Double.isNaN(value); infinity |= Double.isInfinite(value);
            positiveInfinity |= value == Double.POSITIVE_INFINITY;
            if (Double.isFinite(value)) {
                if (kind == CpuAdvancedReductionIr.Kind.LOG_SUM_EXP) maximum = Math.max(maximum, value);
                else if (kind == CpuAdvancedReductionIr.Kind.L1_NORM)
                    exact = exact.add(new BigDecimal(Math.abs(value)));
                else if (kind == CpuAdvancedReductionIr.Kind.L2_NORM) {
                    double absolute = Math.abs(value);
                    if (absolute != 0.0 && scale < absolute) {
                        double ratio = scale / absolute;
                        scaledSquares = 1.0 + scaledSquares * ratio * ratio; scale = absolute;
                    } else if (absolute != 0.0) {
                        double ratio = absolute / scale; scaledSquares += ratio * ratio;
                    }
                } else exact = exact.add(new BigDecimal(value));
            }
            incrementSelected(domainCoordinates, extents, selected);
        }
        if (nan) return canonicalNaN();
        if (kind == CpuAdvancedReductionIr.Kind.LOG_SUM_EXP) {
            if (positiveInfinity) return Double.POSITIVE_INFINITY;
            if (maximum == Double.NEGATIVE_INFINITY) return Double.NEGATIVE_INFINITY;
            double sum = 0.0, compensation = 0.0; Arrays.fill(domainCoordinates, 0);
            for (int axis = 0; axis < selected.length; axis++) if (!selected[axis])
                domainCoordinates[axis] = coordinates[axis];
            for (long index = 0; index < domain; index++) {
                double term = StrictMath.exp(value(input, offset, strides, domainCoordinates) - maximum);
                double adjusted = term - compensation, next = sum + adjusted;
                compensation = (next - sum) - adjusted; sum = next;
                incrementSelected(domainCoordinates, extents, selected);
            }
            return maximum + StrictMath.log(sum);
        }
        if (infinity) return kind == CpuAdvancedReductionIr.Kind.VARIANCE
                || kind == CpuAdvancedReductionIr.Kind.STANDARD_DEVIATION
                ? canonicalNaN() : Double.POSITIVE_INFINITY;
        if (kind == CpuAdvancedReductionIr.Kind.L1_NORM) return exact.doubleValue();
        if (kind == CpuAdvancedReductionIr.Kind.L2_NORM)
            return scale == 0.0 ? 0.0 : scale * StrictMath.sqrt(scaledSquares);
        double mean = represented(type, exact.divide(BigDecimal.valueOf(domain), DIVISION).doubleValue());
        double deviations = 0.0, deviationCompensation = 0.0;
        double squares = 0.0, squareCompensation = 0.0; Arrays.fill(domainCoordinates, 0);
        for (int axis = 0; axis < selected.length; axis++) if (!selected[axis])
            domainCoordinates[axis] = coordinates[axis];
        for (long index = 0; index < domain; index++) {
            double deviation = value(input, offset, strides, domainCoordinates) - mean;
            double adjusted = deviation - deviationCompensation, next = deviations + adjusted;
            deviationCompensation = (next - deviations) - adjusted; deviations = next;
            double square = deviation * deviation;
            if (square == Double.POSITIVE_INFINITY) {
                squares = Double.POSITIVE_INFINITY; squareCompensation = 0.0;
            } else if (squares != Double.POSITIVE_INFINITY) {
                adjusted = square - squareCompensation; next = squares + adjusted;
                squareCompensation = (next - squares) - adjusted; squares = next;
            }
            incrementSelected(domainCoordinates, extents, selected);
        }
        double numerator = squares == Double.POSITIVE_INFINITY ? Double.POSITIVE_INFINITY
                : squares - deviations * deviations / domain;
        if (numerator < 0.0) numerator = 0.0;
        double result = numerator / (domain - correction);
        return kind == CpuAdvancedReductionIr.Kind.STANDARD_DEVIATION
                ? StrictMath.sqrt(result) : result;
    }

    private static double value(double[] input, long offset, long[] strides, long[] coordinates) {
        long address = offset;
        for (int axis = 0; axis < strides.length; axis++) address = Math.addExact(address,
                Math.multiplyExact(strides[axis], coordinates[axis]));
        return input[Math.toIntExact(address)];
    }

    private static void incrementSelected(long[] coordinates, long[] extents, boolean[] selected) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) if (selected[axis]) {
            if (++coordinates[axis] < extents[axis]) return; coordinates[axis] = 0;
        }
    }

    private static void increment(long[] coordinates, long[] extents) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            if (++coordinates[axis] < extents[axis]) return; coordinates[axis] = 0;
        }
    }

    private static long[] reduced(long[] input, boolean[] selected, boolean keep) {
        int selectedCount = 0; for (boolean value : selected) if (value) selectedCount++;
        long[] result = new long[keep ? input.length : input.length - selectedCount];
        for (int axis = 0, out = 0; axis < input.length; axis++) {
            if (selected[axis]) { if (keep) result[out++] = 1; }
            else result[out++] = input[axis];
        }
        return result;
    }

    private static long count(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1; for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static void validateSpan(int length, long offset, long[] extents, long[] strides) {
        if (count(extents) == 0) return;
        long maximum = offset;
        for (int axis = 0; axis < extents.length; axis++) maximum = Math.addExact(maximum,
                Math.multiplyExact(extents[axis] - 1, strides[axis]));
        if (maximum >= length) throw new IllegalArgumentException("input carrier is too small");
    }

    private static boolean floating(DataType type) { return type == DataType.FLOAT64
            || type == DataType.FLOAT32 || type == DataType.BFLOAT16; }

    private static double represented(DataType type, double value) {
        if (Double.isNaN(value)) return canonicalNaN();
        if (type == DataType.FLOAT64) return value;
        float narrowed = (float) value;
        if (type == DataType.FLOAT32) return narrowed;
        int bits = Float.floatToRawIntBits(narrowed);
        if ((bits & 0x7fffffff) > 0x7f800000) return Float.intBitsToFloat(0x7fc00000);
        bits += 0x7fff + (bits >>> 16 & 1);
        return Float.intBitsToFloat(bits & 0xffff0000);
    }

    private static double canonicalNaN() { return Double.longBitsToDouble(CANONICAL_NAN); }
}
