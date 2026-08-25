package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.*;

/** Independent clean-Java mathematical oracle for trailing Layer and RMS normalization. */
public final class CpuTrailingNormalizationReferenceKernel {
    private static final MathContext DIVISION = new MathContext(110, RoundingMode.HALF_EVEN);
    private CpuTrailingNormalizationReferenceKernel() { }

    /**
     * Evaluates one semantic occurrence into dense logical row-major result order.
     *
     * @param kind exact Layer or RMS family
     * @param form exact visible operand form
     * @param inputTypes represented type of each semantic input
     * @param resultType promoted result type
     * @param epsilon exact positive result-type epsilon projected as binary64
     * @param inputs semantic input carriers in occurrence order
     * @param extents semantic input extents in occurrence order
     * @param offsets non-negative element offsets in occurrence order
     * @param strides semantic input element strides in occurrence order
     * @param normalizedRank positive trailing normalized rank
     * @return newly allocated dense represented result
     * @throws NullPointerException if a required reference or nested input is {@code null}
     * @throws IllegalArgumentException if family, form, input cardinality, epsilon, rank, layout,
     *     carrier span, or output-size facts are invalid
     * @throws ArithmeticException if exact element, address, or ordinal arithmetic overflows
     */
    public static double[] evaluate(CpuTrailingNormalizationIr.Kind kind,
            CpuTrailingNormalizationIr.Form form, List<DataType> inputTypes, DataType resultType,
            double epsilon, double[][] inputs, long[][] extents, long[] offsets, long[][] strides,
            int normalizedRank) {
        Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(form, "form");
        inputTypes = List.copyOf(inputTypes); Objects.requireNonNull(resultType, "resultType");
        Objects.requireNonNull(inputs, "inputs"); Objects.requireNonNull(extents, "extents");
        Objects.requireNonNull(offsets, "offsets"); Objects.requireNonNull(strides, "strides");
        int expected = switch (form) { case LAYER, RMS -> 1; case RMS_SCALED -> 2;
            case LAYER_AFFINE -> 3; };
        if (inputs.length != expected || inputTypes.size() != expected || extents.length != expected
                || offsets.length != expected || strides.length != expected || normalizedRank <= 0
                || normalizedRank > extents[0].length || !Double.isFinite(epsilon) || epsilon <= 0)
            throw new IllegalArgumentException("trailing-normalization reference facts disagree");
        for (int i = 0; i < expected; i++) validate(inputs[i], extents[i], offsets[i], strides[i]);
        long total = count(extents[0]);
        if (total > Integer.MAX_VALUE) throw new IllegalArgumentException("reference output is too large");
        double[] output = new double[(int) total]; if (total == 0) return output;
        long domain = suffixCount(extents[0], normalizedRank), leading = total / domain;
        long[] inputCoordinates = new long[extents[0].length];
        long[] parameterCoordinates = new long[normalizedRank];
        for (long slice = 0; slice < leading; slice++) {
            decodePrefix(slice, inputCoordinates, extents[0], normalizedRank);
            if (kind == CpuTrailingNormalizationIr.Kind.LAYER) layer(form, inputTypes, resultType,
                    epsilon, inputs, offsets, strides, extents[0], inputCoordinates,
                    parameterCoordinates, domain, output);
            else rms(form, inputTypes, resultType, epsilon, inputs, offsets, strides, extents[0],
                    inputCoordinates, parameterCoordinates, domain, output);
        }
        return output;
    }

    private static void layer(CpuTrailingNormalizationIr.Form form, List<DataType> types,
            DataType resultType, double epsilon, double[][] inputs, long[] offsets,
            long[][] strides, long[] shape, long[] coordinates, long[] parameterCoordinates,
            long domain, double[] output) {
        BigDecimal sum = BigDecimal.ZERO; boolean nan = false, infinity = false;
        double first = 0; boolean constant = true;
        Arrays.fill(parameterCoordinates, 0);
        for (long i = 0; i < domain; i++) {
            applySuffix(coordinates, parameterCoordinates);
            double value = promoted(types.get(0), resultType,
                    read(inputs[0], offsets[0], strides[0], coordinates));
            nan |= Double.isNaN(value); infinity |= Double.isInfinite(value);
            if (i == 0) first = value; else constant &= value == first;
            if (Double.isFinite(value)) sum = sum.add(new BigDecimal(value));
            increment(parameterCoordinates, trailing(shape, parameterCoordinates.length));
        }
        DataType computationType = resultType == DataType.FLOAT64
                ? DataType.FLOAT64 : DataType.FLOAT32;
        double mean = represented(computationType,
                sum.divide(BigDecimal.valueOf(domain), DIVISION).doubleValue());
        double deviations = 0, dc = 0, squares = 0, sc = 0;
        Arrays.fill(parameterCoordinates, 0);
        if (!nan && !infinity && !constant) for (long i = 0; i < domain; i++) {
            applySuffix(coordinates, parameterCoordinates);
            double deviation = promoted(types.get(0), resultType,
                    read(inputs[0], offsets[0], strides[0], coordinates)) - mean;
            double adjusted = deviation - dc, next = deviations + adjusted;
            dc = (next - deviations) - adjusted; deviations = next;
            double square = deviation * deviation;
            if (Double.isInfinite(square)) { squares = Double.POSITIVE_INFINITY; sc = 0; }
            else if (!Double.isInfinite(squares)) { adjusted = square - sc; next = squares + adjusted;
                sc = (next - squares) - adjusted; squares = next; }
            increment(parameterCoordinates, trailing(shape, parameterCoordinates.length));
        }
        double numerator = squares == Double.POSITIVE_INFINITY ? squares
                : squares - deviations * deviations / domain;
        if (numerator < 0) numerator = 0;
        double denominator = StrictMath.sqrt(numerator / domain + epsilon);
        Arrays.fill(parameterCoordinates, 0);
        for (long i = 0; i < domain; i++) {
            applySuffix(coordinates, parameterCoordinates);
            double standardized = nan || infinity ? Double.NaN : constant ? 0.0
                    : arithmetic(resultType, arithmetic(resultType, promoted(types.get(0), resultType,
                        read(inputs[0], offsets[0], strides[0], coordinates)), mean, '-'), denominator, '/');
            double result = standardized;
            if (form == CpuTrailingNormalizationIr.Form.LAYER_AFFINE) {
                double scale = promoted(types.get(1), resultType,
                        read(inputs[1], offsets[1], strides[1], parameterCoordinates));
                double bias = promoted(types.get(2), resultType,
                        read(inputs[2], offsets[2], strides[2], parameterCoordinates));
                result = arithmetic(resultType, standardized, scale, '*');
                result = arithmetic(resultType, result, bias, '+');
            }
            output[Math.toIntExact(ordinal(coordinates, shape))] = represented(resultType, result);
            increment(parameterCoordinates, trailing(shape, parameterCoordinates.length));
        }
    }

    private static void rms(CpuTrailingNormalizationIr.Form form, List<DataType> types,
            DataType resultType, double epsilon, double[][] inputs, long[] offsets,
            long[][] strides, long[] shape, long[] coordinates, long[] parameterCoordinates,
            long domain, double[] output) {
        boolean nan = false, infinity = false; double scaleState = 0, scaledSquares = 0;
        Arrays.fill(parameterCoordinates, 0);
        for (long i = 0; i < domain; i++) {
            applySuffix(coordinates, parameterCoordinates);
            double value = promoted(types.get(0), resultType,
                    read(inputs[0], offsets[0], strides[0], coordinates));
            nan |= Double.isNaN(value); infinity |= Double.isInfinite(value);
            double absolute = Math.abs(value);
            if (Double.isFinite(absolute) && absolute != 0 && scaleState < absolute) {
                double ratio = scaleState / absolute;
                scaledSquares = 1 + scaledSquares * ratio * ratio; scaleState = absolute;
            } else if (Double.isFinite(absolute) && absolute != 0) {
                double ratio = absolute / scaleState; scaledSquares += ratio * ratio;
            }
            increment(parameterCoordinates, trailing(shape, parameterCoordinates.length));
        }
        DataType computationType = resultType == DataType.FLOAT64
                ? DataType.FLOAT64 : DataType.FLOAT32;
        double root = nan ? Double.NaN : infinity ? Double.POSITIVE_INFINITY
                : represented(computationType, StrictMath.hypot(
                        scaleState * StrictMath.sqrt(scaledSquares / domain),
                        StrictMath.sqrt(epsilon)));
        Arrays.fill(parameterCoordinates, 0);
        for (long i = 0; i < domain; i++) {
            applySuffix(coordinates, parameterCoordinates);
            double value = promoted(types.get(0), resultType,
                    read(inputs[0], offsets[0], strides[0], coordinates));
            double result = arithmetic(resultType, value, root, '/');
            if (form == CpuTrailingNormalizationIr.Form.RMS_SCALED) result = arithmetic(resultType,
                    result, promoted(types.get(1), resultType, read(inputs[1], offsets[1], strides[1],
                        parameterCoordinates)), '*');
            output[Math.toIntExact(ordinal(coordinates, shape))] = represented(resultType, result);
            increment(parameterCoordinates, trailing(shape, parameterCoordinates.length));
        }
    }

    private static double arithmetic(DataType type, double a, double b, char operation) {
        if (type == DataType.FLOAT64) return switch (operation) { case '+' -> a + b;
            case '-' -> a - b; case '*' -> a * b; default -> a / b; };
        float x = (float) a, y = (float) b;
        return switch (operation) { case '+' -> x + y; case '-' -> x - y;
            case '*' -> x * y; default -> x / y; };
    }
    private static double read(double[] values, long offset, long[] strides, long[] coordinates) {
        long address = offset;
        for (int i = 0; i < coordinates.length; i++) address = Math.addExact(address,
                Math.multiplyExact(coordinates[i], strides[i]));
        return values[Math.toIntExact(address)];
    }
    private static void validate(double[] values, long[] extents, long offset, long[] strides) {
        if (extents.length != strides.length || offset < 0 || Arrays.stream(extents).anyMatch(v -> v < 0)
                || Arrays.stream(strides).anyMatch(v -> v < 0)) throw new IllegalArgumentException(
                        "reference layout is invalid");
        if (count(extents) == 0) return; long max = offset;
        for (int i = 0; i < extents.length; i++) max = Math.addExact(max,
                Math.multiplyExact(extents[i] - 1, strides[i]));
        if (max >= values.length) throw new IllegalArgumentException("reference input span is too small");
    }
    private static void decodePrefix(long slice, long[] coordinates, long[] shape, int normalizedRank) {
        for (int axis = shape.length - normalizedRank - 1; axis >= 0; axis--) {
            coordinates[axis] = slice % shape[axis]; slice /= shape[axis];
        }
    }
    private static void applySuffix(long[] coordinates, long[] suffix) {
        System.arraycopy(suffix, 0, coordinates, coordinates.length - suffix.length, suffix.length);
    }
    private static long[] trailing(long[] shape, int rank) {
        return Arrays.copyOfRange(shape, shape.length - rank, shape.length);
    }
    private static void increment(long[] coordinates, long[] extents) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--)
            if (++coordinates[axis] < extents[axis]) return; else coordinates[axis] = 0;
    }
    private static long ordinal(long[] coordinates, long[] extents) {
        long value = 0; for (int axis = 0; axis < extents.length; axis++)
            value = Math.addExact(Math.multiplyExact(value, extents[axis]), coordinates[axis]);
        return value;
    }
    private static long suffixCount(long[] extents, int rank) {
        long result = 1; for (int i = extents.length - rank; i < extents.length; i++)
            result = Math.multiplyExact(result, extents[i]); return result;
    }
    private static long count(long[] extents) { for (long e : extents) if (e == 0) return 0;
        long result = 1; for (long e : extents) result = Math.multiplyExact(result, e); return result; }
    private static double represented(DataType type, double value) {
        if (type == DataType.FLOAT64) return value; float narrowed = (float) value;
        if (type == DataType.FLOAT32) return narrowed;
        int bits = Float.floatToRawIntBits(narrowed);
        if ((bits & 0x7fffffff) > 0x7f800000) return Float.intBitsToFloat(0x7fc00000);
        bits += 0x7fff + (bits >>> 16 & 1); return Float.intBitsToFloat(bits & 0xffff0000);
    }

    private static double promoted(DataType source, DataType result, double value) {
        return represented(result, represented(source, value));
    }
}
