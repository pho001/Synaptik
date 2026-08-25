package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Independent clean-Java oracle for first-class batch-normalization inference.
 *
 * <p>The oracle owns no caller storage and returns a new dense logical output. It preserves each
 * source representation, applies ordered floating promotion, and rounds every arithmetic step at
 * the FLOAT32 boundary for BFLOAT16/FLOAT32 results or the FLOAT64 boundary for FLOAT64 results.
 * It is deliberately independent of generated Class-File traversal and carrier code.</p>
 */
public final class CpuBatchNormInferenceReferenceKernel {
    /** Prevents instantiation of this stateless reference-function holder. */
    private CpuBatchNormInferenceReferenceKernel() { }

    /**
     * Evaluates the exact five-input formula into dense logical row-major output order.
     *
     * @param inputTypes five represented input types in occurrence order
     * @param resultType exact ordered-promotion result type
     * @param epsilon exact positive result-type epsilon represented as binary64; callers must
     *     supply the value decoded from the result-type bits
     * @param inputs five semantic carriers represented as binary64 values
     * @param extents five semantic input Shapes
     * @param offsets five non-negative element offsets
     * @param strides five semantic element-stride vectors
     * @param channelAxis normalized input channel axis
     * @return a newly allocated, non-null dense represented output in logical row-major order
     * @throws NullPointerException if a required reference or nested carrier is null
     * @throws IllegalArgumentException if cardinality, type, Shape, layout, promotion, epsilon,
     *     or carrier span facts disagree
     * @throws ArithmeticException if exact count or address arithmetic overflows
     */
    public static double[] evaluate(List<DataType> inputTypes, DataType resultType, double epsilon,
            double[][] inputs, long[][] extents, long[] offsets, long[][] strides,
            int channelAxis) {
        inputTypes = List.copyOf(inputTypes);
        Objects.requireNonNull(resultType, "resultType");
        Objects.requireNonNull(inputs, "inputs"); Objects.requireNonNull(extents, "extents");
        Objects.requireNonNull(offsets, "offsets"); Objects.requireNonNull(strides, "strides");
        DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
        for (int index = 1; index < inputTypes.size(); index++) {
            promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(index));
        }
        if (inputTypes.size() != 5 || inputs.length != 5 || extents.length != 5
                || offsets.length != 5 || strides.length != 5 || promoted != resultType
                || channelAxis < 0 || extents[0].length < 2 || channelAxis >= extents[0].length
                || !Double.isFinite(epsilon) || epsilon <= 0.0) {
            throw new IllegalArgumentException("batch-normalization reference facts disagree");
        }
        long channels = extents[0][channelAxis];
        for (int position = 0; position < 5; position++) {
            validate(inputs[position], extents[position], offsets[position], strides[position]);
            if (position > 0 && (extents[position].length != 1
                    || extents[position][0] != channels)) {
                throw new IllegalArgumentException("batch-normalization vector Shape disagrees");
            }
        }
        long count = count(extents[0]);
        if (count > Integer.MAX_VALUE) throw new IllegalArgumentException("reference output is too large");
        double[] output = new double[(int) count];
        long[] coordinates = new long[extents[0].length];
        for (long logical = 0; logical < count; logical++) {
            long channel = coordinates[channelAxis];
            double input = promoted(inputTypes.get(0), resultType,
                    read(inputs[0], offsets[0], strides[0], coordinates));
            double scale = promoted(inputTypes.get(1), resultType,
                    readVector(inputs[1], offsets[1], strides[1], channel));
            double bias = promoted(inputTypes.get(2), resultType,
                    readVector(inputs[2], offsets[2], strides[2], channel));
            double mean = promoted(inputTypes.get(3), resultType,
                    readVector(inputs[3], offsets[3], strides[3], channel));
            double variance = promoted(inputTypes.get(4), resultType,
                    readVector(inputs[4], offsets[4], strides[4], channel));
            double centered = arithmetic(resultType, input, mean, '-');
            double radicand = arithmetic(resultType, variance, epsilon, '+');
            double denominator = representedComputation(resultType, StrictMath.sqrt(radicand));
            double standardized = arithmetic(resultType, centered, denominator, '/');
            double scaled = arithmetic(resultType, standardized, scale, '*');
            output[(int) logical] = represented(resultType,
                    arithmetic(resultType, scaled, bias, '+'));
            increment(coordinates, extents[0]);
        }
        return output;
    }

    private static double arithmetic(DataType type, double left, double right, char operation) {
        if (type == DataType.FLOAT64) return switch (operation) {
            case '+' -> left + right; case '-' -> left - right;
            case '*' -> left * right; default -> left / right;
        };
        float a = (float) left, b = (float) right;
        return switch (operation) {
            case '+' -> a + b; case '-' -> a - b; case '*' -> a * b; default -> a / b;
        };
    }

    private static double representedComputation(DataType type, double value) {
        return type == DataType.FLOAT64 ? value : (float) value;
    }

    private static double promoted(DataType source, DataType result, double value) {
        return representedComputation(result, represented(source, value));
    }

    private static double represented(DataType type, double value) {
        if (type == DataType.FLOAT64) return value;
        float narrowed = (float) value;
        if (type == DataType.FLOAT32) return narrowed;
        int bits = Float.floatToRawIntBits(narrowed);
        if ((bits & 0x7fff_ffff) > 0x7f80_0000) return Float.intBitsToFloat(0x7fc0_0000);
        bits += 0x7fff + ((bits >>> 16) & 1);
        return Float.intBitsToFloat(bits & 0xffff_0000);
    }

    private static double read(double[] values, long offset, long[] strides, long[] coordinates) {
        long address = offset;
        for (int axis = 0; axis < coordinates.length; axis++) {
            address = Math.addExact(address, Math.multiplyExact(coordinates[axis], strides[axis]));
        }
        return values[Math.toIntExact(address)];
    }

    private static double readVector(double[] values, long offset, long[] strides, long channel) {
        return values[Math.toIntExact(Math.addExact(offset, Math.multiplyExact(channel, strides[0])))];
    }

    private static void validate(double[] values, long[] extents, long offset, long[] strides) {
        Objects.requireNonNull(values, "input"); Objects.requireNonNull(extents, "extents");
        Objects.requireNonNull(strides, "strides");
        if (extents.length != strides.length || offset < 0
                || Arrays.stream(extents).anyMatch(value -> value < 0)
                || Arrays.stream(strides).anyMatch(value -> value < 0)) {
            throw new IllegalArgumentException("reference layout is invalid");
        }
        if (count(extents) == 0) return;
        long maximum = offset;
        for (int axis = 0; axis < extents.length; axis++) {
            maximum = Math.addExact(maximum,
                    Math.multiplyExact(extents[axis] - 1, strides[axis]));
        }
        if (maximum >= values.length) throw new IllegalArgumentException("reference span is too small");
    }

    private static long count(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent);
        return result;
    }

    private static void increment(long[] coordinates, long[] extents) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            if (++coordinates[axis] < extents[axis]) return;
            coordinates[axis] = 0;
        }
    }
}
