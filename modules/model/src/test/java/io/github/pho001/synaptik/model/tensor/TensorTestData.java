package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

/**
 * Prepares copied strict or cyclic primitive prefixes for same-package tensor test fixtures.
 *
 * <p>This field-free test-source helper preserves the historical prefix preparation behavior
 * without exposing it from production. Every successful entry creates one exact carrier and
 * delegates final construction to the matching public {@link TensorFactory#fromFlatArray}
 * overload.</p>
 */
final class TensorTestData {
    /** Prevents instances because fixture preparation is stateless. */
    private TensorTestData() {
    }

    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, double[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT64, requiresGrad, source.length, false);
        return TensorFactory.fromFlatArray(
                descriptor, label, Arrays.copyOf(source, elementCount(descriptor)));
    }

    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, float[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT32, requiresGrad, source.length, false);
        return TensorFactory.fromFlatArray(
                descriptor, label, Arrays.copyOf(source, elementCount(descriptor)));
    }

    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, short[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BFLOAT16, requiresGrad, source.length, false);
        return TensorFactory.fromFlatArray(
                descriptor, label, Arrays.copyOf(source, elementCount(descriptor)));
    }

    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, int[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT32, requiresGrad, source.length, false);
        return TensorFactory.fromFlatArray(
                descriptor, label, Arrays.copyOf(source, elementCount(descriptor)));
    }

    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, long[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT64, requiresGrad, source.length, false);
        return TensorFactory.fromFlatArray(
                descriptor, label, Arrays.copyOf(source, elementCount(descriptor)));
    }

    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, byte[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BOOL, requiresGrad, source.length, false);
        return TensorFactory.fromFlatArray(
                descriptor, label, Arrays.copyOf(source, elementCount(descriptor)));
    }

    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, double[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT64, requiresGrad, source.length, true);
        double[] values = new double[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, float[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT32, requiresGrad, source.length, true);
        float[] values = new float[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, short[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BFLOAT16, requiresGrad, source.length, true);
        short[] values = new short[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, int[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT32, requiresGrad, source.length, true);
        int[] values = new int[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, long[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT64, requiresGrad, source.length, true);
        long[] values = new long[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, byte[] source) {
        requireInputs(shape, label, source);
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BOOL, requiresGrad, source.length, true);
        byte[] values = new byte[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    private static void requireInputs(Shape shape, Optional<String> label, Object source) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(source, "source");
    }

    private static TensorDescriptor prefixDescriptor(
            Shape shape,
            DataType dataType,
            boolean requiresGrad,
            int sourceLength,
            boolean cyclic) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "prefix tensor creation requires a fully static shape: " + shape);
        }
        long count = shape.knownElementCount().orElseThrow();
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "prefix tensor element count exceeds Java array limit: required="
                            + count
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }
        if (!cyclic && sourceLength < count) {
            throw new IllegalArgumentException(
                    "strict flat prefix source is too short: required="
                            + count
                            + ", actual="
                            + sourceLength);
        }
        if (cyclic && count > 0 && sourceLength == 0) {
            throw new IllegalArgumentException(
                    "cyclic flat prefix source must not be empty for non-empty output");
        }
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad);
    }

    private static int elementCount(TensorDescriptor descriptor) {
        return Math.toIntExact(descriptor.shape().knownElementCount().orElseThrow());
    }
}
