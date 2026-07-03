package io.github.pho001.synaptik.model.layout;

import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Arrays;

/**
 * Performs deterministic, checked calculations for resolved logical element layouts.
 *
 * <p>This package-private helper is stateless and owns no storage or execution policy. Callers must
 * supply a fully static shape and validated non-negative strides and offsets. All returned arrays
 * are newly allocated.</p>
 */
final class LayoutGeometry {
    private LayoutGeometry() {
        throw new AssertionError("No instances");
    }

    /**
     * Calculates canonical row-major element strides for a fully static shape.
     *
     * <p>Only products needed to produce actual stride entries are evaluated. In particular, the
     * first dimension size is never multiplied solely to calculate a total element count.</p>
     *
     * @param shape non-null fully static logical shape
     * @return newly allocated canonical strides, with one entry per shape axis
     * @throws IllegalArgumentException if the shape contains a dynamic dimension
     * @throws ArithmeticException if a product required for a stride exceeds {@link Long#MAX_VALUE}
     */
    static long[] canonicalStrides(Shape shape) {
        long[] sizes = staticSizes(shape);
        long[] strides = new long[sizes.length];
        if (sizes.length == 0) {
            return strides;
        }

        strides[sizes.length - 1] = 1;
        for (int axis = sizes.length - 2; axis >= 0; axis--) {
            strides[axis] = Math.multiplyExact(strides[axis + 1], sizes[axis + 1]);
        }
        return strides;
    }

    /**
     * Classifies supplied resolved geometry using canonical, broadcast, then strided precedence.
     *
     * @param shape non-null fully static logical shape
     * @param strides non-null validated non-negative element strides whose length equals shape rank
     * @param storageOffset validated non-negative storage offset measured in elements
     * @return non-null geometry classification derived from the supplied values
     * @throws IllegalArgumentException if the shape contains a dynamic dimension
     * @throws ArithmeticException if canonical stride calculation overflows
     */
    static LayoutKind classify(Shape shape, long[] strides, long storageOffset) {
        if (Arrays.equals(canonicalStrides(shape), strides)) {
            return storageOffset == 0
                    ? LayoutKind.DENSE_CONTIGUOUS
                    : LayoutKind.DENSE_WITH_OFFSET;
        }
        if (hasBroadcastZeroStride(shape, strides)) {
            return LayoutKind.BROADCAST_ZERO_STRIDE;
        }
        return LayoutKind.STRIDED;
    }

    /**
     * Reports whether any supplied element stride is zero, independent of dimension size.
     *
     * @param strides non-null element strides
     * @return {@code true} if at least one stride is zero; otherwise {@code false}
     */
    static boolean hasZeroStride(long[] strides) {
        for (long stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Reports whether stride zero repeats a static dimension larger than one.
     *
     * @param shape non-null fully static logical shape
     * @param strides non-null element strides whose length equals shape rank
     * @return {@code true} if a non-singleton, non-empty dimension has stride zero
     * @throws IllegalArgumentException if the shape contains a dynamic dimension
     */
    static boolean hasBroadcastZeroStride(Shape shape, long[] strides) {
        long[] sizes = staticSizes(shape);
        for (int axis = 0; axis < sizes.length; axis++) {
            if (sizes[axis] > 1 && strides[axis] == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Calculates the minimum storage element count that contains every referenced element index.
     *
     * @param shape non-null fully static logical shape
     * @param strides non-null validated non-negative element strides whose length equals shape rank
     * @param storageOffset validated non-negative storage offset measured in elements
     * @return zero when the shape contains a zero-sized dimension, or the greatest referenced
     *     element index plus one
     * @throws IllegalArgumentException if the shape contains a dynamic dimension
     * @throws ArithmeticException if checked multiplication or addition exceeds
     *     {@link Long#MAX_VALUE}
     */
    static long referencedElementSpan(Shape shape, long[] strides, long storageOffset) {
        long[] sizes = staticSizes(shape);
        for (long size : sizes) {
            if (size == 0) {
                return 0;
            }
        }

        long greatestIndex = storageOffset;
        for (int axis = 0; axis < sizes.length; axis++) {
            long axisExtent = Math.multiplyExact(sizes[axis] - 1, strides[axis]);
            greatestIndex = Math.addExact(greatestIndex, axisExtent);
        }
        return Math.addExact(greatestIndex, 1);
    }

    private static long[] staticSizes(Shape shape) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException("Layout requires a fully static shape: " + shape);
        }
        return shape.toLongArray();
    }
}
