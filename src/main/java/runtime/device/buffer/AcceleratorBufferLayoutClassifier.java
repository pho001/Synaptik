package runtime.device.buffer;

import tensor.DataType;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.Objects;

/**
 * Pure backend-neutral classifier for logical tensor layout facts.
 */
public final class AcceleratorBufferLayoutClassifier {
    private AcceleratorBufferLayoutClassifier() {
    }

    public static AcceleratorBufferLayout describe(
            DataType dataType,
            int[] shape,
            int[] strides,
            int storageOffset,
            long logicalElementCount
    ) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        int[] normalizedShape = AcceleratorBufferLayout.normalizeShape(shape);
        if (normalizedShape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length");
        }
        if (logicalElementCount < 0) {
            throw new IllegalArgumentException("logicalElementCount cannot be negative");
        }

        int[] strideCopy = strides.clone();
        AcceleratorBufferLayoutClass layoutClass = classify(normalizedShape, strideCopy, storageOffset, logicalElementCount);
        long logicalByteLength = AcceleratorBufferLayout.byteLength(dataType, logicalElementCount);
        return new AcceleratorBufferLayout(
                dataType,
                normalizedShape,
                strideCopy,
                storageOffset,
                logicalElementCount,
                logicalByteLength,
                layoutClass
        );
    }

    private static AcceleratorBufferLayoutClass classify(
            int[] shape,
            int[] strides,
            int storageOffset,
            long logicalElementCount
    ) {
        if (hasNegativeStride(strides) || product(shape) != logicalElementCount) {
            return AcceleratorBufferLayoutClass.UNSUPPORTED;
        }
        if (hasZeroStride(strides)) {
            return AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW;
        }
        int[] denseStrides = TensorMetadata.computeStrides(shape);
        if (Arrays.equals(strides, denseStrides)) {
            return storageOffset == 0
                    ? AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS
                    : AcceleratorBufferLayoutClass.NON_ZERO_OFFSET_VIEW;
        }
        if (storageOffset == 0 && allPositive(strides) && monotonicNonIncreasing(strides)) {
            return AcceleratorBufferLayoutClass.ZERO_OFFSET_VIEW;
        }
        return AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW;
    }

    private static boolean hasNegativeStride(int[] strides) {
        for (int stride : strides) {
            if (stride < 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasZeroStride(int[] strides) {
        for (int stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean allPositive(int[] strides) {
        for (int stride : strides) {
            if (stride <= 0) {
                return false;
            }
        }
        return true;
    }

    private static boolean monotonicNonIncreasing(int[] strides) {
        for (int i = 1; i < strides.length; i++) {
            if (strides[i - 1] < strides[i]) {
                return false;
            }
        }
        return true;
    }

    private static long product(int[] shape) {
        long product = 1L;
        for (int dimension : shape) {
            product = Math.multiplyExact(product, dimension);
        }
        return product;
    }
}
