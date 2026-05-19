package tensor.layout;

import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorMetadata;
import tensor.dtype.BFloat16Bits;
import tensor.storage.TensorStorageSupport;

public final class TensorShape {
    private TensorShape() {
    }

    public static int[] normalize(int[] shape) {
        if (shape == null) {
            throw new IllegalArgumentException("Shape cannot be null.");
        }
        if (shape.length == 0) {
            return new int[]{1};
        }
        int[] normalized = shape.clone();
        for (int dimension : normalized) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("Shape dimensions must be positive: " + dimension);
            }
        }
        checkedFlatSize(normalized);
        return normalized;
    }

    public static int checkedFlatSize(int[] shape) {
        int[] normalized = normalizeForProduct(shape);
        long size = 1L;
        for (int dimension : normalized) {
            size = Math.multiplyExact(size, (long) dimension);
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Tensor shape is too large: flatSize=" + size);
            }
        }
        return (int) size;
    }

    public static int[] contiguousStrides(int[] shape) {
        int[] normalized = normalize(shape);
        int[] strides = new int[normalized.length];
        long stride = 1L;
        for (int i = normalized.length - 1; i >= 0; i--) {
            if (stride > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Tensor strides overflow int range.");
            }
            strides[i] = (int) stride;
            stride = Math.multiplyExact(stride, (long) normalized[i]);
            if (stride > Integer.MAX_VALUE && i > 0) {
                throw new IllegalArgumentException("Tensor strides overflow int range.");
            }
        }
        return strides;
    }

    private static int[] normalizeForProduct(int[] shape) {
        if (shape == null) {
            throw new IllegalArgumentException("Shape cannot be null.");
        }
        if (shape.length == 0) {
            return new int[]{1};
        }
        int[] normalized = shape.clone();
        for (int dimension : normalized) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("Shape dimensions must be positive: " + dimension);
            }
        }
        return normalized;
    }
}
