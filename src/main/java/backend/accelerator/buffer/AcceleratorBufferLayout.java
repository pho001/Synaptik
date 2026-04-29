package backend.accelerator.buffer;

import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Backend-neutral logical tensor layout metadata for an accelerator-visible buffer.
 */
public record AcceleratorBufferLayout(
        DataType dataType,
        int[] shape,
        int[] strides,
        int storageOffset,
        long logicalElementCount,
        long logicalByteLength,
        AcceleratorBufferLayoutClass layoutClass
) {
    public AcceleratorBufferLayout {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        Objects.requireNonNull(layoutClass, "layoutClass cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have the same length");
        }
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        if (logicalElementCount < 0) {
            throw new IllegalArgumentException("logicalElementCount cannot be negative");
        }
        if (logicalByteLength < 0) {
            throw new IllegalArgumentException("logicalByteLength cannot be negative");
        }
        long expectedByteLength = byteLength(dataType, logicalElementCount);
        if (logicalByteLength != expectedByteLength) {
            throw new IllegalArgumentException("logicalByteLength " + logicalByteLength
                    + " does not match dtype/element count byte length " + expectedByteLength);
        }
        shape = normalizeShape(shape);
        strides = strides.clone();
    }

    public static AcceleratorBufferLayout fromTensor(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return AcceleratorBufferLayoutClassifier.describe(
                tensor.getDataType(),
                tensor.getShape(),
                tensor.getStrides(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize()
        );
    }

    public static AcceleratorBufferLayout of(
            DataType dataType,
            int[] shape,
            int[] strides,
            int storageOffset,
            long logicalElementCount
    ) {
        return AcceleratorBufferLayoutClassifier.describe(
                dataType,
                shape,
                strides,
                storageOffset,
                logicalElementCount
        );
    }

    public static long byteLength(DataType dataType, long logicalElementCount) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        if (logicalElementCount < 0) {
            throw new IllegalArgumentException("logicalElementCount cannot be negative");
        }
        return Math.multiplyExact(logicalElementCount, bytesPerElement(dataType));
    }

    static int bytesPerElement(DataType dataType) {
        return switch (Objects.requireNonNull(dataType, "dataType cannot be null")) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case INT32 -> Integer.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    public String describe() {
        return "dtype=" + dataType
                + ", layoutClass=" + layoutClass
                + ", shape=" + Arrays.toString(shape)
                + ", strides=" + Arrays.toString(strides)
                + ", storageOffset=" + storageOffset
                + ", elements=" + logicalElementCount
                + ", bytes=" + logicalByteLength;
    }

    static int[] normalizeShape(int[] shape) {
        if (shape.length == 0) {
            return new int[]{1};
        }
        return shape.clone();
    }
}
