package backend.cpu.storage;

import tensor.DataType;
import tensor.layout.TensorShape;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

public record CpuStorageView(
        DataType dtype,
        CpuStorageKind kind,
        Object array,
        MemorySegment segment,
        int[] shape,
        int[] strides,
        int storageOffset,
        int logicalSize
) {
    public CpuStorageView {
        dtype = Objects.requireNonNull(dtype, "dtype cannot be null");
        kind = Objects.requireNonNull(kind, "kind cannot be null");
        shape = TensorShape.normalize(shape);
        strides = normalizeStrides(shape, strides);
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative: " + storageOffset);
        }
        int shapeSize = TensorShape.checkedFlatSize(shape);
        if (logicalSize != shapeSize) {
            throw new IllegalArgumentException("logicalSize must match shape flat size. logicalSize="
                    + logicalSize + ", shapeSize=" + shapeSize);
        }
        if (kind == CpuStorageKind.JAVA_ARRAY) {
            array = Objects.requireNonNull(array, "array cannot be null for JAVA_ARRAY storage");
            if (segment != null) {
                throw new IllegalArgumentException("segment must be null for JAVA_ARRAY storage");
            }
            validateArray(dtype, array, requiredElementCapacity(shape, strides, storageOffset));
        } else {
            segment = Objects.requireNonNull(segment, "segment cannot be null for MEMORY_SEGMENT storage");
            if (array != null) {
                throw new IllegalArgumentException("array must be null for MEMORY_SEGMENT storage");
            }
            validateSegment(dtype, segment, requiredElementCapacity(shape, strides, storageOffset));
        }
    }

    public static CpuStorageView array(
            DataType dtype,
            Object array,
            int[] shape,
            int[] strides,
            int storageOffset,
            int logicalSize
    ) {
        return new CpuStorageView(
                dtype,
                CpuStorageKind.JAVA_ARRAY,
                array,
                null,
                shape,
                strides,
                storageOffset,
                logicalSize
        );
    }

    public static CpuStorageView segment(
            DataType dtype,
            MemorySegment segment,
            int[] shape,
            int[] strides,
            int storageOffset,
            int logicalSize
    ) {
        return new CpuStorageView(
                dtype,
                CpuStorageKind.MEMORY_SEGMENT,
                null,
                segment,
                shape,
                strides,
                storageOffset,
                logicalSize
        );
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    public boolean isArray() {
        return kind == CpuStorageKind.JAVA_ARRAY;
    }

    public boolean isMemorySegment() {
        return kind == CpuStorageKind.MEMORY_SEGMENT;
    }

    public Object requireArray() {
        if (kind != CpuStorageKind.JAVA_ARRAY) {
            throw new IllegalStateException("JAVA_ARRAY storage required, actual=" + kind);
        }
        return array;
    }

    public float[] requireF32Array() {
        Object value = requireArray();
        if (dtype != DataType.FLOAT32 || !(value instanceof float[] data)) {
            throw wrongArrayType(DataType.FLOAT32);
        }
        return data;
    }

    public double[] requireF64Array() {
        Object value = requireArray();
        if (dtype != DataType.FLOAT64 || !(value instanceof double[] data)) {
            throw wrongArrayType(DataType.FLOAT64);
        }
        return data;
    }

    public short[] requireBF16Array() {
        Object value = requireArray();
        if (dtype != DataType.BFLOAT16 || !(value instanceof short[] data)) {
            throw wrongArrayType(DataType.BFLOAT16);
        }
        return data;
    }

    public int[] requireI32Array() {
        Object value = requireArray();
        if (dtype != DataType.INT32 || !(value instanceof int[] data)) {
            throw wrongArrayType(DataType.INT32);
        }
        return data;
    }

    public long[] requireI64Array() {
        Object value = requireArray();
        if (dtype != DataType.INT64 || !(value instanceof long[] data)) {
            throw wrongArrayType(DataType.INT64);
        }
        return data;
    }

    public byte[] requireBoolArray() {
        Object value = requireArray();
        if (dtype != DataType.BOOL || !(value instanceof byte[] data)) {
            throw wrongArrayType(DataType.BOOL);
        }
        return data;
    }

    public MemorySegment requireSegment() {
        if (kind != CpuStorageKind.MEMORY_SEGMENT) {
            throw new IllegalStateException("MEMORY_SEGMENT storage required, actual=" + kind);
        }
        return segment;
    }

    private IllegalStateException wrongArrayType(DataType expected) {
        return new IllegalStateException(expected + " array storage required, actual dtype=" + dtype
                + ", array=" + array.getClass().getSimpleName());
    }

    private static int[] normalizeStrides(int[] normalizedShape, int[] strides) {
        if (strides == null) {
            throw new IllegalArgumentException("strides cannot be null");
        }
        if (strides.length != normalizedShape.length) {
            throw new IllegalArgumentException("strides length must match shape length");
        }
        int[] copy = strides.clone();
        for (int stride : copy) {
            if (stride < 0) {
                throw new IllegalArgumentException("strides cannot be negative: " + stride);
            }
        }
        return copy;
    }

    private static int requiredElementCapacity(int[] shape, int[] strides, int storageOffset) {
        long maxOffset = storageOffset;
        for (int i = 0; i < shape.length; i++) {
            maxOffset = Math.addExact(maxOffset, Math.multiplyExact((long) shape[i] - 1L, strides[i]));
            if (maxOffset > Integer.MAX_VALUE - 1L) {
                throw new IllegalArgumentException("storage element capacity overflows int range");
            }
        }
        return (int) maxOffset + 1;
    }

    private static void validateArray(DataType dtype, Object array, int requiredElementCapacity) {
        int arrayLength = switch (dtype) {
            case FLOAT64 -> array instanceof double[] data ? data.length : -1;
            case FLOAT32 -> array instanceof float[] data ? data.length : -1;
            case BFLOAT16 -> array instanceof short[] data ? data.length : -1;
            case INT32 -> array instanceof int[] data ? data.length : -1;
            case INT64 -> array instanceof long[] data ? data.length : -1;
            case BOOL -> array instanceof byte[] data ? data.length : -1;
        };
        if (arrayLength < 0) {
            throw new IllegalArgumentException("Array type does not match dtype " + dtype + ": "
                    + array.getClass().getSimpleName());
        }
        if (arrayLength < requiredElementCapacity) {
            throw new IllegalArgumentException("Array storage is too small. requiredElements="
                    + requiredElementCapacity + ", actualElements=" + arrayLength);
        }
    }

    private static void validateSegment(DataType dtype, MemorySegment segment, int requiredElementCapacity) {
        long requiredBytes = Math.multiplyExact((long) requiredElementCapacity, elementSizeBytes(dtype));
        if (segment.byteSize() < requiredBytes) {
            throw new IllegalArgumentException("MemorySegment storage is too small. requiredBytes="
                    + requiredBytes + ", actualBytes=" + segment.byteSize());
        }
    }

    private static int elementSizeBytes(DataType dtype) {
        return switch (dtype) {
            case FLOAT64, INT64 -> Long.BYTES;
            case FLOAT32, INT32 -> Integer.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
        };
    }
}
