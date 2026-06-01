package backend.cpu1.exec;

import backend.cpu1.storage.Cpu1StorageKind;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * CPU buffer view used by cpu1 kernels.
 */
public final class Cpu1BufferView {
    private final DataType dataType;
    private final Cpu1StorageKind storageKind;
    private final Object array;
    private final MemorySegment segment;

    private Cpu1BufferView(DataType dataType, Cpu1StorageKind storageKind, Object array, MemorySegment segment) {
        this.dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
        this.array = array;
        this.segment = segment;
        validate();
    }

    public static Cpu1BufferView array(DataType dataType, Object array) {
        return new Cpu1BufferView(dataType, Cpu1StorageKind.JAVA_ARRAY, Objects.requireNonNull(array), null);
    }

    public static Cpu1BufferView segment(DataType dataType, MemorySegment segment) {
        return new Cpu1BufferView(dataType, Cpu1StorageKind.MEMORY_SEGMENT, null, Objects.requireNonNull(segment));
    }

    public DataType dataType() {
        return dataType;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public float[] float32Array() {
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY || dataType != DataType.FLOAT32 || !(array instanceof float[] values)) {
            throw new IllegalStateException("Buffer is not FLOAT32.");
        }
        return values;
    }

    public double[] float64Array() {
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY || dataType != DataType.FLOAT64 || !(array instanceof double[] values)) {
            throw new IllegalStateException("Buffer is not FLOAT64.");
        }
        return values;
    }

    public short[] bfloat16Array() {
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY || dataType != DataType.BFLOAT16 || !(array instanceof short[] values)) {
            throw new IllegalStateException("Buffer is not BFLOAT16.");
        }
        return values;
    }

    public byte[] boolArray() {
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY || dataType != DataType.BOOL || !(array instanceof byte[] values)) {
            throw new IllegalStateException("Buffer is not BOOL.");
        }
        return values;
    }

    public int[] int32Array() {
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY || dataType != DataType.INT32 || !(array instanceof int[] values)) {
            throw new IllegalStateException("Buffer is not INT32.");
        }
        return values;
    }

    public long[] int64Array() {
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY || dataType != DataType.INT64 || !(array instanceof long[] values)) {
            throw new IllegalStateException("Buffer is not INT64.");
        }
        return values;
    }

    public MemorySegment segment() {
        if (storageKind != Cpu1StorageKind.MEMORY_SEGMENT) {
            throw new IllegalStateException("Buffer is not MEMORY_SEGMENT.");
        }
        return segment;
    }

    private void validate() {
        if (storageKind == Cpu1StorageKind.JAVA_ARRAY) {
            Objects.requireNonNull(array, "array cannot be null for JAVA_ARRAY storage");
            if (segment != null) {
                throw new IllegalArgumentException("segment must be null for JAVA_ARRAY storage");
            }
            if (!matchesArrayType()) {
                throw new IllegalArgumentException("Array type does not match dtype " + dataType + ": "
                        + array.getClass().getSimpleName());
            }
            return;
        }
        Objects.requireNonNull(segment, "segment cannot be null for MEMORY_SEGMENT storage");
        if (array != null) {
            throw new IllegalArgumentException("array must be null for MEMORY_SEGMENT storage");
        }
    }

    private boolean matchesArrayType() {
        return switch (dataType) {
            case FLOAT32 -> array instanceof float[];
            case FLOAT64 -> array instanceof double[];
            case BFLOAT16 -> array instanceof short[];
            case BOOL -> array instanceof byte[];
            case INT32 -> array instanceof int[];
            case INT64 -> array instanceof long[];
        };
    }
}
