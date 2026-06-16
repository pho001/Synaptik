package tensor.storage;

import tensor.DataType;

import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * {@link DataType#INT32} tensor storage backed by native CPU memory.
 */
public final class NativeInt32Storage extends AbstractNativeTensorStorage {
    public NativeInt32Storage(int size, NativeMemoryAllocation allocation) {
        this(size, allocation, 0L, true);
    }

    public NativeInt32Storage(int size, NativeMemoryAllocation allocation, long byteOffset, boolean ownsSegment) {
        super(DataType.INT32, size, Integer.BYTES, allocation, byteOffset, ownsSegment);
    }

    public int getInt32At(int flatIndex) {
        return segment().get(JAVA_INT, byteOffsetForIndex(flatIndex));
    }

    public void setInt32At(int flatIndex, int value) {
        segment().set(JAVA_INT, byteOffsetForIndex(flatIndex), value);
        markModified();
    }
}
