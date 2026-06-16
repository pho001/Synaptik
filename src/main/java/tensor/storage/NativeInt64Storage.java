package tensor.storage;

import tensor.DataType;

import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * {@link DataType#INT64} tensor storage backed by native CPU memory.
 */
public final class NativeInt64Storage extends AbstractNativeTensorStorage {
    public NativeInt64Storage(int size, NativeMemoryAllocation allocation) {
        this(size, allocation, 0L, true);
    }

    public NativeInt64Storage(int size, NativeMemoryAllocation allocation, long byteOffset, boolean ownsSegment) {
        super(DataType.INT64, size, Long.BYTES, allocation, byteOffset, ownsSegment);
    }

    public long getInt64At(int flatIndex) {
        return segment().get(JAVA_LONG, byteOffsetForIndex(flatIndex));
    }

    public void setInt64At(int flatIndex, long value) {
        segment().set(JAVA_LONG, byteOffsetForIndex(flatIndex), value);
        markModified();
    }
}
