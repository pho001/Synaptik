package tensor.storage;

import tensor.DataType;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * {@link DataType#FLOAT32} tensor storage backed by native CPU memory.
 */
public final class NativeFloat32Storage extends AbstractNativeTensorStorage {
    public NativeFloat32Storage(int size, NativeMemoryAllocation allocation) {
        this(size, allocation, 0L, true);
    }

    public NativeFloat32Storage(int size, NativeMemoryAllocation allocation, long byteOffset, boolean ownsSegment) {
        super(DataType.FLOAT32, size, Float.BYTES, allocation, byteOffset, ownsSegment);
    }

    public float getFloat32At(int flatIndex) {
        return segment().get(JAVA_FLOAT, byteOffsetForIndex(flatIndex));
    }

    public void setFloat32At(int flatIndex, float value) {
        segment().set(JAVA_FLOAT, byteOffsetForIndex(flatIndex), value);
        markModified();
    }
}
