package tensor.storage;

import tensor.DataType;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

/**
 * {@link DataType#BOOL} tensor storage backed by native CPU memory.
 *
 * <p>Each logical boolean is stored as one byte. Writes normalize values to {@code 0} or {@code 1} so the
 * native representation matches the public BOOL array contract.</p>
 */
public final class NativeBoolStorage extends AbstractNativeTensorStorage {
    public NativeBoolStorage(int size, NativeMemoryAllocation allocation) {
        this(size, allocation, 0L, true);
    }

    public NativeBoolStorage(int size, NativeMemoryAllocation allocation, long byteOffset, boolean ownsSegment) {
        super(DataType.BOOL, size, Byte.BYTES, allocation, byteOffset, ownsSegment);
    }

    public byte getBoolAt(int flatIndex) {
        return segment().get(JAVA_BYTE, byteOffsetForIndex(flatIndex));
    }

    public void setBoolAt(int flatIndex, byte value) {
        segment().set(JAVA_BYTE, byteOffsetForIndex(flatIndex), value == 0 ? (byte) 0 : (byte) 1);
        markModified();
    }
}
