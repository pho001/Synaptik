package tensor.storage;

import tensor.DataType;

import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * {@link DataType#BFLOAT16} tensor storage backed by native CPU memory.
 *
 * <p>Each element is a raw bfloat16 bit pattern stored in a 16-bit integer slot. The {@code short}
 * carrier is not FP16 and no arithmetic conversion is implied by this storage class.</p>
 */
public final class NativeBFloat16Storage extends AbstractNativeTensorStorage {
    public NativeBFloat16Storage(int size, NativeMemoryAllocation allocation) {
        this(size, allocation, 0L, true);
    }

    public NativeBFloat16Storage(int size, NativeMemoryAllocation allocation, long byteOffset, boolean ownsSegment) {
        super(DataType.BFLOAT16, size, Short.BYTES, allocation, byteOffset, ownsSegment);
    }

    public short getBFloat16BitsAt(int flatIndex) {
        return segment().get(JAVA_SHORT, byteOffsetForIndex(flatIndex));
    }

    public void setBFloat16BitsAt(int flatIndex, short bits) {
        segment().set(JAVA_SHORT, byteOffsetForIndex(flatIndex), bits);
        markModified();
    }
}
