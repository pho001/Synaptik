package tensor;

import backend.cpu.nativecpu.NativeCpuAllocation;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;

/**
 * {@link DataType#FLOAT64} tensor storage backed by native CPU memory.
 */
public final class NativeFloat64Storage extends AbstractNativeTensorStorage {
    public NativeFloat64Storage(int size, NativeCpuAllocation allocation) {
        this(size, allocation, 0L, true);
    }

    public NativeFloat64Storage(int size, NativeCpuAllocation allocation, long byteOffset, boolean ownsSegment) {
        super(DataType.FLOAT64, size, Double.BYTES, allocation, byteOffset, ownsSegment);
    }

    public double getFloat64At(int flatIndex) {
        return segment().get(JAVA_DOUBLE, byteOffsetForIndex(flatIndex));
    }

    public void setFloat64At(int flatIndex, double value) {
        segment().set(JAVA_DOUBLE, byteOffsetForIndex(flatIndex), value);
        markModified();
    }
}
