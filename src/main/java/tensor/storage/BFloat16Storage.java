package tensor.storage;

import tensor.DataType;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Mutable {@link DataType#BFLOAT16} tensor storage backed by raw bfloat16 bits.
 *
 * <p>Each {@code short} stores the upper 16 bits of a single precision value.
 * The backing array is exposed directly and is not synchronized. Direct array
 * writes require a matching {@link #markModified()} call if version tracking
 * matters to the caller.</p>
 */
public class BFloat16Storage implements TensorStorage {
    private final short[] data;
    private final AtomicLong version = new AtomicLong();

    /**
     * Allocates zero-filled bfloat16 storage.
     *
     * @param size number of elements; negative values are rejected by the JVM
     */
    public BFloat16Storage(int size) {
        this.data = new short[size];
    }

    /**
     * Wraps existing bfloat16 bits without copying.
     *
     * @param data mutable backing array of raw bfloat16 bit patterns; must be non-null
     */
    public BFloat16Storage(short[] data) {
        this.data = data;
    }

    @Override
    public DataType getType() {
        return DataType.BFLOAT16;
    }

    @Override
    public int getSize() {
        return data.length;
    }

    @Override
    public long version() {
        return version.get();
    }

    @Override
    public void markModified() {
        version.incrementAndGet();
    }

    /**
     * Reads one raw bfloat16 element.
     *
     * @param flatIndex zero-based physical storage index
     * @return raw bfloat16 bits
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public short getBFloat16BitsAt(int flatIndex) {
        return data[flatIndex];
    }

    /**
     * Writes one raw bfloat16 element.
     *
     * @param flatIndex zero-based physical storage index
     * @param value raw bfloat16 bits to store
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public void setBFloat16BitsAt(int flatIndex, short value) {
        data[flatIndex] = value;
    }

    /**
     * Returns the mutable backing array without copying.
     *
     * @return non-null storage-order array of raw bfloat16 bits
     */
    public short[] getShortArray() {
        return data;
    }
}
