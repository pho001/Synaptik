package tensor.storage;

import tensor.DataType;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Mutable {@link DataType#INT32} tensor storage backed by an {@code int[]}.
 *
 * <p>INT32 tensors are primarily used as index tensors. The backing array is
 * exposed directly and is not synchronized. Direct array writes require a
 * matching {@link #markModified()} call if version tracking matters.</p>
 */
public final class Int32Storage implements TensorStorage {
    private final int[] data;
    private final AtomicLong version = new AtomicLong();

    /**
     * Allocates zero-filled int storage.
     *
     * @param size number of elements; negative values are rejected by the JVM
     */
    public Int32Storage(int size) {
        this.data = new int[size];
    }

    /**
     * Wraps an existing int array without copying.
     *
     * @param data mutable backing array; must be non-null
     */
    public Int32Storage(int[] data) {
        this.data = data;
    }

    @Override
    public DataType getType() {
        return DataType.INT32;
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
     * Reads one physical storage element.
     *
     * @param flatIndex zero-based physical storage index
     * @return stored integer value
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public int getInt32At(int flatIndex) {
        return data[flatIndex];
    }

    /**
     * Writes one physical storage element.
     *
     * @param flatIndex zero-based physical storage index
     * @param value value to store
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public void setInt32At(int flatIndex, int value) {
        data[flatIndex] = value;
    }

    /**
     * Returns the mutable backing array without copying.
     *
     * @return non-null storage-order array
     */
    public int[] getIntArray() {
        return data;
    }
}
