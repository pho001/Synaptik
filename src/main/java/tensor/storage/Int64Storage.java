package tensor.storage;

import tensor.DataType;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Mutable {@link DataType#INT64} tensor storage backed by a {@code long[]}.
 *
 * <p>INT64 tensors are primarily used for ONNX-compatible index and shape-like
 * runtime values. The backing array is exposed directly and is not synchronized.
 * Direct array writes require a matching {@link #markModified()} call if version
 * tracking matters.</p>
 */
public final class Int64Storage implements TensorStorage {
    private final long[] data;
    private final AtomicLong version = new AtomicLong();

    /**
     * Allocates zero-filled long storage.
     *
     * @param size number of elements; negative values are rejected by the JVM
     */
    public Int64Storage(int size) {
        this.data = new long[size];
    }

    /**
     * Wraps an existing long array without copying.
     *
     * @param data mutable backing array; must be non-null
     */
    public Int64Storage(long[] data) {
        this.data = data;
    }

    @Override
    public DataType getType() {
        return DataType.INT64;
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
     */
    public long getInt64At(int flatIndex) {
        return data[flatIndex];
    }

    /**
     * Writes one physical storage element.
     *
     * @param flatIndex zero-based physical storage index
     * @param value value to store
     */
    public void setInt64At(int flatIndex, long value) {
        data[flatIndex] = value;
    }

    /**
     * Returns the mutable backing array without copying.
     *
     * @return non-null storage-order array
     */
    public long[] getLongArray() {
        return data;
    }
}
