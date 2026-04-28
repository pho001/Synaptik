package tensor;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Mutable {@link DataType#FLOAT32} tensor storage backed by a {@code float[]}.
 *
 * <p>The backing array is exposed directly and is not synchronized. Direct array
 * writes require a matching {@link #markModified()} call if version tracking
 * matters to the caller.</p>
 */
public class Float32Storage implements TensorStorage {
    private final float[] data;
    private final AtomicLong version = new AtomicLong();

    /**
     * Allocates zero-filled float storage.
     *
     * @param size number of elements; negative values are rejected by the JVM
     */
    public Float32Storage(int size) {
        this.data = new float[size];
    }

    /**
     * Wraps an existing float array without copying.
     *
     * @param data mutable backing array; must be non-null
     */
    public Float32Storage(float[] data) {
        this.data = data;
    }

    @Override
    public DataType getType() {
        return DataType.FLOAT32;
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
     * @return stored float value
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public float getFloat32At(int flatIndex) {
        return data[flatIndex];
    }

    /**
     * Writes one physical storage element.
     *
     * @param flatIndex zero-based physical storage index
     * @param value value to store
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public void setFloat32At(int flatIndex, float value) {
        data[flatIndex] = value;
    }

    /**
     * Returns the mutable backing array without copying.
     *
     * @return non-null storage-order array
     */
    public float[] getFloatArray() {
        return data;
    }
}
