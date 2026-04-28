package tensor;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Mutable {@link DataType#FLOAT64} tensor storage backed by a {@code double[]}.
 *
 * <p>The backing array is exposed directly and is not synchronized. Direct array
 * writes require a matching {@link #markModified()} call if version tracking
 * matters to the caller.</p>
 */
public class Float64Storage implements TensorStorage {
    private final double[] data;
    private final AtomicLong version = new AtomicLong();

    /**
     * Allocates zero-filled double storage.
     *
     * @param size number of elements; negative values are rejected by the JVM
     */
    public Float64Storage(int size) {
        this.data = new double[size];
    }

    /**
     * Wraps an existing double array without copying.
     *
     * @param data mutable backing array; must be non-null
     * @throws NullPointerException if {@code data} is null and later accessed
     */
    public Float64Storage(double[] data) {
        this.data = data;
    }

    @Override
    public DataType getType() {
        return DataType.FLOAT64;
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
     * @return stored double value
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public double getFloat64At(int flatIndex) {
        return data[flatIndex];
    }

    /**
     * Writes one physical storage element.
     *
     * @param flatIndex zero-based physical storage index
     * @param value value to store
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public void setFloat64At(int flatIndex, double value) {
        data[flatIndex] = value;
    }

    /**
     * Returns the mutable backing array without copying.
     *
     * @return non-null storage-order array
     */
    public double[] getDoubleArray() {
        return data;
    }
}
