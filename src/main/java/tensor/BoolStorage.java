package tensor;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Mutable {@link DataType#BOOL} tensor storage backed by a normalized byte array.
 *
 * <p>Boolean false is stored as {@code 0}; true is stored as {@code 1}. The
 * backing array is exposed directly and is not synchronized. Direct array writes
 * require a matching {@link #markModified()} call if version tracking matters.</p>
 */
public final class BoolStorage implements TensorStorage {
    private final byte[] data;
    private final AtomicLong version = new AtomicLong();

    /**
     * Allocates false-filled boolean storage.
     *
     * @param size number of elements; negative values are rejected by the JVM
     */
    public BoolStorage(int size) {
        this.data = new byte[size];
    }

    /**
     * Wraps an existing byte array without copying.
     *
     * <p>Use {@link #setBoolAt(int, byte)} to normalize future writes; this
     * constructor preserves existing byte values as supplied.</p>
     *
     * @param data mutable backing array; must be non-null
     */
    public BoolStorage(byte[] data) {
        this.data = data;
    }

    @Override
    public DataType getType() {
        return DataType.BOOL;
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
     * Reads one physical boolean storage element.
     *
     * @param flatIndex zero-based physical storage index
     * @return stored byte value, normally {@code 0} or {@code 1}
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public byte getBoolAt(int flatIndex) {
        return data[flatIndex];
    }

    /**
     * Writes one physical boolean storage element.
     *
     * @param flatIndex zero-based physical storage index
     * @param value any non-zero byte is normalized to {@code 1}
     * @throws ArrayIndexOutOfBoundsException if {@code flatIndex} is outside the backing array
     */
    public void setBoolAt(int flatIndex, byte value) {
        data[flatIndex] = value == 0 ? (byte) 0 : (byte) 1;
    }

    /**
     * Returns the mutable backing array without copying.
     *
     * @return non-null storage-order array
     */
    public byte[] getByteArray() {
        return data;
    }
}
