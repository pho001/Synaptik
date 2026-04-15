package tensor;

import java.util.concurrent.atomic.AtomicLong;

public final class Int32Storage implements TensorStorage {
    private final int[] data;
    private final AtomicLong version = new AtomicLong();

    public Int32Storage(int size) {
        this.data = new int[size];
    }

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

    public int getInt32At(int flatIndex) {
        return data[flatIndex];
    }

    public void setInt32At(int flatIndex, int value) {
        data[flatIndex] = value;
    }

    public int[] getIntArray() {
        return data;
    }
}
