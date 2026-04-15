package tensor;

import java.util.concurrent.atomic.AtomicLong;

public final class BoolStorage implements TensorStorage {
    private final byte[] data;
    private final AtomicLong version = new AtomicLong();

    public BoolStorage(int size) {
        this.data = new byte[size];
    }

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

    public byte getBoolAt(int flatIndex) {
        return data[flatIndex];
    }

    public void setBoolAt(int flatIndex, byte value) {
        data[flatIndex] = value == 0 ? (byte) 0 : (byte) 1;
    }

    public byte[] getByteArray() {
        return data;
    }
}
