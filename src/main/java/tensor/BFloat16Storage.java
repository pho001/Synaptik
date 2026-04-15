package tensor;

import java.util.concurrent.atomic.AtomicLong;

public class BFloat16Storage implements TensorStorage {
    private final short[] data;
    private final AtomicLong version = new AtomicLong();

    public BFloat16Storage(int size) {
        this.data = new short[size];
    }

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

    public short getBFloat16BitsAt(int flatIndex) {
        return data[flatIndex];
    }

    public void setBFloat16BitsAt(int flatIndex, short value) {
        data[flatIndex] = value;
    }

    public short[] getShortArray() {
        return data;
    }
}
