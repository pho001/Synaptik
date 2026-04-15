package tensor;

import java.util.concurrent.atomic.AtomicLong;

public class Float64Storage implements TensorStorage {
    private final double[] data;
    private final AtomicLong version = new AtomicLong();

    public Float64Storage(int size) {
        this.data = new double[size];
    }

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

    public double getFloat64At(int flatIndex) {
        return data[flatIndex];
    }

    public void setFloat64At(int flatIndex, double value) {
        data[flatIndex] = value;
    }

    public double[] getDoubleArray() {
        return data;
    }
}
