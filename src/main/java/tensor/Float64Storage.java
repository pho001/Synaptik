package tensor;

public class Float64Storage implements TensorStorage {
    private final double[] data;

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
    public double getAsDoubleAt(int flatIndex) {
        return data[flatIndex];
    }

    @Override
    public void setAsDoubleAt(int flatIndex, double value) {
        data[flatIndex] = value;
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
