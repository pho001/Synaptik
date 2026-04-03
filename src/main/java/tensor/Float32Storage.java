package tensor;

public class Float32Storage implements TensorStorage {
    private final float[] data;

    public Float32Storage(int size) {
        this.data = new float[size];
    }

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

    public float getFloat32At(int flatIndex) {
        return data[flatIndex];
    }

    public void setFloat32At(int flatIndex, float value) {
        data[flatIndex] = value;
    }

    public float[] getFloatArray() {
        return data;
    }
}
