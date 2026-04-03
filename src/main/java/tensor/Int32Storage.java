package tensor;

public final class Int32Storage implements TensorStorage {
    private final int[] data;

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
