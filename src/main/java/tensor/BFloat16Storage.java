package tensor;

public class BFloat16Storage implements TensorStorage {
    private final short[] data;

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
