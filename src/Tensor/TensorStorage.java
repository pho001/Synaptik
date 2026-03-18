package Tensor;

public interface TensorStorage {
    DataType getType();
    int getSize();

    double getAsDoubleAt(int flatIndex);
    void setAsDoubleAt(int flatIndex, double value);
}
