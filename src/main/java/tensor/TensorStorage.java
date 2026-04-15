package tensor;

public interface TensorStorage {
    DataType getType();
    int getSize();
    long version();
    void markModified();
}
