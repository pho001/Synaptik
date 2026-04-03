package tensor;

final class TensorDataTypeUtil {
    private TensorDataTypeUtil() {}

    static DataType promote(DataType left, DataType right) {
        if (left == DataType.BOOL || right == DataType.BOOL || left == DataType.INT32 || right == DataType.INT32) {
            throw new IllegalArgumentException("Only floating numeric dtypes are supported by numeric promotion.");
        }
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) return DataType.FLOAT64;
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) return DataType.FLOAT32;
        return DataType.FLOAT16;
    }

    static DataType binary(Tensor first, Tensor second) {
        return promote(first.getDataType(), second.getDataType());
    }

    static DataType unary(Tensor input) {
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("Only floating numeric dtypes are supported by numeric unary dtype resolution.");
        }
        return input.getDataType();
    }
}
