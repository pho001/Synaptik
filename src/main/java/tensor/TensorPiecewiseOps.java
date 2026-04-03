package tensor;

final class TensorPiecewiseOps {
    private TensorPiecewiseOps() {}

    static Tensor minimum(Tensor first, Tensor second) {
        validateNumericBinary(first, second, "minimum");
        return Tensor.where(first.lessThan(second), first, second);
    }

    static Tensor maximum(Tensor first, Tensor second) {
        validateNumericBinary(first, second, "maximum");
        return Tensor.where(first.greaterThan(second), first, second);
    }

    private static void validateNumericBinary(Tensor first, Tensor second, String opName) {
        if (first == null || second == null) {
            throw new IllegalArgumentException(opName + " inputs cannot be null");
        }
        if (first.getDataType() == DataType.BOOL || second.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException(opName + " requires numeric inputs.");
        }
    }
}
