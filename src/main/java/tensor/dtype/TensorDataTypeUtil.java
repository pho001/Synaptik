package tensor.dtype;

import tensor.DataType;
import tensor.Tensor;

/**
 * Dtype resolution helpers used by public tensor operations.
 */
public final class TensorDataTypeUtil {
    private TensorDataTypeUtil() {}

    /**
     * Promotes two floating dtypes to the result dtype used by binary operations.
     *
     * @param left left dtype; must be FLOAT64, FLOAT32, or BFLOAT16
     * @param right right dtype; must be FLOAT64, FLOAT32, or BFLOAT16
     * @return widest supported floating dtype, preferring FLOAT64 then FLOAT32 then BFLOAT16
     * @throws IllegalArgumentException if either dtype is BOOL, INT32, or INT64
     */
    public static DataType promote(DataType left, DataType right) {
        if (left == DataType.BOOL || right == DataType.BOOL
                || left == DataType.INT32 || right == DataType.INT32
                || left == DataType.INT64 || right == DataType.INT64) {
            throw new IllegalArgumentException("Only floating numeric dtypes are supported by numeric promotion.");
        }
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) return DataType.FLOAT64;
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) return DataType.FLOAT32;
        return DataType.BFLOAT16;
    }

    /**
     * Resolves the promoted dtype for a binary operation.
     *
     * @param first first tensor; must be non-null and floating
     * @param second second tensor; must be non-null and floating
     * @return promoted floating dtype
     */
    public static DataType binary(Tensor first, Tensor second) {
        return promote(first.getDataType(), second.getDataType());
    }

    /**
     * Resolves the output dtype for a unary floating operation.
     *
     * @param input input tensor; must be non-null and floating
     * @return input dtype
     * @throws IllegalArgumentException if the input dtype is BOOL, INT32, or INT64
     */
    public static DataType unary(Tensor input) {
        if (input.getDataType() == DataType.BOOL
                || input.getDataType() == DataType.INT32
                || input.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("Only floating numeric dtypes are supported by numeric unary dtype resolution.");
        }
        return input.getDataType();
    }
}
