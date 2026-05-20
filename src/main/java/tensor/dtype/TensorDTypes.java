package tensor.dtype;

import tensor.DataType;

import java.util.Objects;

/**
 * General dtype contracts shared by tensor operation builders.
 */
public final class TensorDTypes {
    private TensorDTypes() {
    }

    /**
     * Requires a floating dtype and returns it for fluent validation.
     *
     * @param dataType dtype to validate
     * @return the validated dtype
     * @throws IllegalArgumentException if dtype is integral or boolean
     */
    public static DataType requireFloating(DataType dataType) {
        DataType value = Objects.requireNonNull(dataType, "dataType cannot be null");
        if (!value.isFloating()) {
            throw new IllegalArgumentException("Only floating numeric dtypes are supported.");
        }
        return value;
    }

    /**
     * Requires an integral dtype and returns it for fluent validation.
     *
     * @param dataType dtype to validate
     * @return the validated dtype
     * @throws IllegalArgumentException if dtype is floating or boolean
     */
    public static DataType requireIntegral(DataType dataType) {
        DataType value = Objects.requireNonNull(dataType, "dataType cannot be null");
        if (!value.isIntegral()) {
            throw new IllegalArgumentException("Only integral dtypes are supported.");
        }
        return value;
    }

    /**
     * Requires a boolean dtype and returns it for fluent validation.
     *
     * @param dataType dtype to validate
     * @return the validated dtype
     * @throws IllegalArgumentException if dtype is not BOOL
     */
    public static DataType requireBoolean(DataType dataType) {
        DataType value = Objects.requireNonNull(dataType, "dataType cannot be null");
        if (!value.isBoolean()) {
            throw new IllegalArgumentException("Only BOOL dtype is supported.");
        }
        return value;
    }

    /**
     * Promotes two floating dtypes to the widest supported floating dtype.
     *
     * @param left left dtype
     * @param right right dtype
     * @return FLOAT64 when either side is FLOAT64, otherwise FLOAT32, otherwise BFLOAT16
     */
    public static DataType promoteFloating(DataType left, DataType right) {
        DataType leftValue = requireFloating(left);
        DataType rightValue = requireFloating(right);
        if (leftValue == DataType.FLOAT64 || rightValue == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (leftValue == DataType.FLOAT32 || rightValue == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }
}
