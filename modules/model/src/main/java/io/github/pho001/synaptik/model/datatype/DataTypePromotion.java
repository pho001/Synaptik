package io.github.pho001.synaptik.model.datatype;

import java.util.Objects;

/**
 * Applies backend-independent promotion rules to Synaptik data types.
 *
 * <p>The initial contract intentionally supports floating-point promotion only. Integral, boolean,
 * and cross-category conversions require an explicit cast and are therefore rejected rather than
 * inferred by this class.</p>
 */
public final class DataTypePromotion {
    /** Prevents instantiation of this stateless promotion utility. */
    private DataTypePromotion() {
    }

    /**
     * Promotes two floating data types to the widest precision required by either input.
     *
     * <p>The stable order is {@link DataType#BFLOAT16} &lt; {@link DataType#FLOAT32} &lt;
     * {@link DataType#FLOAT64}. The operation is symmetric and idempotent.</p>
     *
     * @param left non-null floating data type of the left operand
     * @param right non-null floating data type of the right operand
     * @return non-null promoted floating data type that can represent both input precisions
     * @throws NullPointerException if {@code left} or {@code right} is {@code null}
     * @throws IllegalArgumentException if either input is integral or boolean
     */
    public static DataType promoteFloating(DataType left, DataType right) {
        DataType checkedLeft = requireFloating(Objects.requireNonNull(left, "left"), "left");
        DataType checkedRight = requireFloating(Objects.requireNonNull(right, "right"), "right");

        if (checkedLeft == DataType.FLOAT64 || checkedRight == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (checkedLeft == DataType.FLOAT32 || checkedRight == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    /**
     * Validates one promotion operand without introducing implicit cross-category conversion.
     *
     * @param dataType non-null data type to validate
     * @param parameterName non-null parameter name used in the failure message
     * @return the same non-null floating data type instance
     * @throws IllegalArgumentException if {@code dataType} is not floating
     */
    private static DataType requireFloating(DataType dataType, String parameterName) {
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    parameterName + " must be a floating data type, but was " + dataType);
        }
        return dataType;
    }
}
