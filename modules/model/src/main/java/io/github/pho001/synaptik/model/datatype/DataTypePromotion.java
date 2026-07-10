package io.github.pho001.synaptik.model.datatype;

import java.util.Objects;

/**
 * Applies backend-independent promotion rules to Synaptik data types.
 *
 * <p>The floating-only contract preserves the established floating hierarchy. The broader
 * numeric contract accepts two operands only when both are floating or both are signed integral.
 * It never treats boolean as numeric or inserts an implicit cross-category cast.</p>
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
     * Promotes two data types from the same numeric category to their common operation type.
     *
     * <p>Floating pairs retain the {@link #promoteFloating(DataType, DataType)} hierarchy.
     * Integral pairs use {@link DataType#INT32} &lt; {@link DataType#INT64}; choosing
     * {@code INT64} means that an {@code INT32} operand participates in the signed, sign-extended
     * {@code INT64} operation domain. The operation is symmetric and idempotent. It never
     * converts between floating and integral categories, and it does not treat
     * {@link DataType#BOOL} as numeric. This method selects metadata only and neither converts nor
     * evaluates values.</p>
     *
     * @param left non-null floating or integral data type of the left operand
     * @param right non-null floating or integral data type of the right operand
     * @return non-null widest data type in the operands' shared numeric category
     * @throws NullPointerException if {@code left} or {@code right} is {@code null}, checked in
     *     that order
     * @throws IllegalArgumentException if either operand is boolean or the operands belong to
     *     different numeric categories
     */
    public static DataType promoteNumeric(DataType left, DataType right) {
        DataType checkedLeft = Objects.requireNonNull(left, "left");
        DataType checkedRight = Objects.requireNonNull(right, "right");
        if (!checkedLeft.isFloating() && !checkedLeft.isIntegral()) {
            throw new IllegalArgumentException(
                    "left must be a numeric data type, but was " + checkedLeft);
        }
        if (!checkedRight.isFloating() && !checkedRight.isIntegral()) {
            throw new IllegalArgumentException(
                    "right must be a numeric data type, but was " + checkedRight);
        }
        if (checkedLeft.category() != checkedRight.category()) {
            throw new IllegalArgumentException(
                    "numeric data types must share a category, but were "
                            + checkedLeft + " and " + checkedRight);
        }
        if (checkedLeft.isFloating()) {
            return promoteFloating(checkedLeft, checkedRight);
        }
        return checkedLeft == DataType.INT64 || checkedRight == DataType.INT64
                ? DataType.INT64
                : DataType.INT32;
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
