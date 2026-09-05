package io.github.pho001.synaptik.model.operation.elementwise.cast;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the exact target data type for an explicit elementwise cast.
 *
 * <p>The immutable record accepts every current {@link DataType} and retains the supplied enum
 * reference unchanged. The source data type is intentionally absent because it belongs to the
 * later input Tensor or graph value descriptor; duplicating it here would create two sources of
 * truth. Every current source and target pair is valid, and {@link CastValueConversions} defines
 * its exact scalar value meaning. The attributes do not perform conversion or promise that a
 * backend implements it.</p>
 *
 * <p>Record-generated equality and hashing use the target enum value. Record-generated text is
 * diagnostic only and is not a serialization, parsing, backend-dispatch, or conversion-policy
 * contract.</p>
 *
 * @param targetDataType the non-null target data type; the exact enum reference is retained
 */
public record CastAttrs(DataType targetDataType) implements OperationAttrs {
    /**
     * Creates cast attributes for an exact target data type.
     *
     * <p>The reference is validated for presence and retained without normalization, defaulting,
     * source-type lookup, conversion, or backend-capability validation.</p>
     *
     * @param targetDataType the non-null target data type to retain unchanged
     * @throws NullPointerException if {@code targetDataType} is {@code null}, with message
     *     {@code targetDataType}
     */
    public CastAttrs {
        targetDataType = Objects.requireNonNull(targetDataType, "targetDataType");
    }

    /**
     * Returns the exact target data type supplied at construction.
     *
     * <p>The result carries no source-type knowledge and performs no conversion or
     * backend-capability validation.</p>
     *
     * @return the exact stored non-null target data type reference
     */
    @Override
    public DataType targetDataType() {
        return targetDataType;
    }
}
