package io.github.pho001.synaptik.model.operation.attention;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable configuration for one scaled dot-product attention occurrence.
 *
 * <p>An empty scale means the semantic default {@code 1 / sqrt(E)}, where {@code E} is the
 * positive query/key embedding extent after binding. A present scale retains its exact floating
 * type and bits and must be finite and strictly positive. Causal selection admits a key position
 * {@code j} for a query position {@code i} exactly when {@code j <= i}. These attributes contain
 * no Tensor, dropout state, backend choice, execution algorithm, or gradient rule.</p>
 *
 * @param scale non-null optional exact FLOAT64, FLOAT32, or BFLOAT16 positive finite scale; FLOAT32
 *     is the scalar metadata format for a FLOAT16 or OFP8 result, and empty selects the embedding-derived default
 * @param causal whether top-left-aligned causal eligibility is combined with any explicit mask
 */
public record ScaledDotProductAttentionAttrs(Optional<ScalarValue> scale, boolean causal)
        implements OperationAttrs {
    /**
     * Creates validated immutable attention attributes.
     *
     * @param scale non-null optional exact FLOAT64, FLOAT32 or BFLOAT16 scale; a present value
     *     must be finite and strictly positive; FLOAT16 and OFP8 results require FLOAT32 metadata
     * @param causal whether key position {@code j} is additionally restricted to {@code j <= i}
     * @throws NullPointerException if {@code scale} is null
     * @throws IllegalArgumentException if a present scale has an unsupported metadata type,
     *     is non-finite or is not positive
     */
    public ScaledDotProductAttentionAttrs {
        Objects.requireNonNull(scale, "scale");
        if (scale.isPresent()) {
            ScalarValue value = scale.orElseThrow();
            DataType dataType = value.dataType();
            if (!dataType.isFloating()) {
                throw new IllegalArgumentException(
                        "scale must have a floating data type, but was " + dataType);
            }
            if (dataType != DataType.FLOAT64 && dataType != DataType.FLOAT32
                    && dataType != DataType.BFLOAT16) {
                throw new IllegalArgumentException(
                        "scale metadata must be FLOAT64, FLOAT32 or BFLOAT16, but was " + dataType);
            }
            double decoded = switch (dataType) {
                case FLOAT64 -> value.float64Value();
                case FLOAT32 -> value.float32Value();
                case BFLOAT16 -> BFloat16Bits.toFloat(value.bfloat16Bits());
                default -> throw new AssertionError("unreachable floating data type: " + dataType);
            };
            if (!Double.isFinite(decoded) || decoded <= 0.0d) {
                throw new IllegalArgumentException(
                        "scale must be finite and positive: " + decoded);
            }
        }
    }
}
