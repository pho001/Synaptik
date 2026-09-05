package io.github.pho001.synaptik.model.operation.elementwise.cast;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent, parameterized elementwise data-type conversion semantics.
 *
 * <p>A cast applies independently to corresponding values from one logical input. Its signature
 * declares one input and one output. The input and its source data type
 * belong to the Tensor or graph value supplied by a later expression contract, while {@link
 * CastAttrs} carries the requested target data type. This kind stores neither input, source type,
 * target attributes, result descriptor, nor graph-occurrence identity.</p>
 *
 * <p>The valid family composition pairs {@link #CAST} with {@link CastAttrs}; {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} construction enforces that exact
 * pairing.</p>
 *
 * <p>Every current source and target data-type pair is valid. {@link CastValueConversions}
 * defines the exact scalar value meaning, including identity-preserving same-type conversion,
 * direct floating rounding, deterministic NaN handling, saturating floating-to-integral
 * conversion, integral width conversion, and Boolean truthiness. Tensor result inference and
 * gradient eligibility belong to the expression contract; execution and backend availability
 * remain backend-specific. Enum identity supplies typed equality and hashing. The inherited {@link
 * #name()} and {@link #toString()} text is diagnostic only, not a serialization, parsing,
 * registry, reflection, dispatch, or kernel contract.</p>
 */
public enum CastKind implements OperationKind {
    /**
     * Converts each value from one logical input to the target data type in {@link CastAttrs}.
     *
     * <p>The source data type remains a fact of the later input descriptor and is not duplicated
     * in the operation attributes. All current source and target pairs use the value contract in
     * {@link CastValueConversions}. Result inference and gradient behavior belong to the Tensor
     * and Compiler contracts; execution and availability remain backend-specific.</p>
     */
    CAST;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(CastAttrs.class, 1, 1));

    /**
     * Returns the exact cast-attributes, one-input, one-output structural variant.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
