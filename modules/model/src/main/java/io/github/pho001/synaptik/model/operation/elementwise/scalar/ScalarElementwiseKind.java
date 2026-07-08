package io.github.pho001.synaptik.model.operation.elementwise.scalar;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent, parameterized, one-input scalar elementwise semantics.
 *
 * <p>Each kind describes how one scalar parameter, or one ordered pair of scalar bounds, refines
 * a computation over one logical Tensor input. The scalar attributes are semantic parameters;
 * they are not additional Tensor inputs. Each kind declares a one-input, one-output signature and
 * retains no input, graph occurrence, result fact, executable behavior, or backend information.</p>
 *
 * <p>The valid kind-to-attributes pairings are {@link #MUL}, {@link #POW}, {@link #CLAMP_MIN},
 * and {@link #CLAMP_MAX} with {@link ScalarValueAttrs}, and {@link #CLAMP} with {@link
 * ClampRangeAttrs}. {@link io.github.pho001.synaptik.model.operation.Operation Operation}
 * construction enforces these exact pairings through the family-owned signatures.</p>
 *
 * <p>Enum identity supplies typed equality and hashing, so equally named constants in another
 * operation family remain different semantic values. The inherited {@link #name()} and {@link
 * #toString()} text is diagnostic vocabulary only, not a serialization token, registry key,
 * reflective identifier, or backend-dispatch contract.</p>
 */
public enum ScalarElementwiseKind implements OperationKind {
    /**
     * Multiplies each input value by the scalar multiplier in {@link ScalarValueAttrs}.
     *
     * <p>The multiplier is a semantic parameter rather than a second Tensor input. Input
     * eligibility, scalar conversion, result type and shape, numerical edge behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    MUL,

    /**
     * Raises each input value, as the base, to the scalar exponent in {@link ScalarValueAttrs}.
     *
     * <p>The base-and-exponent roles are semantic. Input eligibility, scalar conversion, result
     * type and shape, domain and numerical edge behavior, gradients, execution, and backend
     * support belong to later owning contracts.</p>
     */
    POW,

    /**
     * Constrains each input value to the inclusive lower and upper bounds in {@link
     * ClampRangeAttrs}.
     *
     * <p>The bounds are semantic parameters rather than Tensor inputs. Input eligibility, scalar
     * conversion, result type and shape, special-value and boundary behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    CLAMP,

    /**
     * Constrains each input value to be no lower than the minimum in {@link ScalarValueAttrs}.
     *
     * <p>The minimum is a semantic parameter rather than a Tensor input. Input eligibility,
     * scalar conversion, result type and shape, special-value and boundary behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    CLAMP_MIN,

    /**
     * Constrains each input value to be no greater than the maximum in {@link ScalarValueAttrs}.
     *
     * <p>The maximum is a semantic parameter rather than a Tensor input. Input eligibility,
     * scalar conversion, result type and shape, special-value and boundary behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    CLAMP_MAX;

    private static final List<OperationSignature> SCALAR_SIGNATURES =
            List.of(OperationSignature.fixed(ScalarValueAttrs.class, 1, 1));
    private static final List<OperationSignature> CLAMP_SIGNATURES =
            List.of(OperationSignature.fixed(ClampRangeAttrs.class, 1, 1));

    /**
     * Returns the exact one-input, one-output attributes variant accepted by this scalar kind.
     *
     * @return the stable clamp-range signature for {@link #CLAMP}, otherwise the stable scalar-
     *     value signature
     */
    @Override
    public List<OperationSignature> signatures() {
        return this == CLAMP ? CLAMP_SIGNATURES : SCALAR_SIGNATURES;
    }
}
