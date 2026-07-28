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
 * <p>The valid kind-to-attributes pairings are {@link #ADD}, {@link #SUB}, {@link #MUL},
 * {@link #DIV}, {@link #MIN}, {@link #MAX}, and {@link #POW} with {@link ScalarValueAttrs}, and
 * {@link #CLAMP} with {@link ClampRangeAttrs}. {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} construction enforces these
 * exact pairings through the family-owned signatures.</p>
 *
 * <p>Enum identity supplies typed equality and hashing, so equally named constants in another
 * operation family remain different semantic values. The inherited {@link #name()} and {@link
 * #toString()} text is diagnostic vocabulary only, not a serialization token, registry key,
 * reflective identifier, or backend-dispatch contract.</p>
 */
public enum ScalarElementwiseKind implements OperationKind {
    /**
     * Adds the scalar addend in {@link ScalarValueAttrs} to each input value.
     *
     * <p>The request is ordinary ordered IEEE-754 addition in the input data type. It stores the
     * addend as metadata and promises no NaN payload, intermediate precision, exact instruction,
     * bitwise result, gradient rule, or executable backend route.</p>
     */
    ADD,

    /**
     * Subtracts the scalar subtrahend in {@link ScalarValueAttrs} from each input value.
     *
     * <p>The input-minus-scalar order is semantic. This kind stores no input or result facts and
     * promises no numerical algorithm, gradient rule, execution, or backend support.</p>
     */
    SUB,

    /**
     * Multiplies each input value by the scalar multiplier in {@link ScalarValueAttrs}.
     *
     * <p>The multiplier is a semantic parameter rather than a second Tensor input. Input
     * eligibility, scalar conversion, result type and shape, numerical edge behavior, gradients,
     * execution, and backend support belong to later owning contracts.</p>
     */
    MUL,

    /**
     * Divides each input value by the scalar denominator in {@link ScalarValueAttrs}.
     *
     * <p>The input-divided-by-scalar order is semantic. This kind stores no input or result facts
     * and promises no numerical algorithm, gradient rule, execution, or backend support.</p>
     */
    DIV,

    /**
     * Selects the minimum of each input value and the scalar candidate in {@link
     * ScalarValueAttrs}.
     *
     * <p>The candidate is compared as the exact represented numeric value of its same-typed
     * {@code ScalarValue}; exact raw bits remain metadata identity and do not make the comparison
     * bitwise. If either candidate is NaN, the result is NaN. Opposite signed zeros produce
     * negative zero independent of operand order. Infinities and unequal non-NaN values use
     * ordinary numeric order; equal nonzero candidates produce that numeric value. The request
     * promises no NaN payload, source-operand selection, or bitwise result. Integral minimum
     * retains ordinary signed order. This kind records no evaluation algorithm, gradient rule, or
     * backend implementation.</p>
     */
    MIN,

    /**
     * Selects the maximum of each input value and the scalar candidate in {@link
     * ScalarValueAttrs}.
     *
     * <p>The candidate is compared as the exact represented numeric value of its same-typed
     * {@code ScalarValue}; exact raw bits remain metadata identity and do not make the comparison
     * bitwise. If either candidate is NaN, the result is NaN. Opposite signed zeros produce
     * positive zero independent of operand order. Infinities and unequal non-NaN values use
     * ordinary numeric order; equal nonzero candidates produce that numeric value. The request
     * promises no NaN payload, source-operand selection, or bitwise result. Integral maximum
     * retains ordinary signed order. This kind records no evaluation algorithm, gradient rule, or
     * backend implementation.</p>
     */
    MAX,

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
     * <p>The bounds are semantic parameters rather than Tensor inputs. Its value meaning is
     * exactly ordered {@code MIN(MAX(input, minValue), maxValue)} under this family's extrema
     * rules, while remaining one operation occurrence with no stored intermediate producers. A
     * NaN input or bound produces NaN without a payload promise. Equal non-NaN bounds with the
     * same representation produce that bound for every non-NaN input. Bounds {@code [-0, +0]}
     * produce negative zero for negative inputs and negative zero, and positive zero for positive
     * inputs and positive zero. Bounds {@code [+0, -0]} produce negative zero for every non-NaN
     * input. Bound validation, input eligibility, result metadata, gradients, execution, and
     * backend support belong to their owning contracts.</p>
     */
    CLAMP;

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
