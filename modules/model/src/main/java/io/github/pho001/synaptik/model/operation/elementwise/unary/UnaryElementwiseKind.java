package io.github.pho001.synaptik.model.operation.elementwise.unary;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies nineteen backend-independent parameterless unary elementwise semantics.
 *
 * <p>Each kind describes the mathematical or activation meaning applied independently to one
 * logical input. The shared signature declares one input and one output; the kind does not retain
 * input provenance, identify a graph occurrence, infer a result, execute mathematics,
 * define gradients, or report backend support. Those responsibilities belong to the public
 * expression, compiler, autograd, execution, and backend layers that consume this vocabulary.</p>
 *
 * <p>All kinds in this family have no intrinsic parameters. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} therefore represents one of them
 * with {@link io.github.pho001.synaptik.model.operation.NoOperationAttrs#INSTANCE
 * NoOperationAttrs.INSTANCE}. The enum stores no result facts, numerical policy, algorithm, or
 * backend implementation route in its structural signature.</p>
 *
 * <p>Enum identity supplies typed equality and hashing, so an equally named constant in another
 * operation family remains a different semantic value. The inherited {@link #name()} and {@link
 * #toString()} text is stable diagnostic vocabulary only, not a serialization token, registry
 * key, or string-dispatch contract.</p>
 *
 * <p>Under the Model-owned numerical-profile contract, {@code STRICT_IEEE} retains each formula
 * and its signed-zero, infinity, NaN, and domain rules below. For {@code ACCELERATOR FLOAT32},
 * named arithmetic sites may use DAZ/FTZ; logarithmic, exponential, error-function,
 * root/reciprocal-root, sigmoid/tanh, GELU, and SiLU elementary sites have the five-ULP
 * primitive-site ceiling. Composite activation formulas recurse through their named sites and
 * gain no final-output envelope. Ordinary finite values cannot change class except by actual
 * overflow or a domain path after DAZ, NaN cannot become ordinary, and non-FLOAT32 behavior stays
 * strict. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
 */
public enum UnaryElementwiseKind implements OperationKind {
    /**
     * Produces the absolute magnitude of each input value.
     *
     * <p>The mathematical target maps either zero to positive zero, either infinity to positive
     * infinity, and NaN to NaN; a NaN payload is not part of the request. Input/result
     * eligibility, differentiation, execution, and backend availability belong to later owning
     * contracts.</p>
     */
    ABS,

    /**
     * Produces the additive inverse of each input value.
     *
     * <p>The mathematical target flips the sign of finite values, signed zero, and infinity and
     * maps NaN to NaN without promising a payload. Input/result eligibility, differentiation,
     * execution, and backend availability belong to later owning contracts.</p>
     */
    NEG,

    /**
     * Produces the multiplicative reciprocal of each input value.
     *
     * <p>The mathematical target is {@code 1 / x}: signed zero maps to same-signed infinity,
     * signed infinity maps to same-signed zero, and NaN maps to NaN. Input/result eligibility,
     * differentiation, execution, and backend availability belong to later owning contracts.</p>
     */
    RECIPROCAL,

    /**
     * Produces the natural logarithm of each input value.
     *
     * <p>The mathematical target maps either signed zero to negative infinity, negative finite
     * values and negative infinity to NaN, positive infinity to positive infinity, and NaN to NaN.
     * Positive finite values use the natural logarithm. Input/result eligibility,
     * differentiation, execution, and backend availability belong to later owning contracts.</p>
     */
    LOG,

    /**
     * Produces the natural logarithm of one plus each input value.
     *
     * <p>This portable mathematical request preserves signed zero, produces negative infinity at
     * negative one, produces NaN below negative one and for NaN input, and maps positive infinity
     * to positive infinity. It does not select an algorithm or promise correct rounding, a
     * bitwise result, an accuracy bound, a gradient rule, an execution route, or backend
     * availability.</p>
     */
    LOG1P,

    /**
     * Produces the natural exponential of each input value.
     *
     * <p>The mathematical target maps negative infinity to positive zero, positive infinity to
     * positive infinity, and NaN to NaN; finite overflow and underflow follow the represented
     * result type. It selects no algorithm, gradient rule, route, or backend availability.</p>
     */
    EXP,

    /**
     * Produces the natural exponential of each input value minus one.
     *
     * <p>This portable mathematical request preserves signed zero, maps negative infinity to
     * negative one and positive infinity to positive infinity, and produces NaN for NaN input.
     * It does not select an algorithm or promise correct rounding, a bitwise result, an accuracy
     * bound, a gradient rule, an execution route, or backend availability.</p>
     */
    EXPM1,

    /**
     * Produces the Gaussian error function of each input value.
     *
     * <p>The mathematical target is odd, preserves signed zero, maps signed infinity to the
     * same-signed unit value, and maps NaN to NaN. It selects no algorithm, gradient rule,
     * execution route, or backend availability.</p>
     */
    ERF,

    /**
     * Produces the principal square root of each input value.
     *
     * <p>The mathematical target preserves signed zero and positive infinity; negative finite
     * values and negative infinity map to NaN, and NaN maps to NaN. Input/result eligibility,
     * differentiation, execution, and backend availability belong to later owning contracts.</p>
     */
    SQRT,

    /**
     * Produces the reciprocal of the principal square root of each input value.
     *
     * <p>This is one first-class mathematical request. Positive and negative zero map to
     * same-signed infinity, positive infinity maps to positive zero, negative finite values and
     * negative infinity produce NaN, and NaN produces NaN. It does not store a square-root
     * operation followed by a reciprocal operation or promise correct rounding, a bitwise result,
     * or an accuracy bound. Gradients, execution, and backend availability belong to later owning
     * contracts.</p>
     */
    RSQRT,

    /**
     * Produces the greatest integer-valued result not greater than each input value.
     *
     * <p>The mathematical target preserves signed zero and signed infinity, maps NaN to NaN, and
     * returns the greatest integral-valued floating result not greater than a finite input.
     * Differentiation, execution, and backend availability belong to later owning contracts.</p>
     */
    FLOOR,

    /**
     * Produces the least integer-valued result not less than each input value.
     *
     * <p>The mathematical target preserves signed zero and signed infinity, maps NaN to NaN, and
     * returns the least integral-valued floating result not less than a finite input.
     * Differentiation, execution, and backend availability belong to later owning contracts.</p>
     */
    CEIL,

    /**
     * Classifies each input value as negative, zero, or positive and represents that sign
     * numerically.
     *
     * <p>The mathematical target returns negative one for negative nonzero values, positive one
     * for positive nonzero values, preserves signed zero, and maps NaN to NaN. Differentiation,
     * execution, and backend availability belong to later owning contracts.</p>
     */
    SIGN,

    /**
     * Applies the rectified linear unit activation to each input value.
     *
     * <p>The mathematical target is {@code max(x, +0)} under the family extrema rules: negative
     * finite values and negative infinity map to positive zero, either zero maps to positive zero,
     * positive values pass through, and NaN maps to NaN. The zero gradient convention, execution,
     * and backend availability belong to later owning contracts.</p>
     */
    RELU,

    /**
     * Applies the logistic sigmoid activation to each input value.
     *
     * <p>The mathematical target is {@code 1 / (1 + exp(-x))}: negative infinity maps to positive
     * zero, positive infinity maps to one, and NaN maps to NaN. This identity selects no stability
     * algorithm, gradient rule, execution route, or backend availability.</p>
     */
    SIGMOID,

    /**
     * Applies the hyperbolic tangent function to each input value.
     *
     * <p>The mathematical target preserves signed zero, maps signed infinity to the same-signed
     * unit value, and maps NaN to NaN. It selects no algorithm, gradient rule, execution route, or
     * backend availability.</p>
     */
    TANH,

    /**
     * Applies the exact Gaussian error linear unit to each input value.
     *
     * <p>The selected mathematical target is {@code x * Phi(x)}, equivalently
     * {@code 0.5 * x * (1 + erf(x / sqrt(2)))}. Its continuous extension maps negative infinity
     * to negative zero, preserves signed zero, maps positive infinity to positive infinity, and
     * produces NaN for NaN input. This first-class request does not prescribe composition,
     * rounding, an approximation algorithm, a gradient rule, execution, or backend support.</p>
     */
    GELU,

    /**
     * Applies the fixed conventional hyperbolic-tangent GELU approximation to each input value.
     *
     * <p>The selected mathematical target is
     * {@code 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))}. Its continuous extension
     * maps negative infinity to negative zero, preserves signed zero, maps positive infinity to
     * positive infinity, and produces NaN for NaN input. This is a distinct parameterless
     * semantic request, not configurable permission to choose another approximation. It does not
     * prescribe composition, rounding, an evaluation algorithm, a gradient rule, execution, or
     * backend support.</p>
     */
    GELU_TANH_APPROXIMATION,

    /**
     * Applies the sigmoid linear unit activation to each input value.
     *
     * <p>The selected mathematical target is {@code x * sigmoid(x)}, equivalently
     * {@code x / (1 + exp(-x))}. Its continuous extension maps negative infinity to negative
     * zero, preserves signed zero, maps positive infinity to positive infinity, and produces NaN
     * for NaN input. This first-class request does not prescribe literal composition, rounding,
     * an evaluation algorithm, a gradient rule, execution, or backend support.</p>
     */
    SILU;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

    /**
     * Returns the parameterless one-input, one-output structural variant shared by this family.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
