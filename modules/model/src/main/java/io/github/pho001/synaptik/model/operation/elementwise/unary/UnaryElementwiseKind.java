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
 * <p>Under the Model-owned numerical-profile contract, {@code STRICT_IEEE} has an explicit
 * allowed-result baseline for all nineteen kinds. It preserves normal and subnormal values,
 * domains, special classes, and required zero signs. Exact/discrete kinds retain their represented
 * result; selected Java elementary operations retain their documented accuracy; reciprocal square
 * root, error function, sigmoid, and the three composite activations retain the fixed scalar or
 * typed-lane realizations named below. Loose backend conformance tolerances are not public result
 * envelopes. For {@code ACCELERATOR FLOAT32}, only {@code LOG}, {@code LOG1P}, {@code EXP},
 * {@code EXPM1}, {@code ERF}, {@code SQRT}, and {@code TANH} are irreducible elementary sites.
 * {@code RSQRT} expands through square root and division; sigmoid expands through its exact sign
 * guard and selected exponential/add/divide branch. The GELU and SiLU formulas recurse through
 * every named constant, {@code x*x}, {@code xSquared*x}, root, division, exponential/error-function/
 * tanh, addition, and multiplication site in their fixed formula and gain no final-output
 * envelope. See the
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
     * <p>The target is the first-class {@code log1p(x)} function. It preserves signed zero,
     * produces negative infinity at negative one, produces NaN below negative one and for NaN
     * input, and maps positive infinity to positive infinity. The strict result is the selected
     * Java scalar or lane operation under the family baseline; it is not a stored addition
     * followed by {@link #LOG}. It selects no gradient rule, execution route, or backend
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
     * <p>The target is the first-class {@code expm1(x)} function. It preserves signed zero, maps
     * negative infinity to negative one and positive infinity to positive infinity, and produces
     * NaN for NaN input. The strict result is the selected Java scalar or lane operation under the
     * family baseline; it is not a stored {@link #EXP} followed by subtraction. It selects no
     * gradient rule, execution route, or backend availability.</p>
     */
    EXPM1,

    /**
     * Produces the Gaussian error function of each input value.
     *
     * <p>The mathematical target is odd, preserves signed zero, maps signed infinity to the
     * same-signed unit value, and maps NaN to NaN. Strict results are the selected scalar
     * Cephes-derived piecewise realization or selected typed lane realization, including their
     * fixed coefficients, branches, operation order, and special corrections. It selects no
     * gradient rule, execution route, or backend availability.</p>
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
     * <p>This is one first-class {@code 1 / sqrt(x)} request. Positive and negative zero map to
     * same-signed infinity, positive infinity maps to positive zero, negative finite values and
     * negative infinity produce NaN, and NaN produces NaN. Strict results are the retained scalar
     * or typed-lane square-root-then-division realization; FLOAT32 scalar execution widens the
     * input, performs both binary64 operations, and narrows only the final result. The kind stores
     * neither primitive operation and selects no gradient rule, route, or backend availability.</p>
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
     * <p>The selected stable target is {@code 1 / (1 + exp(-x))} for nonnegative input and
     * {@code exp(x) / (1 + exp(x))} for negative input. Negative infinity maps to positive zero,
     * positive infinity maps to one, and NaN maps to NaN. The sign comparison and each negation,
     * exponential, addition, and division are explicit formula sites; the kind selects no
     * gradient rule, execution route, or backend availability.</p>
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
     * <p>The selected target is {@code 0.5 * x * (1 + erf(x / sqrt(2)))}. Its typed {@code 0.5},
     * {@code 1}, and {@code 2} constants, square root, division, error-function, addition, and two
     * multiplications are the complete primitive sites. Strict uses the fixed selected operation
     * order and strict {@link #ERF} realization. The continuous extension maps negative infinity
     * to negative zero, preserves signed zero, maps positive infinity to positive infinity, and
     * produces NaN for NaN input. The kind selects no gradient rule, route, or backend support.</p>
     */
    GELU,

    /**
     * Applies the fixed conventional hyperbolic-tangent GELU approximation to each input value.
     *
     * <p>The selected target is
     * {@code 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))}. The typed
     * {@code 0.5}, {@code 1}, {@code 2}, {@code pi}, and {@code 0.044715} constants; {@code x*x}
     * and {@code x^2*x}; root, division, additions, remaining multiplications, and {@link #TANH}
     * are the complete primitive sites in the fixed strict order. Its continuous extension maps
     * negative infinity to negative zero, preserves signed zero, maps positive infinity to
     * positive infinity, and produces NaN for NaN input. This is not permission to select another
     * approximation, gradient rule, route, or backend.</p>
     */
    GELU_TANH_APPROXIMATION,

    /**
     * Applies the sigmoid linear unit activation to each input value.
     *
     * <p>The selected stable target is {@code x / (1 + exp(-x))} for nonnegative input and
     * {@code x * exp(x) / (1 + exp(x))} for negative input. The sign comparison, negation,
     * exponential, additions, multiplication, and divisions are the complete formula sites.
     * Its continuous extension maps negative infinity to negative zero, preserves signed zero,
     * maps positive infinity to positive infinity, and produces NaN for NaN input. The kind
     * selects no gradient rule, route, or backend support.</p>
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
