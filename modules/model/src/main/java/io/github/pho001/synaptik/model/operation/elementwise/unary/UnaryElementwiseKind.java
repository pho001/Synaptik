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
 * <p>Each of the nineteen kinds has one Model formula for accepted BFLOAT16, FLOAT16, FLOAT32, and
 * FLOAT64 values. ABS and NEG transform non-NaN represented sign bits exactly; NaN class is
 * preserved without a payload or sign promise. SIGN, FLOOR, CEIL, and RELU retain exact
 * represented-value rules. None of these six uses arithmetic DAZ or FTZ. RECIPROCAL is
 * a named division. LOG, LOG1P, EXP, EXPM1, ERF, SQRT, and TANH retain their mathematical
 * reference, domain, class, range, and signed-zero rules; finite accuracy is qualified per backend
 * route in tests rather than by a public ordered-distance or error envelope. RSQRT, SIGMOID, GELU,
 * tanh GELU, and SiLU recurse through their named guards, constants, elementary functions, and
 * rounded arithmetic primitives; they gain no blanket output tolerance. Only named arithmetic
 * primitive inputs/results may use DAZ/FTZ for FLOAT32, BFLOAT16, and FLOAT16, never FLOAT64; exact
 * unary sites and stored outputs do not inherit that permission. Homogeneous low arithmetic uses
 * FLOAT32 working values and one final ties-to-even low narrowing. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
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
     * <p>The mathematical reference is the exact natural logarithm. Either signed zero maps to
     * negative infinity, negative finite values and negative infinity map to NaN, positive
     * infinity remains positive infinity, and NaN remains NaN. For a positive finite input,
     * finite accuracy is qualified per backend route in tests rather than by a public bound.
     * Input/result eligibility, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    LOG,

    /**
     * Produces the natural logarithm of one plus each input value.
     *
     * <p>The target is the first-class exact {@code log1p(x)} function, without a separately
     * rounded addition. It preserves signed zero, produces negative infinity at negative one,
     * produces NaN below negative one and for NaN input, and maps positive infinity to positive
     * infinity. For an in-domain finite input, accuracy is qualified per backend route in tests.
     * It is not a stored addition followed by {@link #LOG} and selects no gradient rule, execution
     * route, or backend availability.</p>
     */
    LOG1P,

    /**
     * Produces the natural exponential of each input value.
     *
     * <p>The mathematical reference is the exact natural exponential. Negative infinity maps to
     * positive zero, positive infinity remains positive infinity, and NaN remains NaN; finite
     * overflow and underflow follow the represented result type. Finite accuracy is qualified per
     * backend route in tests. It selects no algorithm, gradient rule, route, or backend
     * availability.</p>
     */
    EXP,

    /**
     * Produces the natural exponential of each input value minus one.
     *
     * <p>The target is the first-class exact {@code expm1(x)} function, without a separately
     * rounded subtraction. It preserves signed zero, maps negative infinity to negative one and
     * positive infinity to positive infinity, and produces NaN for NaN input. Finite accuracy is
     * qualified per backend route in tests. It is not a stored {@link #EXP} followed by
     * subtraction and selects no gradient rule, execution route, or backend availability.</p>
     */
    EXPM1,

    /**
     * Produces the Gaussian error function of each input value.
     *
     * <p>The exact mathematical reference is
     * {@code 2/sqrt(pi) * integral[0,x](exp(-t*t)) dt}. It is odd, preserves signed zero, maps
     * signed infinity to the same-signed unit value, and maps NaN to NaN. Its mathematical
     * reference lies in {@code [-1,1]}; finite accuracy is qualified per backend route in tests,
     * not by a public coefficient bound. It selects no algorithm, coefficient table, gradient
     * rule, execution route, or backend availability.</p>
     */
    ERF,

    /**
     * Produces the principal square root of each input value.
     *
     * <p>The exact mathematical reference preserves signed zero and positive infinity; negative
     * finite values and negative infinity map to NaN, and NaN remains NaN. Finite accuracy is
     * qualified per backend route in tests. Input/result eligibility, differentiation, execution,
     * and backend availability belong to
     * later owning contracts.</p>
     */
    SQRT,

    /**
     * Produces the reciprocal of the principal square root of each input value.
     *
     * <p>This is one first-class {@code 1 / sqrt(x)} request. Positive and negative zero map to
     * same-signed infinity, positive infinity maps to positive zero, negative finite values and
     * negative infinity produce NaN, and NaN remains NaN. Its formula has a named square-root
     * site followed by typed-one division. Only named arithmetic primitive inputs and results
     * may use dtype-specific DAZ/FTZ; homogeneous low work uses FLOAT32 and one final narrowing.
     * The kind stores neither primitive operation and selects no algorithm, gradient rule, route,
     * or backend availability.</p>
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
     * <p>The stable target is {@code 1 / (1 + exp(-x))} for nonnegative input and {@code exp(x) /
     * (1 + exp(x))} for negative input. Negative infinity maps to positive zero, either zero maps
     * to {@code 0.5}, positive infinity maps to one, and NaN remains NaN. The exact sign guard
     * selects the corresponding {@link #EXP}, typed-one addition, and division sites; only named
     * arithmetic primitives inherit dtype-specific DAZ/FTZ. The kind selects no gradient rule,
     * execution route, or backend availability.</p>
     */
    SIGMOID,

    /**
     * Applies the hyperbolic tangent function to each input value.
     *
     * <p>The exact mathematical reference preserves signed zero, maps signed infinity to the
     * same-signed unit value, and maps NaN to NaN. Its mathematical reference lies in
     * {@code [-1,1]}; finite accuracy is qualified per backend route in tests. It selects no
     * algorithm, gradient rule, execution route, or backend availability.</p>
     */
    TANH,

    /**
     * Applies the exact Gaussian error linear unit to each input value.
     *
     * <p>The target is {@code 0.5 * x * (1 + erf(x / sqrt(2)))}. Its constants, square root,
     * division, error-function, addition, and two multiplications are the complete sites. Only
     * named arithmetic primitives inherit dtype-specific DAZ/FTZ; the composite gains no blanket
     * output tolerance. The continuous extension maps negative infinity to negative zero,
     * preserves signed zero, maps positive infinity to positive infinity, and produces NaN for
     * NaN input. The kind selects no gradient rule, route, or backend support.</p>
     */
    GELU,

    /**
     * Applies the fixed conventional hyperbolic-tangent GELU approximation to each input value.
     *
     * <p>The target is
     * {@code 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))}. The constants;
     * {@code x*x} and {@code x^2*x}; root, division, additions, remaining multiplications, and
     * {@link #TANH} are the complete sites. Only named arithmetic primitives inherit
     * dtype-specific DAZ/FTZ; the fixed composite gains no blanket output tolerance. Its
     * continuous extension maps negative infinity to negative zero, preserves signed zero, maps
     * positive infinity to positive infinity, and produces NaN for NaN input. This is not
     * permission to select another approximation, gradient rule, route, or backend.</p>
     */
    GELU_TANH_APPROXIMATION,

    /**
     * Applies the sigmoid linear unit activation to each input value.
     *
     * <p>The stable target is {@code x / (1 + exp(-x))} for nonnegative input and
     * {@code x * exp(x) / (1 + exp(x))} for negative input. Its exact sign guard selects the
     * corresponding {@link #EXP}, typed-one addition, multiplication, and division sites; only
     * named arithmetic primitives inherit dtype-specific DAZ/FTZ. Its continuous extension maps
     * negative infinity to negative zero, preserves signed zero, maps positive infinity to
     * positive infinity, and produces NaN for NaN input. The kind selects no gradient rule, route,
     * or backend support.</p>
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
