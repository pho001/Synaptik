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
 */
public enum UnaryElementwiseKind implements OperationKind {
    /**
     * Produces the absolute magnitude of each input value.
     *
     * <p>This kind defines mathematical absolute value only. Input and result types, signed-zero
     * and special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    ABS,

    /**
     * Produces the additive inverse of each input value.
     *
     * <p>This kind defines mathematical negation only. Input and result types, overflow and
     * special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    NEG,

    /**
     * Produces the multiplicative reciprocal of each input value.
     *
     * <p>This kind defines the mathematical reciprocal only. Input and result types, zero and
     * special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    RECIPROCAL,

    /**
     * Produces the natural logarithm of each input value.
     *
     * <p>This kind does not define an accepted domain, result type, special-value behavior,
     * accuracy, differentiation, execution, or backend availability. Later owning contracts
     * define those rules.</p>
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
     * <p>This portable mathematical request does not select an algorithm or promise a bitwise
     * result, approximation bound, or backend route. Result type, overflow, underflow,
     * special-value behavior, accuracy, differentiation, execution, and backend availability
     * belong to later owning contracts.</p>
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
     * <p>This kind defines the mathematical function only. Input and result types, accuracy,
     * special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    ERF,

    /**
     * Produces the principal square root of each input value.
     *
     * <p>This kind does not define an accepted domain, result type, signed-zero or special-value
     * behavior, accuracy, differentiation, execution, or backend availability. Later owning
     * contracts define those rules.</p>
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
     * <p>This kind defines mathematical floor only. Input and result types, representation,
     * special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    FLOOR,

    /**
     * Produces the least integer-valued result not less than each input value.
     *
     * <p>This kind defines mathematical ceiling only. Input and result types, representation,
     * special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    CEIL,

    /**
     * Classifies each input value as negative, zero, or positive and represents that sign
     * numerically.
     *
     * <p>This kind does not define the result type, exact numeric representation, signed-zero or
     * NaN behavior, differentiation, execution, or backend availability. Later owning contracts
     * define those rules.</p>
     */
    SIGN,

    /**
     * Applies the rectified linear unit activation to each input value.
     *
     * <p>This kind defines the activation identity only. Input and result types, behavior at zero
     * and for special values, gradient convention, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    RELU,

    /**
     * Applies the logistic sigmoid activation to each input value.
     *
     * <p>This kind defines the activation identity only. Input and result types, numerical
     * stability, accuracy, special-value behavior, differentiation, execution, and backend
     * availability belong to later owning contracts.</p>
     */
    SIGMOID,

    /**
     * Applies the hyperbolic tangent function to each input value.
     *
     * <p>This portable mathematical request does not select an algorithm or promise a bitwise
     * result, approximation bound, or backend route. Input and result types, accuracy,
     * special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
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
