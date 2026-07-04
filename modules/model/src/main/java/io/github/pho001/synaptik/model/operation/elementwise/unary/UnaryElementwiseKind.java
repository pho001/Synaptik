package io.github.pho001.synaptik.model.operation.elementwise.unary;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent parameterless unary elementwise semantics.
 *
 * <p>Each kind describes the mathematical or activation meaning applied independently to one
 * logical input. One-input arity is family context rather than stored metadata: the kind does not
 * retain input provenance, identify a graph occurrence, infer a result, execute mathematics,
 * define gradients, or report backend support. Those responsibilities belong to the public
 * expression, compiler, autograd, execution, and backend layers that consume this vocabulary.</p>
 *
 * <p>All kinds in this family have no intrinsic parameters. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} therefore represents one of them
 * with {@link io.github.pho001.synaptik.model.operation.NoOperationAttrs#INSTANCE
 * NoOperationAttrs.INSTANCE}. The enum stores no input count, result facts, numerical policy, or
 * approximation implementation as attributes or metadata.</p>
 *
 * <p>{@link #FAST_EXP} and {@link #FAST_TANH} are explicit approximate semantic requests distinct
 * from {@link #EXP} and {@link #TANH}; later numerical and backend contracts define their accuracy
 * and implementation. Enum identity supplies typed equality and hashing, so an equally named
 * constant in another operation family remains a different semantic value. The inherited {@link
 * #name()} and {@link #toString()} text is stable diagnostic vocabulary only, not a serialization
 * token, registry key, or string-dispatch contract.</p>
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
     * <p>This kind defines mathematical inversion only. Input and result types, zero and
     * special-value behavior, differentiation, execution, and backend availability belong to
     * later owning contracts.</p>
     */
    INV,

    /**
     * Produces the natural logarithm of each input value.
     *
     * <p>This kind does not define an accepted domain, result type, special-value behavior,
     * accuracy, differentiation, execution, or backend availability. Later owning contracts
     * define those rules.</p>
     */
    LOG,

    /**
     * Produces the natural exponential of each input value.
     *
     * <p>This strict semantic request is distinct from {@link #FAST_EXP}. Result type, overflow,
     * underflow, special-value behavior, accuracy, differentiation, execution, and backend
     * availability belong to later owning contracts.</p>
     */
    EXP,

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
     * <p>This strict semantic request is distinct from {@link #FAST_TANH}. Input and result types,
     * accuracy, special-value behavior, differentiation, execution, and backend availability
     * belong to later owning contracts.</p>
     */
    TANH,

    /**
     * Requests an explicitly approximate natural exponential for each input value.
     *
     * <p>This kind is a distinct semantic request rather than an alias or backend flag for {@link
     * #EXP}. Its accepted types, accuracy bounds, approximation algorithm, special-value behavior,
     * differentiation, execution, and backend availability belong to later owning contracts.</p>
     */
    FAST_EXP,

    /**
     * Requests an explicitly approximate hyperbolic tangent for each input value.
     *
     * <p>This kind is a distinct semantic request rather than an alias or backend flag for {@link
     * #TANH}. Its accepted types, accuracy bounds, approximation algorithm, special-value
     * behavior, differentiation, execution, and backend availability belong to later owning
     * contracts.</p>
     */
    FAST_TANH
}
