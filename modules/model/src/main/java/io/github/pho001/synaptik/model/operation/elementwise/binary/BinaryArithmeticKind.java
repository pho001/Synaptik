package io.github.pho001.synaptik.model.operation.elementwise.binary;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent tensor-to-tensor elementwise binary arithmetic semantics.
 *
 * <p>Each kind describes the mathematical relationship between an ordered left operand and right
 * operand. The kind does not store either operand, identify a graph occurrence, create tensor
 * provenance, infer a result, execute arithmetic, or report backend support. Those responsibilities
 * belong to the public expression, compiler, and execution layers that consume this semantic
 * vocabulary.</p>
 *
 * <p>All kinds in this family have no intrinsic parameters. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} therefore represents one of them
 * with {@link io.github.pho001.synaptik.model.operation.NoOperationAttrs#INSTANCE
 * NoOperationAttrs.INSTANCE}. Broadcast geometry is derived from operand shapes and is not stored
 * as an attribute or as mutable state on the kind.</p>
 *
 * <p>The arithmetic kinds use their named ADD, SUB, MUL, DIV, or POW site after exact broadcasting
 * and promotion. FLOAT32, BFLOAT16, and FLOAT16 may treat a subnormal input to a named floating
 * arithmetic primitive as same-signed zero (DAZ) and flush a finite subnormal primitive result to
 * zero (FTZ), subject to the kind's signed-zero rule; FLOAT64 may do neither. Homogeneous low
 * arithmetic uses FLOAT32 working values and one final ties-to-even low narrowing. POW retains its
 * mathematical function, domain, and special-value rules; finite accuracy is qualified per route in
 * tests, not by a public distance bound. MIN and MAX instead compare stored represented values,
 * including subnormals, and preserve their NaN, signed-zero, and candidate-selection rules without
 * DAZ or FTZ. No arithmetic freedom crosses an observable intermediate. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 *
 * <p>Enum identity supplies typed equality and hashing, so an equally named constant in another
 * operation family remains a different semantic value. The inherited {@link #name()} and
 * {@link #toString()} text is stable diagnostic vocabulary only; it is not a serialization token,
 * registry key, or string-dispatch contract.</p>
 */
public enum BinaryArithmeticKind implements OperationKind {
    /**
     * Adds the left element value to the corresponding right element value.
     *
     * <p>The semantic request is ordered floating addition at the declared working type, including
     * its NaN, infinity, signed-zero, overflow, and underflow classifications. In particular,
     * {@code -0 + +0} produces {@code +0}. It promises no NaN payload, exact instruction, or
     * bitwise reproducibility beyond these requirements. Operand eligibility, broadcasting,
     * result-data-type derivation, gradients, execution, and
     * backend availability belong to their owning contracts.</p>
     */
    ADD,

    /**
     * Subtracts the right element value from the corresponding left element value.
     *
     * <p>The left-minus-right order and the named subtraction site's working-type and special-value
     * rules are semantic. The request promises no NaN payload, exact instruction, or bitwise
     * reproducibility beyond its special-value rules. Operand eligibility, broadcasting, result-data-type
     * derivation, gradients, execution, and backend availability belong to their owners.</p>
     */
    SUB,

    /**
     * Multiplies the left element value by the corresponding right element value.
     *
     * <p>The semantic request is ordered multiplication at the declared working type, including
     * the low-type arithmetic-site DAZ/FTZ permission above. It promises no NaN payload, exact
     * instruction, or bitwise reproducibility beyond its special-value rules. Operand eligibility,
     * broadcasting, result-data-type derivation, gradients,
     * execution, and backend availability belong to their owning contracts.</p>
     */
    MUL,

    /**
     * Divides the left element value by the corresponding right element value.
     *
     * <p>The left-divided-by-right order and the named division site's working-type and special-value
     * rules are semantic. The request promises no NaN payload, exact instruction, or bitwise
     * reproducibility beyond its special-value rules. Operand eligibility, broadcasting, result-data-type
     * derivation, gradients, execution, and backend availability belong to their owners.</p>
     */
    DIV,

    /**
     * Selects the mathematical minimum of the corresponding left and right element values.
     *
     * <p>FLOAT64 and FLOAT32 compare the represented IEEE-754 value; FLOAT16 and BFLOAT16 compare
     * the values represented by their exact 16-bit storage after the existing promotion. If either
     * candidate is NaN, the result is NaN. Opposite signed zeros produce negative zero,
     * independent of operand order. Infinities and unequal non-NaN values use ordinary numeric
     * order; equal nonzero candidates produce that numeric value. The request promises no NaN
     * payload, source-operand selection, or bitwise result. Integral minimum retains ordinary
     * signed order after promotion. Operand eligibility, broadcasting, result-data-type
     * derivation, gradients, execution, and backend availability belong to their owning
     * contracts.</p>
     */
    MIN,

    /**
     * Selects the mathematical maximum of the corresponding left and right element values.
     *
     * <p>FLOAT64 and FLOAT32 compare the represented IEEE-754 value; FLOAT16 and BFLOAT16 compare
     * the values represented by their exact 16-bit storage after the existing promotion. If either
     * candidate is NaN, the result is NaN. Opposite signed zeros produce positive zero,
     * independent of operand order. Infinities and unequal non-NaN values use ordinary numeric
     * order; equal nonzero candidates produce that numeric value. The request promises no NaN
     * payload, source-operand selection, or bitwise result. Integral maximum retains ordinary
     * signed order after promotion. Operand eligibility, broadcasting, result-data-type
     * derivation, gradients, execution, and backend availability belong to their owning
     * contracts.</p>
     */
    MAX,

    /**
     * Raises the left element value, as the base, to the corresponding right element value, as the
     * exponent.
     *
     * <p>The left-base and right-exponent roles are semantic. Operand eligibility, broadcasting,
     * result data type, numeric edge behavior, differentiation, execution, and backend availability
     * are defined by later owning contracts.</p>
     */
    POW;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 1));

    /**
     * Returns the parameterless two-input, one-output structural variant shared by this family.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
