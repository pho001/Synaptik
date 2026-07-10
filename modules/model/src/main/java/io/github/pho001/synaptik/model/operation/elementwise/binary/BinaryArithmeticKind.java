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
 * <p>Enum identity supplies typed equality and hashing, so an equally named constant in another
 * operation family remains a different semantic value. The inherited {@link #name()} and
 * {@link #toString()} text is stable diagnostic vocabulary only; it is not a serialization token,
 * registry key, or string-dispatch contract.</p>
 */
public enum BinaryArithmeticKind implements OperationKind {
    /**
     * Adds the left element value to the corresponding right element value.
     *
     * <p>The semantic request is ordinary ordered IEEE-754 addition in the eventual result data
     * type, including its NaN, infinity, signed-zero, overflow, and underflow classifications.
     * It promises no NaN payload, intermediate precision, exact instruction, or bitwise result.
     * Operand eligibility, broadcasting, result-data-type derivation, gradients, execution, and
     * backend availability belong to their owning contracts.</p>
     */
    ADD,

    /**
     * Subtracts the right element value from the corresponding left element value.
     *
     * <p>The left-minus-right order and ordinary IEEE-754 subtraction in the eventual result data
     * type are semantic. The request promises no NaN payload, intermediate precision, exact
     * instruction, or bitwise result. Operand eligibility, broadcasting, result-data-type
     * derivation, gradients, execution, and backend availability belong to their owners.</p>
     */
    SUB,

    /**
     * Multiplies the left element value by the corresponding right element value.
     *
     * <p>The semantic request is ordinary ordered IEEE-754 multiplication in the eventual result
     * data type. It promises no NaN payload, intermediate precision, exact instruction, or
     * bitwise result. Operand eligibility, broadcasting, result-data-type derivation, gradients,
     * execution, and backend availability belong to their owning contracts.</p>
     */
    MUL,

    /**
     * Divides the left element value by the corresponding right element value.
     *
     * <p>The left-divided-by-right order and ordinary IEEE-754 division in the eventual result
     * data type are semantic. The request promises no NaN payload, intermediate precision, exact
     * instruction, or bitwise result. Operand eligibility, broadcasting, result-data-type
     * derivation, gradients, execution, and backend availability belong to their owners.</p>
     */
    DIV,

    /**
     * Selects the mathematical minimum of the corresponding left and right element values.
     *
     * <p>The portable request propagates NaN, orders infinities normally, and selects negative
     * zero when comparing opposite signed zeros. It promises no NaN payload or bitwise result.
     * Operand eligibility, broadcasting, result-data-type derivation, gradients, execution, and
     * backend availability belong to their owning contracts.</p>
     */
    MIN,

    /**
     * Selects the mathematical maximum of the corresponding left and right element values.
     *
     * <p>The portable request propagates NaN, orders infinities normally, and selects positive
     * zero when comparing opposite signed zeros. It promises no NaN payload or bitwise result.
     * Operand eligibility, broadcasting, result-data-type derivation, gradients, execution, and
     * backend availability belong to their owning contracts.</p>
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
