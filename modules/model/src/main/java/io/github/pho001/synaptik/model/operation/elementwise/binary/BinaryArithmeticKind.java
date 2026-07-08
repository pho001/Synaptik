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
     * <p>This kind defines operand order and mathematical addition only. Operand eligibility,
     * broadcasting, result data type, numeric edge behavior, differentiation, execution, and
     * backend availability are defined by later owning contracts.</p>
     */
    ADD,

    /**
     * Subtracts the right element value from the corresponding left element value.
     *
     * <p>The left-minus-right order is semantic. Operand eligibility, broadcasting, result data
     * type, numeric edge behavior, differentiation, execution, and backend availability are
     * defined by later owning contracts.</p>
     */
    SUB,

    /**
     * Multiplies the left element value by the corresponding right element value.
     *
     * <p>This kind defines operand order and mathematical multiplication only. Operand
     * eligibility, broadcasting, result data type, numeric edge behavior, differentiation,
     * execution, and backend availability are defined by later owning contracts.</p>
     */
    MUL,

    /**
     * Divides the left element value by the corresponding right element value.
     *
     * <p>The left-divided-by-right order is semantic. Operand eligibility, broadcasting, result
     * data type, integer and floating-point edge behavior, differentiation, execution, and backend
     * availability are defined by later owning contracts.</p>
     */
    DIV,

    /**
     * Selects the mathematical minimum of the corresponding left and right element values.
     *
     * <p>This kind does not define operand eligibility, broadcasting, result data type, tie, NaN,
     * or signed-zero behavior, differentiation, execution, or backend availability. Later owning
     * contracts define those semantics.</p>
     */
    MIN,

    /**
     * Selects the mathematical maximum of the corresponding left and right element values.
     *
     * <p>This kind does not define operand eligibility, broadcasting, result data type, tie, NaN,
     * or signed-zero behavior, differentiation, execution, or backend availability. Later owning
     * contracts define those semantics.</p>
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
