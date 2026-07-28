package io.github.pho001.synaptik.model.operation.elementwise.comparison;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent, parameterless, ordered binary comparison semantics.
 *
 * <p>Each kind describes the mathematical relation between a left element value and the
 * corresponding right element value. The kind stores neither operand, broadcast geometry, nor a
 * result descriptor. It does not identify a graph occurrence, create Tensor provenance, infer a
 * {@code BOOL} result descriptor, execute a comparison, or report backend support. Public
 * expression, compiler, and execution layers own those responsibilities.</p>
 *
 * <p>All kinds in this family have no intrinsic parameters. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} therefore represents one of them
 * explicitly with {@link io.github.pho001.synaptik.model.operation.NoOperationAttrs#INSTANCE
 * NoOperationAttrs.INSTANCE}. Operation construction enforces that exact pairing, and the shared
 * signature declares two inputs and one output.</p>
 *
 * <p>For floating operands, every relation compares the represented numeric values after the
 * existing same-category promotion. FLOAT64 and FLOAT32 use their represented IEEE-754 binary64
 * and binary32 values; BFLOAT16 uses the value represented by its exact 16-bit storage. Ordered
 * relations are false when either operand is NaN. Negative and positive zero compare equal, so
 * neither strict relation holds and both inclusive relations hold in either operand order.
 * Equality is exact numeric equality rather than bit or tolerance equality: NaN is unequal to
 * every value, including itself, and opposite signed zeros are equal. Inequality is its logical
 * complement. Integral operands retain exact signed comparison after promotion.</p>
 *
 * <p>Enum identity supplies typed equality and hashing, so an equally named constant in another
 * operation family remains a different semantic value. The inherited {@link #name()} and
 * {@link #toString()} text is stable diagnostic vocabulary only; it is not a serialization token,
 * registry key, reflection identifier, backend-dispatch key, or kernel name.</p>
 */
public enum BinaryComparisonKind implements OperationKind {
    /**
     * Tests whether the left element value is strictly greater than the right element value.
     *
     * <p>For floating values, the result is false if either operand is NaN and false for either
     * ordering of negative and positive zero. Finite values and infinities otherwise use ordinary
     * numeric order. Operand eligibility, promotion, broadcasting, {@code BOOL} result
     * representation, differentiation, execution, and backend availability belong to their
     * owning contracts.</p>
     */
    GREATER_THAN,

    /**
     * Tests whether the left element value is greater than or equal to the right element value.
     *
     * <p>For floating values, the result is false if either operand is NaN and true for either
     * ordering of negative and positive zero. Finite values and infinities otherwise use ordinary
     * inclusive numeric order. Operand eligibility, promotion, broadcasting, {@code BOOL} result
     * representation, differentiation, execution, and backend availability belong to their
     * owning contracts.</p>
     */
    GREATER_OR_EQUAL,

    /**
     * Tests whether the left element value is strictly less than the right element value.
     *
     * <p>For floating values, the result is false if either operand is NaN and false for either
     * ordering of negative and positive zero. Finite values and infinities otherwise use ordinary
     * numeric order. Operand eligibility, promotion, broadcasting, {@code BOOL} result
     * representation, differentiation, execution, and backend availability belong to their
     * owning contracts.</p>
     */
    LESS_THAN,

    /**
     * Tests whether the left element value is less than or equal to the right element value.
     *
     * <p>For floating values, the result is false if either operand is NaN and true for either
     * ordering of negative and positive zero. Finite values and infinities otherwise use ordinary
     * inclusive numeric order. Operand eligibility, promotion, broadcasting, {@code BOOL} result
     * representation, differentiation, execution, and backend availability belong to their
     * owning contracts.</p>
     */
    LESS_OR_EQUAL,

    /**
     * Tests whether the left and right element values compare equal.
     *
     * <p>Floating equality is exact represented numeric equality, not raw-bit or tolerance
     * equality. NaN is unequal to every value, including itself, while negative zero equals
     * positive zero. Operand eligibility, promotion, broadcasting, {@code BOOL} result
     * representation, differentiation, execution, and backend availability belong to their
     * owning contracts.</p>
     */
    EQUAL,

    /**
     * Tests whether the left and right element values compare unequal.
     *
     * <p>Floating inequality is the exact logical complement of represented numeric equality, not
     * raw-bit or tolerance inequality. It is true if either operand is NaN and false for either
     * ordering of negative and positive zero. Operand eligibility, promotion, broadcasting,
     * {@code BOOL} result representation, differentiation, execution, and backend availability
     * belong to their owning contracts.</p>
     */
    NOT_EQUAL;

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
