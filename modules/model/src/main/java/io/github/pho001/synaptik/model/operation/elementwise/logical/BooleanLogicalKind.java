package io.github.pho001.synaptik.model.operation.elementwise.logical;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent, parameterless elementwise boolean logical semantics.
 *
 * <p>{@link #AND} and {@link #OR} describe conjunction and disjunction over two logical inputs,
 * while {@link #NOT} describes negation over one logical input. The family-owned signatures
 * declare those cardinalities without storing inputs, identifying a graph occurrence, or creating
 * Tensor provenance or a result descriptor.</p>
 *
 * <p>All kinds in this family have no intrinsic parameters. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} therefore represents one of them
 * explicitly with {@link io.github.pho001.synaptik.model.operation.NoOperationAttrs#INSTANCE
 * NoOperationAttrs.INSTANCE}. Operation construction enforces the exact attributes pairing, and
 * a compiled-node occurrence enforces the selected signature's input and output counts.</p>
 *
 * <p>This vocabulary defines boolean truth meaning only. It does not define descriptor
 * eligibility, broadcasting or shape preservation, {@code BOOL} storage representation, numeric
 * truthiness, gradients, execution, or backend availability. Enum identity supplies typed
 * equality and hashing, so an equally named constant in another operation family remains a
 * different semantic value. The inherited {@link #name()} and {@link #toString()} text is stable
 * diagnostic vocabulary only; it is not a serialization, parsing, registry, or dispatch
 * contract.</p>
 */
public enum BooleanLogicalKind implements OperationKind {
    /**
     * Produces the conjunction of corresponding left and right logical input values.
     *
     * <p>The result is true exactly when both input values are true. Its signature declares two
     * inputs and one output. Descriptor eligibility,
     * broadcasting, {@code BOOL} representation, gradients, execution, and backend availability
     * belong to later owning contracts.</p>
     */
    AND,

    /**
     * Produces the disjunction of corresponding left and right logical input values.
     *
     * <p>The result is true exactly when at least one input value is true. Its signature declares
     * two inputs and one output. Descriptor eligibility,
     * broadcasting, {@code BOOL} representation, gradients, execution, and backend availability
     * belong to later owning contracts.</p>
     */
    OR,

    /**
     * Produces the negation of each logical input value.
     *
     * <p>The result is true exactly when the one input value is false. Its signature declares one
     * input and one output. Descriptor eligibility,
     * shape preservation, {@code BOOL} representation, gradients, execution, and backend
     * availability belong to later owning contracts.</p>
     */
    NOT;

    private static final List<OperationSignature> BINARY_SIGNATURES =
            List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 1));
    private static final List<OperationSignature> UNARY_SIGNATURES =
            List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

    /**
     * Returns the parameterless signature matching this kind's documented logical arity.
     *
     * @return the stable immutable binary signature for {@link #AND} and {@link #OR}, or the
     *     unary signature for {@link #NOT}
     */
    @Override
    public List<OperationSignature> signatures() {
        return this == NOT ? UNARY_SIGNATURES : BINARY_SIGNATURES;
    }
}
