package io.github.pho001.synaptik.model.operation.elementwise.logical;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent, parameterless elementwise boolean logical semantics.
 *
 * <p>{@link #AND} and {@link #OR} describe conjunction and disjunction over two logical inputs,
 * while {@link #NOT} describes negation over one logical input. These input roles are family
 * context rather than stored arity metadata. A kind stores no inputs, validates no input count,
 * identifies no graph occurrence, and creates no Tensor provenance or result descriptor.</p>
 *
 * <p>All kinds in this family have no intrinsic parameters. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} therefore represents one of them
 * explicitly with {@link io.github.pho001.synaptik.model.operation.NoOperationAttrs#INSTANCE
 * NoOperationAttrs.INSTANCE}. The generic operation descriptor validates component presence but
 * does not enforce family-specific arity or kind-to-attributes compatibility.</p>
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
     * <p>The result is true exactly when both input values are true. Two-input context is
     * documented rather than stored or validated as arity metadata. Descriptor eligibility,
     * broadcasting, {@code BOOL} representation, gradients, execution, and backend availability
     * belong to later owning contracts.</p>
     */
    AND,

    /**
     * Produces the disjunction of corresponding left and right logical input values.
     *
     * <p>The result is true exactly when at least one input value is true. Two-input context is
     * documented rather than stored or validated as arity metadata. Descriptor eligibility,
     * broadcasting, {@code BOOL} representation, gradients, execution, and backend availability
     * belong to later owning contracts.</p>
     */
    OR,

    /**
     * Produces the negation of each logical input value.
     *
     * <p>The result is true exactly when the one input value is false. One-input context is
     * documented rather than stored or validated as arity metadata. Descriptor eligibility,
     * shape preservation, {@code BOOL} representation, gradients, execution, and backend
     * availability belong to later owning contracts.</p>
     */
    NOT
}
