package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent meanings for joining tensors or selecting one unstack output.
 *
 * <p>{@link #CONCAT} and {@link #STACK} consume ordered, non-empty logical input sequences. Input
 * order is semantic: concatenating conceptual Shapes {@code [2, 3]} and {@code [2, 5]} on
 * existing axis one places the first input's three positions before the second input's five
 * positions and conceptually produces Shape {@code [2, 8]}. Stacking two conceptual Shapes
 * {@code [2, 3]} at insertion position one instead creates a new axis and conceptually produces
 * Shape {@code [2, 2, 3]}. Both kinds pair with {@link CompositionAxisAttrs}; their kind
 * distinguishes an existing input axis from a newly inserted result-axis position.</p>
 *
 * <p>{@link #UNSTACK} represents one individually indexed result of a public logical multi-result
 * request and pairs with {@link UnstackOutputAttrs}. For conceptual Shape {@code [2, 3, 4]},
 * unstacking source axis one yields output indices zero through two, each with conceptual Shape
 * {@code [2, 4]}. The output index lets each public result tensor carry distinguishable semantics
 * in the current one-provenance-value-per-tensor model; it is not a graph output slot or producer
 * grouping identity.</p>
 *
 * <p>STACK remains a first-class semantic kind even though a compiler may later choose an
 * expand-dimensions-plus-concat decomposition. The generic {@code Operation} descriptor accepts
 * any non-null kind and attributes value and therefore does not enforce these family pairings,
 * input count, rank, or result count.</p>
 *
 * <p>This enum stores no Tensor, input list, Shape, descriptor, layout, provenance, graph state,
 * grouping identity, gradient, compiler policy, backend or ONNX behavior, or execution state.
 * Its inherited enum name is diagnostic text rather than a serialization, parsing, dispatch, or
 * kernel identifier.</p>
 */
public enum TensorCompositionKind implements OperationKind {
    /**
     * Joins ordered, non-empty inputs along the existing normalized axis in
     * {@link CompositionAxisAttrs} while preserving rank.
     *
     * <p>The kind states logical ordering only. It does not retain the inputs, validate their
     * count or compatibility, derive a result Shape, or execute concatenation.</p>
     */
    CONCAT,

    /**
     * Joins ordered, non-empty same-shaped inputs along the newly inserted normalized result-axis
     * position in {@link CompositionAxisAttrs}, increasing rank by one.
     *
     * <p>This first-class meaning does not prescribe compiler decomposition, retain inputs,
     * validate their Shapes, derive a result, or execute stacking.</p>
     */
    STACK,

    /**
     * Selects one result by fixing the normalized source axis and logical coordinate described by
     * {@link UnstackOutputAttrs}, then removing that axis conceptually.
     *
     * <p>The indexed result is distinguishable in current public Tensor provenance without
     * defining producer grouping or graph output-slot state. The kind performs no bound check,
     * result construction, grouping, graph capture, or execution.</p>
     */
    UNSTACK
}
