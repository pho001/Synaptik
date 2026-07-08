package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

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
 * expand-dimensions-plus-concat decomposition. Family-owned signatures enforce the exact
 * attributes pairings and declare variadic non-empty input for CONCAT and STACK and one input for
 * UNSTACK. They do not validate rank or result-shape compatibility.</p>
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
    UNSTACK;

    private static final List<OperationSignature> VARIADIC_SIGNATURES = List.of(
            OperationSignature.inputRange(
                    CompositionAxisAttrs.class, 1, Integer.MAX_VALUE, 1));
    private static final List<OperationSignature> UNSTACK_SIGNATURES =
            List.of(OperationSignature.fixed(UnstackOutputAttrs.class, 1, 1));

    /**
     * Returns the variadic composition or independently indexed unstack-output signature.
     *
     * @return the stable unstack signature for {@link #UNSTACK}, otherwise the stable non-empty
     *     variadic-input signature
     */
    @Override
    public List<OperationSignature> signatures() {
        return this == UNSTACK ? UNSTACK_SIGNATURES : VARIADIC_SIGNATURES;
    }
}
