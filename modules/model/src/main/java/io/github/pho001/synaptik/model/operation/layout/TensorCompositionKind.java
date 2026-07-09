package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent meanings for joining tensors.
 *
 * <p>{@link #CONCAT} and {@link #STACK} consume ordered, non-empty logical input sequences. Input
 * order is semantic: concatenating conceptual Shapes {@code [2, 3]} and {@code [2, 5]} on
 * existing axis one places the first input's three positions before the second input's five
 * positions and conceptually produces Shape {@code [2, 8]}. Stacking two conceptual Shapes
 * {@code [2, 3]} at insertion position one instead creates a new axis and conceptually produces
 * Shape {@code [2, 2, 3]}. Both kinds pair with {@link CompositionAxisAttrs}; their kind
 * distinguishes an existing input axis from a newly inserted result-axis position.</p>
 *
 * <p>STACK remains a first-class semantic kind even though a compiler may later choose an
 * expand-dimensions-plus-concat decomposition. Family-owned signatures enforce the exact
 * attributes pairings and declare variadic non-empty input for CONCAT and STACK. They do not
 * validate rank or result-shape compatibility.</p>
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
    STACK;

    private static final List<OperationSignature> VARIADIC_SIGNATURES = List.of(
            OperationSignature.inputRange(
                    CompositionAxisAttrs.class, 1, Integer.MAX_VALUE, 1));

    /**
     * Returns the variadic composition signature.
     *
     * @return the stable non-empty variadic-input signature
     */
    @Override
    public List<OperationSignature> signatures() {
        return VARIADIC_SIGNATURES;
    }
}
