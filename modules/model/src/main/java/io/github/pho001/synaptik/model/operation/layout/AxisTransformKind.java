package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies backend-independent, one-input transformations expressed through normalized axis
 * coordinates.
 *
 * <p>Each kind has one exact attributes pairing:</p>
 *
 * <pre>{@code
 * new Operation(AxisTransformKind.PERMUTE, permutationAttrs)
 * new Operation(AxisTransformKind.EXPAND_DIMS, axisTransformAttrs)
 * new Operation(AxisTransformKind.SQUEEZE, axisTransformAttrs)
 * }</pre>
 *
 * <p>{@link #PERMUTE} uses {@link PermutationAttrs}, while {@link #EXPAND_DIMS} and
 * {@link #SQUEEZE} use {@link AxisTransformAttrs}. The generic {@code Operation} descriptor
 * accepts any non-null kind and attributes value, so it does not enforce these family-specific
 * pairings or one-input context.</p>
 *
 * <p>A future rank-two {@code transpose()} convenience is represented by {@code PERMUTE} with
 * {@code PermutationAttrs(List.of(1, 0))}; transpose is not a separate semantic kind. This enum
 * stores no input, rank, Shape, layout, result descriptor, storage view, provenance, gradient,
 * compiler state, backend support, or execution behavior. Inherited enum names are diagnostic
 * text rather than serialization, parsing, registry, dispatch, reflection, or kernel identifiers.
 * </p>
 */
public enum AxisTransformKind implements OperationKind {
    /**
     * Reorders axes according to the complete output-to-input mapping in
     * {@link PermutationAttrs} while preserving each logical value's coordinate association.
     *
     * <p>Output axis {@code i} corresponds to input axis {@code axes[i]}. The kind does not
     * inspect an input rank or derive a result Shape or layout.</p>
     */
    PERMUTE,

    /**
     * Inserts one extent-one axis at the normalized output position in
     * {@link AxisTransformAttrs} while preserving logical values.
     *
     * <p>The kind does not validate the insertion position against an input rank or construct the
     * resulting Shape or layout.</p>
     */
    EXPAND_DIMS,

    /**
     * Removes the selected extent-one input axis identified by the normalized position in
     * {@link AxisTransformAttrs} while preserving logical values.
     *
     * <p>The kind does not validate the position against an input rank or prove that the selected
     * input dimension has extent one.</p>
     */
    SQUEEZE
}
