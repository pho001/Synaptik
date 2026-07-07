package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies the three backend-independent meanings for indexing tensor data along one axis with
 * a tensor of indices.
 *
 * <p>Every kind has the ordered logical inputs {@code [data, indices]}: {@code data} supplies the
 * values being indexed, and {@code indices} supplies logical coordinates on the normalized axis
 * carried by {@link IndexAxisAttrs}. The shape relationships below explain each meaning; neither
 * this enum nor the attributes inspect inputs, validate shapes, or construct a result.</p>
 *
 * <ul>
 *   <li>{@link #GATHER} requires the indices shape to equal the data shape with the selected axis
 *       removed, and its conceptual result has that same reduced shape. For data shape
 *       {@code [2, 3, 4]}, axis {@code 1}, and indices shape {@code [2, 4]}, the conceptual result
 *       shape is {@code [2, 4]}.</li>
 *   <li>{@link #GATHER_AXIS} replaces the selected data axis with the complete indices shape. For
 *       data shape {@code [2, 3, 4]}, axis {@code 1}, and indices shape {@code [5, 6]}, the
 *       conceptual result shape is {@code [2, 5, 6, 4]}. A future public {@code take} method is an
 *       alias for this exact meaning, not another enum constant.</li>
 *   <li>{@link #TAKE_ALONG_AXIS} aligns same-rank indices with data coordinates away from the
 *       selected axis and has the exact indices shape as its conceptual result. For data shape
 *       {@code [2, 3, 4]}, axis {@code 1}, and indices shape {@code [2, 7, 4]}, the conceptual
 *       result shape is {@code [2, 7, 4]}, subject to later non-axis compatibility checks.</li>
 * </ul>
 *
 * <p>All three kinds pair explicitly with {@link IndexAxisAttrs}; the generic operation
 * descriptor does not enforce those pairings or the ordered two-input context. Axis gather is
 * distinct from scalar {@link SelectKind#SELECT}, whose coordinate is an intrinsic scalar and
 * removes an axis, and from gather-ND, which uses multi-axis index tuples. It is also distinct
 * from functional scatter, which writes or combines updates rather than reading indexed data.</p>
 *
 * <p>This enum stores no input, shape, data type, bounds, result descriptor, provenance, gradient,
 * graph or compiler policy, backend support, or execution state. Task 0018D's later public
 * Tensor-expression contract validates that indices use {@code INT32} or {@code INT64},
 * normalizes a caller-facing axis, checks input-dependent shape and bounds rules, and constructs
 * result metadata. The inherited enum name is diagnostic text rather than a serialization,
 * dispatch, registry, route, or kernel identifier.</p>
 */
public enum AxisGatherKind implements OperationKind {
    /**
     * Uses one index for every coordinate of the data shape with the selected axis removed and
     * produces that same reduced conceptual shape.
     *
     * <p>The ordered logical inputs are {@code [data, indices]}. For data {@code [2, 3, 4]}, axis
     * {@code 1}, and indices {@code [2, 4]}, the conceptual result is {@code [2, 4]}. This kind
     * does not validate that relationship or read indexed values.</p>
     */
    GATHER,

    /**
     * Replaces the selected data axis with the complete indices shape in ONNX-style axis-gather
     * semantics.
     *
     * <p>The ordered logical inputs are {@code [data, indices]}. For data {@code [2, 3, 4]}, axis
     * {@code 1}, and indices {@code [5, 6]}, the conceptual result is {@code [2, 5, 6, 4]}. A
     * future public {@code take} method names this same operation; there is deliberately no
     * separate {@code TAKE} kind. This enum performs no shape validation or result construction.</p>
     */
    GATHER_AXIS,

    /**
     * Reads indices aligned with same-rank data coordinates away from the selected axis and
     * produces the exact indices shape conceptually.
     *
     * <p>The ordered logical inputs are {@code [data, indices]}. For data {@code [2, 3, 4]}, axis
     * {@code 1}, and indices {@code [2, 7, 4]}, the conceptual result is {@code [2, 7, 4]}, subject
     * to later non-axis compatibility checks. This kind performs neither those checks nor value
     * selection.</p>
     */
    TAKE_ALONG_AXIS
}
