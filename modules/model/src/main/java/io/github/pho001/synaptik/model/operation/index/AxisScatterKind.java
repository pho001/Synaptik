package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies three backend-independent meanings for functionally scattering tensor updates along
 * one data axis.
 *
 * <p>A functional scatter has ordered logical inputs {@code [data, indices, updates]}.
 * {@code data} is the base value and supplies the exact result shape, {@code indices} selects a
 * target coordinate on the normalized axis, and {@code updates} supplies values to write or
 * combine at those targets. The operation conceptually starts with {@code data} and produces a
 * new value with exactly the data shape; it never mutates {@code data} in place. A duplicate
 * target occurs when multiple index entries address the same result coordinate. A reduction
 * defines how the base and all updates for one target are combined.</p>
 *
 * <p>The shape relationships distinguish the three meanings:</p>
 *
 * <ul>
 *   <li>{@link #SCATTER_ADD} requires
 *       {@code indices.shape == updates.shape == remove(data.shape, axis)} and uses fixed
 *       addition. For data shape {@code [2, 3, 4]}, axis {@code 1}, and indices and updates shapes
 *       {@code [2, 4]}, the conceptual result shape is {@code [2, 3, 4]}. At reduced coordinate
 *       {@code [0, 2]}, index {@code 1} adds update {@code [0, 2]} to data coordinate
 *       {@code [0, 1, 2]}.</li>
 *   <li>{@link #SCATTER_AXIS_ADD} requires the updates shape to equal the shape produced by
 *       replacing the selected data axis with the complete indices shape, and uses fixed
 *       addition. For data shape {@code [2, 3, 4]}, axis {@code 1}, indices shape {@code [5, 6]},
 *       and updates shape {@code [2, 5, 6, 4]}, the conceptual result shape is
 *       {@code [2, 3, 4]}. Update {@code [0, i, j, 2]} targets data coordinate
 *       {@code [0, indices[i, j], 2]}.</li>
 *   <li>{@link #SCATTER_ELEMENTS} requires indices and updates to have equal rank and shape and to
 *       match data away from the selected axis. It uses a caller-selected
 *       {@link ScatterReduction}. For data shape {@code [2, 3, 4]}, axis {@code 1}, and equal
 *       indices and updates shapes {@code [2, 5, 4]}, the conceptual result shape is
 *       {@code [2, 3, 4]}. At update coordinate {@code [0, 4, 2]}, the corresponding index selects
 *       the middle coordinate of target {@code [0, index, 2]}.</li>
 * </ul>
 *
 * <p>These relationships explain semantics only. This enum does not inspect operands, validate
 * ranks, shapes, data types, index values, bounds, or duplicate targets, and does not construct or
 * execute a result. {@code SCATTER_ADD} and {@code SCATTER_AXIS_ADD} pair with
 * {@link IndexAxisAttrs}; their addition is intrinsic to their kinds. {@code SCATTER_ELEMENTS}
 * pairs with {@link ScatterElementsAttrs}, which carries its selected reduction. Generic operation
 * composition does not enforce these pairings or the ordered three-input context.</p>
 *
 * <p>Axis scatter differs from {@link AxisGatherKind axis gather} and
 * {@link GatherNdKind#GATHER_ND Gather-ND}, which read selected data; scatter-ND, which uses
 * multi-axis coordinate tuples; fold, which reconstructs overlapping windows; and in-place
 * mutation. The later public Tensor-expression contract in task 0018H owns caller-axis
 * normalization and input-aware index-type, data-type, and shape validation. This enum stores no
 * operands, shapes, result metadata, provenance, gradient policy, graph or compiler behavior,
 * backend support, numerical policy, or execution state.</p>
 */
public enum AxisScatterKind implements OperationKind {
    /**
     * Adds reduced-rank updates to targets selected by one index at each non-axis data coordinate.
     *
     * <p>The ordered logical inputs are {@code [data, indices, updates]}; indices and updates have
     * the data shape with the selected axis removed, and the conceptual result has the exact data
     * shape. Addition is fixed by this kind and is not configurable attribute state.</p>
     */
    SCATTER_ADD,

    /**
     * Adds rank-changing updates aligned like the inverse of complete-shape axis gather.
     *
     * <p>The ordered logical inputs are {@code [data, indices, updates]}; the complete indices
     * shape replaces the selected data axis in the updates shape, and the conceptual result has
     * the exact data shape. Addition is fixed by this kind and is not configurable attribute
     * state.</p>
     */
    SCATTER_AXIS_ADD,

    /**
     * Writes or reduces same-rank updates at axis coordinates supplied by matching indices.
     *
     * <p>The ordered logical inputs are {@code [data, indices, updates]}; indices and updates have
     * equal shapes, match data away from the selected axis, and reduce into the exact data shape.
     * {@link ScatterElementsAttrs} supplies the selected reduction.</p>
     */
    SCATTER_ELEMENTS
}
