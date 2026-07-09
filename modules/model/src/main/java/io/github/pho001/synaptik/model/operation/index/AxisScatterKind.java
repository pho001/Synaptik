package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies the backend-independent meaning for functionally scattering tensor updates along
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
 * <p>The shape relationship is:</p>
 *
 * <ul>
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
 * execute a result. {@code SCATTER_ELEMENTS} pairs with {@link ScatterElementsAttrs}, which
 * carries its selected reduction. Family-owned
 * signatures enforce those exact pairings and declare the ordered three-input, one-output
 * occurrence.</p>
 *
 * <p>Axis scatter differs from {@link AxisGatherKind axis gather} and
 * {@link GatherNdKind#GATHER_ND Gather-ND}, which read selected data; scatter-ND, which uses
 * multi-axis coordinate tuples; fold, which reconstructs overlapping windows; and in-place
 * mutation. The public Tensor-expression contract owns caller-axis normalization and input-aware
 * index-type, data-type, and shape validation. This enum stores no
 * operands, shapes, result metadata, provenance, gradient policy, graph or compiler behavior,
 * backend support, numerical policy, or execution state.</p>
 */
public enum AxisScatterKind implements OperationKind {
    /**
     * Writes or reduces same-rank updates at axis coordinates supplied by matching indices.
     *
     * <p>The ordered logical inputs are {@code [data, indices, updates]}; indices and updates have
     * equal shapes, match data away from the selected axis, and reduce into the exact data shape.
     * {@link ScatterElementsAttrs} supplies the selected reduction.</p>
     */
    SCATTER_ELEMENTS;

    private static final List<OperationSignature> ELEMENTS_SIGNATURES =
            List.of(OperationSignature.fixed(ScatterElementsAttrs.class, 3, 1));

    /**
     * Returns the exact three-input, one-output attributes variant accepted by this scatter kind.
     *
     * @return the stable scatter-elements signature
     */
    @Override
    public List<OperationSignature> signatures() {
        return ELEMENTS_SIGNATURES;
    }
}
