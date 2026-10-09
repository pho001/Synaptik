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
 * target occurs when multiple index entries address the same result coordinate. For
 * {@code SCATTER_ELEMENTS}, every update position contributes one scalar to its selected result
 * coordinate; duplicate targets therefore contribute distinct multiset members even when their
 * values are equal. A non-replacement reduction combines the base exactly once with every such
 * contribution exactly once, and an unaddressed coordinate retains the exact base
 * representation.</p>
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
 *   <li>{@link #SCATTER_ADD} requires updates to have the Shape that {@link AxisGatherKind#GATHER}
 *       would produce from data and indices. It adds every update to its addressed base value,
 *       including accumulation of duplicate indices.</li>
 * </ul>
 *
 * <p>These relationships explain semantics only. This enum does not inspect operands, validate
 * ranks, shapes, data types, index values, bounds, or duplicate targets, and does not construct or
 * execute a result. {@code SCATTER_ELEMENTS} pairs with {@link ScatterElementsAttrs}, which
 * carries its selected reduction, while {@code SCATTER_ADD} pairs with {@link IndexAxisAttrs} and
 * has intrinsic addition. Family-owned signatures enforce those exact pairings and declare the
 * ordered three-input, one-output occurrence.</p>
 *
 * <p>Axis scatter differs from {@link AxisGatherKind axis gather} and
 * {@link GatherNdKind#GATHER_ND Gather-ND}, which read selected data; scatter-ND, which uses
 * multi-axis coordinate tuples; fold, which reconstructs overlapping windows; and in-place
 * mutation. The public Tensor-expression contract owns caller-axis normalization and input-aware
 * index-type, data-type, and shape validation. This enum stores no
 * operands, shapes, result metadata, provenance, gradient or subgradient policy, graph or compiler
 * behavior, backend support, numerical algorithm, or execution state. The portable represented-
 * value rules for configurable reductions are defined by {@link ScatterReduction}; they are
 * independent of encounter, layout, atomic, tree, and backend order.</p>
 *
 * <p>Axis indices, bounds, target mapping, duplicate membership, and raw unaddressed cells remain
 * exact. NONE replacement preserves the original update representation; MIN and MAX select original
 * candidates using stored represented values and their NaN, signed-zero, and tie rules. Floating
 * ADD and MUL combine the base and every addressed update exactly once in a rounded tree; named
 * arithmetic primitives may use dtype-specific DAZ/FTZ for FLOAT32, BFLOAT16, and FLOAT16, but
 * never FLOAT64. A rounded MUL tree may return NaN or positive zero for maxFinite * 2 * +0 through
 * intermediate overflow or a different grouping. Integral reductions use exact fixed-width modular
 * arithmetic. Homogeneous low arithmetic accumulates in FLOAT32 and narrows once at the declared
 * output. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 */
public enum AxisScatterKind implements OperationKind {
    /**
     * Writes or reduces same-rank updates at axis coordinates supplied by matching indices.
     *
     * <p>The ordered logical inputs are {@code [data, indices, updates]}; indices and updates have
     * equal shapes, match data away from the selected axis, and reduce into the exact data shape.
     * Each updates coordinate contributes exactly once to the result coordinate obtained by
     * replacing its selected-axis coordinate with the corresponding index value.
     * {@link ScatterElementsAttrs} supplies the selected reduction.</p>
     */
    SCATTER_ELEMENTS,

    /**
     * Adds Gather-compatible updates into the exact data Shape.
     *
     * <p>The ordered logical inputs are {@code [data, indices, updates]}. The complete indices
     * Shape replaces the selected data axis in the required updates Shape. Each update is added
     * to the base value addressed by its index, and duplicate targets accumulate.</p>
     */
    SCATTER_ADD;

    private static final List<OperationSignature> ELEMENTS_SIGNATURES =
            List.of(OperationSignature.fixed(ScatterElementsAttrs.class, 3, 1));
    private static final List<OperationSignature> ADD_SIGNATURES =
            List.of(OperationSignature.fixed(IndexAxisAttrs.class, 3, 1));

    /**
     * Returns the exact three-input, one-output attributes variant accepted by this scatter kind.
     *
     * @return the stable signature for this exact scatter kind
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case SCATTER_ELEMENTS -> ELEMENTS_SIGNATURES;
            case SCATTER_ADD -> ADD_SIGNATURES;
        };
    }
}
