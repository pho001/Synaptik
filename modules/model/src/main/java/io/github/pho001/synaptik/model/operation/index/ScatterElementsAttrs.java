package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries the normalized data axis and explicit reduction for
 * {@link AxisScatterKind#SCATTER_ELEMENTS}.
 *
 * <p>The {@link #axis()} value is already normalized, zero-based, and non-negative. The
 * {@link #reduction()} value selects replacement, addition, multiplication, maximum, or minimum
 * semantics for updates that address a result target. The attributes are used with ordered
 * logical inputs {@code [data, indices, updates]}; they store none of those operands. The
 * conceptual functional result starts from {@code data}, has exactly the data shape, and leaves
 * the data input unchanged. Every updates coordinate contributes one scalar to the result
 * coordinate obtained by replacing that updates coordinate's selected-axis position with the
 * corresponding index value. Duplicate targets remain distinct contributions. A
 * non-replacement reduction combines the base exactly once with every addressed contribution
 * exactly once; an unaddressed coordinate retains the exact base representation. The portable
 * represented-value rules are defined by {@link ScatterReduction}.</p>
 *
 * <p>This value stores no input rank or shape, so it cannot prove that the axis exists or validate
 * that indices and updates have equal shapes and match data away from the axis. Zero, positive
 * values, and {@link Integer#MAX_VALUE} are structurally valid axes. The current public
 * scatter-elements expressions own caller-axis normalization, exact integral index-type checks,
 * exact data/update-type matching, reduction eligibility, and input-aware rank and shape checks.
 * Index bounds and duplicate-target detection require values and do not occur during this
 * metadata construction. In particular,
 * {@link ScatterReduction#NONE} is structurally valid here even though duplicate targets make a
 * concrete replacement operation invalid.</p>
 *
 * <p>The immutable record retains both components unchanged. Record-generated equality and
 * hashing use both components, and generated text is diagnostic rather than a serialization,
 * parsing, compiler-dispatch, backend, or execution contract. These attributes define no Tensor
 * construction, result descriptor, provenance, derivative or subgradient policy, numerical
 * algorithm, materialization, graph or compiler behavior, backend support, or execution.</p>
 *
 * @param axis the already normalized, zero-based, non-negative data-axis index
 * @param reduction the non-null replacement or reduction meaning applied at addressed targets
 */
public record ScatterElementsAttrs(int axis, ScatterReduction reduction)
        implements OperationAttrs {
    /**
     * Creates immutable normalized axis and reduction parameters for scatter-elements.
     *
     * <p>The axis is validated before the reduction reference. Both valid values are then retained
     * unchanged. Construction does not normalize a raw caller axis, inspect operands, validate
     * ranks, shapes, data types, bounds, or duplicate targets, or construct a result.</p>
     *
     * @param axis the already normalized data-axis index; must be non-negative
     * @param reduction the replacement or reduction meaning; must not be {@code null}
     * @throws IllegalArgumentException if {@code axis} is negative, with message
     *     {@code axis must be non-negative: <axis>}
     * @throws NullPointerException if {@code reduction} is {@code null}, with message
     *     {@code reduction}
     */
    public ScatterElementsAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        Objects.requireNonNull(reduction, "reduction");
    }

    /**
     * Returns the already normalized, zero-based data-axis index.
     *
     * @return the exact non-negative axis supplied at construction; it has not been validated
     *     against an input rank by this attributes value
     */
    @Override
    public int axis() {
        return axis;
    }

    /**
     * Returns the selected replacement or reduction meaning.
     *
     * @return the exact non-null {@link ScatterReduction} supplied at construction
     */
    @Override
    public ScatterReduction reduction() {
        return reduction;
    }
}
