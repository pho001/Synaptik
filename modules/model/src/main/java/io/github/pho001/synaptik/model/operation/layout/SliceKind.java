package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent signed slice extraction and functional replacement meanings.
 *
 * <p>The family composes explicitly with normalized {@link SliceAttrs}:</p>
 *
 * <pre>{@code
 * SliceAttrs attrs = new SliceAttrs(starts, lengths, axes, steps);
 * Operation operation = new Operation(SliceKind.SLICE, attrs);
 * }</pre>
 *
 * <p>For entry {@code i}, the coordinate sequence begins at
 * {@code starts[i]}, advancing by signed non-zero {@code steps[i]}, for exactly
 * {@code lengths[i]} coordinates along normalized input axis {@code axes[i]}. Extraction selects
 * those positions from one input. Functional replacement maps a same-rank update into those
 * positions of a base while retaining values outside the region. {@link CropToShapeAttrs}
 * separately represents an exact target-relative extraction whose extents may remain symbolic.</p>
 *
 * <p>General slice, both single-axis conveniences, and multi-axis flip use this same kind; flip is
 * represented by negative-step entries rather than another semantic kind. This enum calculates no
 * Shape, creates no layout or view,
 * and defines no Tensor construction, storage, materialization, gradient, compiler, backend,
 * ONNX, or execution behavior. Its inherited enum name is diagnostic text rather than a
 * serialization, parsing, registry, dispatch, reflection, or kernel identifier.</p>
 */
public enum SliceKind implements OperationKind {
    /**
     * Extracts either the finite coordinate sequences described by {@link SliceAttrs} or the
     * target-relative region described by {@link CropToShapeAttrs} from one input.
     *
     * <p>The kind describes logical meaning only. It does not normalize raw coordinates, inspect
     * input rank or dimensions, derive a result Shape, or decide layout and materialization.</p>
     */
    SLICE,

    /**
     * Functionally replaces the finite base-coordinate sequences described by {@link SliceAttrs}
     * with values from a same-rank update input.
     *
     * <p>The ordered inputs are base then update and the sole result has the base Shape. This is
     * logical replacement, not mutation, addition, storage writing, or a backward-only kind.</p>
     */
    SLICE_UPDATE;

    private static final List<OperationSignature> SLICE_SIGNATURES = List.of(
            OperationSignature.fixed(SliceAttrs.class, 1, 1),
            OperationSignature.fixed(CropToShapeAttrs.class, 1, 1));
    private static final List<OperationSignature> SLICE_UPDATE_SIGNATURES =
            List.of(OperationSignature.fixed(SliceAttrs.class, 2, 1));

    /**
     * Returns the immutable structural variants accepted by this exact slice-family kind.
     *
     * @return the stable immutable extraction or functional-replacement signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return this == SLICE ? SLICE_SIGNATURES : SLICE_UPDATE_SIGNATURES;
    }
}
