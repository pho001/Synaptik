package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies the backend-independent, one-input meaning of a positive-step logical slice.
 *
 * <p>The kind composes explicitly with normalized {@link SliceAttrs}:</p>
 *
 * <pre>{@code
 * SliceAttrs attrs = new SliceAttrs(starts, ends, axes, steps);
 * Operation operation = new Operation(SliceKind.SLICE, attrs);
 * }</pre>
 *
 * <p>For entry {@code i}, the same-rank result selects input coordinates beginning at
 * {@code starts[i]}, advancing by {@code steps[i]}, and remaining below {@code ends[i]} along
 * normalized input axis {@code axes[i]}. Axes without an entry retain their full logical
 * coordinate range. The generic {@code Operation} descriptor accepts any non-null kind and
 * attributes value, so it does not enforce this family-specific pairing or one-input context.</p>
 *
 * <p>A future single-axis convenience uses this same kind with one attributes entry whose step is
 * one; it is not another semantic kind. This enum calculates no Shape, creates no layout or view,
 * and defines no Tensor construction, storage, materialization, gradient, compiler, backend,
 * ONNX, or execution behavior. Its inherited enum name is diagnostic text rather than a
 * serialization, parsing, registry, dispatch, reflection, or kernel identifier.</p>
 */
public enum SliceKind implements OperationKind {
    /**
     * Selects the parallel half-open coordinate intervals described by {@link SliceAttrs} while
     * leaving unlisted axes unrestricted.
     *
     * <p>The kind describes logical meaning only. It does not normalize raw coordinates, inspect
     * input rank or dimensions, derive a result Shape, or decide layout and materialization.</p>
     */
    SLICE
}
