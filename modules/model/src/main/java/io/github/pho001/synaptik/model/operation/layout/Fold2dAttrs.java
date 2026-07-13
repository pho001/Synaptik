package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the explicit logical result Shape and shared window geometry for two-dimensional fold.
 *
 * <p>{@link WindowTransformKind#FOLD2D} interprets its eventual rank-three input as canonical
 * im2col columns and performs overlap-summing col2im into {@link #outputShape()}. Multiple column
 * entries targeting the same NCHW output coordinate are added, uncovered positions remain zero,
 * and no division by overlap count occurs. For 2-by-2 columns derived from conceptual
 * {@code [1, 1, 3, 3]} geometry, the center receives four contributions while corners receive
 * one.</p>
 *
 * <p>Both immutable references are retained exactly. This structural layer deliberately accepts
 * every current Shape category, including scalar, non-rank-four, zero-extent, and dynamic Shapes.
 * Public Tensor expression construction proves the rank-four NCHW and exact structural column
 * compatibility boundary, using checked static geometry or canonical symbolic formulas. This
 * record does not copy or canonicalize either value or compare Shape against the window
 * geometry.</p>
 *
 * <p>The attributes define no Tensor construction, Shape calculation, sampling or accumulation,
 * DataType, layout, storage, provenance, gradient, graph/compiler, planning, prepare, runtime,
 * backend, ONNX, or execution behavior.</p>
 *
 * @param outputShape the non-null explicit logical result Shape; the exact immutable reference is
 *     retained
 * @param window the non-null two-dimensional window geometry; the exact immutable reference is
 *     retained
 */
public record Fold2dAttrs(Shape outputShape, Window2dAttrs window) implements OperationAttrs {
    /**
     * Creates immutable structural parameters for two-dimensional fold.
     *
     * <p>References are null-checked in component order and retained without copying,
     * reconstruction, canonicalization, rank checking, or compatibility validation.</p>
     *
     * @param outputShape the explicit logical result Shape; must be non-null and is stored by
     *     exact reference
     * @param window the shared window geometry; must be non-null and is stored by exact reference
     * @throws NullPointerException if {@code outputShape} is {@code null}, with message
     *     {@code outputShape}
     * @throws NullPointerException if {@code window} is {@code null} after output Shape validation,
     *     with message {@code window}
     */
    public Fold2dAttrs {
        Objects.requireNonNull(outputShape, "outputShape");
        Objects.requireNonNull(window, "window");
    }

    /**
     * Returns the explicit logical fold result Shape.
     *
     * @return the exact non-null immutable Shape reference supplied at construction
     */
    @Override
    public Shape outputShape() {
        return outputShape;
    }

    /**
     * Returns the shared two-dimensional window geometry.
     *
     * @return the exact non-null immutable window reference supplied at construction
     */
    @Override
    public Window2dAttrs window() {
        return window;
    }
}
