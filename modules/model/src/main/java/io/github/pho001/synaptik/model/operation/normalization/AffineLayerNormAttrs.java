package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the exact trailing normalized Shape and positive typed epsilon for affine layer
 * normalization.
 *
 * <p>This distinct attributes type selects the exact three-input signature
 * {@code [input, scale, bias]}. The normalized Shape has positive rank; epsilon is finite,
 * floating, and strictly positive. Operand Shapes, ordered floating promotion, and result metadata
 * are validated by Tensor expression construction because this record contains no operand. The
 * record retains both immutable references and owns no statistics, gradient, or execution state.</p>
 *
 * @param normalizedShape non-null positive-rank Shape corresponding to exact trailing input axes;
 *     retained unchanged
 * @param epsilon non-null exact finite floating value strictly greater than positive zero;
 *     retained unchanged
 */
public record AffineLayerNormAttrs(Shape normalizedShape, ScalarValue epsilon)
        implements OperationAttrs {
    /**
     * Creates validated affine layer-normalization parameters.
     *
     * @param normalizedShape non-null Shape with positive rank; retained unchanged
     * @param epsilon non-null floating, finite, strictly positive typed value; retained unchanged
     * @throws NullPointerException if {@code normalizedShape} or {@code epsilon} is null, checked
     *     in declaration order
     * @throws IllegalArgumentException if {@code normalizedShape} has rank zero, or if
     *     {@code epsilon} is non-floating, non-finite, negative, or either signed zero
     */
    public AffineLayerNormAttrs {
        normalizedShape = Objects.requireNonNull(normalizedShape, "normalizedShape");
        epsilon = Objects.requireNonNull(epsilon, "epsilon");
        LayerNormAttrs.validate(normalizedShape, epsilon);
    }
}
