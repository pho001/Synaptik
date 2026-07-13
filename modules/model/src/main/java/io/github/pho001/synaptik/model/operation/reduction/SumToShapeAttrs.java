package io.github.pho001.synaptik.model.operation.reduction;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the exact result Shape for one binding-aware {@link AggregateReductionKind#SUM}
 * occurrence.
 *
 * <p>The source Shape is right-aligned with {@link #targetShape()}. Every leading source axis is
 * reduced and removed. At concrete binding, an aligned target extent of one reduces its source
 * axis and retains one position, while an extent equal to the source extent preserves that axis.
 * Every other concrete pair is invalid. Model construction rejects only statically provable
 * incompatibility; unresolved pairs retain this obligation for later binding validation.</p>
 *
 * <p>The immutable record retains the exact target Shape reference and contains no source,
 * resolved axis set, binding, constraint, descriptor, provenance, compiler state, or executable
 * behavior. Record-generated equality and hashing use structural Shape equality; generated text
 * is diagnostic and is not serialization, dispatch, or a reduction plan.</p>
 *
 * @param targetShape the non-null exact semantic result Shape retained unchanged; scalar,
 *     zero-extent, static, named-dynamic, and expression Shapes are accepted
 */
public record SumToShapeAttrs(Shape targetShape) implements OperationAttrs {
    /**
     * Creates immutable binding-aware sum-to-Shape attributes.
     *
     * @param targetShape the non-null exact semantic result Shape to retain unchanged
     * @throws NullPointerException if {@code targetShape} is null, with message
     *     {@code targetShape}
     */
    public SumToShapeAttrs {
        targetShape = Objects.requireNonNull(targetShape, "targetShape");
    }

    /**
     * Returns the exact semantic result Shape supplied at construction.
     *
     * @return the exact stored non-null target Shape reference, carrying any unresolved
     *     target-one-or-source-equal binding obligation
     */
    @Override
    public Shape targetShape() {
        return targetShape;
    }
}
