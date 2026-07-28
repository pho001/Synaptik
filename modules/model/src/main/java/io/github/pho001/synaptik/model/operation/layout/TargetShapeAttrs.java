package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the exact normalized target shape for a reshape or expansion.
 *
 * <p>The immutable record accepts every valid {@link Shape}: rank-zero scalar, zero-extent,
 * ordinary static, mixed static and dynamic, and fully dynamic shapes. The exact immutable Shape
 * reference is retained. Input-aware expansion construction may retain this exact target while a
 * right-aligned source dimension still has to bind either to one or to its target dimension.
 * Construction of this record does not inspect an input, compare element counts, validate that
 * obligation, bind dynamic dimensions, create a proof or constraint, or derive result layout.</p>
 *
 * <p>The stored Shape is normalized model semantics, not raw public reshape-request syntax.
 * Dynamic dimensions use explicit symbols, and static dimensions are non-negative, so a numeric
 * {@code -1} inference sentinel cannot occur in this value. A later public request boundary owns
 * any inference and normalization before these attributes are constructed.</p>
 *
 * <p>Valid operations explicitly pair this value with either
 * {@link ShapeTransformKind#RESHAPE} or {@link ShapeTransformKind#EXPAND}. Their shared
 * family-owned signature enforces those exact pairings and declares one input and one output.
 * Record-generated equality and hashing use structural Shape equality; generated text is
 * diagnostic only and is not public request syntax, serialization, parser input, backend
 * dispatch, or a layout plan.</p>
 *
 * @param targetShape the non-null normalized semantic result Shape; every valid Shape category is
 *     accepted and the exact immutable reference is retained
 */
public record TargetShapeAttrs(Shape targetShape) implements OperationAttrs {
    /**
     * Creates immutable target-shape attributes.
     *
     * <p>The Shape reference is checked for presence and retained without copying, request
     * inference, compatibility validation, symbolic binding, deferred-predicate creation, or
     * layout derivation.</p>
     *
     * @param targetShape the non-null normalized semantic result Shape to retain unchanged
     * @throws NullPointerException if {@code targetShape} is {@code null}, with message
     *     {@code targetShape}
     */
    public TargetShapeAttrs {
        targetShape = Objects.requireNonNull(targetShape, "targetShape");
    }

    /**
     * Returns the exact normalized semantic target Shape supplied at construction.
     *
     * <p>The returned immutable value is not raw request syntax and carries no input compatibility
     * proof, deferred predicate, inferred-axis marker, layout, provenance, or executable
     * state.</p>
     *
     * @return the exact stored non-null target Shape reference
     */
    @Override
    public Shape targetShape() {
        return targetShape;
    }
}
