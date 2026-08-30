package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;

/**
 * Carries the exact NCDHW target Shape and window geometry for volumetric overlap fold.
 *
 * <p>Fold geometrically excludes padded and terminal ceil-tail coordinates, initializes target
 * coordinates to represented positive zero, and accumulates in-range canonical-column
 * contributions in input order. Both immutable component references are retained exactly. This
 * structural record performs no rank, type, geometry, column-compatibility, or execution work.</p>
 *
 * @param outputShape non-null explicit rank-five target Shape retained by reference
 * @param window non-null three-dimensional window geometry retained by reference
 */
public record Fold3dAttrs(Shape outputShape, Window3dAttrs window) implements OperationAttrs {
    /**
     * Creates immutable three-dimensional fold metadata.
     *
     * @param outputShape non-null exact target Shape retained without copying
     * @param window non-null exact geometry retained without copying
     * @throws NullPointerException if {@code outputShape} or {@code window} is null, checked in
     *     component order with the component name as message
     */
    public Fold3dAttrs {
        Objects.requireNonNull(outputShape, "outputShape");
        Objects.requireNonNull(window, "window");
    }
}
