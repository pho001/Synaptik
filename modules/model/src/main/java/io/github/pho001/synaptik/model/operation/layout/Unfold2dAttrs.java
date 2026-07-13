package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries two-dimensional window geometry and an exact typed out-of-domain sample value.
 *
 * <p>The padding value supplies every sampled position outside the logical unpadded input,
 * including symmetric padding and terminal ceil-grid positions beyond the padded extent. Its
 * exact data type and primitive bits are retained, including NaN payloads, infinities, and signed
 * zero. This structural record accepts every current scalar type because it has no input tensor;
 * public Tensor construction performs the exact input-type compatibility check.</p>
 *
 * <p>These attributes are the explicit-padding variant for
 * {@link WindowTransformKind#UNFOLD2D}; the direct {@link Window2dAttrs} variant retains its
 * conceptual positive-zero padding. This record performs no input-aware rank, Shape, data-type,
 * geometry, or compatibility validation. It defines semantic metadata only and selects no
 * pooling behavior, backend fill policy, kernel, or execution.</p>
 *
 * @param window the non-null exact two-dimensional window geometry retained by reference
 * @param paddingValue the non-null exact typed padding value retained by reference
 */
public record Unfold2dAttrs(Window2dAttrs window, ScalarValue paddingValue)
        implements OperationAttrs {
    /**
     * Creates immutable configurable-padding unfold metadata.
     *
     * @param window the exact window geometry; must be non-null
     * @param paddingValue the exact typed padding value; must be non-null
     * @throws NullPointerException if {@code window} or {@code paddingValue} is {@code null},
     *     checked in component order with the component name as message
     */
    public Unfold2dAttrs {
        Objects.requireNonNull(window, "window");
        Objects.requireNonNull(paddingValue, "paddingValue");
    }

    /**
     * Returns the shared two-dimensional window geometry.
     *
     * @return the exact non-null window reference supplied at construction
     */
    @Override
    public Window2dAttrs window() {
        return window;
    }

    /**
     * Returns the typed value used for every out-of-domain sample.
     *
     * @return the exact non-null scalar-value reference supplied at construction
     */
    @Override
    public ScalarValue paddingValue() {
        return paddingValue;
    }
}
