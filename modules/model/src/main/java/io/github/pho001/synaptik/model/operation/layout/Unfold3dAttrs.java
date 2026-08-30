package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.Objects;

/**
 * Carries NCDHW window geometry and one exact typed out-of-domain sample value.
 *
 * <p>The scalar supplies symmetric padding and every terminal literal-ceil sample outside the
 * unpadded input. Its exact data type, primitive bits, and object reference are retained. This
 * structural value performs no input-aware rank, type, Shape, or geometry validation and selects
 * no pooling, backend, or execution behavior.</p>
 *
 * @param window non-null exact three-dimensional window geometry retained by reference
 * @param paddingValue non-null exact typed padding scalar retained by reference
 */
public record Unfold3dAttrs(Window3dAttrs window, ScalarValue paddingValue)
        implements OperationAttrs {
    /**
     * Creates immutable explicit-padding unfold metadata.
     *
     * @param window non-null exact geometry retained without copying
     * @param paddingValue non-null exact typed scalar retained without conversion
     * @throws NullPointerException if {@code window} or {@code paddingValue} is null, checked in
     *     component order with the component name as message
     */
    public Unfold3dAttrs {
        Objects.requireNonNull(window, "window");
        Objects.requireNonNull(paddingValue, "paddingValue");
    }
}
