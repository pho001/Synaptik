package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable, backend-independent description of the logical facts of one tensor value.
 *
 * <p>The layout component is present when numeric element geometry has been resolved and empty
 * when that geometry remains unresolved. Dynamic shapes must remain unresolved, while a fully
 * static shape may also remain unresolved; this descriptor neither infers nor supplies a default
 * layout. When a layout is present, reconstructing it against the paired shape proves compatibility
 * of its public geometry, but it does not prove the identity of the shape from which the layout was
 * originally created.</p>
 *
 * <p>A descriptor is not the public mutable {@code Tensor}, a graph value or node, or a storage
 * object. It neither allocates storage nor associates the logical value with a graph occurrence.
 * Compiler contracts own inference and symbolic binding. Planning owns logical materialization
 * requirements, while prepare, runtime, and backend contracts own executable materialization and
 * execution state. This value contains none of those cross-layer facts or decisions.</p>
 *
 * <p>Record-generated equality and hashing include all four components. The generated
 * {@link #toString()} is diagnostic text that exposes the component values; it is not a serialized
 * representation or wire format.</p>
 *
 * @param dataType non-null immutable element type owned by the model data-type contract
 * @param shape non-null immutable logical shape owned by the model shape contract
 * @param layout non-null value-based optional; a present value is the exact immutable resolved
 *     layout supplied by the caller, while an empty value means layout geometry is unresolved;
 *     optional-container identity is not part of the contract
 * @param requiresGrad {@code true} to request model-level gradient eligibility, which is permitted
 *     only for a differentiable {@code dataType}
 */
public record TensorDescriptor(
        DataType dataType,
        Shape shape,
        Optional<LayoutDescriptor> layout,
        boolean requiresGrad) {
    /**
     * Creates a tensor descriptor and validates relationships between its model values.
     *
     * <p>A present layout is reconstructed from its public strides, element offset, and view flag
     * against {@code shape}. Complete value equality with that reconstruction validates rank,
     * strides, offset, view metadata, derived kind, and referenced span. This proves geometric
     * compatibility through public contracts, not that {@code shape} originally created the
     * supplied layout. The supplied values are otherwise retained through ordinary record
     * assignment.</p>
     *
     * @param dataType non-null immutable logical element type
     * @param shape non-null immutable logical shape; static shapes may still have unresolved layout
     * @param layout non-null value-based optional containing the resolved immutable layout or empty
     *     when layout geometry is unresolved; the optional is retained without making its
     *     container identity part of the contract, and a present value retains the exact supplied
     *     layout reference
     * @param requiresGrad {@code true} only when {@code dataType} is differentiable
     * @throws NullPointerException if {@code dataType}, {@code shape}, or {@code layout} is
     *     {@code null}
     * @throws IllegalArgumentException if a layout is present for a dynamic shape, reconstruction
     *     rejects or changes the supplied layout geometry, or gradient eligibility is requested for
     *     a non-differentiable data type
     * @throws ArithmeticException if checked layout reconstruction arithmetic overflows
     */
    public TensorDescriptor {
        dataType = Objects.requireNonNull(dataType, "dataType");
        shape = Objects.requireNonNull(shape, "shape");
        layout = Objects.requireNonNull(layout, "layout");

        if (layout.isPresent()) {
            if (!shape.isFullyStatic()) {
                throw new IllegalArgumentException(
                        "Resolved layout requires a fully static shape: " + shape);
            }

            LayoutDescriptor suppliedLayout = layout.orElseThrow();
            LayoutDescriptor reconstructedLayout = LayoutDescriptor.of(
                    shape,
                    suppliedLayout.strides(),
                    suppliedLayout.storageOffset(),
                    suppliedLayout.isView());
            if (!suppliedLayout.equals(reconstructedLayout)) {
                throw new IllegalArgumentException(
                        "Resolved layout is incompatible with shape " + shape + ": "
                                + suppliedLayout);
            }
        }

        if (requiresGrad && !dataType.isDifferentiable()) {
            throw new IllegalArgumentException(
                    "Gradient eligibility requires a differentiable data type: " + dataType);
        }
    }

    /**
     * Returns the stored model-owned logical element type.
     *
     * @return non-null immutable data type supplied at construction, describing every logical
     *     tensor element
     */
    public DataType dataType() {
        return dataType;
    }

    /**
     * Returns the stored model-owned logical shape.
     *
     * @return non-null immutable shape supplied at construction, describing the tensor's ordered
     *     logical dimensions
     */
    public Shape shape() {
        return shape;
    }

    /**
     * Returns the explicit resolved-or-unresolved layout state.
     *
     * <p>The non-null result is present for resolved numeric geometry and empty for unresolved
     * geometry, including permitted fully static unresolved descriptors. Callers must compare the
     * value or inspect its presence and must not rely on identity of the value-based optional
     * container. A present result contains the exact immutable {@link LayoutDescriptor} reference
     * supplied at construction.</p>
     *
     * @return non-null stored optional, compared by value, containing the exact supplied layout
     *     reference when resolved or empty when unresolved
     */
    public Optional<LayoutDescriptor> layout() {
        return layout;
    }

    /**
     * Reports whether model-level gradient eligibility was requested.
     *
     * <p>A true value is possible only for a differentiable data type. It does not assert that a
     * particular operation has a gradient rule or that a backend can differentiate it.</p>
     *
     * @return {@code true} when gradient eligibility was requested; otherwise {@code false}
     */
    public boolean requiresGrad() {
        return requiresGrad;
    }
}
