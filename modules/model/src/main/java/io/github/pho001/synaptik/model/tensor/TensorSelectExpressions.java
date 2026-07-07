package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally normalized, storage-free scalar-select expressions for {@link Tensor}.
 *
 * <p>Selection fixes one scalar coordinate and removes its source axis. Static selected extents
 * support one-time negative-index normalization and immediate bounds checks. Dynamic selected
 * extents accept a non-negative coordinate with its upper bound deferred, but reject a negative
 * coordinate because no numeric extent exists for local normalization. Unaffected Dimension
 * references remain exact, and rank-one selection produces the canonical scalar Shape.</p>
 *
 * <p>Resolved input geometry with a non-empty result produces one new logical view after removing
 * the selected stride and checked-advancing the element offset. Unresolved input geometry or an
 * empty result remain unresolved. Every result is fresh, unlabeled, storage-free, and records
 * exact SELECT semantics with one-input provenance. This field-free helper does not inspect values
 * or storage, promise a physical alias, define gradients, capture a graph, choose materialization
 * or lowering, or execute work.</p>
 */
final class TensorSelectExpressions {
    /** Prevents instantiation because scalar-select expression construction owns no state. */
    private TensorSelectExpressions() {
    }

    /**
     * Validates one input-aware scalar-select request and creates its fresh expression tensor.
     *
     * <p>The input is null-checked before its exact descriptor and Shape are read once. The raw
     * axis is normalized exactly once through {@link Shape#normalizeAxis(int)}, then the selected
     * Dimension is read once. Static index normalization, normalized attributes, axis removal,
     * conditional view geometry, and semantic construction all complete before the single final
     * derived-factory call, so local failure consumes no tensor identifier.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input and never mutated
     * @param axis positive or negative existing input axis normalized once against the input rank
     * @param index scalar coordinate normalized against a static selected extent, or a
     *     non-negative coordinate retained for a dynamic selected extent
     * @return a non-null fresh unlabeled and storage-free SELECT expression
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IndexOutOfBoundsException if the axis is invalid or a static selected coordinate is
     *     outside its extent
     * @throws IllegalArgumentException if a negative index targets a dynamic selected extent
     * @throws ArithmeticException if checked result-element-count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor apply(Tensor input, int axis, long index) {
        Objects.requireNonNull(input, "input");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        Dimension selected = inputShape.dimensions().get(normalizedAxis);
        long normalizedIndex = normalizeIndex(selected, normalizedAxis, index);
        SelectAttrs attrs = new SelectAttrs(normalizedAxis, normalizedIndex);
        Shape resultShape = removeAxis(inputShape, normalizedAxis);
        Optional<LayoutDescriptor> resultLayout =
                resolveViewLayout(inputDescriptor, resultShape, normalizedAxis, normalizedIndex);
        return create(input, inputDescriptor, resultShape, resultLayout, attrs);
    }

    /**
     * Normalizes and validates one scalar coordinate for its selected Dimension category.
     *
     * <p>For a static extent, one negative coordinate adds the extent once, after which the result
     * must lie in {@code [0, extent)}. This naturally rejects every coordinate for extent zero.
     * For a dynamic extent, a non-negative coordinate is already normalized and is retained while
     * upper-bound validation is deferred. A negative dynamic coordinate is rejected because no
     * concrete extent exists to normalize it.</p>
     *
     * @param selected non-null exact selected input Dimension
     * @param normalizedAxis normalized non-negative axis used in failure diagnostics
     * @param rawIndex caller-supplied signed scalar coordinate
     * @return the non-negative normalized coordinate
     * @throws IndexOutOfBoundsException if a statically normalized coordinate is outside its
     *     selected extent
     * @throws IllegalArgumentException if {@code rawIndex} is negative and {@code selected} is
     *     dynamic
     */
    private static long normalizeIndex(
            Dimension selected, int normalizedAxis, long rawIndex) {
        if (selected instanceof StaticDimension staticDimension) {
            long size = staticDimension.size();
            long normalized = rawIndex;
            if (normalized < 0) {
                normalized += size;
            }
            if (normalized < 0 || normalized >= size) {
                throw new IndexOutOfBoundsException(
                        "select index " + rawIndex + " is outside axis " + normalizedAxis
                                + " extent " + size);
            }
            return normalized;
        }
        if (rawIndex < 0) {
            throw new IllegalArgumentException(
                    "select index " + rawIndex + " cannot be normalized against dynamic axis "
                            + normalizedAxis);
        }
        return rawIndex;
    }

    /**
     * Removes one selected input axis while preserving every unaffected Dimension reference.
     *
     * <p>Exactly one new Dimension array is populated in original axis order and passed once to
     * {@link Shape#ofDimensions(Dimension...)}. Removing the only rank-one axis therefore returns
     * the canonical scalar Shape; removing a dynamic selected Dimension does not alter any
     * unselected static or dynamic reference.</p>
     *
     * @param inputShape non-null exact source Shape
     * @param normalizedAxis normalized existing source axis to remove
     * @return a non-null rank-minus-one Shape with exact unaffected references
     */
    private static Shape removeAxis(Shape inputShape, int normalizedAxis) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank() - 1];
        for (int inputAxis = 0, resultAxis = 0;
                inputAxis < inputShape.rank(); inputAxis++) {
            if (inputAxis != normalizedAxis) {
                resultDimensions[resultAxis++] = inputShape.dimensions().get(inputAxis);
            }
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Derives checked scalar-select view geometry when the result references storage elements.
     *
     * <p>An unresolved input or a result with known element count zero returns unresolved layout.
     * Otherwise the input strides are copied once, the selected stride is removed in order, and
     * the result offset is the checked sum of the input offset and normalized index times selected
     * stride. For contiguous Shape {@code [2, 3, 4]}, strides {@code [12, 4, 1]}, axis {@code 1},
     * and index {@code 2}, the result strides are {@code [12, 1]} and the offset is
     * {@code 0 + 2 * 4 = 8}. The new descriptor is logical view metadata only.</p>
     *
     * @param inputDescriptor non-null exact descriptor supplying optional input geometry
     * @param resultShape non-null derived rank-minus-one result Shape
     * @param normalizedAxis normalized source axis whose stride is removed
     * @param normalizedIndex non-negative selected scalar coordinate
     * @return a non-null optional containing one new view layout, or empty for unresolved or empty
     *     geometry
     * @throws ArithmeticException if checked result-element-count, offset, classification, or
     *     referenced-span arithmetic overflows
     */
    private static Optional<LayoutDescriptor> resolveViewLayout(
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            int normalizedAxis,
            long normalizedIndex) {
        Optional<LayoutDescriptor> inputLayout = inputDescriptor.layout();
        if (inputLayout.isEmpty()) {
            return Optional.empty();
        }
        if (resultShape.knownElementCount().orElseThrow() == 0L) {
            return Optional.empty();
        }

        LayoutDescriptor resolvedInputLayout = inputLayout.orElseThrow();
        long[] inputStrides = resolvedInputLayout.strides();
        long[] resultStrides = new long[inputStrides.length - 1];
        for (int inputAxis = 0, resultAxis = 0;
                inputAxis < inputStrides.length; inputAxis++) {
            if (inputAxis != normalizedAxis) {
                resultStrides[resultAxis++] = inputStrides[inputAxis];
            }
        }
        long resultOffset = Math.addExact(
                resolvedInputLayout.storageOffset(),
                Math.multiplyExact(normalizedIndex, inputStrides[normalizedAxis]));
        return Optional.of(
                LayoutDescriptor.of(resultShape, resultStrides, resultOffset, true));
    }

    /**
     * Creates the exact descriptor, SELECT operation, one-input provenance, and fresh Tensor.
     *
     * <p>The descriptor retains the exact input data type and gradient eligibility with the
     * derived Shape and resolved-or-unresolved layout. The operation retains the exact normalized
     * attributes, provenance is ordered {@code [input]}, and the single derived-factory call uses
     * no label or storage. Repeated, nested, and same-coordinate requests remain separate fresh
     * expressions rather than being folded or canonicalized.</p>
     *
     * @param input non-null exact sole provenance input
     * @param inputDescriptor non-null exact descriptor supplying type and gradient eligibility
     * @param resultShape non-null derived result Shape
     * @param resultLayout non-null resolved-view or unresolved-layout optional
     * @param attrs non-null exact normalized SELECT attributes retained by the operation
     * @return a non-null fresh factory-derived Tensor without label or storage
     * @throws IllegalArgumentException if descriptor invariants reject supplied metadata
     * @throws ArithmeticException if descriptor layout reconstruction arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at factory delegation
     */
    private static Tensor create(
            Tensor input,
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            Optional<LayoutDescriptor> resultLayout,
            SelectAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                resultShape,
                resultLayout,
                inputDescriptor.requiresGrad());
        Operation operation = new Operation(SelectKind.SELECT, attrs);
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}
