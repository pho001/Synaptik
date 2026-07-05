package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs storage-free expressions that insert or remove one singleton tensor axis.
 *
 * <p>Insertion axes are normalized against the result-rank positions, while removal axes use the
 * existing-axis contract of {@link Shape}. Squeeze accepts only a selected dimension whose size
 * is statically known as one; a dynamic dimension is not treated as an implicit constraint. Both
 * transformations preserve the exact immutable references of every unaffected {@link Dimension}.
 * </p>
 *
 * <p>For every resolved input layout kind, this field-free package-private boundary creates a new
 * same-offset view layout by inserting or removing one element stride. Unresolved geometry stays
 * unresolved. View metadata attaches no host storage, proves no physical alias, and guarantees no
 * copy-free execution. Construction reads no values or storage and performs no canonicalization,
 * graph capture, gradient work, materialization, backend lowering, or execution.</p>
 */
final class TensorRankEditingExpressions {
    /** Prevents instantiation because rank-editing expression construction owns no state. */
    private TensorRankEditingExpressions() {
    }

    /**
     * Inserts one singleton axis and creates a fresh expand-dimensions expression.
     *
     * <p>Validation checks {@code input}, reads its exact descriptor and Shape once, normalizes the
     * raw insertion position, derives Shape and optional layout metadata, and delegates once to
     * the derived factory. For input rank {@code r}, raw axes in {@code [-r - 1, r]} are valid;
     * scalar axes {@code -1} and {@code 0} both select its only insertion position. Failures before
     * the final delegation consume no tensor identity.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input; it is not mutated
     * @param axis raw insertion position relative to the result rank
     * @return non-null fresh unlabeled, storage-free expand-dimensions expression retaining input
     *     data type and gradient eligibility
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IndexOutOfBoundsException if {@code axis} is outside the insertion range
     * @throws ArithmeticException if inserted-stride, layout-classification, or span arithmetic
     *     overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at final creation
     */
    static Tensor expandDims(Tensor input, int axis) {
        Objects.requireNonNull(input, "input");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        int normalizedAxis = normalizeInsertionAxis(axis, inputShape.rank());
        Shape resultShape = insertSingleton(inputShape, normalizedAxis);
        Optional<LayoutDescriptor> resultLayout = resolveInsertedLayout(
                inputDescriptor, inputShape, resultShape, normalizedAxis);
        return create(
                input,
                inputDescriptor,
                resultShape,
                resultLayout,
                AxisTransformKind.EXPAND_DIMS,
                normalizedAxis);
    }

    /**
     * Removes one statically known singleton axis and creates a fresh squeeze expression.
     *
     * <p>The raw axis is normalized exactly once through {@link Shape#normalizeAxis(int)}. The
     * selected dimension must be {@code StaticDimension(1)}; zero, another static extent, and every
     * dynamic dimension are rejected rather than inspected at runtime or converted into a hidden
     * symbolic constraint. Failures before final factory delegation consume no tensor identity.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input; it is not mutated
     * @param axis positive or negative existing input axis to remove
     * @return non-null fresh unlabeled, storage-free squeeze expression retaining input data type
     *     and gradient eligibility
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IndexOutOfBoundsException if {@code axis} is outside the input Shape rank, including
     *     every axis for a scalar
     * @throws IllegalArgumentException if the selected dimension is not statically known as one
     * @throws ArithmeticException if result-layout classification or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at final creation
     */
    static Tensor squeeze(Tensor input, int axis) {
        Objects.requireNonNull(input, "input");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        int normalizedAxis = inputShape.normalizeAxis(axis);
        validateSqueezableDimension(inputShape, normalizedAxis);
        Shape resultShape = removeSingleton(inputShape, normalizedAxis);
        Optional<LayoutDescriptor> resultLayout =
                resolveSqueezedLayout(inputDescriptor, resultShape, normalizedAxis);
        return create(
                input,
                inputDescriptor,
                resultShape,
                resultLayout,
                AxisTransformKind.SQUEEZE,
                normalizedAxis);
    }

    /**
     * Normalizes a raw insertion position against {@code rank + 1} possible output positions.
     *
     * <p>Normalization uses {@code long}, adds {@code rank + 1} exactly once for a negative raw
     * value, and accepts the inclusive normalized range {@code [0, rank]}.</p>
     *
     * @param axis raw insertion position
     * @param rank non-negative input Shape rank
     * @return normalized insertion position in the inclusive range {@code [0, rank]}
     * @throws IndexOutOfBoundsException if the raw position does not normalize into that range,
     *     with message {@code Axis <axis> is outside insertion range for shape rank <rank>}
     */
    private static int normalizeInsertionAxis(int axis, int rank) {
        long normalizedAxis = axis;
        if (normalizedAxis < 0) {
            normalizedAxis += (long) rank + 1;
        }
        if (normalizedAxis < 0 || normalizedAxis > rank) {
            throw new IndexOutOfBoundsException(
                    "Axis " + axis + " is outside insertion range for shape rank " + rank);
        }
        return (int) normalizedAxis;
    }

    /**
     * Requires the selected existing input dimension to be exactly a static singleton.
     *
     * @param inputShape non-null input Shape containing the already normalized axis
     * @param normalizedAxis existing non-negative input axis
     * @throws IllegalArgumentException if the selected dimension is zero, a static extent other
     *     than one, or dynamic, with a message naming the normalized axis and input Shape
     */
    private static void validateSqueezableDimension(
            Shape inputShape, int normalizedAxis) {
        Dimension selected = inputShape.dimensions().get(normalizedAxis);
        if (!(selected instanceof StaticDimension staticDimension)
                || staticDimension.size() != 1) {
            throw new IllegalArgumentException(
                    "cannot squeeze axis "
                            + normalizedAxis
                            + " of "
                            + inputShape
                            + ": dimension must be statically known as 1");
        }
    }

    /**
     * Inserts one new static singleton while preserving exact unaffected Dimension references.
     *
     * @param inputShape non-null exact input Shape; not mutated or retained as an array
     * @param normalizedAxis insertion position in the inclusive range {@code [0, inputRank]}
     * @return non-null result Shape with rank increased by one and exactly one new
     *     {@link StaticDimension} of size one
     */
    private static Shape insertSingleton(Shape inputShape, int normalizedAxis) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank() + 1];
        for (int inputAxis = 0; inputAxis < normalizedAxis; inputAxis++) {
            resultDimensions[inputAxis] = inputShape.dimensions().get(inputAxis);
        }
        resultDimensions[normalizedAxis] = new StaticDimension(1);
        for (int inputAxis = normalizedAxis; inputAxis < inputShape.rank(); inputAxis++) {
            resultDimensions[inputAxis + 1] = inputShape.dimensions().get(inputAxis);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Removes one singleton while preserving exact unaffected Dimension references in order.
     *
     * @param inputShape non-null input Shape whose selected dimension has already been validated
     * @param normalizedAxis existing non-negative input axis to omit
     * @return non-null result Shape with rank reduced by one; rank-one input produces the canonical
     *     scalar Shape
     */
    private static Shape removeSingleton(Shape inputShape, int normalizedAxis) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank() - 1];
        for (int inputAxis = 0; inputAxis < normalizedAxis; inputAxis++) {
            resultDimensions[inputAxis] = inputShape.dimensions().get(inputAxis);
        }
        for (int inputAxis = normalizedAxis + 1; inputAxis < inputShape.rank(); inputAxis++) {
            resultDimensions[inputAxis - 1] = inputShape.dimensions().get(inputAxis);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Inserts deterministic resolved stride geometry or preserves unresolved layout state.
     *
     * <p>The input layout optional is read exactly once. When present, exact input strides are
     * copied around the insertion. An insertion before input axis {@code i} uses checked product
     * {@code inputStride(i) * inputExtent(i)}; insertion at the end uses stride one. The exact
     * element offset is retained in one new view-marked descriptor for every input layout kind.
     * This metadata does not attach storage or guarantee an executable alias.</p>
     *
     * @param inputDescriptor non-null exact input descriptor supplying optional layout
     * @param inputShape non-null exact input Shape supplying the following static extent
     * @param resultShape non-null inserted result Shape used to classify the new layout
     * @param normalizedAxis insertion position in the inclusive range {@code [0, inputRank]}
     * @return non-null optional containing one new resolved same-offset view layout, or empty when
     *     input layout is unresolved
     * @throws ArithmeticException if inserted-stride multiplication, classification, or referenced
     *     span arithmetic overflows
     */
    private static Optional<LayoutDescriptor> resolveInsertedLayout(
            TensorDescriptor inputDescriptor,
            Shape inputShape,
            Shape resultShape,
            int normalizedAxis) {
        Optional<LayoutDescriptor> inputLayout = inputDescriptor.layout();
        if (inputLayout.isEmpty()) {
            return Optional.empty();
        }

        LayoutDescriptor resolvedInputLayout = inputLayout.orElseThrow();
        long[] resultStrides = new long[resultShape.rank()];
        for (int inputAxis = 0; inputAxis < normalizedAxis; inputAxis++) {
            resultStrides[inputAxis] = resolvedInputLayout.stride(inputAxis);
        }
        resultStrides[normalizedAxis] = normalizedAxis == inputShape.rank()
                ? 1
                : Math.multiplyExact(
                        resolvedInputLayout.stride(normalizedAxis),
                        ((StaticDimension) inputShape.dimensions().get(normalizedAxis)).size());
        for (int inputAxis = normalizedAxis; inputAxis < inputShape.rank(); inputAxis++) {
            resultStrides[inputAxis + 1] = resolvedInputLayout.stride(inputAxis);
        }
        return Optional.of(LayoutDescriptor.of(
                resultShape,
                resultStrides,
                resolvedInputLayout.storageOffset(),
                true));
    }

    /**
     * Removes one resolved stride entry or preserves unresolved layout state.
     *
     * <p>The input layout optional is read exactly once. When present, every stride except the
     * selected one is copied exactly and in order into a newly allocated vector. One new
     * view-marked descriptor preserves the exact element offset and derives kind and span. The
     * input descriptor is never reused as result layout, and no storage is observed or attached.</p>
     *
     * @param inputDescriptor non-null exact input descriptor supplying optional layout
     * @param resultShape non-null squeezed result Shape used to classify the new layout
     * @param normalizedAxis existing non-negative input axis whose stride is omitted
     * @return non-null optional containing one new resolved same-offset view layout, or empty when
     *     input layout is unresolved
     * @throws ArithmeticException if result-layout classification or referenced-span arithmetic
     *     overflows
     */
    private static Optional<LayoutDescriptor> resolveSqueezedLayout(
            TensorDescriptor inputDescriptor, Shape resultShape, int normalizedAxis) {
        Optional<LayoutDescriptor> inputLayout = inputDescriptor.layout();
        if (inputLayout.isEmpty()) {
            return Optional.empty();
        }

        LayoutDescriptor resolvedInputLayout = inputLayout.orElseThrow();
        long[] resultStrides = new long[resultShape.rank()];
        for (int inputAxis = 0; inputAxis < normalizedAxis; inputAxis++) {
            resultStrides[inputAxis] = resolvedInputLayout.stride(inputAxis);
        }
        for (int inputAxis = normalizedAxis + 1; inputAxis < resolvedInputLayout.rank(); inputAxis++) {
            resultStrides[inputAxis - 1] = resolvedInputLayout.stride(inputAxis);
        }
        return Optional.of(LayoutDescriptor.of(
                resultShape,
                resultStrides,
                resolvedInputLayout.storageOffset(),
                true));
    }

    /**
     * Creates exact rank-edit attributes, descriptor, operation, provenance, and Tensor once.
     *
     * <p>The result retains exact input data type and gradient eligibility with the supplied Shape
     * and resolved-or-unresolved layout. It records exact {@code kind}, one
     * {@link AxisTransformAttrs} containing {@code normalizedAxis}, and ordered provenance
     * {@code [input]}. The single derived-factory call uses no label or storage and therefore gives
     * every successful request a fresh identity without canonicalizing repeated, nested, or
     * inverse-like edits.</p>
     *
     * @param input non-null exact sole provenance input; not inspected or mutated
     * @param inputDescriptor non-null descriptor supplying exact data type and gradient eligibility
     * @param resultShape non-null inserted or squeezed Shape retained by the result descriptor
     * @param resultLayout non-null resolved-view or unresolved-layout value retained by descriptor
     * @param kind non-null exact {@link AxisTransformKind#EXPAND_DIMS} or
     *     {@link AxisTransformKind#SQUEEZE} selected by the entry method
     * @param normalizedAxis non-negative normalized insertion or removal position
     * @return non-null fresh factory-derived Tensor with exact one-input provenance and no label or
     *     host storage
     * @throws IllegalArgumentException if semantic or descriptor invariants reject supplied metadata
     * @throws ArithmeticException if descriptor layout reconstruction arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at the single factory
     *     delegation
     */
    private static Tensor create(
            Tensor input,
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            Optional<LayoutDescriptor> resultLayout,
            AxisTransformKind kind,
            int normalizedAxis) {
        AxisTransformAttrs attrs = new AxisTransformAttrs(normalizedAxis);
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                resultShape,
                resultLayout,
                inputDescriptor.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}
