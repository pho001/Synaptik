package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally normalized, storage-free positive-step slice expressions for {@link Tensor}.
 *
 * <p>A request supplies four parallel arrays. At each entry, the raw axis and half-open bounds are
 * normalized against one statically known input dimension, the bounds are clamped into that
 * dimension, and the positive step determines a new static result extent. Unselected dimensions
 * retain their exact immutable references, so rank and any unselected dynamic symbols are
 * preserved. Empty arrays describe an explicit identity slice, while start greater than or equal
 * to end produces a valid zero extent.</p>
 *
 * <p>For a non-empty result with resolved input geometry, construction advances the element
 * offset by each normalized start times the original input stride and multiplies selected strides
 * by their steps using checked arithmetic. Every input layout kind is accepted and the result is
 * marked as a logical view. Unresolved input geometry and empty results remain unresolved. This
 * metadata neither attaches storage nor promises physical aliasing, materialization, gradient,
 * compiler, backend, or execution behavior.</p>
 */
final class TensorSliceExpressions {
    /** Prevents instantiation because slice-expression construction owns no state. */
    private TensorSliceExpressions() {
    }

    /**
     * Normalizes one general request and creates a fresh slice expression.
     *
     * <p>References are null-checked in parameter order and equal lengths are required before the
     * four caller-owned arrays are cloned in declaration order. The exact input descriptor and
     * Shape are then read once. Every subsequent failure occurs before the single derived-factory
     * call and therefore consumes no tensor identifier.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input; not mutated
     * @param starts non-null caller-owned inclusive raw starts, paired by entry
     * @param ends non-null caller-owned exclusive raw ends, paired by entry
     * @param axes non-null caller-owned raw positive or negative axes, paired by entry
     * @param steps non-null caller-owned strictly positive steps, paired by entry
     * @return a non-null fresh unlabeled, storage-free SLICE tensor with normalized attributes
     * @throws NullPointerException if any reference is null, with its parameter name as message
     * @throws IllegalArgumentException if array lengths differ, an axis is invalid or repeated, a
     *     step is non-positive, or a selected dimension is dynamic
     * @throws ArithmeticException if checked result element-count, layout-offset, stride,
     *     classification, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor apply(
            Tensor input, long[] starts, long[] ends, int[] axes, long[] steps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(starts, "starts");
        Objects.requireNonNull(ends, "ends");
        Objects.requireNonNull(axes, "axes");
        Objects.requireNonNull(steps, "steps");
        if (starts.length != ends.length
                || starts.length != axes.length
                || starts.length != steps.length) {
            throw new IllegalArgumentException(
                    "starts, ends, axes, and steps must have matching lengths");
        }

        long[] privateStarts = starts.clone();
        long[] privateEnds = ends.clone();
        int[] privateAxes = axes.clone();
        long[] privateSteps = steps.clone();
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        SliceAttrs attrs = normalize(
                inputShape, privateStarts, privateEnds, privateAxes, privateSteps);
        Shape resultShape = deriveShape(inputShape, attrs);
        Optional<LayoutDescriptor> resultLayout =
                resolveViewLayout(inputDescriptor, resultShape, attrs);
        return create(input, inputDescriptor, resultShape, resultLayout, attrs);
    }

    /**
     * Creates a one-axis, step-one request through the general construction path.
     *
     * <p>No independent normalization or semantic kind exists. The four private one-element
     * arrays make this exactly one {@link SliceKind#SLICE} entry with step one.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input
     * @param axis raw positive or negative selected input axis
     * @param fromInclusive raw inclusive start normalized and clamped by the general path
     * @param toExclusive raw exclusive end normalized and clamped by the general path
     * @return a non-null fresh one-axis SLICE expression
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalArgumentException if the axis is invalid, selected dimension is dynamic, or
     *     the delegated general request is otherwise invalid
     * @throws ArithmeticException if checked result element-count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor applyAxis(
            Tensor input, int axis, long fromInclusive, long toExclusive) {
        return apply(
                input,
                new long[] {fromInclusive},
                new long[] {toExclusive},
                new int[] {axis},
                new long[] {1L});
    }

    /**
     * Normalizes axes and bounds and validates one private general request.
     *
     * <p>Entries remain in caller order. A negative axis adds rank once using {@code long}; a
     * negative bound adds the selected static dimension size once and is then clamped. Duplicate
     * detection uses normalized axes. Exactly one immutable {@link SliceAttrs} snapshot is created
     * after all entries pass.</p>
     *
     * @param inputShape non-null exact input Shape used for rank and selected dimensions
     * @param starts non-null private inclusive starts with matching length
     * @param ends non-null private exclusive ends with matching length
     * @param axes non-null private raw axes with matching length
     * @param steps non-null private steps with matching length
     * @return non-null normalized immutable attributes in original request order
     * @throws IllegalArgumentException if an axis is outside rank or repeated after normalization,
     *     a step is non-positive, or a selected dimension is not static
     */
    private static SliceAttrs normalize(
            Shape inputShape, long[] starts, long[] ends, int[] axes, long[] steps) {
        int rank = inputShape.rank();
        boolean[] seenAxes = new boolean[rank];
        List<Long> normalizedStarts = new ArrayList<>(starts.length);
        List<Long> normalizedEnds = new ArrayList<>(ends.length);
        List<Integer> normalizedAxes = new ArrayList<>(axes.length);
        List<Long> normalizedSteps = new ArrayList<>(steps.length);
        for (int index = 0; index < starts.length; index++) {
            int rawAxis = axes[index];
            long normalizedAxis = rawAxis;
            if (normalizedAxis < 0) {
                normalizedAxis += rank;
            }
            if (normalizedAxis < 0 || normalizedAxis >= rank) {
                throw new IllegalArgumentException(
                        "slice axis " + rawAxis + " at index " + index
                                + " is outside rank " + rank);
            }
            int axis = (int) normalizedAxis;
            if (seenAxes[axis]) {
                throw new IllegalArgumentException(
                        "slice contains duplicate normalized axis " + axis
                                + " at index " + index);
            }
            seenAxes[axis] = true;

            long step = steps[index];
            if (step <= 0) {
                throw new IllegalArgumentException(
                        "steps[" + index + "] must be positive: " + step);
            }

            Dimension selectedDimension = inputShape.dimensions().get(axis);
            if (!(selectedDimension instanceof StaticDimension staticDimension)) {
                throw new IllegalArgumentException(
                        "slice axis " + axis + " at index " + index
                                + " must have a statically known dimension");
            }
            long dimensionSize = staticDimension.size();
            normalizedStarts.add(normalizeBound(starts[index], dimensionSize));
            normalizedEnds.add(normalizeBound(ends[index], dimensionSize));
            normalizedAxes.add(axis);
            normalizedSteps.add(step);
        }
        return new SliceAttrs(
                normalizedStarts, normalizedEnds, normalizedAxes, normalizedSteps);
    }

    /**
     * Normalizes one raw bound once against a selected static extent, then clamps it.
     *
     * @param rawBound any signed long request coordinate
     * @param dimensionSize non-negative selected static dimension extent
     * @return normalized bound in the inclusive range {@code [0, dimensionSize]}
     */
    private static long normalizeBound(long rawBound, long dimensionSize) {
        long normalizedBound = rawBound;
        if (normalizedBound < 0) {
            normalizedBound += dimensionSize;
        }
        if (normalizedBound < 0) {
            return 0;
        }
        if (normalizedBound > dimensionSize) {
            return dimensionSize;
        }
        return normalizedBound;
    }

    /**
     * Calculates the number of coordinates in one normalized positive-step half-open interval.
     *
     * @param start normalized inclusive start
     * @param end normalized exclusive end
     * @param step strictly positive coordinate increment
     * @return zero when {@code start >= end}; otherwise the overflow-safe positive extent
     */
    private static long sliceExtent(long start, long end, long step) {
        if (start >= end) {
            return 0;
        }
        return 1L + (end - 1L - start) / step;
    }

    /**
     * Derives a same-rank Shape while preserving exact unselected Dimension references.
     *
     * <p>Each selected axis is replaced by a new static extent. For input Shape {@code [3, 6]},
     * bounds {@code [0, 1]} to {@code [3, 6]} on axes {@code [0, 1]} with steps {@code [1, 2]}
     * produce Shape {@code [3, 3]}. Empty attributes preserve all references; a reversed normalized
     * interval creates extent zero.</p>
     *
     * @param inputShape non-null exact input Shape
     * @param attrs non-null normalized slice attributes
     * @return non-null same-rank Shape with new selected static extents
     */
    private static Shape deriveShape(Shape inputShape, SliceAttrs attrs) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < resultDimensions.length; axis++) {
            resultDimensions[axis] = inputShape.dimensions().get(axis);
        }
        for (int index = 0; index < attrs.axes().size(); index++) {
            int axis = attrs.axes().get(index);
            long extent = sliceExtent(
                    attrs.starts().get(index),
                    attrs.ends().get(index),
                    attrs.steps().get(index));
            resultDimensions[axis] = new StaticDimension(extent);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Derives checked positive-step view geometry for a resolved, non-empty result.
     *
     * <p>The input layout optional and strides are each copied once. Every resolved layout kind is
     * accepted. Each selected start advances the exact input offset by start times that axis's
     * original stride, and each result stride is the original stride times the positive step.
     * {@link LayoutDescriptor} reclassifies the geometry and marks it as a view. An unresolved
     * input or a known zero element count yields unresolved result layout because an empty result
     * references no storage element and needs no arbitrary one-past-end geometry.</p>
     *
     * @param inputDescriptor non-null exact input descriptor supplying optional geometry
     * @param resultShape non-null derived same-rank result Shape
     * @param attrs non-null normalized slice attributes
     * @return non-null optional containing one new view layout, or empty for unresolved/empty cases
     * @throws ArithmeticException if checked offset, stride, classification, or span arithmetic
     *     overflows
     */
    private static Optional<LayoutDescriptor> resolveViewLayout(
            TensorDescriptor inputDescriptor, Shape resultShape, SliceAttrs attrs) {
        Optional<LayoutDescriptor> inputLayout = inputDescriptor.layout();
        if (inputLayout.isEmpty()) {
            return Optional.empty();
        }
        if (resultShape.knownElementCount().orElseThrow() == 0L) {
            return Optional.empty();
        }

        LayoutDescriptor resolvedInputLayout = inputLayout.orElseThrow();
        long[] resultStrides = resolvedInputLayout.strides();
        long resultOffset = resolvedInputLayout.storageOffset();
        for (int index = 0; index < attrs.axes().size(); index++) {
            int axis = attrs.axes().get(index);
            long inputStride = resultStrides[axis];
            resultOffset = Math.addExact(
                    resultOffset,
                    Math.multiplyExact(attrs.starts().get(index), inputStride));
            resultStrides[axis] = Math.multiplyExact(inputStride, attrs.steps().get(index));
        }
        return Optional.of(LayoutDescriptor.of(resultShape, resultStrides, resultOffset, true));
    }

    /**
     * Creates exact descriptor, slice semantics, provenance, and one fresh Tensor.
     *
     * <p>Data type and gradient eligibility are retained. The result has the supplied Shape and
     * resolved-or-unresolved layout, absent label and storage, exact {@link SliceKind#SLICE} with
     * the normalized attributes, and ordered provenance {@code [input]}. The central derived
     * factory is called exactly once, including for identity, repeated, nested, and empty slices.</p>
     *
     * @param input non-null exact sole provenance input
     * @param inputDescriptor non-null descriptor supplying exact type and eligibility
     * @param resultShape non-null derived result Shape
     * @param resultLayout non-null resolved-view or unresolved-layout optional
     * @param attrs non-null exact normalized attributes retained by the Operation
     * @return non-null fresh factory-derived Tensor without label or storage
     * @throws IllegalArgumentException if completed descriptor invariants reject supplied metadata
     * @throws ArithmeticException if descriptor layout reconstruction arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at factory delegation
     */
    private static Tensor create(
            Tensor input,
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            Optional<LayoutDescriptor> resultLayout,
            SliceAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                resultShape,
                resultLayout,
                inputDescriptor.requiresGrad());
        Operation operation = new Operation(SliceKind.SLICE, attrs);
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}
