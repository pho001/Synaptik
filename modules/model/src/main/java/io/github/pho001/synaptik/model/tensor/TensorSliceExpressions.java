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
 * Constructs locally normalized, storage-free signed-step slice expressions for {@link Tensor}.
 *
 * <p>The raw-bound path normalizes one directional half-open interval against each selected static
 * input Dimension and calculates its finite length. The length-defined path instead accepts
 * already normalized non-negative starts and explicit finite lengths. It therefore accepts an
 * unresolved selected Dimension after proving the first and final coordinates non-negative and
 * defers only the upper bound against that unresolved extent. Neither path stores a deferred
 * constraint object.</p>
 *
 * <p>Both paths normalize caller-facing axes once, require distinct axes and signed non-zero
 * steps, preserve exact unselected Dimension references, and treat empty arrays as a fresh
 * explicit identity slice. A zero-length entry stores canonical start zero because it selects no
 * coordinate.</p>
 *
 * <p>For a non-empty result with resolved input geometry, construction advances the element
 * offset by each normalized start times the original input stride and multiplies selected strides
 * by their positive steps using checked arithmetic. Every resolved input layout kind is accepted
 * and the result is marked as a logical view. Unresolved input geometry, empty results, and every
 * request containing a negative step remain layout-unresolved because the current descriptor
 * forbids negative strides. This metadata neither attaches storage nor chooses a reverse copy,
 * physical alias, materialization, gradient, compiler, backend, or execution behavior.</p>
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
     * @param steps non-null caller-owned signed non-zero steps, paired by entry
     * @return a non-null fresh unlabeled, storage-free SLICE tensor with normalized attributes
     * @throws NullPointerException if any reference is null, with its parameter name as message
     * @throws IllegalArgumentException if array lengths differ, an axis is invalid or repeated, a
     *     step is zero, or a selected dimension is dynamic
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
     * Validates normalized starts and finite lengths, then creates a fresh slice expression.
     *
     * <p>The four caller-owned arrays are null-checked and length-checked before they are cloned
     * once each in parameter order; no array is retained or mutated. Entries are checked in caller
     * order: axis range, duplicate normalized axis, non-negative start, non-negative length,
     * non-zero step, then coordinate fit. A non-empty entry uses checked
     * {@code start + (length - 1) * step}; its first and final coordinates must be non-negative.
     * Static selected extents are checked completely, while only the upper bound against an
     * unresolved selected extent is deferred. A zero-length entry performs no coordinate-bound
     * proof and stores canonical start zero.</p>
     *
     * <p>The result has static selected lengths and exact unaffected Dimension references.
     * Resolved view layout is derived only for a non-empty result, resolved input geometry, and
     * all-positive steps. Every failure precedes final derived creation and consumes no Tensor
     * identifier; a success creates one fresh canonical output wrapper.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input; not mutated
     * @param starts non-null caller-owned normalized non-negative inclusive starts; cloned and
     *     never retained or mutated
     * @param lengths non-null caller-owned finite non-negative coordinate counts; cloned and
     *     never retained or mutated
     * @param axes non-null caller-owned positive or negative axes; cloned and normalized once
     * @param steps non-null caller-owned signed non-zero coordinate increments; cloned unchanged
     * @return a non-null fresh canonical, unlabeled, storage-free SLICE tensor with static selected
     *     lengths, exact input type and eligibility, and output-index-zero provenance
     * @throws NullPointerException if a reference is null, checked in parameter order
     * @throws IllegalArgumentException if lengths differ or an entry violates the local contract
     * @throws ArithmeticException if checked coordinate, Shape, or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor applyByLength(
            Tensor input, long[] starts, long[] lengths, int[] axes, long[] steps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(starts, "starts");
        Objects.requireNonNull(lengths, "lengths");
        Objects.requireNonNull(axes, "axes");
        Objects.requireNonNull(steps, "steps");
        if (starts.length != lengths.length
                || starts.length != axes.length
                || starts.length != steps.length) {
            throw new IllegalArgumentException(
                    "starts, lengths, axes, and steps must have matching lengths");
        }

        long[] privateStarts = starts.clone();
        long[] privateLengths = lengths.clone();
        int[] privateAxes = axes.clone();
        long[] privateSteps = steps.clone();
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        SliceAttrs attrs = normalizeByLength(
                inputShape, privateStarts, privateLengths, privateAxes, privateSteps);
        Shape resultShape = deriveShape(inputShape, attrs);
        Optional<LayoutDescriptor> resultLayout =
                resolveViewLayout(inputDescriptor, resultShape, attrs);
        return create(input, inputDescriptor, resultShape, resultLayout, attrs);
    }

    /**
     * Creates a one-axis signed-step request through the general construction path.
     *
     * <p>No independent normalization or semantic kind exists. The four private one-element
     * arrays make this exactly one {@link SliceKind#SLICE} occurrence with the supplied step.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input
     * @param axis raw positive or negative selected input axis
     * @param fromInclusive raw inclusive start normalized and clamped by the general path
     * @param toExclusive raw exclusive end normalized and clamped by the general path
     * @param step signed non-zero distance between selected coordinates
     * @return a non-null fresh one-axis SLICE expression
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalArgumentException if the axis is invalid, selected dimension is dynamic, or
     *     {@code step} is zero, or the delegated general request is otherwise invalid
     * @throws ArithmeticException if checked result element-count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor applyAxis(
            Tensor input, int axis, long fromInclusive, long toExclusive, long step) {
        return apply(
                input,
                new long[] {fromInclusive},
                new long[] {toExclusive},
                new int[] {axis},
                new long[] {step});
    }

    /**
     * Creates one normalized negative-step slice that reverses all requested axes.
     *
     * <p>The caller array is cloned once and processed in order. Negative axes add rank once; the
     * first invalid or repeated normalized axis fails. A selected static extent {@code D > 0}
     * contributes start {@code D - 1}, length {@code D}, and step {@code -1}; a zero extent uses
     * canonical start and length zero. Empty axes are an explicit identity flip, including for a
     * scalar, and still create one fresh {@link SliceKind#SLICE} occurrence.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input; not mutated
     * @param axes non-null caller-owned positive or negative axes; cloned and never retained
     * @return a non-null fresh unlabeled, storage-free SLICE tensor; layout is unresolved whenever
     *     at least one axis is selected
     * @throws NullPointerException if {@code input} or {@code axes} is null, in that order, with
     *     the parameter name as message
     * @throws IllegalArgumentException if an axis is outside rank, repeated after normalization,
     *     or selects a dynamic dimension
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor flip(Tensor input, int[] axes) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(axes, "axes");
        int[] privateAxes = axes.clone();
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        int rank = inputShape.rank();
        boolean[] seenAxes = new boolean[rank];
        List<Long> starts = new ArrayList<>(privateAxes.length);
        List<Long> lengths = new ArrayList<>(privateAxes.length);
        List<Integer> normalizedAxes = new ArrayList<>(privateAxes.length);
        List<Long> steps = new ArrayList<>(privateAxes.length);
        for (int index = 0; index < privateAxes.length; index++) {
            int rawAxis = privateAxes[index];
            long normalizedAxis = rawAxis;
            if (normalizedAxis < 0) {
                normalizedAxis += rank;
            }
            if (normalizedAxis < 0 || normalizedAxis >= rank) {
                throw new IllegalArgumentException(
                        "flip axis " + rawAxis + " at index " + index
                                + " is outside rank " + rank);
            }
            int axis = (int) normalizedAxis;
            if (seenAxes[axis]) {
                throw new IllegalArgumentException(
                        "flip contains duplicate normalized axis " + axis
                                + " at index " + index);
            }
            seenAxes[axis] = true;
            Dimension selectedDimension = inputShape.dimensions().get(axis);
            if (!(selectedDimension instanceof StaticDimension staticDimension)) {
                throw new IllegalArgumentException(
                        "flip axis " + axis + " at index " + index
                                + " must have a statically known dimension");
            }
            long dimensionSize = staticDimension.size();
            starts.add(dimensionSize == 0 ? 0L : Math.subtractExact(dimensionSize, 1L));
            lengths.add(dimensionSize);
            normalizedAxes.add(axis);
            steps.add(-1L);
        }
        SliceAttrs attrs = new SliceAttrs(starts, lengths, normalizedAxes, steps);
        Shape resultShape = deriveShape(inputShape, attrs);
        Optional<LayoutDescriptor> resultLayout =
                resolveViewLayout(inputDescriptor, resultShape, attrs);
        return create(input, inputDescriptor, resultShape, resultLayout, attrs);
    }

    /**
     * Normalizes axes and bounds and validates one private general request.
     *
     * <p>Entries remain in caller order. A negative axis adds rank once using {@code long}; a
     * negative bound adds the selected static dimension size once and is then clamped according to
     * direction and whether it is a start or exclusive end. Duplicate detection uses normalized
     * axes. A negative-step zero extent bypasses bound arithmetic and emits canonical empty state.
     * Exactly one immutable {@link SliceAttrs} snapshot is created after all entries pass.</p>
     *
     * @param inputShape non-null exact input Shape used for rank and selected dimensions
     * @param starts non-null private inclusive starts with matching length
     * @param ends non-null private exclusive ends with matching length
     * @param axes non-null private raw axes with matching length
     * @param steps non-null private steps with matching length
     * @return non-null normalized immutable attributes in original request order
     * @throws IllegalArgumentException if an axis is outside rank or repeated after normalization,
     *     a step is zero, or a selected dimension is not static
     * @throws ArithmeticException if checked bound or length arithmetic overflows
     */
    private static SliceAttrs normalize(
            Shape inputShape, long[] starts, long[] ends, int[] axes, long[] steps) {
        int rank = inputShape.rank();
        boolean[] seenAxes = new boolean[rank];
        List<Long> normalizedStarts = new ArrayList<>(starts.length);
        List<Long> normalizedLengths = new ArrayList<>(ends.length);
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
            if (step == 0) {
                throw new IllegalArgumentException(
                        "steps[" + index + "] must be non-zero: " + step);
            }

            Dimension selectedDimension = inputShape.dimensions().get(axis);
            if (!(selectedDimension instanceof StaticDimension staticDimension)) {
                throw new IllegalArgumentException(
                        "slice axis " + axis + " at index " + index
                                + " must have a statically known dimension");
            }
            long dimensionSize = staticDimension.size();
            if (step < 0 && dimensionSize == 0) {
                normalizedStarts.add(0L);
                normalizedLengths.add(0L);
                normalizedAxes.add(axis);
                normalizedSteps.add(step);
                continue;
            }
            long start = normalizeBound(starts[index], dimensionSize, step, true);
            long end = normalizeBound(ends[index], dimensionSize, step, false);
            long length = sliceLength(start, end, step);
            normalizedStarts.add(length == 0 ? 0L : start);
            normalizedLengths.add(length);
            normalizedAxes.add(axis);
            normalizedSteps.add(step);
        }
        return new SliceAttrs(
                normalizedStarts, normalizedLengths, normalizedAxes, normalizedSteps);
    }

    /**
     * Normalizes axes and validates one private normalized-start, finite-length request.
     *
     * <p>Entries are processed in ascending caller index. Axis and duplicate checks precede
     * start, length, step, and coordinate checks at the same entry. Non-empty coordinate
     * arithmetic is exact. Zero length stores start zero and does not inspect the selected
     * extent for bounds. One immutable attributes value is created only after every entry
     * succeeds.</p>
     *
     * @param inputShape non-null exact input Shape used for rank and selected dimensions
     * @param starts non-null private normalized non-negative starts with matching length
     * @param lengths non-null private finite non-negative lengths with matching length
     * @param axes non-null private raw positive or negative axes with matching length
     * @param steps non-null private signed non-zero steps with matching length
     * @return one non-null immutable normalized {@link SliceAttrs} value whose empty starts are
     *     canonical zero
     * @throws IllegalArgumentException if an entry violates axis, coordinate, or static-fit rules
     * @throws ArithmeticException if final-coordinate arithmetic overflows
     */
    private static SliceAttrs normalizeByLength(
            Shape inputShape, long[] starts, long[] lengths, int[] axes, long[] steps) {
        int rank = inputShape.rank();
        boolean[] seenAxes = new boolean[rank];
        List<Long> normalizedStarts = new ArrayList<>(starts.length);
        List<Long> normalizedLengths = new ArrayList<>(lengths.length);
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
                        "slice by length axis " + rawAxis + " at index " + index
                                + " is outside rank " + rank);
            }
            int axis = (int) normalizedAxis;
            if (seenAxes[axis]) {
                throw new IllegalArgumentException(
                        "slice by length contains duplicate normalized axis " + axis
                                + " at index " + index);
            }
            seenAxes[axis] = true;

            long start = starts[index];
            if (start < 0) {
                throw new IllegalArgumentException(
                        "starts[" + index + "] must be non-negative: " + start);
            }
            long length = lengths[index];
            if (length < 0) {
                throw new IllegalArgumentException(
                        "lengths[" + index + "] must be non-negative: " + length);
            }
            long step = steps[index];
            if (step == 0) {
                throw new IllegalArgumentException(
                        "steps[" + index + "] must be non-zero: 0");
            }

            Dimension selectedDimension = inputShape.dimensions().get(axis);
            if (length > 0) {
                long last = Math.addExact(
                        start,
                        Math.multiplyExact(Math.subtractExact(length, 1L), step));
                boolean outside = last < 0;
                Object extent = selectedDimension;
                if (selectedDimension instanceof StaticDimension staticDimension) {
                    extent = staticDimension.size();
                    outside = outside
                            || start >= staticDimension.size()
                            || last >= staticDimension.size();
                }
                if (outside) {
                    throw new IllegalArgumentException(
                            "slice by length coordinates at index " + index
                                    + " do not fit input extent " + extent + ": start=" + start
                                    + ", length=" + length + ", step=" + step);
                }
            }

            normalizedStarts.add(length == 0 ? 0L : start);
            normalizedLengths.add(length);
            normalizedAxes.add(axis);
            normalizedSteps.add(step);
        }
        return new SliceAttrs(
                normalizedStarts, normalizedLengths, normalizedAxes, normalizedSteps);
    }

    /**
     * Normalizes one raw bound once against a selected static extent, then clamps it.
     *
     * @param rawBound any signed long request coordinate
     * @param dimensionSize positive selected static dimension extent for negative steps, or any
     *     non-negative selected static extent for positive steps
     * @param step signed non-zero coordinate increment selecting the clamp direction
     * @param startBound true for an inclusive start, false for an exclusive end
     * @return positive-step bound in {@code [0, dimensionSize]}; negative-step start in
     *     {@code [0, dimensionSize - 1]}; or negative-step end in
     *     {@code [-1, dimensionSize - 1]}
     * @throws ArithmeticException if checked normalization or upper-bound arithmetic overflows
     */
    private static long normalizeBound(
            long rawBound, long dimensionSize, long step, boolean startBound) {
        long normalizedBound = rawBound;
        if (normalizedBound < 0) {
            normalizedBound = Math.addExact(normalizedBound, dimensionSize);
        }
        if (step > 0) {
            if (normalizedBound < 0) {
                return 0;
            }
            if (normalizedBound > dimensionSize) {
                return dimensionSize;
            }
            return normalizedBound;
        }
        long upperBound = Math.subtractExact(dimensionSize, 1L);
        long lowerBound = startBound ? 0L : -1L;
        if (normalizedBound < lowerBound) {
            return lowerBound;
        }
        return Math.min(normalizedBound, upperBound);
    }

    /**
     * Calculates the number of coordinates in one normalized directional half-open request.
     *
     * @param start normalized inclusive start
     * @param end normalized exclusive end
     * @param step signed non-zero coordinate increment
     * @return zero when the start is empty in the step's direction; otherwise the checked positive
     *     selected-coordinate count
     * @throws ArithmeticException if checked numerator or result arithmetic overflows
     */
    private static long sliceLength(long start, long end, long step) {
        if (step > 0) {
            if (start >= end) {
                return 0;
            }
            long numerator = Math.subtractExact(
                    Math.subtractExact(end, 1L), start);
            return Math.addExact(1L, numerator / step);
        }
        if (start <= end) {
            return 0;
        }
        long numerator = Math.subtractExact(
                Math.subtractExact(start, 1L), end);
        return Math.subtractExact(1L, numerator / step);
    }

    /**
     * Derives a same-rank Shape while preserving exact unselected Dimension references.
     *
     * <p>Each selected axis is replaced by a new static extent. For input Shape {@code [3, 6]},
     * starts {@code [0, 4]}, lengths {@code [3, 5]}, axes {@code [0, 1]}, and steps
     * {@code [1, -1]} produce Shape {@code [3, 5]}. Empty attributes preserve all references.</p>
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
            resultDimensions[axis] = new StaticDimension(attrs.lengths().get(index));
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Derives checked positive-step view geometry when the complete result is representable.
     *
     * <p>The input layout optional and strides are each copied once. Every resolved layout kind is
     * accepted. Each selected start advances the exact input offset by start times that axis's
     * original stride, and each result stride is the original stride times the positive step.
     * {@link LayoutDescriptor} reclassifies the geometry and marks it as a view. An unresolved
     * input or a known zero element count yields unresolved result layout because an empty result
     * references no storage element and needs no arbitrary one-past-end geometry. Any negative
     * step also yields unresolved layout because {@link LayoutDescriptor} accepts only
     * non-negative strides; this method does not choose a copy or reverse kernel.</p>
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
        for (long step : attrs.steps()) {
            if (step < 0) {
                return Optional.empty();
            }
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
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
