package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
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
 * Constructs locally validated functional slice updates and target-relative crop expressions.
 *
 * <p>Slice update maps a same-rank update into normalized signed coordinate sequences of a base
 * and retains the exact base Shape. Target-relative crop selects an exact target Shape after the
 * per-axis extents in a prefix Shape. Static bounds are proved locally; unresolved upper bounds
 * remain obligations for later binding or execution.</p>
 *
 * <p>Both transformations create storage-free metadata and exact producer provenance. This
 * field-free helper reads no values or storage, mutates no input, promises no view or copy, and
 * performs no binding, graph capture, gradient construction, lowering, or execution.</p>
 */
final class TensorSlicePlacementExpressions {
    /** Prevents instantiation because slice-placement construction owns no state. */
    private TensorSlicePlacementExpressions() {
    }

    /**
     * Creates one functional signed, strided, multi-axis slice update.
     *
     * <p>References are null-checked in parameter order, array lengths are checked, and all three
     * arrays are cloned in declaration order before descriptors are read. Selected update
     * Dimensions must be static and supply the finite sequence lengths. Negative starts normalize
     * once against static base extents; unresolved base extents permit only non-negative starts
     * and defer upper-bound proof. Every local failure precedes Tensor identifier allocation.</p>
     *
     * @param base the non-null base Tensor retained as provenance input zero and never mutated
     * @param update the non-null same-type, same-rank update Tensor retained as input one
     * @param starts the non-null caller-owned raw inclusive starts paired by entry; never retained
     * @param axes the non-null caller-owned positive or negative base axes paired by entry
     * @param steps the non-null caller-owned signed non-zero coordinate increments paired by entry
     * @return a non-null fresh, unlabeled, storage-free SLICE_UPDATE Tensor with exact base Shape
     *     and type, unresolved layout, combined gradient eligibility, and output-index-zero
     *     provenance
     * @throws NullPointerException if a reference is null, checked in parameter order with the
     *     parameter name as the message
     * @throws IllegalArgumentException if array lengths, data types, ranks, an axis, duplicate
     *     axis, step, selected update extent, start, coordinate sequence, or complete update Shape
     *     violates the functional-update contract
     * @throws ArithmeticException if checked coordinate arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted at final construction
     */
    static Tensor update(
            Tensor base, Tensor update, long[] starts, int[] axes, long[] steps) {
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(update, "update");
        Objects.requireNonNull(starts, "starts");
        Objects.requireNonNull(axes, "axes");
        Objects.requireNonNull(steps, "steps");
        if (starts.length != axes.length || starts.length != steps.length) {
            throw new IllegalArgumentException(
                    "starts, axes, and steps must have matching lengths");
        }

        long[] privateStarts = starts.clone();
        int[] privateAxes = axes.clone();
        long[] privateSteps = steps.clone();
        TensorDescriptor baseDescriptor = base.descriptor();
        TensorDescriptor updateDescriptor = update.descriptor();
        if (baseDescriptor.dataType() != updateDescriptor.dataType()) {
            throw new IllegalArgumentException(
                    "slice update data types must match: base=" + baseDescriptor.dataType()
                            + ", update=" + updateDescriptor.dataType());
        }
        Shape baseShape = baseDescriptor.shape();
        Shape updateShape = updateDescriptor.shape();
        if (updateShape.rank() != baseShape.rank()) {
            throw new IllegalArgumentException(
                    "slice update rank must match base rank: base=" + baseShape.rank()
                            + ", update=" + updateShape.rank());
        }

        SliceAttrs attrs = normalizeUpdate(
                baseShape, updateShape, privateStarts, privateAxes, privateSteps);
        Shape expected = expectedUpdateShape(baseShape, updateShape, attrs);
        if (!updateShape.equals(expected)) {
            throw new IllegalArgumentException(
                    "slice update shape must match base Shape with selected axes replaced: "
                            + "expected=" + expected + ", actual=" + updateShape);
        }
        return createUpdate(base, update, baseDescriptor, updateDescriptor, attrs);
    }

    /**
     * Creates one target-relative crop whose result retains the exact target Shape reference.
     *
     * <p>Input, target, and prefix ranks must match. Each axis whose three extents are static is
     * checked with exact addition; any axis involving an unresolved extent retains its bound as a
     * later obligation. No crop coordinate is clamped, shifted, inferred, or bound.</p>
     *
     * @param input the non-null source Tensor retained as the sole provenance input and not mutated
     * @param targetShape the non-null exact logical result Shape retained in attributes/descriptor
     * @param prefixShape the non-null exact per-axis prefix-extent Shape retained in attributes
     * @return a non-null fresh, unlabeled, storage-free SLICE Tensor with exact target Shape, input
     *     type and eligibility, unresolved layout, and output-index-zero provenance
     * @throws NullPointerException if a reference is null, checked in parameter order with the
     *     parameter name as the message
     * @throws IllegalArgumentException if target or prefix rank differs from input rank, or the
     *     first fully static crop region exceeds its input extent
     * @throws ArithmeticException if checked static prefix-plus-target addition overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted at final construction
     */
    static Tensor cropToShape(Tensor input, Shape targetShape, Shape prefixShape) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(targetShape, "targetShape");
        Objects.requireNonNull(prefixShape, "prefixShape");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        if (targetShape.rank() != inputShape.rank()) {
            throw new IllegalArgumentException(
                    "crop target rank must match input rank: input=" + inputShape.rank()
                            + ", target=" + targetShape.rank());
        }
        if (prefixShape.rank() != inputShape.rank()) {
            throw new IllegalArgumentException(
                    "crop prefix rank must match input rank: input=" + inputShape.rank()
                            + ", prefix=" + prefixShape.rank());
        }
        validateStaticCropBounds(inputShape, targetShape, prefixShape);
        CropToShapeAttrs attrs = new CropToShapeAttrs(targetShape, prefixShape);
        return createCrop(input, inputDescriptor, attrs);
    }

    /**
     * Normalizes update entries in caller order and creates one exact immutable attributes value.
     *
     * @param baseShape the non-null exact base Shape
     * @param updateShape the non-null exact same-rank update Shape
     * @param starts private raw inclusive-start snapshot
     * @param axes private raw-axis snapshot
     * @param steps private signed-step snapshot
     * @return one non-null SliceAttrs containing normalized starts, derived lengths, axes, and steps
     * @throws IllegalArgumentException if an entry violates the documented local contract
     * @throws ArithmeticException if checked coordinate arithmetic overflows
     */
    private static SliceAttrs normalizeUpdate(
            Shape baseShape,
            Shape updateShape,
            long[] starts,
            int[] axes,
            long[] steps) {
        ArrayList<Long> normalizedStarts = new ArrayList<>(starts.length);
        ArrayList<Long> lengths = new ArrayList<>(starts.length);
        ArrayList<Integer> normalizedAxes = new ArrayList<>(starts.length);
        ArrayList<Long> normalizedSteps = new ArrayList<>(starts.length);
        boolean[] seenAxes = new boolean[baseShape.rank()];

        for (int index = 0; index < starts.length; index++) {
            int rawAxis = axes[index];
            long normalizedAxis = rawAxis;
            if (normalizedAxis < 0) {
                normalizedAxis += baseShape.rank();
            }
            if (normalizedAxis < 0 || normalizedAxis >= baseShape.rank()) {
                throw new IllegalArgumentException(
                        "slice update axis " + rawAxis + " at index " + index
                                + " is outside rank " + baseShape.rank());
            }
            int axis = (int) normalizedAxis;
            if (seenAxes[axis]) {
                throw new IllegalArgumentException(
                        "slice update contains duplicate normalized axis " + axis
                                + " at index " + index);
            }
            seenAxes[axis] = true;
            long step = steps[index];
            if (step == 0) {
                throw new IllegalArgumentException(
                        "steps[" + index + "] must be non-zero: 0");
            }
            Dimension updateDimension = updateShape.dimension(axis);
            if (!(updateDimension instanceof StaticDimension updateStatic)) {
                throw new IllegalArgumentException(
                        "slice update axis " + axis + " at index " + index
                                + " must have a statically known update dimension");
            }
            long length = updateStatic.size();
            long start = normalizeUpdateStart(
                    starts[index], baseShape.dimension(axis), axis, index, length);
            if (length > 0) {
                long last = Math.addExact(
                        start,
                        Math.multiplyExact(Math.subtractExact(length, 1L), step));
                Dimension baseDimension = baseShape.dimension(axis);
                boolean outside = start < 0 || last < 0;
                Object extent = baseDimension;
                if (baseDimension instanceof StaticDimension baseStatic) {
                    extent = baseStatic.size();
                    outside = outside || start >= baseStatic.size() || last >= baseStatic.size();
                }
                if (outside) {
                    throw new IllegalArgumentException(
                            "slice update coordinates at index " + index
                                    + " do not fit base extent " + extent + ": start=" + start
                                    + ", length=" + length + ", step=" + step);
                }
            }
            normalizedStarts.add(start);
            lengths.add(length);
            normalizedAxes.add(axis);
            normalizedSteps.add(step);
        }
        return new SliceAttrs(
                normalizedStarts, lengths, normalizedAxes, normalizedSteps);
    }

    /**
     * Canonicalizes an empty selection or normalizes one raw start exactly once.
     *
     * @param rawStart caller-supplied inclusive start
     * @param baseDimension exact selected base Dimension
     * @param axis normalized selected base axis used in diagnostics
     * @param index caller entry index used in diagnostics
     * @param length non-negative selected update extent
     * @return zero for an empty selection; otherwise the normalized start
     * @throws IllegalArgumentException if a negative start targets an unresolved base Dimension
     */
    private static long normalizeUpdateStart(
            long rawStart, Dimension baseDimension, int axis, int index, long length) {
        if (length == 0) {
            return 0;
        }
        if (rawStart >= 0) {
            return rawStart;
        }
        if (!(baseDimension instanceof StaticDimension baseStatic)) {
            throw new IllegalArgumentException(
                    "slice update start " + rawStart + " at index " + index
                            + " cannot be negative for dynamic base axis " + axis);
        }
        return Math.addExact(rawStart, baseStatic.size());
    }

    /**
     * Builds the exact required update Shape from base and selected update Dimensions.
     *
     * @param baseShape exact base Shape supplying every unselected Dimension reference
     * @param updateShape exact update Shape supplying selected static Dimension references
     * @param attrs normalized slice entries identifying selected axes
     * @return a non-null same-rank Shape used for complete structural equality validation
     */
    private static Shape expectedUpdateShape(
            Shape baseShape, Shape updateShape, SliceAttrs attrs) {
        Dimension[] dimensions = new Dimension[baseShape.rank()];
        for (int axis = 0; axis < baseShape.rank(); axis++) {
            dimensions[axis] = baseShape.dimension(axis);
        }
        for (int index = 0; index < attrs.axes().size(); index++) {
            int axis = attrs.axes().get(index);
            dimensions[axis] = updateShape.dimension(axis);
        }
        return Shape.ofDimensions(dimensions);
    }

    /**
     * Validates every crop bound whose input, prefix, and target extents are all static.
     *
     * @param inputShape exact source Shape
     * @param targetShape exact result Shape of equal rank
     * @param prefixShape exact per-axis prefix Shape of equal rank
     * @throws IllegalArgumentException if the first fully static crop exceeds the input extent
     * @throws ArithmeticException if checked prefix-plus-target addition overflows
     */
    private static void validateStaticCropBounds(
            Shape inputShape, Shape targetShape, Shape prefixShape) {
        for (int axis = 0; axis < inputShape.rank(); axis++) {
            Dimension inputDimension = inputShape.dimension(axis);
            Dimension prefixDimension = prefixShape.dimension(axis);
            Dimension targetDimension = targetShape.dimension(axis);
            if (inputDimension instanceof StaticDimension inputStatic
                    && prefixDimension instanceof StaticDimension prefixStatic
                    && targetDimension instanceof StaticDimension targetStatic
                    && Math.addExact(prefixStatic.size(), targetStatic.size())
                            > inputStatic.size()) {
                throw new IllegalArgumentException(
                        "crop region exceeds input extent at axis " + axis + ": input="
                                + inputStatic.size() + ", prefix=" + prefixStatic.size()
                                + ", target=" + targetStatic.size());
            }
        }
    }

    /**
     * Constructs exact functional-update metadata and one fresh derived Tensor.
     *
     * @param base validated exact provenance input zero and result-Shape source
     * @param update validated exact provenance input one
     * @param baseDescriptor exact base descriptor supplying type, Shape, and eligibility
     * @param updateDescriptor exact update descriptor supplying eligibility
     * @param attrs exact normalized finite coordinate sequences
     * @return the non-null fresh storage-free SLICE_UPDATE Tensor
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    private static Tensor createUpdate(
            Tensor base,
            Tensor update,
            TensorDescriptor baseDescriptor,
            TensorDescriptor updateDescriptor,
            SliceAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                baseDescriptor.dataType(),
                baseDescriptor.shape(),
                Optional.empty(),
                baseDescriptor.requiresGrad() || updateDescriptor.requiresGrad());
        Operation operation = new Operation(SliceKind.SLICE_UPDATE, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(base, update));
    }

    /**
     * Constructs exact target-relative extraction metadata and one fresh derived Tensor.
     *
     * @param input validated exact sole provenance input
     * @param inputDescriptor exact input descriptor supplying type and eligibility
     * @param attrs exact target/prefix Shapes retained by the operation
     * @return the non-null fresh storage-free SLICE Tensor with exact target Shape
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    private static Tensor createCrop(
            Tensor input, TensorDescriptor inputDescriptor, CropToShapeAttrs attrs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                attrs.targetShape(),
                Optional.empty(),
                inputDescriptor.requiresGrad());
        Operation operation = new Operation(SliceKind.SLICE, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(input));
    }
}
