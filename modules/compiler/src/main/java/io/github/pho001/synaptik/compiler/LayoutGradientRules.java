package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;

/**
 * Builds the closed logical layout, slice, composition, and window first-order formulas.
 *
 * <p>The selected rows are {@code CONTIGUOUS}, {@code RESHAPE}, {@code EXPAND},
 * {@code EXPAND_DIMS}, {@code SQUEEZE}, {@code PERMUTE}, both {@code SLICE}/{@code
 * SLICE_UPDATE} attribute forms, {@code SELECT}, {@code PAD}, {@code TILE}, {@code CONCAT},
 * {@code STACK}, and the axis, two-dimensional, and three-dimensional window transforms.
 * Formulas reverse only
 * logical Shape, axis, coordinate, composition, and window metadata through public Tensor
 * operations; they neither select physical aliasing or materialization nor read storage, lower
 * work, or execute computation. Preflight owns type, Shape, attributes, role, local
 * constructibility, and policy validation.</p>
 *
 * <p>{@code SLICE} writes {@code g} into an exact typed zero shaped like the input.
 * {@code SLICE_UPDATE} writes a zero shaped like the update into {@code g} for the base
 * cotangent, and extracts the stored finite coordinate sequence from {@code g} with
 * {@code sliceByLength} for the update cotangent. This uses the exact recorded lengths for empty,
 * positive-step, negative-step, and unresolved selected-base regions; it does not reconstruct a
 * raw end. The target-relative form uses the exact target and prefix Shapes for placement or
 * extraction.</p>
 *
 * <p>{@code SELECT} writes an axis-restored {@code g} into an exact typed zero, {@code PAD}
 * crops by the before-width prefix, and {@code TILE} interleaves repeat/input axes, sums the
 * repeat axes, then restores the input Shape. {@code CONCAT} crops each selected input from its
 * ordered symbolic prefix; {@code STACK} selects the corresponding inserted-axis coordinate.
 * Repeated input positions remain separate contributions for the reverse accumulator.
 * {@code UNFOLD_AXIS}/{@code FOLD_AXIS}, {@code UNFOLD2D}/{@code FOLD2D}, and
 * {@code UNFOLD3D}/{@code FOLD3D} use their exact public overlap-add or extraction counterpart;
 * typed unfold padding remains scalar metadata and receives no cotangent. Fold adjoints use the
 * direct positive-zero unfold overload so discarded padded and ceil-tail columns receive zero.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class LayoutGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    LayoutGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds the existing ordinary layout VJP with explicit primal value bindings.
     * @param producer exact original layout occurrence
     * @param gradient accumulated cotangent
     * @param selectedInputs positional route flags
     * @param primal total binding of original operands to lexical formals
     * @return ordered cotangents with null unselected entries
     * @throws IllegalStateException if the occurrence is outside the preflight-approved matrix
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            java.util.function.Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        if (producer.operation().kind() == ContiguousKind.CONTIGUOUS) {
            return ops.result(gradient);
        }
        if (producer.operation().kind() == ShapeTransformKind.RESHAPE) {
            return ops.result(ops.reshape(gradient, ops.descriptor(input).shape()));
        }
        if (producer.operation().kind() == ShapeTransformKind.EXPAND) {
            return ops.result(ops.sumToShape(gradient, ops.descriptor(input).shape()));
        }
        if (producer.operation().kind() == AxisTransformKind.EXPAND_DIMS) {
            int axis = ((AxisTransformAttrs) producer.operation().attrs()).axis();
            return ops.result(ops.squeeze(gradient, axis));
        }
        if (producer.operation().kind() == AxisTransformKind.SQUEEZE) {
            int axis = ((AxisTransformAttrs) producer.operation().attrs()).axis();
            return ops.result(ops.expandDims(gradient, axis));
        }
        if (producer.operation().kind() == AxisTransformKind.PERMUTE) {
            PermutationAttrs attrs = (PermutationAttrs) producer.operation().attrs();
            int[] inverse = new int[attrs.axes().size()];
            for (int outputAxis = 0; outputAxis < attrs.axes().size(); outputAxis++) {
                inverse[attrs.axes().get(outputAxis)] = outputAxis;
            }
            return ops.result(ops.permute(gradient, inverse));
        }
        if (producer.operation().kind() == SliceKind.SLICE
                && producer.operation().attrs() instanceof CropToShapeAttrs attrs) {
            return ops.result(ops.sliceUpdate(ops.zeroLike(input), gradient, attrs.prefixShape()));
        }
        if (producer.operation().kind() == SliceKind.SLICE) {
            SliceAttrs attrs = (SliceAttrs) producer.operation().attrs();
            return ops.result(ops.sliceUpdate(ops.zeroLike(input), gradient, attrs));
        }
        if (producer.operation().kind() == SliceKind.SLICE_UPDATE
                && producer.operation().attrs() instanceof CropToShapeAttrs attrs) {
            V update = primal.apply(producer.inputs().get(1));
            V baseGradient = selectedInputs[0]
                    ? ops.sliceUpdate(gradient, ops.zeroLike(update), attrs.prefixShape())
                    : null;
            V updateGradient = selectedInputs[1]
                    ? ops.cropToShape(gradient, attrs.targetShape(), attrs.prefixShape())
                    : null;
            return ops.result(baseGradient, updateGradient);
        }
        if (producer.operation().kind() == SliceKind.SLICE_UPDATE) {
            SliceAttrs attrs = (SliceAttrs) producer.operation().attrs();
            V update = primal.apply(producer.inputs().get(1));
            V baseGradient = selectedInputs[0]
                    ? ops.sliceUpdate(gradient, ops.zeroLike(update), attrs)
                    : null;
            V updateGradient = selectedInputs[1]
                    ? ops.sliceByLength(gradient, attrs)
                    : null;
            return ops.result(baseGradient, updateGradient);
        }
        if (producer.operation().kind() == SelectKind.SELECT) {
            SelectAttrs attrs = (SelectAttrs) producer.operation().attrs();
            Dimension[] prefix = new Dimension[ops.descriptor(input).shape().rank()];
            for (int axis = 0; axis < prefix.length; axis++) {
                prefix[axis] = new StaticDimension(axis == attrs.axis() ? attrs.index() : 0);
            }
            return ops.result(ops.sliceUpdate(ops.zeroLike(input), ops.expandDims(gradient, attrs.axis()), Shape.ofDimensions(prefix)));
        }
        if (producer.operation().kind() == PadKind.PAD) {
            PadAttrs attrs = (PadAttrs) producer.operation().attrs();
            return ops.result(ops.cropToShape(gradient, ops.descriptor(input).shape(), Shape.of(longs(attrs.before()))));
        }
        if (producer.operation().kind() == TileKind.TILE) {
            return ops.result(tileGradient(
                    input, gradient, (TileAttrs) producer.operation().attrs()));
        }
        if (producer.operation().kind() == TensorCompositionKind.CONCAT) {
            int axis = ((CompositionAxisAttrs) producer.operation().attrs()).axis();
            V[] result = ops.array(producer.inputs().size());
            Dimension prefix = new StaticDimension(0);
            for (int index = 0; index < producer.inputs().size(); index++) {
                V compositionInput = primal.apply(producer.inputs().get(index));
                if (selectedInputs[index]) {
                    Dimension[] prefixDimensions =
                            new Dimension[ops.descriptor(compositionInput).shape().rank()];
                    for (int candidate = 0; candidate < prefixDimensions.length; candidate++) {
                        prefixDimensions[candidate] =
                                candidate == axis ? prefix : new StaticDimension(0);
                    }
                    result[index] = ops.cropToShape(gradient, ops.descriptor(compositionInput).shape(), Shape.ofDimensions(prefixDimensions));
                }
                prefix = DimensionExpressions.add(
                        prefix, ops.descriptor(compositionInput).shape().dimension(axis));
            }
            return result;
        }
        if (producer.operation().kind() == TensorCompositionKind.STACK) {
            int axis = ((CompositionAxisAttrs) producer.operation().attrs()).axis();
            V[] result = ops.array(producer.inputs().size());
            for (int index = 0; index < result.length; index++) {
                if (selectedInputs[index]) {
                    result[index] = ops.select(gradient, axis, index);
                }
            }
            return result;
        }
        if (producer.operation().kind() == WindowTransformKind.UNFOLD_AXIS) {
            UnfoldAxisAttrs attrs = (UnfoldAxisAttrs) producer.operation().attrs();
            long outputSize = ((StaticDimension)
                            ops.descriptor(input).shape().dimension(attrs.axis()))
                    .size();
            return ops.result(ops.foldAxis(gradient, attrs.axis(), outputSize, attrs.step()));
        }
        if (producer.operation().kind() == WindowTransformKind.FOLD_AXIS) {
            FoldAxisAttrs attrs = (FoldAxisAttrs) producer.operation().attrs();
            long windowSize = ((StaticDimension) ops.descriptor(input)
                            .shape()
                            .dimension(ops.descriptor(input).shape().rank() - 1))
                    .size();
            return ops.result(ops.unfold(gradient, attrs.axis(), windowSize, attrs.step()));
        }
        if (producer.operation().kind() == WindowTransformKind.UNFOLD2D) {
            Window2dAttrs window = producer.operation().attrs() instanceof Unfold2dAttrs attrs
                    ? attrs.window()
                    : (Window2dAttrs) producer.operation().attrs();
            return ops.result(ops.fold2d(gradient, ops.descriptor(input).shape(), window));
        }
        if (producer.operation().kind() == WindowTransformKind.FOLD2D) {
            Fold2dAttrs attrs = (Fold2dAttrs) producer.operation().attrs();
            return ops.result(ops.unfold2d(gradient, attrs.window()));
        }
        if (producer.operation().kind() == WindowTransformKind.UNFOLD3D) {
            Window3dAttrs window = producer.operation().attrs() instanceof Unfold3dAttrs attrs
                    ? attrs.window()
                    : (Window3dAttrs) producer.operation().attrs();
            return ops.result(ops.fold3d(gradient, ops.descriptor(input).shape(), window));
        }
        if (producer.operation().kind() == WindowTransformKind.FOLD3D) {
            Fold3dAttrs attrs = (Fold3dAttrs) producer.operation().attrs();
            return ops.result(ops.unfold3d(gradient, attrs.window()));
        }
        throw new IllegalStateException(
                "layout operation was not preflight-approved: " + producer.operation());
    }

    /**
     * Builds the exact inverse complete-pattern repetition formula for one TILE input.
     *
     * @param input non-null exact original input Tensor
     * @param gradient non-null output cotangent with the preflight-approved tiled Shape
     * @param attrs non-null rank-aligned positive repeat attributes
     * @return a new public-Tensor expression with the exact input Shape, or {@code gradient}
     *     itself for a scalar input
     */
    private V tileGradient(V input, V gradient, TileAttrs attrs) {
        int rank = ops.descriptor(input).shape().rank();
        if (rank == 0) {
            return gradient;
        }
        Dimension[] interleaved = new Dimension[rank * 2];
        int[] repeatAxes = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            interleaved[2 * axis] = new StaticDimension(attrs.repeats().get(axis));
            interleaved[2 * axis + 1] = ops.descriptor(input).shape().dimension(axis);
            repeatAxes[axis] = 2 * axis;
        }
        return ops.reshape(ops.sum(ops.reshape(gradient, Shape.ofDimensions(interleaved)), repeatAxes, false), ops.descriptor(input).shape());
    }

    /**
     * Copies boxed long metadata into an independently owned primitive array.
     *
     * @param values non-null immutable list with no null element
     * @return a new same-order primitive array
     */
    private static long[] longs(List<Long> values) {
        long[] result = new long[values.size()];
        for (int index = 0; index < result.length; index++) {
            result[index] = values.get(index);
        }
        return result;
    }
}
