package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

/**
 * Builds first-order aggregate-reduction, scan, and softmax formulas selected by preflight.
 *
 * <p>All formulas are ordinary public Tensor expressions. Product formulas are division-free
 * across represented zeros, extrema share exact numeric ties, cumulative product uses safe
 * products plus cumulative zero counts, and softmax formulas consume the exact canonical forward
 * output. Statistical correction is represented by reducing an exact typed logical-one
 * expression, never by converting a host integer to floating point.</p>
 *
 * <p>Standard deviation and L2 norm select an exact zero contribution when their saved result is
 * not strictly positive; L1 norm selects zero at signed zero and NaN. Other listed exceptional
 * values follow the ordinary public Tensor operations in formula order. These are fixed
 * first-order compiler policies, not changes to forward reduction or normalization semantics.</p>
 *
 * <p>This owner constructs metadata only. It does not inspect values or storage, choose an
 * execution algorithm, capture a graph, lower work, or apply gradient-specific simplification.
 * Preflight owns exact kind, attributes, role, Shape, type, and boundary-policy selection.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class ReductionGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    ReductionGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds selected input cotangents for one approved reduction, scan, or softmax occurrence.
     *
     * @param producer non-null exact original producer occurrence
     * @param outputIndex exact selected canonical output slot, which must be zero for this family
     * @param gradient non-null accumulated cotangent for the selected output
     * @param selectedInputs non-null input-position-aligned selected-role flags from successful
     *     preflight
     * @param primal total lexical binding of original value operands and saved outputs
     * @return a new input-position-aligned array containing selected contributions and
     *     {@code null} for non-differentiable or unselected roles
     * @throws IllegalStateException if called for a row not approved by preflight
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            int outputIndex,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        if (outputIndex != 0) {
            throw new IllegalStateException("reduction/scan/softmax output slot must be zero");
        }
        V input = primal.apply(producer.inputs().getFirst());
        if (producer.operation().kind() instanceof CumulativeScanKind kind) {
            return ops.result(kind == CumulativeScanKind.CUM_SUM
                    ? cumulativeSum(producer, gradient)
                    : cumulativeProduct(producer, gradient, primal));
        }
        if (producer.operation().kind() instanceof SoftmaxKind kind) {
            V output = primal.apply(producer.output(0));
            int axis = ((SoftmaxAttrs) producer.operation().attrs()).axis();
            V contribution = kind == SoftmaxKind.SOFTMAX
                    ? ops.mul(output, ops.sub(gradient, ops.sum(ops.mul(gradient, output), axis, true)))
                    : ops.sub(gradient, ops.mul(ops.exp(output), ops.sum(gradient, axis, true)));
            return ops.result(contribution);
        }
        if (!(producer.operation().kind() instanceof AggregateReductionKind kind)) {
            throw new IllegalStateException(
                    "reduction operation was not preflight-approved: "
                            + producer.operation().kind());
        }
        if (producer.operation().attrs() instanceof SumToShapeAttrs) {
            return ops.result(ops.expand(gradient, ops.descriptor(input).shape()));
        }
        if (producer.operation().attrs() instanceof MaskedReductionAttrs attrs) {
            return masked(producer, gradient, attrs, primal);
        }

        Axes axes = axes(producer);
        return ops.result(switch (kind) {
            case SUM -> restore(input, gradient, axes.values(), axes.keepDimensions());
            case MEAN -> mean(input, gradient, axes);
            case PROD -> product(input, gradient, axes);
            case MIN, MAX -> extrema(producer, gradient, axes, primal);
            case LOG_SUM_EXP -> logSumExp(producer, gradient, axes, primal);
            case VARIANCE -> variance(producer, gradient, axes, primal);
            case STANDARD_DEVIATION ->
                    standardDeviation(producer, gradient, axes, primal);
            case L1_NORM -> l1Norm(input, gradient, axes);
            case L2_NORM -> l2Norm(producer, gradient, axes, primal);
            case ALL, ANY, ARG_MIN, ARG_MAX -> throw new IllegalStateException(
                    "non-differentiable aggregate was preflight-approved: " + kind);
        });
    }


    private V cumulativeSum(OrdinaryConstruction.Source<V> producer, V gradient) {
        CumulativeScanAttrs attrs =
                (CumulativeScanAttrs) producer.operation().attrs();
        return ops.cumSum(gradient, attrs.axis(), attrs.exclusive(), !attrs.reverse());
    }

    private V cumulativeProduct(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        CumulativeScanAttrs attrs =
                (CumulativeScanAttrs) producer.operation().attrs();
        V zero = ops.zeroLike(input);
        V one = ops.oneLike(input);
        V isZero = ops.equalTo(input, zero);
        V safeInput = ops.where(isZero, one, input);
        V zeroPrefix = ops.cumSum(ops.where(isZero, one, zero), attrs.axis(), attrs.exclusive(), attrs.reverse());
        V safeProduct = ops.cumProd(safeInput, attrs.axis(), attrs.exclusive(), attrs.reverse());
        V q0 = ops.where(ops.equalTo(zeroPrefix, zero), ops.mul(gradient, safeProduct), zero);
        V q1 = ops.where(ops.equalTo(zeroPrefix, one), ops.mul(gradient, safeProduct), zero);
        V s0 = ops.cumSum(q0, attrs.axis(), attrs.exclusive(), !attrs.reverse());
        V s1 = ops.cumSum(q1, attrs.axis(), attrs.exclusive(), !attrs.reverse());
        return ops.where(isZero, s1, ops.div(s0, safeInput));
    }

    private V[] masked(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            MaskedReductionAttrs attrs,
            Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        V mask = primal.apply(producer.inputs().get(1));
        V restored = restore(input, gradient, List.of(attrs.axis()), false);
        V zero = ops.zeroLike(input);
        if (producer.operation().kind() == AggregateReductionKind.MEAN) {
            V count = ops.expand(ops.sum(ops.where(mask, ops.oneLike(input), zero), attrs.axis(), true), ops.descriptor(input).shape());
            return ops.result(ops.where(mask, ops.div(restored, count), zero), null);
        }
        return ops.result(ops.where(mask, restored, zero), null);
    }

    private V mean(V input, V gradient, Axes axes) {
        V restored =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        V count = count(input, axes.values());
        return ops.div(restored, ops.expand(count, ops.descriptor(input).shape()));
    }

    private V product(V input, V gradient, Axes axes) {
        if (axes.values().isEmpty()) {
            return gradient;
        }
        List<V> stages = new ArrayList<>(axes.values().size() + 1);
        stages.add(input);
        V stage = input;
        for (int axis : axes.values()) {
            stage = ops.prod(stage, axis, true);
            stages.add(stage);
        }
        V result = restoreToShape(
                gradient,
                axes.values(),
                axes.keepDimensions(),
                ops.descriptor(stages.getLast()).shape());
        for (int index = axes.values().size() - 1; index >= 0; index--) {
            V stageInput = stages.get(index);
            int axis = axes.values().get(index);
            V prefix = ops.cumProd(stageInput, axis, true, false);
            V suffix = ops.cumProd(stageInput, axis, true, true);
            result = ops.mul(ops.mul(ops.expand(result, ops.descriptor(stageInput).shape()), prefix), suffix);
        }
        return result;
    }

    private V extrema(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Axes axes,
            Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        V restoredGradient =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        V restoredOutput = restore(
                input, primal.apply(producer.output(0)), axes.values(), axes.keepDimensions());
        V matches = ops.equalTo(input, restoredOutput);
        V zero = ops.zeroLike(input);
        V tieCount = ops.expand(ops.sum(ops.where(matches, ops.oneLike(input), zero), ints(axes.values()), true), ops.descriptor(input).shape());
        return ops.where(matches, ops.div(restoredGradient, tieCount), zero);
    }

    private V logSumExp(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Axes axes,
            Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        V restoredGradient =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        V restoredOutput = restore(
                input, primal.apply(producer.output(0)), axes.values(), axes.keepDimensions());
        return ops.mul(restoredGradient, ops.exp(ops.sub(input, restoredOutput)));
    }

    private V variance(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Axes axes,
            Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        StatisticalReductionAttrs attrs =
                (StatisticalReductionAttrs) producer.operation().attrs();
        V mean = ops.mean(input, ints(axes.values()), true);
        V denominator = ops.sub(count(input, axes.values()), correction(input, attrs.correction()));
        return ops.div(ops.mul(ops.mul(restore(input, gradient, axes.values(), axes.keepDimensions()), ops.two(ops.descriptor(input).dataType())), ops.sub(input, mean)), ops.expand(denominator, ops.descriptor(input).shape()));
    }

    private V standardDeviation(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Axes axes,
            Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        StatisticalReductionAttrs attrs =
                (StatisticalReductionAttrs) producer.operation().attrs();
        V mean = ops.mean(input, ints(axes.values()), true);
        V restoredOutput = restore(
                input, primal.apply(producer.output(0)), axes.values(), axes.keepDimensions());
        V denominator = ops.expand(ops.sub(count(input, axes.values()), correction(input, attrs.correction())), ops.descriptor(input).shape());
        V regular = ops.div(ops.mul(restore(input, gradient, axes.values(), axes.keepDimensions()), ops.sub(input, mean)), ops.mul(denominator, restoredOutput));
        return ops.where(ops.greaterThan(restoredOutput, ops.zeroLike(input)), regular, ops.zeroLike(input));
    }

    private V l1Norm(V input, V gradient, Axes axes) {
        V restored =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        V zero = ops.zeroLike(input);
        return ops.where(ops.greaterThan(input, zero), restored, ops.where(ops.lessThan(input, zero), ops.neg(restored), zero));
    }

    private V l2Norm(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            Axes axes,
            Function<V, V> primal) {
        V input = primal.apply(producer.inputs().getFirst());
        V restoredOutput = restore(
                input, primal.apply(producer.output(0)), axes.values(), axes.keepDimensions());
        V regular = ops.div(ops.mul(restore(input, gradient, axes.values(), axes.keepDimensions()), input), restoredOutput);
        return ops.where(ops.greaterThan(restoredOutput, ops.zeroLike(input)), regular, ops.zeroLike(input));
    }

    private V correction(V input, long correction) {
        return ops.sum(ops.expand(ops.oneBase(ops.descriptor(input).dataType()), Shape.of(correction)));
    }

    private V count(V input, List<Integer> axes) {
        return ops.sum(ops.oneLike(input), ints(axes), true);
    }

    private V restore(V input, V tensor, List<Integer> axes, boolean keepDimensions) {
        return restoreToShape(
                tensor, axes, keepDimensions, ops.descriptor(input).shape());
    }

    private V restoreToShape(V tensor, List<Integer> axes, boolean keepDimensions, Shape targetShape) {
        V restored = tensor;
        if (!keepDimensions) {
            List<Integer> sorted = new ArrayList<>(axes);
            sorted.sort(Comparator.naturalOrder());
            for (int axis : sorted) {
                restored = ops.expandDims(restored, axis);
            }
        }
        return ops.descriptor(restored).shape().equals(targetShape)
                ? restored
                : ops.contiguous(ops.expand(restored, targetShape));
    }

    private Axes axes(OrdinaryConstruction.Source<V> producer) {
        Shape inputShape = ops.descriptor(producer.inputs().getFirst()).shape();
        if (producer.operation().attrs() == NoOperationAttrs.INSTANCE) {
            List<Integer> axes = new ArrayList<>(inputShape.rank());
            for (int axis = 0; axis < inputShape.rank(); axis++) {
                axes.add(axis);
            }
            return new Axes(List.copyOf(axes), false);
        }
        if (producer.operation().attrs() instanceof AxisReductionAttrs attrs) {
            return new Axes(List.of(attrs.axis()), attrs.keepDimensions());
        }
        if (producer.operation().attrs() instanceof MultiAxisReductionAttrs attrs) {
            return new Axes(attrs.axes(), attrs.keepDimensions());
        }
        StatisticalReductionAttrs attrs =
                (StatisticalReductionAttrs) producer.operation().attrs();
        return new Axes(attrs.axes(), attrs.keepDimensions());
    }

    private static int[] ints(List<Integer> axes) {
        return axes.stream().mapToInt(Integer::intValue).toArray();
    }

    private record Axes(List<Integer> values, boolean keepDimensions) {}
}
