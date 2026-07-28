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
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

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
 */
final class ReductionGradientRules {
    private ReductionGradientRules() {}

    /**
     * Builds selected input cotangents for one approved reduction, scan, or softmax occurrence.
     *
     * @param producer non-null exact original producer occurrence
     * @param outputIndex exact selected canonical output slot, which must be zero for this family
     * @param gradient non-null accumulated cotangent for the selected output
     * @param selectedInputs non-null input-position-aligned selected-role flags from successful
     *     preflight
     * @param constants non-null request-local exact typed logical-splat owner
     * @return a new input-position-aligned array containing selected contributions and
     *     {@code null} for non-differentiable or unselected roles
     * @throws IllegalStateException if called for a row not approved by preflight
     */
    static Tensor[] apply(
            TensorProducer producer,
            int outputIndex,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        if (outputIndex != 0) {
            throw new IllegalStateException("reduction/scan/softmax output slot must be zero");
        }
        Tensor input = producer.inputs().getFirst();
        if (producer.operation().kind() instanceof CumulativeScanKind kind) {
            return new Tensor[] {kind == CumulativeScanKind.CUM_SUM
                    ? cumulativeSum(producer, gradient)
                    : cumulativeProduct(producer, gradient, constants)};
        }
        if (producer.operation().kind() instanceof SoftmaxKind kind) {
            Tensor output = producer.output(0);
            int axis = ((SoftmaxAttrs) producer.operation().attrs()).axis();
            Tensor contribution = kind == SoftmaxKind.SOFTMAX
                    ? output.mul(gradient.sub(gradient.mul(output).sum(axis, true)))
                    : gradient.sub(output.exp().mul(gradient.sum(axis, true)));
            return new Tensor[] {contribution};
        }
        if (!(producer.operation().kind() instanceof AggregateReductionKind kind)) {
            throw new IllegalStateException(
                    "reduction operation was not preflight-approved: "
                            + producer.operation().kind());
        }
        if (producer.operation().attrs() instanceof SumToShapeAttrs) {
            return new Tensor[] {gradient.expand(input.descriptor().shape())};
        }
        if (producer.operation().attrs() instanceof MaskedReductionAttrs attrs) {
            return masked(producer, gradient, attrs, constants);
        }

        Axes axes = axes(producer);
        return new Tensor[] {switch (kind) {
            case SUM -> restore(input, gradient, axes.values(), axes.keepDimensions());
            case MEAN -> mean(input, gradient, axes, constants);
            case PROD -> product(input, gradient, axes);
            case MIN, MAX -> extrema(producer, gradient, axes, constants);
            case LOG_SUM_EXP -> logSumExp(producer, gradient, axes);
            case VARIANCE -> variance(producer, gradient, axes, constants);
            case STANDARD_DEVIATION ->
                    standardDeviation(producer, gradient, axes, constants);
            case L1_NORM -> l1Norm(input, gradient, axes, constants);
            case L2_NORM -> l2Norm(producer, gradient, axes, constants);
            case ALL, ANY, ARG_MIN, ARG_MAX -> throw new IllegalStateException(
                    "non-differentiable aggregate was preflight-approved: " + kind);
        }};
    }

    private static Tensor cumulativeSum(TensorProducer producer, Tensor gradient) {
        CumulativeScanAttrs attrs =
                (CumulativeScanAttrs) producer.operation().attrs();
        return gradient.cumSum(attrs.axis(), attrs.exclusive(), !attrs.reverse());
    }

    private static Tensor cumulativeProduct(
            TensorProducer producer,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        CumulativeScanAttrs attrs =
                (CumulativeScanAttrs) producer.operation().attrs();
        Tensor zero = constants.zeroLike(input);
        Tensor one = constants.oneLike(input);
        Tensor isZero = input.equalTo(zero);
        Tensor safeInput = Tensor.where(isZero, one, input);
        Tensor zeroPrefix = Tensor.where(isZero, one, zero)
                .cumSum(attrs.axis(), attrs.exclusive(), attrs.reverse());
        Tensor safeProduct = safeInput.cumProd(
                attrs.axis(), attrs.exclusive(), attrs.reverse());
        Tensor q0 = Tensor.where(
                zeroPrefix.equalTo(zero), gradient.mul(safeProduct), zero);
        Tensor q1 = Tensor.where(
                zeroPrefix.equalTo(one), gradient.mul(safeProduct), zero);
        Tensor s0 = q0.cumSum(attrs.axis(), attrs.exclusive(), !attrs.reverse());
        Tensor s1 = q1.cumSum(attrs.axis(), attrs.exclusive(), !attrs.reverse());
        return Tensor.where(isZero, s1, s0.div(safeInput));
    }

    private static Tensor[] masked(
            TensorProducer producer,
            Tensor gradient,
            MaskedReductionAttrs attrs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        Tensor mask = producer.inputs().get(1);
        Tensor restored = restore(input, gradient, List.of(attrs.axis()), false);
        Tensor zero = constants.zeroLike(input);
        if (producer.operation().kind() == AggregateReductionKind.MEAN) {
            Tensor count = Tensor.where(mask, constants.oneLike(input), zero)
                    .sum(attrs.axis(), true)
                    .expand(input.descriptor().shape());
            return new Tensor[] {Tensor.where(mask, restored.div(count), zero), null};
        }
        return new Tensor[] {Tensor.where(mask, restored, zero), null};
    }

    private static Tensor mean(
            Tensor input,
            Tensor gradient,
            Axes axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor restored =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        Tensor count = count(input, axes.values(), constants);
        return restored.div(count.expand(input.descriptor().shape()));
    }

    private static Tensor product(Tensor input, Tensor gradient, Axes axes) {
        if (axes.values().isEmpty()) {
            return gradient;
        }
        List<Tensor> stages = new ArrayList<>(axes.values().size() + 1);
        stages.add(input);
        Tensor stage = input;
        for (int axis : axes.values()) {
            stage = stage.prod(axis, true);
            stages.add(stage);
        }
        Tensor result = restoreToShape(
                gradient,
                axes.values(),
                axes.keepDimensions(),
                stages.getLast().descriptor().shape());
        for (int index = axes.values().size() - 1; index >= 0; index--) {
            Tensor stageInput = stages.get(index);
            int axis = axes.values().get(index);
            Tensor prefix = stageInput.cumProd(axis, true, false);
            Tensor suffix = stageInput.cumProd(axis, true, true);
            result = result.expand(stageInput.descriptor().shape())
                    .mul(prefix)
                    .mul(suffix);
        }
        return result;
    }

    private static Tensor extrema(
            TensorProducer producer,
            Tensor gradient,
            Axes axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        Tensor restoredGradient =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        Tensor restoredOutput = restore(
                input, producer.output(0), axes.values(), axes.keepDimensions());
        Tensor matches = input.equalTo(restoredOutput);
        Tensor zero = constants.zeroLike(input);
        Tensor tieCount = Tensor.where(matches, constants.oneLike(input), zero)
                .sum(ints(axes.values()), true)
                .expand(input.descriptor().shape());
        return Tensor.where(matches, restoredGradient.div(tieCount), zero);
    }

    private static Tensor logSumExp(
            TensorProducer producer, Tensor gradient, Axes axes) {
        Tensor input = producer.inputs().getFirst();
        Tensor restoredGradient =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        Tensor restoredOutput = restore(
                input, producer.output(0), axes.values(), axes.keepDimensions());
        return restoredGradient.mul(input.sub(restoredOutput).exp());
    }

    private static Tensor variance(
            TensorProducer producer,
            Tensor gradient,
            Axes axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        StatisticalReductionAttrs attrs =
                (StatisticalReductionAttrs) producer.operation().attrs();
        Tensor mean = input.mean(ints(axes.values()), true);
        Tensor denominator = count(input, axes.values(), constants)
                .sub(correction(input, attrs.correction(), constants));
        return restore(input, gradient, axes.values(), axes.keepDimensions())
                .mul(constants.two(input.descriptor().dataType()))
                .mul(input.sub(mean))
                .div(denominator.expand(input.descriptor().shape()));
    }

    private static Tensor standardDeviation(
            TensorProducer producer,
            Tensor gradient,
            Axes axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        StatisticalReductionAttrs attrs =
                (StatisticalReductionAttrs) producer.operation().attrs();
        Tensor mean = input.mean(ints(axes.values()), true);
        Tensor restoredOutput = restore(
                input, producer.output(0), axes.values(), axes.keepDimensions());
        Tensor denominator = count(input, axes.values(), constants)
                .sub(correction(input, attrs.correction(), constants))
                .expand(input.descriptor().shape());
        Tensor regular = restore(input, gradient, axes.values(), axes.keepDimensions())
                .mul(input.sub(mean))
                .div(denominator.mul(restoredOutput));
        return Tensor.where(
                restoredOutput.greaterThan(constants.zeroLike(input)),
                regular,
                constants.zeroLike(input));
    }

    private static Tensor l1Norm(
            Tensor input,
            Tensor gradient,
            Axes axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor restored =
                restore(input, gradient, axes.values(), axes.keepDimensions());
        Tensor zero = constants.zeroLike(input);
        return Tensor.where(
                input.greaterThan(zero),
                restored,
                Tensor.where(input.lessThan(zero), restored.neg(), zero));
    }

    private static Tensor l2Norm(
            TensorProducer producer,
            Tensor gradient,
            Axes axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        Tensor restoredOutput = restore(
                input, producer.output(0), axes.values(), axes.keepDimensions());
        Tensor regular = restore(input, gradient, axes.values(), axes.keepDimensions())
                .mul(input)
                .div(restoredOutput);
        return Tensor.where(
                restoredOutput.greaterThan(constants.zeroLike(input)),
                regular,
                constants.zeroLike(input));
    }

    private static Tensor correction(
            Tensor input,
            long correction,
            FirstOrderAutograd.DerivativeConstants constants) {
        return constants.oneBase(input.descriptor().dataType())
                .expand(Shape.of(correction))
                .sum();
    }

    private static Tensor count(
            Tensor input,
            List<Integer> axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        return constants.oneLike(input).sum(ints(axes), true);
    }

    private static Tensor restore(
            Tensor input, Tensor tensor, List<Integer> axes, boolean keepDimensions) {
        return restoreToShape(
                tensor, axes, keepDimensions, input.descriptor().shape());
    }

    private static Tensor restoreToShape(
            Tensor tensor,
            List<Integer> axes,
            boolean keepDimensions,
            Shape targetShape) {
        Tensor restored = tensor;
        if (!keepDimensions) {
            List<Integer> sorted = new ArrayList<>(axes);
            sorted.sort(Comparator.naturalOrder());
            for (int axis : sorted) {
                restored = restored.expandDims(axis);
            }
        }
        return restored.descriptor().shape().equals(targetShape)
                ? restored
                : restored.expand(targetShape);
    }

    private static Axes axes(TensorProducer producer) {
        Shape inputShape = producer.inputs().getFirst().descriptor().shape();
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
