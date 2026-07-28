package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.List;

/**
 * Builds first-order Layer, RMS, and batch-normalization formulas selected by preflight.
 *
 * <p>Mixed-floating occurrences compute in the exact selected forward-output type and cast only
 * completed Shape-correct contributions back to their selected input types. Affine and statistic
 * vectors are aligned through visible reshape and expand expressions. Batch-training formulas
 * consume the exact canonical saved mean and inverse-standard-deviation wrappers from output
 * slots three and four of the same producer occurrence.</p>
 *
 * <p>Batch inference can return contributions for all five floating Tensor inputs. Batch training
 * selects input/scale/bias from output slot zero, input/running mean from slot one, and
 * input/running variance from slot two. Saved slots three and four are formula operands only.
 * Contributions are Shape- and type-normalized before the caller accumulates them.</p>
 *
 * <p>This owner creates ordinary Tensor expression metadata only. It neither recomputes saved
 * batch statistics nor creates a tape, physical buffer, publication requirement, backend route,
 * or gradient-specific numerical policy. Preflight owns exact signatures, output-slot roles,
 * promotion, Shape implications, and failure ordering.</p>
 */
final class NormalizationGradientRules {
    private NormalizationGradientRules() {}

    /**
     * Builds selected input cotangents for one approved normalization output route.
     *
     * @param producer non-null exact original normalization producer occurrence
     * @param outputIndex exact selected canonical output slot; zero for Layer, RMS, and batch
     *     inference, or zero through two for batch training
     * @param gradient non-null accumulated cotangent for that output
     * @param selectedInputs non-null input-position-aligned selected-role flags from successful
     *     preflight
     * @param constants non-null request-local exact typed logical-splat owner
     * @return a new input-position-aligned contribution array with {@code null} for unselected
     *     roles
     * @throws IllegalStateException if called for a row not approved by preflight
     */
    static Tensor[] apply(
            TensorProducer producer,
            int outputIndex,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        if (producer.operation().kind() == LayerNormKind.LAYER_NORM) {
            return layerNorm(producer, gradient, selectedInputs);
        }
        if (producer.operation().kind() == RmsNormKind.RMS_NORM) {
            return rmsNorm(producer, gradient, selectedInputs);
        }
        if (producer.operation().kind() == BatchNormKind.BATCH_NORM_INFERENCE) {
            return batchInference(producer, gradient, selectedInputs, constants);
        }
        if (producer.operation().kind() == BatchNormKind.BATCH_NORM_TRAINING) {
            return batchTraining(
                    producer, outputIndex, gradient, selectedInputs, constants);
        }
        throw new IllegalStateException(
                "normalization operation was not preflight-approved: "
                        + producer.operation().kind());
    }

    private static Tensor[] layerNorm(
            TensorProducer producer, Tensor gradient, boolean[] selectedInputs) {
        Tensor input = producer.inputs().getFirst();
        Tensor output = producer.output(0);
        Shape normalizedShape;
        io.github.pho001.synaptik.model.datatype.ScalarValue epsilon;
        if (producer.operation().attrs() instanceof LayerNormAttrs attrs) {
            normalizedShape = attrs.normalizedShape();
            epsilon = attrs.epsilon();
        } else {
            AffineLayerNormAttrs attrs =
                    (AffineLayerNormAttrs) producer.operation().attrs();
            normalizedShape = attrs.normalizedShape();
            epsilon = attrs.epsilon();
        }
        int[] normalizedAxes = trailingAxes(
                input.descriptor().shape(), normalizedShape.rank());
        int[] prefixAxes = prefixAxes(
                input.descriptor().shape(), normalizedShape.rank());
        DataType outputType = output.descriptor().dataType();
        Tensor promotedInput = cast(input, outputType);
        Tensor mean = promotedInput.mean(normalizedAxes, true);
        Tensor centered = promotedInput.sub(mean);
        Tensor inverseStandardDeviation = centered.mul(centered)
                .mean(normalizedAxes, true)
                .add(epsilon)
                .rsqrt();
        Tensor h = gradient;
        if (producer.inputs().size() == 3) {
            h = h.mul(align(
                    cast(producer.inputs().get(1), outputType),
                    input.descriptor().shape(),
                    normalizedAxes));
        }
        Tensor inputContribution = inverseStandardDeviation.mul(
                h.sub(h.mean(normalizedAxes, true))
                        .sub(centered
                                .mul(inverseStandardDeviation)
                                .mul(inverseStandardDeviation)
                                .mul(h.mul(centered).mean(normalizedAxes, true))));

        Tensor[] result = new Tensor[producer.inputs().size()];
        if (selectedInputs[0]) {
            result[0] = cast(inputContribution, input.descriptor().dataType());
        }
        if (producer.inputs().size() == 3) {
            Tensor normalized = centered.mul(inverseStandardDeviation);
            if (selectedInputs[1]) {
                result[1] = reduceTo(
                        gradient.mul(normalized),
                        producer.inputs().get(1),
                        prefixAxes);
            }
            if (selectedInputs[2]) {
                result[2] = reduceTo(
                        gradient,
                        producer.inputs().get(2),
                        prefixAxes);
            }
        }
        return result;
    }

    private static Tensor[] rmsNorm(
            TensorProducer producer, Tensor gradient, boolean[] selectedInputs) {
        Tensor input = producer.inputs().getFirst();
        Tensor output = producer.output(0);
        RmsNormAttrs attrs = (RmsNormAttrs) producer.operation().attrs();
        int[] normalizedAxes = trailingAxes(
                input.descriptor().shape(), attrs.normalizedShape().rank());
        int[] prefixAxes = prefixAxes(
                input.descriptor().shape(), attrs.normalizedShape().rank());
        DataType outputType = output.descriptor().dataType();
        Tensor promotedInput = cast(input, outputType);
        Tensor inverseRootMeanSquare = promotedInput.mul(promotedInput)
                .mean(normalizedAxes, true)
                .add(attrs.epsilon())
                .rsqrt();
        Tensor h = gradient;
        if (producer.inputs().size() == 2) {
            h = h.mul(align(
                    cast(producer.inputs().get(1), outputType),
                    input.descriptor().shape(),
                    normalizedAxes));
        }
        Tensor inputContribution = inverseRootMeanSquare.mul(h)
                .sub(promotedInput
                        .mul(inverseRootMeanSquare)
                        .mul(inverseRootMeanSquare)
                        .mul(inverseRootMeanSquare)
                        .mul(h.mul(promotedInput).mean(normalizedAxes, true)));

        Tensor[] result = new Tensor[producer.inputs().size()];
        if (selectedInputs[0]) {
            result[0] = cast(inputContribution, input.descriptor().dataType());
        }
        if (producer.inputs().size() == 2 && selectedInputs[1]) {
            result[1] = reduceTo(
                    gradient.mul(promotedInput).mul(inverseRootMeanSquare),
                    producer.inputs().get(1),
                    prefixAxes);
        }
        return result;
    }

    private static Tensor[] batchInference(
            TensorProducer producer,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        Tensor output = producer.output(0);
        BatchNormInferenceAttrs attrs =
                (BatchNormInferenceAttrs) producer.operation().attrs();
        int channelAxis = attrs.channelAxis();
        int[] channelAxes = new int[] {channelAxis};
        int[] reducedAxes = excludingAxis(input.descriptor().shape(), channelAxis);
        DataType outputType = output.descriptor().dataType();
        Tensor promotedInput = cast(input, outputType);
        Tensor scale = align(
                cast(producer.inputs().get(1), outputType),
                input.descriptor().shape(),
                channelAxes);
        Tensor mean = align(
                cast(producer.inputs().get(3), outputType),
                input.descriptor().shape(),
                channelAxes);
        Tensor variance = align(
                cast(producer.inputs().get(4), outputType),
                input.descriptor().shape(),
                channelAxes);
        Tensor inverseStandardDeviation = variance.add(attrs.epsilon()).rsqrt();
        Tensor centered = promotedInput.sub(mean);
        Tensor scaledGradient = gradient.mul(scale).mul(inverseStandardDeviation);

        Tensor[] result = new Tensor[5];
        if (selectedInputs[0]) {
            result[0] = cast(scaledGradient, input.descriptor().dataType());
        }
        if (selectedInputs[1]) {
            result[1] = reduceTo(
                    gradient.mul(centered).mul(inverseStandardDeviation),
                    producer.inputs().get(1),
                    reducedAxes);
        }
        if (selectedInputs[2]) {
            result[2] = reduceTo(
                    gradient,
                    producer.inputs().get(2),
                    reducedAxes);
        }
        if (selectedInputs[3]) {
            result[3] = reduceTo(
                    scaledGradient.neg(),
                    producer.inputs().get(3),
                    reducedAxes);
        }
        if (selectedInputs[4]) {
            result[4] = reduceTo(
                    gradient.mul(constants.negativeHalf(outputType))
                            .mul(scale)
                            .mul(centered)
                            .mul(inverseStandardDeviation)
                            .mul(inverseStandardDeviation)
                            .mul(inverseStandardDeviation),
                    producer.inputs().get(4),
                    reducedAxes);
        }
        return result;
    }

    private static Tensor[] batchTraining(
            TensorProducer producer,
            int outputIndex,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor input = producer.inputs().getFirst();
        Tensor output = producer.output(outputIndex);
        BatchNormTrainingAttrs attrs =
                (BatchNormTrainingAttrs) producer.operation().attrs();
        int channelAxis = attrs.channelAxis();
        int[] channelAxes = new int[] {channelAxis};
        int[] reducedAxes = excludingAxis(input.descriptor().shape(), channelAxis);
        DataType outputType = output.descriptor().dataType();
        Tensor promotedInput = cast(input, outputType);
        Tensor savedMean = align(
                producer.output(3), input.descriptor().shape(), channelAxes);

        Tensor[] result = new Tensor[5];
        if (outputIndex == 0) {
            Tensor savedInverseStandardDeviation = align(
                    producer.output(4), input.descriptor().shape(), channelAxes);
            Tensor scale = align(
                    cast(producer.inputs().get(1), outputType),
                    input.descriptor().shape(),
                    channelAxes);
            Tensor centered = promotedInput.sub(savedMean);
            Tensor h = gradient.mul(scale);
            if (selectedInputs[0]) {
                Tensor contribution = savedInverseStandardDeviation.mul(
                        h.sub(h.mean(reducedAxes, true))
                                .sub(centered
                                        .mul(savedInverseStandardDeviation)
                                        .mul(savedInverseStandardDeviation)
                                        .mul(h.mul(centered).mean(reducedAxes, true))));
                result[0] = cast(contribution, input.descriptor().dataType());
            }
            if (selectedInputs[1]) {
                result[1] = reduceTo(
                        gradient.mul(centered).mul(savedInverseStandardDeviation),
                        producer.inputs().get(1),
                        reducedAxes);
            }
            if (selectedInputs[2]) {
                result[2] = reduceTo(
                        gradient,
                        producer.inputs().get(2),
                        reducedAxes);
            }
            return result;
        }
        Tensor count = count(promotedInput, reducedAxes, constants);
        if (outputIndex == 1) {
            if (selectedInputs[0]) {
                Tensor aligned = align(
                        gradient.mul(attrs.momentum()),
                        input.descriptor().shape(),
                        channelAxes);
                result[0] = cast(
                        aligned.div(count.expand(input.descriptor().shape())),
                        input.descriptor().dataType());
            }
            if (selectedInputs[3]) {
                Tensor oneMinusMomentum =
                        constants.oneLike(gradient).sub(attrs.momentum());
                result[3] = cast(
                        gradient.mul(oneMinusMomentum),
                        producer.inputs().get(3).descriptor().dataType());
            }
            return result;
        }
        if (outputIndex == 2) {
            if (selectedInputs[0]) {
                Tensor numerator = align(
                                gradient.mul(attrs.momentum()),
                                input.descriptor().shape(),
                                channelAxes)
                        .mul(constants.two(outputType))
                        .mul(promotedInput.sub(savedMean));
                Tensor denominator = count.sub(constants.oneLike(count))
                        .expand(input.descriptor().shape());
                result[0] = cast(
                        numerator.div(denominator),
                        input.descriptor().dataType());
            }
            if (selectedInputs[4]) {
                Tensor oneMinusMomentum =
                        constants.oneLike(gradient).sub(attrs.momentum());
                result[4] = cast(
                        gradient.mul(oneMinusMomentum),
                        producer.inputs().get(4).descriptor().dataType());
            }
            return result;
        }
        throw new IllegalStateException(
                "batch-normalization training auxiliary slot was preflight-approved: "
                        + outputIndex);
    }

    private static Tensor count(
            Tensor tensor,
            int[] axes,
            FirstOrderAutograd.DerivativeConstants constants) {
        return constants.oneLike(tensor).sum(axes, true);
    }

    private static Tensor align(Tensor tensor, Shape target, int[] mappedAxes) {
        if (tensor.descriptor().shape().rank() != mappedAxes.length) {
            throw new IllegalStateException("preflight-approved alignment rank changed");
        }
        Dimension[] dimensions = new Dimension[target.rank()];
        for (int axis = 0; axis < dimensions.length; axis++) {
            dimensions[axis] = new StaticDimension(1);
        }
        for (int index = 0; index < mappedAxes.length; index++) {
            dimensions[mappedAxes[index]] =
                    tensor.descriptor().shape().dimension(index);
        }
        Shape alignedShape = Shape.ofDimensions(dimensions);
        Tensor aligned = tensor.descriptor().shape().equals(alignedShape)
                ? tensor
                : tensor.reshape(alignedShape);
        return aligned.descriptor().shape().equals(target)
                ? aligned
                : aligned.expand(target);
    }

    private static Tensor reduceTo(Tensor tensor, Tensor target, int[] axes) {
        Tensor reduced = tensor.sum(axes, false);
        if (!reduced.descriptor().shape().equals(target.descriptor().shape())) {
            reduced = reduced.reshape(target.descriptor().shape());
        }
        return cast(reduced, target.descriptor().dataType());
    }

    private static Tensor cast(Tensor tensor, DataType dataType) {
        return tensor.descriptor().dataType() == dataType
                ? tensor
                : tensor.cast(dataType);
    }

    private static int[] trailingAxes(Shape input, int count) {
        int[] result = new int[count];
        int offset = input.rank() - count;
        for (int index = 0; index < count; index++) {
            result[index] = offset + index;
        }
        return result;
    }

    private static int[] prefixAxes(Shape input, int trailingCount) {
        int count = input.rank() - trailingCount;
        int[] result = new int[count];
        for (int axis = 0; axis < count; axis++) {
            result[axis] = axis;
        }
        return result;
    }

    private static int[] excludingAxis(Shape input, int excludedAxis) {
        List<Integer> result = new ArrayList<>(input.rank() - 1);
        for (int axis = 0; axis < input.rank(); axis++) {
            if (axis != excludedAxis) {
                result.add(axis);
            }
        }
        return result.stream().mapToInt(Integer::intValue).toArray();
    }
}
