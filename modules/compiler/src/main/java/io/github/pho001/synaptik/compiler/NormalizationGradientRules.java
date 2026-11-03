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
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Builds first-order Layer, RMS, and batch-normalization formulas selected by preflight.
 *
 * <p>Mixed-floating occurrences compute in the selected forward-output type, except FLOAT16
 * normalization uses FLOAT32 working expressions with its exact FLOAT32 metadata. Completed
 * Shape-correct contributions are cast back to their selected input types. Affine and statistic
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
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class NormalizationGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    NormalizationGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds selected input cotangents for one approved normalization output route.
     *
     * @param producer non-null exact original normalization producer occurrence
     * @param outputIndex exact selected canonical output slot; zero for Layer, RMS, and batch
     *     inference, or zero through two for batch training
     * @param gradient non-null accumulated cotangent for that output
     * @param selectedInputs non-null input-position-aligned selected-role flags from successful
     *     preflight
     * @param primal total lexical binding of original value operands and saved outputs
     * @return a new input-position-aligned contribution array with {@code null} for unselected
     *     roles
     * @throws IllegalStateException if called for a row not approved by preflight
     */
    V[] apply(OrdinaryConstruction.Source<V> producer, int outputIndex, V gradient, boolean[] selectedInputs, Function<V, V> primal) {
            if (producer.operation().kind() == LayerNormKind.LAYER_NORM) {
                return layerNorm(producer, gradient, selectedInputs, primal);
            }
            if (producer.operation().kind() == RmsNormKind.RMS_NORM) {
                return rmsNorm(producer, gradient, selectedInputs, primal);
            }
            if (producer.operation().kind() == BatchNormKind.BATCH_NORM_INFERENCE) {
                return batchInference(producer, gradient, selectedInputs, primal);
            }
            if (producer.operation().kind() == BatchNormKind.BATCH_NORM_TRAINING) {
                return batchTraining(
                        producer, outputIndex, gradient, selectedInputs, primal);
            }
            throw new IllegalStateException(
                    "normalization operation was not preflight-approved: "
                            + producer.operation().kind());
    }

    private V[] layerNorm(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V input = producer.inputs().getFirst();
        V output = producer.output(0);
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
                ops.descriptor(input).shape(), normalizedShape.rank());
        int[] prefixAxes = prefixAxes(
                ops.descriptor(input).shape(), normalizedShape.rank());
        DataType outputType = workingType(output);
        gradient = cast(gradient, outputType);
        V promotedInput = cast(primal.apply(input), outputType);
        V mean = ops.mean(promotedInput, normalizedAxes, true);
        V centered = ops.sub(promotedInput, mean);
        V inverseStandardDeviation = ops.rsqrt(ops.add(ops.mean(ops.mul(centered, centered), normalizedAxes, true), epsilon));
        V h = gradient;
        if (producer.inputs().size() == 3) {
            h = ops.mul(h, align(
                    cast(primal.apply(producer.inputs().get(1)), outputType),
                    ops.descriptor(input).shape(),
                    normalizedAxes));
        }
        V inputContribution = ops.mul(inverseStandardDeviation, ops.sub(ops.sub(h, ops.mean(h, normalizedAxes, true)), ops.mul(ops.mul(ops.mul(centered, inverseStandardDeviation), inverseStandardDeviation), ops.mean(ops.mul(h, centered), normalizedAxes, true))));
        
        V[] result = ops.array(producer.inputs().size());
        if (selectedInputs[0]) {
            result[0] = cast(inputContribution, ops.descriptor(input).dataType());
        }
        if (producer.inputs().size() == 3) {
            V normalized = ops.mul(centered, inverseStandardDeviation);
            if (selectedInputs[1]) {
                result[1] = reduceTo(
                        ops.mul(gradient, normalized),
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

    private V[] rmsNorm(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V input = producer.inputs().getFirst();
        V output = producer.output(0);
        RmsNormAttrs attrs = (RmsNormAttrs) producer.operation().attrs();
        int[] normalizedAxes = trailingAxes(
                ops.descriptor(input).shape(), attrs.normalizedShape().rank());
        int[] prefixAxes = prefixAxes(
                ops.descriptor(input).shape(), attrs.normalizedShape().rank());
        DataType outputType = workingType(output);
        gradient = cast(gradient, outputType);
        V promotedInput = cast(primal.apply(input), outputType);
        V inverseRootMeanSquare = ops.rsqrt(ops.add(ops.mean(ops.mul(promotedInput, promotedInput), normalizedAxes, true), attrs.epsilon()));
        V h = gradient;
        if (producer.inputs().size() == 2) {
            h = ops.mul(h, align(
                    cast(primal.apply(producer.inputs().get(1)), outputType),
                    ops.descriptor(input).shape(),
                    normalizedAxes));
        }
        V inputContribution = ops.sub(ops.mul(inverseRootMeanSquare, h), ops.mul(ops.mul(ops.mul(ops.mul(promotedInput, inverseRootMeanSquare), inverseRootMeanSquare), inverseRootMeanSquare), ops.mean(ops.mul(h, promotedInput), normalizedAxes, true)));
        
        V[] result = ops.array(producer.inputs().size());
        if (selectedInputs[0]) {
            result[0] = cast(inputContribution, ops.descriptor(input).dataType());
        }
        if (producer.inputs().size() == 2 && selectedInputs[1]) {
            result[1] = reduceTo(
                    ops.mul(ops.mul(gradient, promotedInput), inverseRootMeanSquare),
                    producer.inputs().get(1),
                    prefixAxes);
        }
        return result;
    }

    private V[] batchInference(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V input = producer.inputs().getFirst();
        V output = producer.output(0);
        BatchNormInferenceAttrs attrs =
                (BatchNormInferenceAttrs) producer.operation().attrs();
        int channelAxis = attrs.channelAxis();
        int[] channelAxes = new int[] {channelAxis};
        int[] reducedAxes = excludingAxis(ops.descriptor(input).shape(), channelAxis);
        DataType outputType = workingType(output);
        gradient = cast(gradient, outputType);
        V promotedInput = cast(primal.apply(input), outputType);
        V scale = align(
                cast(primal.apply(producer.inputs().get(1)), outputType),
                ops.descriptor(input).shape(),
                channelAxes);
        V mean = align(
                cast(primal.apply(producer.inputs().get(3)), outputType),
                ops.descriptor(input).shape(),
                channelAxes);
        V variance = align(
                cast(primal.apply(producer.inputs().get(4)), outputType),
                ops.descriptor(input).shape(),
                channelAxes);
        V inverseStandardDeviation = ops.rsqrt(ops.add(variance, attrs.epsilon()));
        V centered = ops.sub(promotedInput, mean);
        V scaledGradient = ops.mul(ops.mul(gradient, scale), inverseStandardDeviation);

        V[] result = ops.array(5);
        if (selectedInputs[0]) {
            result[0] = cast(scaledGradient, ops.descriptor(input).dataType());
        }
        if (selectedInputs[1]) {
            result[1] = reduceTo(
                    ops.mul(ops.mul(gradient, centered), inverseStandardDeviation),
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
                    ops.neg(scaledGradient),
                    producer.inputs().get(3),
                    reducedAxes);
        }
        if (selectedInputs[4]) {
            result[4] = reduceTo(
                    ops.mul(ops.mul(ops.mul(ops.mul(ops.mul(ops.mul(gradient, ops.negativeHalf(outputType)), scale), centered), inverseStandardDeviation), inverseStandardDeviation), inverseStandardDeviation),
                    producer.inputs().get(4),
                    reducedAxes);
        }
        return result;
    }

    private V[] batchTraining(
            OrdinaryConstruction.Source<V> producer,
            int outputIndex,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V input = producer.inputs().getFirst();
        V output = producer.output(outputIndex);
        BatchNormTrainingAttrs attrs =
                (BatchNormTrainingAttrs) producer.operation().attrs();
        int channelAxis = attrs.channelAxis();
        int[] channelAxes = new int[] {channelAxis};
        int[] reducedAxes = excludingAxis(ops.descriptor(input).shape(), channelAxis);
        DataType outputType = workingType(output);
        gradient = cast(gradient, outputType);
        V promotedInput = cast(primal.apply(input), outputType);
        V savedMean = align(
                cast(primal.apply(producer.output(3)), outputType),
                ops.descriptor(input).shape(), channelAxes);

        V[] result = ops.array(5);
        if (outputIndex == 0) {
            V savedInverseStandardDeviation = align(
                    cast(primal.apply(producer.output(4)), outputType),
                    ops.descriptor(input).shape(), channelAxes);
            V scale = align(
                    cast(primal.apply(producer.inputs().get(1)), outputType),
                    ops.descriptor(input).shape(),
                    channelAxes);
            V centered = ops.sub(promotedInput, savedMean);
            V h = ops.mul(gradient, scale);
            if (selectedInputs[0]) {
                V contribution = ops.mul(savedInverseStandardDeviation, ops.sub(ops.sub(h, ops.mean(h, reducedAxes, true)), ops.mul(ops.mul(ops.mul(centered, savedInverseStandardDeviation), savedInverseStandardDeviation), ops.mean(ops.mul(h, centered), reducedAxes, true))));
                result[0] = cast(contribution, ops.descriptor(input).dataType());
            }
            if (selectedInputs[1]) {
                result[1] = reduceTo(
                        ops.mul(ops.mul(gradient, centered), savedInverseStandardDeviation),
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
        V count = count(promotedInput, reducedAxes);
        if (outputIndex == 1) {
            if (selectedInputs[0]) {
                V aligned = align(
                        ops.mul(gradient, attrs.momentum()),
                        ops.descriptor(input).shape(),
                        channelAxes);
                result[0] = cast(
                        ops.div(aligned, ops.expand(count, ops.descriptor(input).shape())),
                        ops.descriptor(input).dataType());
            }
            if (selectedInputs[3]) {
                V oneMinusMomentum =
                        ops.sub(ops.oneLike(gradient), attrs.momentum());
                result[3] = cast(
                        ops.mul(gradient, oneMinusMomentum),
                        ops.descriptor(producer.inputs().get(3)).dataType());
            }
            return result;
        }
        if (outputIndex == 2) {
            if (selectedInputs[0]) {
                V numerator = ops.mul(ops.mul(align(
                                ops.mul(gradient, attrs.momentum()),
                                ops.descriptor(input).shape(),
                                channelAxes), ops.two(outputType)), ops.sub(promotedInput, savedMean));
                V denominator = ops.expand(ops.sub(count, ops.oneLike(count)), ops.descriptor(input).shape());
                result[0] = cast(
                        ops.div(numerator, denominator),
                        ops.descriptor(input).dataType());
            }
            if (selectedInputs[4]) {
                V oneMinusMomentum =
                        ops.sub(ops.oneLike(gradient), attrs.momentum());
                result[4] = cast(
                        ops.mul(gradient, oneMinusMomentum),
                        ops.descriptor(producer.inputs().get(4)).dataType());
            }
            return result;
        }
        throw new IllegalStateException(
                "batch-normalization training auxiliary slot was preflight-approved: "
                        + outputIndex);
    }

    private V count(V tensor, int[] axes) {
        return ops.sum(ops.oneLike(tensor), axes, true);
    }

    private V align(V tensor, Shape target, int[] mappedAxes) {
        if (ops.descriptor(tensor).shape().rank() != mappedAxes.length) {
            throw new IllegalStateException("preflight-approved alignment rank changed");
        }
        Dimension[] dimensions = new Dimension[target.rank()];
        for (int axis = 0; axis < dimensions.length; axis++) {
            dimensions[axis] = new StaticDimension(1);
        }
        for (int index = 0; index < mappedAxes.length; index++) {
            dimensions[mappedAxes[index]] =
                    ops.descriptor(tensor).shape().dimension(index);
        }
        Shape alignedShape = Shape.ofDimensions(dimensions);
        V aligned = ops.descriptor(tensor).shape().equals(alignedShape)
                ? tensor
                : ops.reshape(tensor, alignedShape);
        return ops.descriptor(aligned).shape().equals(target)
                ? aligned
                : ops.expand(aligned, target);
    }

    private V reduceTo(V tensor, V target, int[] axes) {
        V reduced = ops.sum(tensor, axes, false);
        if (!ops.descriptor(reduced).shape().equals(ops.descriptor(target).shape())) {
            reduced = ops.reshape(reduced, ops.descriptor(target).shape());
        }
        return cast(reduced, ops.descriptor(target).dataType());
    }

    private DataType workingType(V output) {
        DataType type = ops.descriptor(output).dataType();
        return type == DataType.FLOAT16 ? DataType.FLOAT32 : type;
    }

    private V cast(V tensor, DataType dataType) {
        return ops.descriptor(tensor).dataType() == dataType
                ? tensor
                : ops.cast(tensor, dataType);
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
