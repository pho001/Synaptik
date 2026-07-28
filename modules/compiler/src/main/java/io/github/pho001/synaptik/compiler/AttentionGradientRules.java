package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds first-order formulas for a two-output scaled-dot-product-attention occurrence.
 *
 * <p>The exact canonical weights wrapper at producer output slot one is the saved softmax result.
 * Selection-safe contractions use public expand, multiplication, {@code where}, and reduction
 * expressions so an exact-zero saved weight excludes an exceptional opposing operand before
 * arithmetic. This owner creates no sibling attention occurrence or physical saved state.</p>
 */
final class AttentionGradientRules {
    /**
     * Prevents construction of this stateless formula owner.
     */
    private AttentionGradientRules() {}

    /**
     * Builds selected query, key, and value cotangents for one approved output slot.
     *
     * @param producer non-null exact original two-output attention occurrence
     * @param outputIndex selected output slot, zero for values or one for weights
     * @param gradient non-null accumulated cotangent for the selected output
     * @param selectedInputs non-null input-position-aligned selected-route flags; observed but
     *     not mutated
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new input-position-aligned array of selected cotangents, with {@code null} for
     *     unselected roles
     */
    static Tensor[] apply(
            TensorProducer producer,
            int outputIndex,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        DataType resultType = producer.output(outputIndex).descriptor().dataType();
        Tensor query = cast(producer.inputs().get(0), resultType);
        Tensor key = cast(producer.inputs().get(1), resultType);
        Tensor value = cast(producer.inputs().get(2), resultType);
        Tensor weights = producer.output(1);
        Tensor[] result = new Tensor[producer.inputs().size()];

        Tensor scoreGradient;
        if (outputIndex == 0) {
            Tensor valueGradient = selectionSafeContractOverQuery(
                    weights, gradient, constants);
            if (selectedInputs[2]) {
                result[2] = normalize(valueGradient, producer.inputs().get(2));
            }
            Tensor upstreamWeights = gradient.matmul(swapLastTwo(value));
            scoreGradient = softmaxPullback(weights, upstreamWeights, constants);
        } else {
            scoreGradient = softmaxPullback(weights, gradient, constants);
        }

        ScaledDotProductAttentionAttrs attrs =
                (ScaledDotProductAttentionAttrs) producer.operation().attrs();
        if (selectedInputs[0]) {
            Tensor queryGradient =
                    selectionSafeContractOverKey(scoreGradient, key, constants);
            result[0] = normalize(
                    applyScale(queryGradient, query, attrs.scale(), constants),
                    producer.inputs().get(0));
        }
        if (selectedInputs[1]) {
            Tensor keyGradient =
                    selectionSafeContractOverQuery(scoreGradient, query, constants);
            result[1] = normalize(
                    applyScale(keyGradient, query, attrs.scale(), constants),
                    producer.inputs().get(1));
        }
        return result;
    }

    /**
     * Applies the softmax Jacobian transpose without allowing an exact-zero weight to select an
     * exceptional upstream value into arithmetic.
     *
     * @param weights non-null canonical same-occurrence attention weights
     * @param upstream non-null cotangent with the weights descriptor
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new score-cotangent expression with the weights descriptor
     */
    private static Tensor softmaxPullback(
            Tensor weights,
            Tensor upstream,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor weighted = selectionSafeProduct(weights, upstream, constants);
        Tensor centered = upstream.sub(weighted.sum(-1, true));
        return selectionSafeProduct(weights, centered, constants);
    }

    /**
     * Contracts scores with an operand over the key-position axis.
     *
     * @param scores non-null score-shaped selector Tensor
     * @param operand non-null key-position-aligned operand
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new selection-safe contraction expression
     */
    private static Tensor selectionSafeContractOverKey(
            Tensor scores,
            Tensor operand,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor expandedScores = scores.expandDims(scores.descriptor().shape().rank());
        Tensor expandedOperand =
                operand.expandDims(operand.descriptor().shape().rank() - 2);
        return selectionSafeProduct(expandedScores, expandedOperand, constants)
                .sum(-2, false);
    }

    /**
     * Contracts scores with an operand over the query-position axis.
     *
     * @param scores non-null score-shaped selector Tensor
     * @param operand non-null query-position-aligned operand
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new selection-safe contraction expression
     */
    private static Tensor selectionSafeContractOverQuery(
            Tensor scores,
            Tensor operand,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor expandedScores = scores.expandDims(scores.descriptor().shape().rank());
        Tensor expandedOperand =
                operand.expandDims(operand.descriptor().shape().rank() - 1);
        return selectionSafeProduct(expandedScores, expandedOperand, constants)
                .sum(-3, false);
    }

    /**
     * Multiplies two broadcast-compatible Tensors and then replaces positions selected by exact
     * zero in the selector with an exact typed positive zero.
     *
     * @param selector non-null floating selector Tensor
     * @param operand non-null broadcast-compatible floating operand
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new Tensor expression with excluded exceptional operands suppressed
     */
    private static Tensor selectionSafeProduct(
            Tensor selector,
            Tensor operand,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor product = selector.mul(operand);
        Tensor excluded = selector.equalTo(constants.zeroLike(selector));
        return Tensor.where(excluded, constants.zeroLike(product), product);
    }

    /**
     * Applies either the configured scalar scale or the symbolic reciprocal square root of the
     * logical query embedding extent.
     *
     * @param value non-null unscaled cotangent
     * @param query non-null query Tensor whose final Dimension defines the default scale
     * @param configuredScale non-null optional exact configured scalar
     * @param constants non-null request-local exact floating logical-splat owner
     * @return a new scaled cotangent expression
     */
    private static Tensor applyScale(
            Tensor value,
            Tensor query,
            java.util.Optional<ScalarValue> configuredScale,
            FirstOrderAutograd.DerivativeConstants constants) {
        if (configuredScale.isPresent()) {
            return value.mul(configuredScale.orElseThrow());
        }
        Dimension embedding =
                query.descriptor().shape().dimension(query.descriptor().shape().rank() - 1);
        Tensor scale = constants.oneBase(query.descriptor().dataType())
                .expand(Shape.ofDimensions(embedding))
                .sum()
                .rsqrt();
        return value.mul(scale);
    }

    /**
     * Converts an operand to the requested result type only when required.
     *
     * @param input non-null floating operand
     * @param dataType non-null floating result type
     * @return {@code input} when its type already matches; otherwise a new cast expression
     */
    private static Tensor cast(Tensor input, DataType dataType) {
        return input.descriptor().dataType() == dataType ? input : input.cast(dataType);
    }

    /**
     * Restores one selected cotangent to an original input's exact Shape and data type.
     *
     * @param gradient non-null selected cotangent
     * @param input non-null original input whose descriptor is restored
     * @return a non-null ordinary Tensor expression with the input's exact descriptor
     */
    private static Tensor normalize(Tensor gradient, Tensor input) {
        Tensor result = gradient.descriptor().shape().equals(input.descriptor().shape())
                ? gradient
                : gradient.sumToShape(input.descriptor().shape());
        return result.descriptor().dataType() == input.descriptor().dataType()
                ? result
                : result.cast(input.descriptor().dataType());
    }

    /**
     * Builds a public permutation expression that exchanges only the final two axes.
     *
     * @param tensor non-null preflight-approved Tensor with rank at least two
     * @return a new Tensor expression with the final two axes exchanged
     */
    private static Tensor swapLastTwo(Tensor tensor) {
        int rank = tensor.descriptor().shape().rank();
        int[] axes = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            axes[axis] = axis;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return tensor.permute(axes);
    }
}
