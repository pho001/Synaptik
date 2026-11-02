package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;

import java.util.function.Function;

/**
 * Builds first-order formulas for a two-output scaled-dot-product-attention occurrence.
 *
 * <p>The exact canonical weights wrapper at producer output slot one is the saved softmax result.
 * Selection-safe contractions use public expand, multiplication, {@code where}, and reduction
 * expressions so an exact-zero saved weight excludes an exceptional opposing operand before
 * arithmetic. This owner creates no sibling attention occurrence or physical saved state.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class AttentionGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    AttentionGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds selected query, key, and value cotangents for one approved output slot.
     *
     * @param producer non-null exact original two-output attention occurrence
     * @param outputIndex selected output slot, zero for values or one for weights
     * @param gradient non-null accumulated cotangent for the selected output
     * @param selectedInputs non-null input-position-aligned selected-route flags; observed but
     *     not mutated
     * @param primal non-null resolver from original primal wrappers to lexical values
     * @return a new input-position-aligned array of selected cotangents, with {@code null} for
     *     unselected roles
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            int outputIndex,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        DataType resultType = ops.descriptor(producer.output(outputIndex)).dataType();
        V query = cast(primal.apply(producer.inputs().get(0)), resultType);
        V key = cast(primal.apply(producer.inputs().get(1)), resultType);
        V value = cast(primal.apply(producer.inputs().get(2)), resultType);
        V weights = primal.apply(producer.output(1));
        V[] result = ops.array(producer.inputs().size());

        V scoreGradient;
        if (outputIndex == 0) {
            V valueGradient = selectionSafeContractOverQuery(
                    weights, gradient);
            if (selectedInputs[2]) {
                result[2] = normalize(valueGradient, producer.inputs().get(2));
            }
            V upstreamWeights = ops.matmul(gradient, swapLastTwo(value));
            scoreGradient = softmaxPullback(weights, upstreamWeights);
        } else {
            scoreGradient = softmaxPullback(weights, gradient);
        }

        ScaledDotProductAttentionAttrs attrs =
                (ScaledDotProductAttentionAttrs) producer.operation().attrs();
        if (selectedInputs[0]) {
            V queryGradient =
                    selectionSafeContractOverKey(scoreGradient, key);
            result[0] = normalize(
                    applyScale(queryGradient, query, attrs.scale()),
                    producer.inputs().get(0));
        }
        if (selectedInputs[1]) {
            V keyGradient =
                    selectionSafeContractOverQuery(scoreGradient, query);
            result[1] = normalize(
                    applyScale(keyGradient, query, attrs.scale()),
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
     * @return a new score-cotangent expression with the weights descriptor
     */
    private V softmaxPullback(V weights, V upstream) {
        V weighted = selectionSafeProduct(weights, upstream);
        V centered = ops.sub(upstream, ops.sum(weighted, ops.descriptor(weighted).shape().rank() - 1, true));
        return selectionSafeProduct(weights, centered);
    }

    /**
     * Contracts scores with an operand over the key-position axis.
     *
     * @param scores non-null score-shaped selector Tensor
     * @param operand non-null key-position-aligned operand
     * @return a new selection-safe contraction expression
     */
    private V selectionSafeContractOverKey(V scores, V operand) {
        V expandedScores = ops.expandDims(scores, ops.descriptor(scores).shape().rank());
        V expandedOperand =
                ops.expandDims(operand, ops.descriptor(operand).shape().rank() - 2);
        V product = selectionSafeProduct(expandedScores, expandedOperand);
    return ops.sum(product, ops.descriptor(product).shape().rank() - 2, false);
    }

    /**
     * Contracts scores with an operand over the query-position axis.
     *
     * @param scores non-null score-shaped selector Tensor
     * @param operand non-null query-position-aligned operand
     * @return a new selection-safe contraction expression
     */
    private V selectionSafeContractOverQuery(V scores, V operand) {
        V expandedScores = ops.expandDims(scores, ops.descriptor(scores).shape().rank());
        V expandedOperand =
                ops.expandDims(operand, ops.descriptor(operand).shape().rank() - 1);
        V product = selectionSafeProduct(expandedScores, expandedOperand);
    return ops.sum(product, ops.descriptor(product).shape().rank() - 3, false);
    }

    /**
     * Multiplies two broadcast-compatible Tensors and then replaces positions selected by exact
     * zero in the selector with an exact typed positive zero.
     *
     * @param selector non-null floating selector Tensor
     * @param operand non-null broadcast-compatible floating operand
     * @return a new Tensor expression with excluded exceptional operands suppressed
     */
    private V selectionSafeProduct(V selector, V operand) {
        V product = ops.mul(selector, operand);
        V excluded = ops.equalTo(selector, ops.zeroLike(selector));
        return ops.where(excluded, ops.zeroLike(product), product);
    }

    /**
     * Applies either the configured scalar scale or the symbolic reciprocal square root of the
     * logical query embedding extent.
     *
     * @param value non-null unscaled cotangent
     * @param query non-null query Tensor whose final Dimension defines the default scale
     * @param configuredScale non-null optional exact configured scalar
     * @return a new scaled cotangent expression
     */
    private V applyScale(
            V value,
            V query,
            java.util.Optional<ScalarValue> configuredScale) {
        if (configuredScale.isPresent()) {
            return ops.mul(value, configuredScale.orElseThrow());
        }
        Dimension embedding =
                ops.descriptor(query).shape().dimension(ops.descriptor(query).shape().rank() - 1);
        V scale = ops.rsqrt(ops.sum(ops.expand(ops.oneBase(ops.descriptor(query).dataType()), Shape.ofDimensions(embedding))));
        return ops.mul(value, scale);
    }

    /**
     * Converts an operand to the requested result type only when required.
     *
     * @param input non-null floating operand
     * @param dataType non-null floating result type
     * @return {@code input} when its type already matches; otherwise a new cast expression
     */
    private V cast(V input, DataType dataType) {
        return ops.descriptor(input).dataType() == dataType ? input : ops.cast(input, dataType);
    }

    /**
     * Restores one selected cotangent to an original input's exact Shape and data type.
     *
     * @param gradient non-null selected cotangent
     * @param input non-null original input whose descriptor is restored
     * @return a non-null ordinary Tensor expression with the input's exact descriptor
     */
    private V normalize(V gradient, V input) {
        V result = ops.descriptor(gradient).shape().equals(ops.descriptor(input).shape())
                ? gradient
                : ops.sumToShape(gradient, ops.descriptor(input).shape());
        return ops.descriptor(result).dataType() == ops.descriptor(input).dataType()
                ? result
                : ops.cast(result, ops.descriptor(input).dataType());
    }

    /**
     * Builds a public permutation expression that exchanges only the final two axes.
     *
     * @param tensor non-null preflight-approved Tensor with rank at least two
     * @return a new Tensor expression with the final two axes exchanged
     */
    private V swapLastTwo(V tensor) {
        int rank = ops.descriptor(tensor).shape().rank();
        int[] axes = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            axes[axis] = axis;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return ops.permute(tensor, axes);
    }
}
