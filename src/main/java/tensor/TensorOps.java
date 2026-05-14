package tensor;

import operations.index.ScatterReduction;
import tensor.ops.binary.TensorBinaryOps;
import tensor.ops.bool.TensorBoolOps;
import tensor.ops.compare.TensorCompareOps;
import tensor.ops.conv.TensorConvOps;
import tensor.ops.dtype.TensorDTypeOps;
import tensor.ops.index.TensorIndexOps;
import tensor.ops.layout.TensorLayoutOps;
import tensor.ops.linalg.TensorAttentionOps;
import tensor.ops.linalg.TensorLinearOps;
import tensor.ops.linalg.TensorMatMulOps;
import tensor.ops.loss.TensorLossOps;
import tensor.ops.normalization.TensorNormalizationOps;
import tensor.ops.pool.TensorPoolOps;
import tensor.ops.reduction.TensorReduceOps;
import tensor.ops.select.TensorSelectOps;
import tensor.ops.unary.TensorUnaryOps;
import tensor.loss.LossReduction;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.List;

/**
 * Static facade for tensor operations.
 *
 * <p>This class delegates to the operation-specific implementations under
 * {@code tensor.ops.*}. It exists for callers that prefer static functions over
 * instance methods on {@link Tensor}. Method contracts, dtype restrictions,
 * broadcasting rules, autograd behavior, and invalid-input behavior match the
 * corresponding {@link Tensor} instance method and concrete operation class.</p>
 */
public final class TensorOps {
    private TensorOps() {}

    public static Tensor contiguous(Tensor input) {
        return TensorLayoutOps.contiguous(input);
    }

    public static Tensor reshape(Tensor input, int[] newShape) {
        return TensorLayoutOps.reshape(input, newShape);
    }

    public static Tensor expand(Tensor input, int[] newShape) {
        return TensorLayoutOps.expand(input, newShape);
    }

    public static Tensor permute(Tensor input, int[] axes) {
        return TensorLayoutOps.permute(input, axes);
    }

    public static Tensor expandDims(Tensor input, int axis) {
        return TensorLayoutOps.expandDims(input, axis);
    }

    public static Tensor squeeze(Tensor input, int axis) {
        return TensorLayoutOps.squeeze(input, axis);
    }

    public static Tensor slice(Tensor input, int[] starts, int[] ends, int[] axes, int[] steps) {
        return TensorLayoutOps.slice(input, starts, ends, axes, steps);
    }

    /**
     * Slices one axis with positive step {@code 1}.
     *
     * <p>This is the static counterpart of {@link Tensor#sliceAxis(int, int, int)}.
     * The output rank is the same as the input rank and only {@code axis} changes
     * length.</p>
     *
     * @param input source tensor
     * @param axis axis to slice; negative axes are normalized
     * @param fromInclusive inclusive start index
     * @param toExclusive exclusive end index
     * @return sliced tensor view
     */
    public static Tensor sliceAxis(Tensor input, int axis, int fromInclusive, int toExclusive) {
        return TensorLayoutOps.slice(
                input,
                new int[]{fromInclusive},
                new int[]{toExclusive},
                new int[]{axis},
                new int[]{1}
        );
    }

    public static Tensor concat(int axis, List<Tensor> inputs) {
        return TensorLayoutOps.concat(axis, inputs);
    }

    /**
     * Inserts a new axis and concatenates same-shaped tensors along it.
     *
     * <p>For example, stacking tensors shaped {@code [batch, features]} at axis
     * {@code 1} produces one tensor shaped {@code [batch, time, features]} where
     * {@code time = inputs.size()}.</p>
     *
     * @param axis insertion axis in {@code [0, rank]}; negative axes are normalized
     * @param inputs non-empty same-shaped, same-dtype input tensors
     * @return tensor with one additional axis
     */
    public static Tensor stack(int axis, List<Tensor> inputs) {
        return TensorLayoutOps.stack(axis, inputs);
    }

    /**
     * Splits a tensor along an axis and removes that axis from each output.
     *
     * @param input source tensor
     * @param axis existing axis to split; negative axes are normalized
     * @return one tensor per position along {@code axis}
     */
    public static Tensor[] unstack(Tensor input, int axis) {
        return TensorLayoutOps.unstack(input, axis);
    }

    public static Tensor pad(Tensor input, int[] before, int[] after, double constantValue) {
        return TensorLayoutOps.pad(input, before, after, constantValue);
    }

    public static Tensor tile(Tensor input, int[] repeats) {
        return TensorLayoutOps.tile(input, repeats);
    }

    public static Tensor cast(Tensor input, DataType targetType) {
        return TensorDTypeOps.cast(input, targetType);
    }

    public static Tensor add(Tensor first, Tensor second) {
        return TensorBinaryOps.add(first, second);
    }

    public static Tensor sub(Tensor first, Tensor second) {
        return TensorBinaryOps.sub(first, second);
    }

    public static Tensor mul(Tensor first, Tensor second) {
        return TensorBinaryOps.mul(first, second);
    }

    public static Tensor div(Tensor first, Tensor second) {
        return TensorBinaryOps.div(first, second);
    }

    public static Tensor min(Tensor first, Tensor second) {
        return TensorBinaryOps.min(first, second);
    }

    public static Tensor max(Tensor first, Tensor second) {
        return TensorBinaryOps.max(first, second);
    }

    public static Tensor pow(Tensor first, Tensor second) {
        return TensorBinaryOps.pow(first, second);
    }

    public static Tensor greaterThan(Tensor first, Tensor second) {
        return TensorCompareOps.greaterThan(first, second);
    }

    public static Tensor lessThan(Tensor first, Tensor second) {
        return TensorCompareOps.lessThan(first, second);
    }

    public static Tensor greaterOrEqual(Tensor first, Tensor second) {
        return TensorCompareOps.greaterOrEqual(first, second);
    }

    public static Tensor lessOrEqual(Tensor first, Tensor second) {
        return TensorCompareOps.lessOrEqual(first, second);
    }

    public static Tensor equalTo(Tensor first, Tensor second) {
        return TensorCompareOps.equalTo(first, second);
    }

    public static Tensor notEqualTo(Tensor first, Tensor second) {
        return TensorCompareOps.notEqualTo(first, second);
    }

    public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        return TensorSelectOps.where(condition, ifTrue, ifFalse);
    }

    public static Tensor select(Tensor input, int dimension, int index) {
        return TensorIndexOps.select(input, dimension, index);
    }

    public static Tensor minimum(Tensor first, Tensor second) {
        return TensorPiecewiseOps.minimum(first, second);
    }

    public static Tensor maximum(Tensor first, Tensor second) {
        return TensorPiecewiseOps.maximum(first, second);
    }

    public static Tensor logicalAnd(Tensor first, Tensor second) {
        return TensorBoolOps.logicalAnd(first, second);
    }

    public static Tensor logicalOr(Tensor first, Tensor second) {
        return TensorBoolOps.logicalOr(first, second);
    }

    public static Tensor logicalNot(Tensor input) {
        return TensorBoolOps.logicalNot(input);
    }

    public static Tensor gather(Tensor input, Tensor indices, int dimension) {
        return TensorIndexOps.gather(input, indices, dimension);
    }

    public static Tensor gatherAxis(Tensor input, Tensor indices, int axis) {
        return TensorIndexOps.gatherAxis(input, indices, axis);
    }

    /**
     * ONNX Gather-style axis take.
     *
     * <p>Result shape is {@code input.shape[:axis] + indices.shape +
     * input.shape[axis + 1:]}.</p>
     *
     * @param input source tensor
     * @param axis source axis; negative axes are normalized
     * @param indices numeric integral index tensor
     * @return gathered tensor
     */
    public static Tensor take(Tensor input, int axis, Tensor indices) {
        return TensorIndexOps.take(input, axis, indices);
    }

    /**
     * ONNX Gather-style axis take using a copied Java INT32 index vector.
     *
     * @param input source tensor
     * @param axis source axis; negative axes are normalized
     * @param indices integer index list; must be non-null and non-empty
     * @return gathered tensor
     */
    public static Tensor take(Tensor input, int axis, int[] indices) {
        return TensorIndexOps.take(input, axis, indices);
    }

    public static Tensor gatherNd(Tensor input, Tensor indices) {
        return TensorIndexOps.gatherNd(input, indices);
    }

    public static Tensor gatherNd(Tensor input, Tensor indices, int batchDims) {
        return TensorIndexOps.gatherNd(input, indices, batchDims);
    }

    public static Tensor scatterAdd(Tensor base, Tensor indices, Tensor src, int dimension) {
        return TensorIndexOps.scatterAdd(base, indices, src, dimension);
    }

    public static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis) {
        return TensorIndexOps.scatterElements(data, indices, updates, axis, ScatterReduction.NONE);
    }

    public static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
        return TensorIndexOps.scatterElements(data, indices, updates, axis, reduction);
    }

    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates) {
        return TensorIndexOps.scatterNd(data, indices, updates, ScatterReduction.NONE);
    }

    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction) {
        return TensorIndexOps.scatterNd(data, indices, updates, reduction);
    }

    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction, int batchDims) {
        return TensorIndexOps.scatterNd(data, indices, updates, reduction, batchDims);
    }

    public static Tensor scatterAxisAdd(Tensor data, Tensor indices, Tensor updates, int axis) {
        return TensorIndexOps.scatterAxisAdd(data, indices, updates, axis);
    }

    public static Tensor takeAlongAxis(Tensor input, Tensor indices, int dimension) {
        return TensorIndexOps.takeAlongAxis(input, indices, dimension);
    }

    public static Tensor abs(Tensor input) {
        return TensorUnaryOps.abs(input);
    }

    public static Tensor matmul(Tensor first, Tensor second) {
        return TensorMatMulOps.matmul(first, second);
    }

    /**
     * Applies an N-D last-dimension linear projection without bias.
     *
     * <p>Input shape is {@code [..., inFeatures]}, weight shape is
     * {@code [inFeatures, outFeatures]}, and output shape is
     * {@code [..., outFeatures]}.</p>
     *
     * @param input floating input tensor with rank at least 2
     * @param weight floating rank-2 weight tensor
     * @return projected tensor
     */
    public static Tensor linear(Tensor input, Tensor weight) {
        return TensorLinearOps.linear(input, weight);
    }

    /**
     * Applies an N-D last-dimension linear projection with bias.
     *
     * <p>Input shape is {@code [..., inFeatures]}, weight shape is
     * {@code [inFeatures, outFeatures]}, optional bias shape is
     * {@code [outFeatures]} or {@code [1, outFeatures]}, and output shape is
     * {@code [..., outFeatures]}.</p>
     *
     * @param input floating input tensor with rank at least 2
     * @param weight floating rank-2 weight tensor
     * @param bias floating bias tensor
     * @return projected tensor plus broadcast bias
     */
    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        return TensorLinearOps.linear(input, weight, bias);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Conv2dOptions options) {
        return TensorConvOps.conv2d(input, weight, options);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, Conv2dOptions options) {
        return TensorConvOps.conv2d(input, weight, bias, options);
    }

    public static Tensor maxPool2d(Tensor input, Pool2dOptions options) {
        return TensorPoolOps.maxPool2d(input, options);
    }

    public static Tensor avgPool2d(Tensor input, Pool2dOptions options) {
        return TensorPoolOps.avgPool2d(input, options);
    }

    public static Tensor scaledDotProductAttention(Tensor query, Tensor key, Tensor value, AttentionOptions options) {
        return TensorAttentionOps.scaledDotProductAttention(query, key, value, options);
    }

    public static Tensor scaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor mask, AttentionOptions options) {
        return TensorAttentionOps.scaledDotProductAttention(query, key, value, mask, options);
    }

    public static Tensor neg(Tensor input) {
        return TensorUnaryOps.neg(input);
    }

    public static Tensor log(Tensor input) {
        return TensorUnaryOps.log(input);
    }

    public static Tensor exp(Tensor input) {
        return TensorUnaryOps.exp(input);
    }

    public static Tensor fastExp(Tensor input) {
        return TensorUnaryOps.fastExp(input);
    }

    public static Tensor erf(Tensor input) {
        return TensorUnaryOps.erf(input);
    }

    public static Tensor fastTanh(Tensor input) {
        return TensorUnaryOps.fastTanh(input);
    }

    public static Tensor relu(Tensor input) {
        return TensorUnaryOps.relu(input);
    }

    public static Tensor clamp(Tensor input, double minValue, double maxValue) {
        return TensorUnaryOps.clamp(input, minValue, maxValue);
    }

    public static Tensor clampMin(Tensor input, double minValue) {
        return TensorUnaryOps.clampMin(input, minValue);
    }

    public static Tensor clampMax(Tensor input, double maxValue) {
        return TensorUnaryOps.clampMax(input, maxValue);
    }

    public static Tensor pow(Tensor input, double exponent) {
        return TensorUnaryOps.pow(input, exponent);
    }

    public static Tensor mulScalar(Tensor input, double scalar) {
        return TensorUnaryOps.mulScalar(input, scalar);
    }

    public static Tensor inv(Tensor input) {
        return TensorUnaryOps.inv(input);
    }

    public static Tensor sqrt(Tensor input) {
        return TensorUnaryOps.sqrt(input);
    }

    public static Tensor floor(Tensor input) {
        return TensorUnaryOps.floor(input);
    }

    public static Tensor ceil(Tensor input) {
        return TensorUnaryOps.ceil(input);
    }

    public static Tensor sign(Tensor input) {
        return TensorUnaryOps.sign(input);
    }

    public static Tensor sigmoid(Tensor input) {
        return TensorUnaryOps.sigmoid(input);
    }

    public static Tensor tanh(Tensor input) {
        return TensorUnaryOps.tanh(input);
    }

    public static Tensor sum(Tensor input, int dimension) {
        return TensorReduceOps.sum(input, dimension);
    }

    public static Tensor sum(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.sum(input, dimension, keepDims);
    }

    /**
     * Sums along one dimension while ignoring positions where {@code mask} is false.
     *
     * @param input floating input tensor
     * @param dimension axis to reduce; negative axes are normalized
     * @param mask BOOL mask broadcastable to {@code input}
     * @return masked sum with {@code dimension} removed
     */
    public static Tensor sum(Tensor input, int dimension, Tensor mask) {
        return TensorReduceOps.sum(input, dimension, mask);
    }

    public static Tensor sumAll(Tensor input) {
        return TensorReduceOps.sumAll(input);
    }

    public static Tensor mean(Tensor input, int dimension) {
        return TensorReduceOps.mean(input, dimension);
    }

    public static Tensor mean(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.mean(input, dimension, keepDims);
    }

    /**
     * Averages along one dimension while ignoring positions where {@code mask} is false.
     *
     * <p>The denominator is the valid mask count, not the full reduced-axis size.
     * All-masked output positions evaluate to zero.</p>
     *
     * @param input floating input tensor
     * @param dimension axis to reduce; negative axes are normalized
     * @param mask BOOL mask broadcastable to {@code input}
     * @return masked mean with {@code dimension} removed
     */
    public static Tensor mean(Tensor input, int dimension, Tensor mask) {
        return TensorReduceOps.mean(input, dimension, mask);
    }

    public static Tensor meanAll(Tensor input) {
        return TensorReduceOps.meanAll(input);
    }

    public static Tensor prod(Tensor input, int dimension) {
        return TensorReduceOps.prod(input, dimension);
    }

    public static Tensor prod(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.prod(input, dimension, keepDims);
    }

    public static Tensor prodAll(Tensor input) {
        return TensorReduceOps.prodAll(input);
    }

    public static Tensor argMax(Tensor input, int dimension) {
        return TensorReduceOps.argMax(input, dimension);
    }

    public static Tensor argMax(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.argMax(input, dimension, keepDims);
    }

    public static Tensor argMax(Tensor input, int dimension, boolean keepDims, operations.reduction.ArgMaxTiePolicy tiePolicy) {
        return TensorReduceOps.argMax(input, dimension, keepDims, tiePolicy);
    }

    public static Tensor cumSum(Tensor input, int axis) {
        return TensorReduceOps.cumSum(input, axis);
    }

    public static Tensor cumSum(Tensor input, int axis, boolean exclusive, boolean reverse) {
        return TensorReduceOps.cumSum(input, axis, exclusive, reverse);
    }

    public static Tensor softmax(Tensor input, int dimension) {
        return TensorReduceOps.softmax(input, dimension);
    }

    public static Tensor logSoftmax(Tensor input, int dimension) {
        return TensorReduceOps.logSoftmax(input, dimension);
    }

    public static Tensor min(Tensor input, int dimension) {
        return TensorReduceOps.min(input, dimension);
    }

    public static Tensor min(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.min(input, dimension, keepDims);
    }

    public static Tensor minAll(Tensor input) {
        return TensorReduceOps.minAll(input);
    }

    public static Tensor max(Tensor input, int dimension) {
        return TensorReduceOps.max(input, dimension);
    }

    public static Tensor max(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.max(input, dimension, keepDims);
    }

    public static Tensor maxAll(Tensor input) {
        return TensorReduceOps.maxAll(input);
    }

    public static Tensor all(Tensor input, int dimension) {
        return TensorReduceOps.all(input, dimension);
    }

    public static Tensor all(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.all(input, dimension, keepDims);
    }

    public static Tensor allAll(Tensor input) {
        return TensorReduceOps.allAll(input);
    }

    public static Tensor any(Tensor input, int dimension) {
        return TensorReduceOps.any(input, dimension);
    }

    public static Tensor any(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.any(input, dimension, keepDims);
    }

    public static Tensor anyAll(Tensor input) {
        return TensorReduceOps.anyAll(input);
    }

    /**
     * Applies batch-normalization arithmetic using statistics computed from input.
     *
     * <p>This is a stateless tensor primitive: it does not own running statistics
     * or layer state. Mean and variance are computed over every axis except
     * {@code channelDimension}.</p>
     *
     * @param input floating input tensor with at least two axes
     * @param gamma rank-1 scale tensor shaped {@code [channels]}
     * @param beta rank-1 bias tensor shaped {@code [channels]}
     * @param channelDimension channel axis; negative axes are normalized
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as {@code input}
     */
    public static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            int channelDimension,
            double epsilon
    ) {
        return TensorNormalizationOps.batchNorm(input, gamma, beta, channelDimension, epsilon);
    }

    /**
     * Applies batch-normalization arithmetic using caller-supplied statistics.
     *
     * <p>This inference-style form expects {@code mean} and {@code variance} to be
     * rank-1 tensors shaped {@code [channels]}. No running-statistics state is
     * updated by this method.</p>
     *
     * @param input floating input tensor
     * @param gamma rank-1 scale tensor shaped {@code [channels]}
     * @param beta rank-1 bias tensor shaped {@code [channels]}
     * @param mean rank-1 mean tensor shaped {@code [channels]}
     * @param variance rank-1 variance tensor shaped {@code [channels]}
     * @param channelDimension channel axis; negative axes are normalized
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as {@code input}
     */
    public static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Tensor mean,
            Tensor variance,
            int channelDimension,
            double epsilon
    ) {
        return TensorNormalizationOps.batchNorm(input, gamma, beta, mean, variance, channelDimension, epsilon);
    }

    /**
     * Applies layer-normalization arithmetic over trailing dimensions.
     *
     * <p>The shape of {@code gamma} and {@code beta} defines the normalized tail
     * of {@code input}. For example, input {@code [batch, time, features]} with
     * parameters {@code [features]} normalizes each feature vector independently.</p>
     *
     * @param input floating input tensor
     * @param gamma scale tensor matching the normalized input tail
     * @param beta bias tensor with the same shape as {@code gamma}
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as {@code input}
     */
    public static Tensor layerNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            double epsilon
    ) {
        return TensorNormalizationOps.layerNorm(input, gamma, beta, epsilon);
    }

    /**
     * Applies RMS-normalization arithmetic over trailing dimensions.
     *
     * <p>The shape of {@code gamma} defines the normalized tail of {@code input}.
     * The operation is stateless and does not represent a model layer object.</p>
     *
     * @param input floating input tensor
     * @param gamma scale tensor matching the normalized input tail
     * @param epsilon positive stability constant
     * @return normalized tensor with the same shape as {@code input}
     */
    public static Tensor rmsNorm(
            Tensor input,
            Tensor gamma,
            double epsilon
    ) {
        return TensorNormalizationOps.rmsNorm(input, gamma, epsilon);
    }

    public static Tensor nllLoss(Tensor logProbs, Tensor targets, int classDimension) {
        return TensorLossOps.nllLoss(logProbs, targets, classDimension);
    }

    public static Tensor crossEntropyLoss(Tensor logits, Tensor targets, int classDimension) {
        return TensorLossOps.crossEntropyLoss(logits, targets, classDimension);
    }

    /**
     * Computes dense-target cross entropy while ignoring masked-out samples.
     *
     * @param logits floating logits tensor
     * @param targets dense floating targets with the same shape as {@code logits}
     * @param classDimension class axis; negative axes are normalized
     * @param mask BOOL mask broadcastable to the per-sample loss shape
     * @return shape {@code [1]} mean loss normalized by valid mask count
     */
    public static Tensor crossEntropyLoss(Tensor logits, Tensor targets, int classDimension, Tensor mask) {
        return TensorLossOps.crossEntropyLoss(logits, targets, classDimension, mask);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension) {
        return TensorLossOps.nllLossFromIndices(logProbs, targetIndices, classDimension);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, LossReduction reduction) {
        return TensorLossOps.nllLossFromIndices(logProbs, targetIndices, classDimension, reduction);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return TensorLossOps.nllLossFromIndices(logProbs, targetIndices, classDimension, classWeights, reduction);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return TensorLossOps.nllLossFromIndices(logProbs, targetIndices, classDimension, ignoreIndex);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return TensorLossOps.nllLossFromIndices(logProbs, targetIndices, classDimension, ignoreIndex, reduction);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return TensorLossOps.nllLossFromIndices(logProbs, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension) {
        return TensorLossOps.crossEntropyLossFromIndices(logits, targetIndices, classDimension);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, LossReduction reduction) {
        return TensorLossOps.crossEntropyLossFromIndices(logits, targetIndices, classDimension, reduction);
    }

    /**
     * Computes index-target cross entropy while ignoring masked-out samples.
     *
     * @param logits floating logits tensor
     * @param targetIndices numeric integral target indices shaped like logits without the class axis
     * @param classDimension class axis; negative axes are normalized
     * @param mask BOOL mask broadcastable to {@code targetIndices}
     * @return shape {@code [1]} mean loss normalized by valid mask count
     */
    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, Tensor mask) {
        return TensorLossOps.crossEntropyLossFromIndices(logits, targetIndices, classDimension, mask);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return TensorLossOps.crossEntropyLossFromIndices(logits, targetIndices, classDimension, classWeights, reduction);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return TensorLossOps.crossEntropyLossFromIndices(logits, targetIndices, classDimension, ignoreIndex);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return TensorLossOps.crossEntropyLossFromIndices(logits, targetIndices, classDimension, ignoreIndex, reduction);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return TensorLossOps.crossEntropyLossFromIndices(logits, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }
}
