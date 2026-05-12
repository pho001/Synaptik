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

    public static Tensor concat(int axis, List<Tensor> inputs) {
        return TensorLayoutOps.concat(axis, inputs);
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

    public static Tensor takeAlongAxis(Tensor input, Tensor indices, int dimension) {
        return TensorIndexOps.takeAlongAxis(input, indices, dimension);
    }

    public static Tensor abs(Tensor input) {
        return TensorUnaryOps.abs(input);
    }

    public static Tensor matmul(Tensor first, Tensor second) {
        return TensorMatMulOps.matmul(first, second);
    }

    public static Tensor linear(Tensor input, Tensor weight) {
        return TensorLinearOps.linear(input, weight);
    }

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

    public static Tensor sumAll(Tensor input) {
        return TensorReduceOps.sumAll(input);
    }

    public static Tensor mean(Tensor input, int dimension) {
        return TensorReduceOps.mean(input, dimension);
    }

    public static Tensor mean(Tensor input, int dimension, boolean keepDims) {
        return TensorReduceOps.mean(input, dimension, keepDims);
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

    public static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            int channelDimension,
            double epsilon
    ) {
        return TensorNormalizationOps.batchNorm(input, gamma, beta, channelDimension, epsilon);
    }

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

    public static Tensor layerNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            double epsilon
    ) {
        return TensorNormalizationOps.layerNorm(input, gamma, beta, epsilon);
    }

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
