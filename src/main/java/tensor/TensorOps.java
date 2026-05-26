package tensor;

import operations.index.ScatterReduction;
import tensor.internal.TensorPiecewiseOps;
import tensor.ops.binary.AddOp;
import tensor.ops.binary.DivOp;
import tensor.ops.binary.MaxOp;
import tensor.ops.binary.MinOp;
import tensor.ops.binary.MulOp;
import tensor.ops.binary.PowTensorOp;
import tensor.ops.binary.SubOp;
import tensor.ops.bool.LogicalAndOp;
import tensor.ops.bool.LogicalNotOp;
import tensor.ops.bool.LogicalOrOp;
import tensor.ops.compare.EqualToOp;
import tensor.ops.compare.GreaterOrEqualOp;
import tensor.ops.compare.GreaterThanOp;
import tensor.ops.compare.LessOrEqualOp;
import tensor.ops.compare.LessThanOp;
import tensor.ops.compare.NotEqualToOp;
import tensor.ops.conv.Conv2dOp;
import tensor.ops.dtype.CastOp;
import tensor.ops.index.GatherNdOp;
import tensor.ops.index.GatherOp;
import tensor.ops.index.ScatterAddOp;
import tensor.ops.index.ScatterAxisAddOp;
import tensor.ops.index.ScatterElementsOp;
import tensor.ops.index.ScatterNdOp;
import tensor.ops.index.SelectOp;
import tensor.ops.index.TakeAlongAxisOp;
import tensor.ops.layout.ConcatOp;
import tensor.ops.layout.ContiguousOp;
import tensor.ops.layout.ExpandDimsOp;
import tensor.ops.layout.ExpandOp;
import tensor.ops.layout.Fold2dOp;
import tensor.ops.layout.PadOp;
import tensor.ops.layout.PermuteOp;
import tensor.ops.layout.ReshapeOp;
import tensor.ops.layout.SliceOp;
import tensor.ops.layout.SqueezeOp;
import tensor.ops.layout.StackOp;
import tensor.ops.layout.TileOp;
import tensor.ops.layout.UnstackOp;
import tensor.ops.layout.UnfoldAxisOp;
import tensor.ops.layout.Unfold2dOp;
import tensor.ops.linalg.LinearOp;
import tensor.ops.linalg.MatMulOp;
import tensor.ops.linalg.ScaledDotProductAttentionOp;
import tensor.ops.loss.CrossEntropyLossFromIndicesOp;
import tensor.ops.loss.DenseCrossEntropyLossOp;
import tensor.ops.loss.DenseNllLossOp;
import tensor.ops.loss.NllLossFromIndicesOp;
import tensor.ops.normalization.BatchNormOp;
import tensor.ops.normalization.LayerNormOp;
import tensor.ops.normalization.RmsNormOp;
import tensor.ops.pool.AvgPool2dOp;
import tensor.ops.pool.MaxPool2dOp;
import tensor.ops.reduction.AllOp;
import tensor.ops.reduction.AnyOp;
import tensor.ops.reduction.ArgMaxOp;
import tensor.ops.reduction.CumSumOp;
import tensor.ops.reduction.LogSoftmaxOp;
import tensor.ops.reduction.MeanOp;
import tensor.ops.reduction.ProdOp;
import tensor.ops.reduction.ReduceMaxOp;
import tensor.ops.reduction.ReduceMinOp;
import tensor.ops.reduction.SoftmaxOp;
import tensor.ops.reduction.SumOp;
import tensor.ops.select.WhereOp;
import tensor.ops.unary.AbsOp;
import tensor.ops.unary.CeilOp;
import tensor.ops.unary.ClampMaxOp;
import tensor.ops.unary.ClampMinOp;
import tensor.ops.unary.ClampOp;
import tensor.ops.unary.ErfOp;
import tensor.ops.unary.ExpOp;
import tensor.ops.unary.FastExpOp;
import tensor.ops.unary.FastTanhOp;
import tensor.ops.unary.FloorOp;
import tensor.ops.unary.InvOp;
import tensor.ops.unary.LogOp;
import tensor.ops.unary.MulScalarOp;
import tensor.ops.unary.NegOp;
import tensor.ops.unary.PowScalarOp;
import tensor.ops.unary.ReluOp;
import tensor.ops.unary.SigmoidOp;
import tensor.ops.unary.SignOp;
import tensor.ops.unary.SqrtOp;
import tensor.ops.unary.TanhOp;
import tensor.loss.LossReduction;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;
import tensor.options.Window2dOptions;

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
        return ContiguousOp.build(input);
    }

    public static Tensor reshape(Tensor input, int[] newShape) {
        return ReshapeOp.build(input, newShape);
    }

    public static Tensor expand(Tensor input, int[] newShape) {
        return ExpandOp.build(input, newShape);
    }

    public static Tensor permute(Tensor input, int[] axes) {
        return PermuteOp.build(input, axes);
    }

    public static Tensor expandDims(Tensor input, int axis) {
        return ExpandDimsOp.build(input, axis);
    }

    public static Tensor squeeze(Tensor input, int axis) {
        return SqueezeOp.build(input, axis);
    }

    public static Tensor slice(Tensor input, int[] starts, int[] ends, int[] axes, int[] steps) {
        return SliceOp.build(input, starts, ends, axes, steps);
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
        return SliceOp.build(
                input,
                new int[]{fromInclusive},
                new int[]{toExclusive},
                new int[]{axis},
                new int[]{1}
        );
    }

    public static Tensor concat(int axis, List<Tensor> inputs) {
        return ConcatOp.build(axis, inputs);
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
        return StackOp.build(axis, inputs);
    }

    /**
     * Splits a tensor along an axis and removes that axis from each output.
     *
     * @param input source tensor
     * @param axis existing axis to split; negative axes are normalized
     * @return one tensor per position along {@code axis}
     */
    public static Tensor[] unstack(Tensor input, int axis) {
        return UnstackOp.build(input, axis);
    }

    public static Tensor pad(Tensor input, int[] before, int[] after, double constantValue) {
        return PadOp.build(input, before, after, constantValue);
    }

    public static Tensor tile(Tensor input, int[] repeats) {
        return TileOp.build(input, repeats);
    }

    public static Tensor unfold(Tensor input, int axis, int size, int step) {
        return UnfoldAxisOp.build(input, axis, size, step);
    }

    public static Tensor unfold2d(Tensor input, Window2dOptions options) {
        return Unfold2dOp.build(input, options);
    }

    public static Tensor fold2d(Tensor input, int[] outputShape, Window2dOptions options) {
        return Fold2dOp.build(input, outputShape, options);
    }

    public static Tensor cast(Tensor input, DataType targetType) {
        return CastOp.build(input, targetType);
    }

    public static Tensor add(Tensor first, Tensor second) {
        return AddOp.build(first, second);
    }

    public static Tensor sub(Tensor first, Tensor second) {
        return SubOp.build(first, second);
    }

    public static Tensor mul(Tensor first, Tensor second) {
        return MulOp.build(first, second);
    }

    public static Tensor div(Tensor first, Tensor second) {
        return DivOp.build(first, second);
    }

    public static Tensor min(Tensor first, Tensor second) {
        return MinOp.build(first, second);
    }

    public static Tensor max(Tensor first, Tensor second) {
        return MaxOp.build(first, second);
    }

    public static Tensor pow(Tensor first, Tensor second) {
        return PowTensorOp.build(first, second);
    }

    public static Tensor greaterThan(Tensor first, Tensor second) {
        return GreaterThanOp.build(first, second);
    }

    public static Tensor lessThan(Tensor first, Tensor second) {
        return LessThanOp.build(first, second);
    }

    public static Tensor greaterOrEqual(Tensor first, Tensor second) {
        return GreaterOrEqualOp.build(first, second);
    }

    public static Tensor lessOrEqual(Tensor first, Tensor second) {
        return LessOrEqualOp.build(first, second);
    }

    public static Tensor equalTo(Tensor first, Tensor second) {
        return EqualToOp.build(first, second);
    }

    public static Tensor notEqualTo(Tensor first, Tensor second) {
        return NotEqualToOp.build(first, second);
    }

    public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        return WhereOp.build(condition, ifTrue, ifFalse);
    }

    public static Tensor select(Tensor input, int dimension, int index) {
        return SelectOp.build(input, dimension, index);
    }

    public static Tensor minimum(Tensor first, Tensor second) {
        return TensorPiecewiseOps.minimum(first, second);
    }

    public static Tensor maximum(Tensor first, Tensor second) {
        return TensorPiecewiseOps.maximum(first, second);
    }

    public static Tensor logicalAnd(Tensor first, Tensor second) {
        return LogicalAndOp.build(first, second);
    }

    public static Tensor logicalOr(Tensor first, Tensor second) {
        return LogicalOrOp.build(first, second);
    }

    public static Tensor logicalNot(Tensor input) {
        return LogicalNotOp.build(input);
    }

    public static Tensor gather(Tensor input, Tensor indices, int dimension) {
        return GatherOp.build(input, indices, dimension);
    }

    public static Tensor gatherAxis(Tensor input, Tensor indices, int axis) {
        return GatherOp.buildAxis(input, indices, axis);
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
        return GatherOp.take(input, axis, indices);
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
        return GatherOp.take(input, axis, indices);
    }

    public static Tensor gatherNd(Tensor input, Tensor indices) {
        return GatherNdOp.build(input, indices);
    }

    public static Tensor gatherNd(Tensor input, Tensor indices, int batchDims) {
        return GatherNdOp.build(input, indices, batchDims);
    }

    public static Tensor scatterAdd(Tensor base, Tensor indices, Tensor src, int dimension) {
        return ScatterAddOp.build(base, indices, src, dimension);
    }

    public static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis) {
        return ScatterElementsOp.build(data, indices, updates, axis, ScatterReduction.NONE);
    }

    public static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
        return ScatterElementsOp.build(data, indices, updates, axis, reduction);
    }

    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates) {
        return ScatterNdOp.build(data, indices, updates, ScatterReduction.NONE);
    }

    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction) {
        return ScatterNdOp.build(data, indices, updates, reduction);
    }

    public static Tensor scatterNd(Tensor data, Tensor indices, Tensor updates, ScatterReduction reduction, int batchDims) {
        return ScatterNdOp.build(data, indices, updates, reduction, batchDims);
    }

    public static Tensor scatterAxisAdd(Tensor data, Tensor indices, Tensor updates, int axis) {
        return ScatterAxisAddOp.build(data, indices, updates, axis);
    }

    public static Tensor takeAlongAxis(Tensor input, Tensor indices, int dimension) {
        return TakeAlongAxisOp.build(input, indices, dimension);
    }

    public static Tensor abs(Tensor input) {
        return AbsOp.build(input);
    }

    public static Tensor matmul(Tensor first, Tensor second) {
        return MatMulOp.build(first, second);
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
        return LinearOp.build(input, weight);
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
        return LinearOp.build(input, weight, bias);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Conv2dOptions options) {
        return Conv2dOp.build(input, weight, options);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, Conv2dOptions options) {
        return Conv2dOp.build(input, weight, bias, options);
    }

    public static Tensor maxPool2d(Tensor input, Pool2dOptions options) {
        return MaxPool2dOp.build(input, options);
    }

    public static Tensor avgPool2d(Tensor input, Pool2dOptions options) {
        return AvgPool2dOp.build(input, options);
    }

    public static Tensor scaledDotProductAttention(Tensor query, Tensor key, Tensor value, AttentionOptions options) {
        return ScaledDotProductAttentionOp.build(query, key, value, options);
    }

    public static Tensor scaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor mask, AttentionOptions options) {
        return ScaledDotProductAttentionOp.build(query, key, value, mask, options);
    }

    public static Tensor neg(Tensor input) {
        return NegOp.build(input);
    }

    public static Tensor log(Tensor input) {
        return LogOp.build(input);
    }

    public static Tensor exp(Tensor input) {
        return ExpOp.build(input);
    }

    public static Tensor fastExp(Tensor input) {
        return FastExpOp.build(input);
    }

    public static Tensor erf(Tensor input) {
        return ErfOp.build(input);
    }

    public static Tensor fastTanh(Tensor input) {
        return FastTanhOp.build(input);
    }

    public static Tensor relu(Tensor input) {
        return ReluOp.build(input);
    }

    public static Tensor clamp(Tensor input, double minValue, double maxValue) {
        return ClampOp.build(input, minValue, maxValue);
    }

    public static Tensor clampMin(Tensor input, double minValue) {
        return ClampMinOp.build(input, minValue);
    }

    public static Tensor clampMax(Tensor input, double maxValue) {
        return ClampMaxOp.build(input, maxValue);
    }

    public static Tensor pow(Tensor input, double exponent) {
        return PowScalarOp.build(input, exponent);
    }

    public static Tensor mulScalar(Tensor input, double scalar) {
        return MulScalarOp.build(input, scalar);
    }

    public static Tensor inv(Tensor input) {
        return InvOp.build(input);
    }

    public static Tensor sqrt(Tensor input) {
        return SqrtOp.build(input);
    }

    public static Tensor floor(Tensor input) {
        return FloorOp.build(input);
    }

    public static Tensor ceil(Tensor input) {
        return CeilOp.build(input);
    }

    public static Tensor sign(Tensor input) {
        return SignOp.build(input);
    }

    public static Tensor sigmoid(Tensor input) {
        return SigmoidOp.build(input);
    }

    public static Tensor tanh(Tensor input) {
        return TanhOp.build(input);
    }

    public static Tensor sum(Tensor input, int dimension) {
        return SumOp.build(input, dimension);
    }

    public static Tensor sum(Tensor input, int dimension, boolean keepDims) {
        return SumOp.build(input, dimension, keepDims);
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
        return SumOp.buildMasked(input, dimension, mask);
    }

    public static Tensor sumAll(Tensor input) {
        return SumOp.buildAll(input);
    }

    public static Tensor mean(Tensor input, int dimension) {
        return MeanOp.build(input, dimension);
    }

    public static Tensor mean(Tensor input, int dimension, boolean keepDims) {
        return MeanOp.build(input, dimension, keepDims);
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
        return MeanOp.buildMasked(input, dimension, mask);
    }

    public static Tensor meanAll(Tensor input) {
        return MeanOp.buildAll(input);
    }

    public static Tensor prod(Tensor input, int dimension) {
        return ProdOp.build(input, dimension);
    }

    public static Tensor prod(Tensor input, int dimension, boolean keepDims) {
        return ProdOp.build(input, dimension, keepDims);
    }

    public static Tensor prodAll(Tensor input) {
        return ProdOp.buildAll(input);
    }

    public static Tensor argMax(Tensor input, int dimension) {
        return ArgMaxOp.build(input, dimension);
    }

    public static Tensor argMax(Tensor input, int dimension, boolean keepDims) {
        return ArgMaxOp.build(input, dimension, keepDims);
    }

    public static Tensor argMax(Tensor input, int dimension, boolean keepDims, operations.reduction.ArgMaxTiePolicy tiePolicy) {
        return ArgMaxOp.build(input, dimension, keepDims, tiePolicy);
    }

    public static Tensor cumSum(Tensor input, int axis) {
        return CumSumOp.build(input, axis);
    }

    public static Tensor cumSum(Tensor input, int axis, boolean exclusive, boolean reverse) {
        return CumSumOp.build(input, axis, exclusive, reverse);
    }

    public static Tensor softmax(Tensor input, int dimension) {
        return SoftmaxOp.build(input, dimension);
    }

    public static Tensor logSoftmax(Tensor input, int dimension) {
        return LogSoftmaxOp.build(input, dimension);
    }

    public static Tensor min(Tensor input, int dimension) {
        return ReduceMinOp.build(input, dimension);
    }

    public static Tensor min(Tensor input, int dimension, boolean keepDims) {
        return ReduceMinOp.build(input, dimension, keepDims);
    }

    public static Tensor minAll(Tensor input) {
        return ReduceMinOp.buildAll(input);
    }

    public static Tensor max(Tensor input, int dimension) {
        return ReduceMaxOp.build(input, dimension);
    }

    public static Tensor max(Tensor input, int dimension, boolean keepDims) {
        return ReduceMaxOp.build(input, dimension, keepDims);
    }

    public static Tensor maxAll(Tensor input) {
        return ReduceMaxOp.buildAll(input);
    }

    public static Tensor all(Tensor input, int dimension) {
        return AllOp.build(input, dimension);
    }

    public static Tensor all(Tensor input, int dimension, boolean keepDims) {
        return AllOp.build(input, dimension, keepDims);
    }

    public static Tensor allAll(Tensor input) {
        return AllOp.buildAll(input);
    }

    public static Tensor any(Tensor input, int dimension) {
        return AnyOp.build(input, dimension);
    }

    public static Tensor any(Tensor input, int dimension, boolean keepDims) {
        return AnyOp.build(input, dimension, keepDims);
    }

    public static Tensor anyAll(Tensor input) {
        return AnyOp.buildAll(input);
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
        return BatchNormOp.build(input, gamma, beta, channelDimension, epsilon);
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
        return BatchNormOp.build(input, gamma, beta, mean, variance, channelDimension, epsilon);
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
        return LayerNormOp.build(input, gamma, beta, epsilon);
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
        return RmsNormOp.build(input, gamma, epsilon);
    }

    public static Tensor nllLoss(Tensor logProbs, Tensor targets, int classDimension) {
        return DenseNllLossOp.build(logProbs, targets, classDimension);
    }

    public static Tensor crossEntropyLoss(Tensor logits, Tensor targets, int classDimension) {
        return DenseCrossEntropyLossOp.build(logits, targets, classDimension);
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
        return DenseCrossEntropyLossOp.build(logits, targets, classDimension, mask);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension) {
        return NllLossFromIndicesOp.build(logProbs, targetIndices, classDimension);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, LossReduction reduction) {
        return NllLossFromIndicesOp.build(logProbs, targetIndices, classDimension, reduction);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return NllLossFromIndicesOp.build(logProbs, targetIndices, classDimension, classWeights, reduction);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return NllLossFromIndicesOp.build(logProbs, targetIndices, classDimension, ignoreIndex);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return NllLossFromIndicesOp.build(logProbs, targetIndices, classDimension, ignoreIndex, reduction);
    }

    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return NllLossFromIndicesOp.build(logProbs, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension) {
        return CrossEntropyLossFromIndicesOp.build(logits, targetIndices, classDimension);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, LossReduction reduction) {
        return CrossEntropyLossFromIndicesOp.build(logits, targetIndices, classDimension, reduction);
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
        return CrossEntropyLossFromIndicesOp.build(logits, targetIndices, classDimension, mask);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return CrossEntropyLossFromIndicesOp.build(logits, targetIndices, classDimension, classWeights, reduction);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return CrossEntropyLossFromIndicesOp.build(logits, targetIndices, classDimension, ignoreIndex);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        return CrossEntropyLossFromIndicesOp.build(logits, targetIndices, classDimension, ignoreIndex, reduction);
    }

    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return CrossEntropyLossFromIndicesOp.build(logits, targetIndices, classDimension, ignoreIndex, classWeights, reduction);
    }
}
