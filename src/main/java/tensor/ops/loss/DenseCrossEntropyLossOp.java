package tensor.ops.loss;

import operations.loss.crossEntropyLoss;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for dense-target cross-entropy loss.
 */
public final class DenseCrossEntropyLossOp {
    private DenseCrossEntropyLossOp() {
    }

    /**
     * Computes cross-entropy loss from logits and dense targets.
     *
     * @param logits floating logits tensor; must be non-null
     * @param targets floating dense target tensor with the same shape as {@code logits}
     * @param classDimension class axis; negative axes are normalized
     * @return shape {@code [1]} mean loss tensor
     * @throws IllegalArgumentException if inputs are null, non-floating, shape-mismatched,
     *                                  or the class axis is invalid
     */
    public static Tensor build(Tensor logits, Tensor targets, int classDimension) {
        if (logits == null || targets == null) {
            throw new IllegalArgumentException("crossEntropyLoss inputs cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || targets.getDataType() == DataType.BOOL
                || logits.getDataType() == DataType.INT32 || targets.getDataType() == DataType.INT32
                || logits.getDataType() == DataType.INT64 || targets.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("crossEntropyLoss requires floating numeric inputs.");
        }
        int[] logitsShape = logits.getShape();
        int[] targetShape = targets.getShape();
        if (logitsShape.length != targetShape.length) {
            throw new IllegalArgumentException("crossEntropyLoss targets rank must match logits rank.");
        }
        for (int i = 0; i < logitsShape.length; i++) {
            if (logitsShape[i] != targetShape[i]) {
                throw new IllegalArgumentException("crossEntropyLoss targets shape must match logits shape.");
            }
        }
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logitsShape.length);
        DataType outputType = TensorDataTypeUtil.binary(logits, targets);
        Tensor out = TensorPrimitiveBuilder.nary(
                new int[]{1},
                LossSupport.asInputs(logits, targets),
                new crossEntropyLoss(normalizedClassDimension),
                "crossEntropyLoss",
                outputType
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            double scale = outGrad.scalarAsDouble() / LossSupport.sampleCount(logitsShape, normalizedClassDimension);
            if (logits.getRequiresGrad()) {
                Tensor grad = logits.softmax(normalizedClassDimension).sub(targets).mul(scale);
                LossSupport.accumulateGradient(logits, grad);
            }
            if (targets.getRequiresGrad()) {
                Tensor grad = logits.logSoftmax(normalizedClassDimension).mul(-scale);
                LossSupport.accumulateGradient(targets, grad);
            }
        });
        return out;
    }

    /**
     * Computes dense-target cross-entropy loss while ignoring masked-out samples.
     *
     * @param logits floating logits tensor; must be non-null
     * @param targets floating dense target tensor with the same shape as {@code logits}
     * @param classDimension class axis; negative axes are normalized
     * @param mask BOOL mask broadcastable to the per-sample loss shape
     * @return shape {@code [1]} mean loss normalized by valid mask count
     * @throws IllegalArgumentException if inputs are null, non-floating, shape-mismatched,
     *                                  mask-incompatible, or the class axis is invalid
     */
    public static Tensor build(Tensor logits, Tensor targets, int classDimension, Tensor mask) {
        validateDenseCrossEntropyInputs(logits, targets);
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logits.getShapeUnsafe().length);
        Tensor perSampleLoss = logits.logSoftmax(normalizedClassDimension)
                .mul(targets)
                .sum(normalizedClassDimension)
                .neg();
        Tensor alignedMask = LossSupport.alignSampleMask(mask, perSampleLoss.getShapeUnsafe(), "crossEntropyLoss");
        Tensor maskedLoss = Tensor.where(alignedMask, perSampleLoss, Tensor.zerosLike(perSampleLoss));
        Tensor valid = Tensor.where(alignedMask, Tensor.onesLike(perSampleLoss), Tensor.zerosLike(perSampleLoss));
        return maskedLoss.sum().div(valid.sum().clampMin(1.0d));
    }

    private static void validateDenseCrossEntropyInputs(Tensor logits, Tensor targets) {
        if (logits == null || targets == null) {
            throw new IllegalArgumentException("crossEntropyLoss inputs cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || targets.getDataType() == DataType.BOOL
                || logits.getDataType() == DataType.INT32 || targets.getDataType() == DataType.INT32
                || logits.getDataType() == DataType.INT64 || targets.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("crossEntropyLoss requires floating numeric inputs.");
        }
        LossSupport.validateShape(
                targets.getShape(),
                logits.getShape(),
                "crossEntropyLoss targets shape must match logits shape."
        );
    }
}
