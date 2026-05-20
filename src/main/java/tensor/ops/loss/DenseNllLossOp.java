package tensor.ops.loss;

import operations.loss.nllLoss;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypes;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for dense-target negative log-likelihood loss.
 */
public final class DenseNllLossOp {
    private DenseNllLossOp() {
    }

    /**
     * Computes negative log-likelihood loss from log-probabilities and dense targets.
     *
     * @param logProbs floating log-probability tensor; must be non-null
     * @param targets floating dense target tensor with the same shape as {@code logProbs}
     * @param classDimension class axis; negative axes are normalized
     * @return shape {@code [1]} mean loss tensor
     * @throws IllegalArgumentException if inputs are null, non-floating, shape-mismatched,
     *                                  or the class axis is invalid
     */
    public static Tensor build(Tensor logProbs, Tensor targets, int classDimension) {
        if (logProbs == null || targets == null) {
            throw new IllegalArgumentException("nllLoss inputs cannot be null");
        }
        if (logProbs.getDataType() == DataType.BOOL || targets.getDataType() == DataType.BOOL
                || logProbs.getDataType() == DataType.INT32 || targets.getDataType() == DataType.INT32
                || logProbs.getDataType() == DataType.INT64 || targets.getDataType() == DataType.INT64) {
            throw new IllegalArgumentException("nllLoss requires floating numeric inputs.");
        }
        int[] logShape = logProbs.getShape();
        int[] targetShape = targets.getShape();
        if (logShape.length != targetShape.length) {
            throw new IllegalArgumentException("nllLoss targets rank must match logProbs rank.");
        }
        for (int i = 0; i < logShape.length; i++) {
            if (logShape[i] != targetShape[i]) {
                throw new IllegalArgumentException("nllLoss targets shape must match logProbs shape.");
            }
        }
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logShape.length);
        DataType outputType = TensorDTypes.promoteFloating(logProbs.getDataType(), targets.getDataType());
        Tensor out = TensorPrimitiveBuilder.nary(
                new int[]{1},
                LossSupport.asInputs(logProbs, targets),
                new nllLoss(normalizedClassDimension),
                "nllLoss",
                outputType
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            double scale = outGrad.scalarAsDouble() / LossSupport.sampleCount(logShape, normalizedClassDimension);
            if (logProbs.getRequiresGrad()) {
                context.accumulate(logProbs, targets.mul(-scale));
            }
            if (targets.getRequiresGrad()) {
                context.accumulate(targets, logProbs.mul(-scale));
            }
        });
        return out;
    }
}
