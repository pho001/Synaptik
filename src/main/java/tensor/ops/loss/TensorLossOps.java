package tensor.ops.loss;

import operations.loss.crossEntropyLoss;
import operations.loss.crossEntropyLossIndices;
import operations.loss.nllLoss;
import graph.optimizer.intent.BackendIntentPropagator;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorDataTypeUtil;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;
import tensor.loss.LossReduction;

import java.util.Arrays;

/**
 * Differentiable loss functions for floating tensors.
 *
 * <p>Class-probability targets use the same shape as the logits/log-probability
 * tensor. Index targets use the input shape with the class axis removed and may
 * be supplied as numeric integral tensors. Methods return graph tensors and do
 * not mutate inputs.</p>
 */
public final class TensorLossOps {
    private TensorLossOps() {
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
    public static Tensor nllLoss(Tensor logProbs, Tensor targets, int classDimension) {
        if (logProbs == null || targets == null) {
            throw new IllegalArgumentException("nllLoss inputs cannot be null");
        }
        if (logProbs.getDataType() == DataType.BOOL || targets.getDataType() == DataType.BOOL
                || logProbs.getDataType() == DataType.INT32 || targets.getDataType() == DataType.INT32) {
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
        DataType outputType = TensorDataTypeUtil.binary(logProbs, targets);
        Tensor out = TensorPrimitiveBuilder.nary(
                new int[]{1},
                LossSupport.asInputs(logProbs, targets),
                new nllLoss(normalizedClassDimension),
                "nllLoss",
                outputType
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            double scale = outGrad.scalarAsDouble() / LossSupport.sampleCount(logShape, normalizedClassDimension);
            if (logProbs.getRequiresGrad()) {
                LossSupport.accumulateGradient(logProbs, targets.mul(-scale));
            }
            if (targets.getRequiresGrad()) {
                LossSupport.accumulateGradient(targets, logProbs.mul(-scale));
            }
        });
        return out;
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
    public static Tensor crossEntropyLoss(Tensor logits, Tensor targets, int classDimension) {
        if (logits == null || targets == null) {
            throw new IllegalArgumentException("crossEntropyLoss inputs cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || targets.getDataType() == DataType.BOOL
                || logits.getDataType() == DataType.INT32 || targets.getDataType() == DataType.INT32) {
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
     * Computes mean NLL loss from integer-like class indices.
     *
     * @param logProbs floating log-probability tensor
     * @param targetIndices numeric integral indices shaped like {@code logProbs} without the class axis
     * @param classDimension class axis; negative axes are normalized
     * @return shape {@code [1]} mean loss tensor
     */
    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, LossReduction.MEAN);
    }

    /**
     * Computes weighted NLL loss from integer-like class indices.
     *
     * @param logProbs floating log-probability tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param classWeights rank-1 floating tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     * @throws IllegalArgumentException if inputs are null, dtypes/shapes are invalid,
     *                                  or reduction is null
     */
    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, classWeights, null, reduction);
    }

    /**
     * Computes NLL loss from integer-like class indices.
     *
     * @param logProbs floating log-probability tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     * @throws IllegalArgumentException if inputs are null, dtypes/shapes are invalid,
     *                                  or reduction is null
     */
    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, LossReduction reduction) {
        if (logProbs == null || targetIndices == null) {
            throw new IllegalArgumentException("nllLossFromIndices inputs cannot be null");
        }
        if (reduction == null) {
            throw new IllegalArgumentException("nllLossFromIndices reduction cannot be null");
        }
        if (logProbs.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("nllLossFromIndices requires numeric tensors and numeric integral indices.");
        }
        int[] logShape = logProbs.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logShape.length);
        int[] expectedIndexShape = LossSupport.reduceShape(logShape, normalizedClassDimension);
        LossSupport.validateShape(targetIndices.getShape(), expectedIndexShape, "nllLossFromIndices targetIndices shape must equal logProbs shape without class axis.");
        Tensor perSampleLoss = logProbs.gather(targetIndices, normalizedClassDimension).neg();
        return LossSupport.applyLossReduction(perSampleLoss, null, reduction);
    }

    /**
     * Computes mean NLL loss while ignoring one target index value.
     *
     * @param logProbs floating log-probability tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from the loss and mean denominator
     * @return shape {@code [1]} mean loss tensor
     */
    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, ignoreIndex, LossReduction.MEAN);
    }

    /**
     * Computes weighted NLL loss while ignoring one target index value.
     *
     * @param logProbs floating log-probability tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from the loss and reduction weights
     * @param classWeights rank-1 floating tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     */
    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, classWeights, Integer.valueOf(ignoreIndex), reduction);
    }

    /**
     * Computes NLL loss while ignoring one target index value.
     *
     * @param logProbs floating log-probability tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from the loss and reduction weights
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     */
    public static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        if (logProbs == null || targetIndices == null) {
            throw new IllegalArgumentException("nllLossFromIndices inputs cannot be null");
        }
        if (reduction == null) {
            throw new IllegalArgumentException("nllLossFromIndices reduction cannot be null");
        }
        if (logProbs.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("nllLossFromIndices requires numeric tensors and numeric integral indices.");
        }
        int[] logShape = logProbs.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logShape.length);
        int[] expectedIndexShape = LossSupport.reduceShape(logShape, normalizedClassDimension);
        LossSupport.validateShape(targetIndices.getShape(), expectedIndexShape, "nllLossFromIndices targetIndices shape must equal logProbs shape without class axis.");

        Tensor validMask = LossSupport.buildIgnoreMask(targetIndices, ignoreIndex);
        Tensor safeIndices = LossSupport.buildSafeIndices(targetIndices, ignoreIndex);
        Tensor gathered = logProbs.gather(safeIndices, normalizedClassDimension);
        Tensor validMaskNumeric = Tensor.where(validMask, Tensor.onesLike(gathered), Tensor.zerosLike(gathered));
        Tensor maskedLoss = Tensor.where(validMask, gathered.neg(), Tensor.zerosLike(gathered));
        return LossSupport.applyLossReduction(maskedLoss, validMaskNumeric, reduction);
    }

    /**
     * Computes mean cross-entropy loss from logits and integer-like class indices.
     *
     * @param logits floating logits tensor
     * @param targetIndices numeric integral indices shaped like {@code logits} without the class axis
     * @param classDimension class axis; negative axes are normalized
     * @return shape {@code [1]} mean loss tensor
     */
    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, LossReduction.MEAN);
    }

    /**
     * Computes weighted cross-entropy loss from logits and integer-like targets.
     *
     * @param logits floating logits tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param classWeights rank-1 floating tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     * @throws IllegalArgumentException if inputs are null, dtypes/shapes are invalid,
     *                                  or reduction is null
     */
    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, classWeights, null, reduction);
    }

    /**
     * Computes cross-entropy loss from logits and integer-like targets.
     *
     * @param logits floating logits tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     * @throws IllegalArgumentException if inputs are null, dtypes/shapes are invalid,
     *                                  or reduction is null
     */
    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, LossReduction reduction) {
        if (logits == null || targetIndices == null) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices inputs cannot be null");
        }
        if (reduction == null) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices reduction cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices requires numeric logits and numeric integral indices.");
        }
        int[] logitsShape = logits.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logitsShape.length);
        int[] expectedIndexShape = LossSupport.reduceShape(logitsShape, normalizedClassDimension);
        LossSupport.validateShape(targetIndices.getShape(), expectedIndexShape, "crossEntropyLossFromIndices targetIndices shape must equal logits shape without class axis.");
        return crossEntropyLossFromIndicesPrimitive(logits, targetIndices, normalizedClassDimension, reduction, null);
    }

    /**
     * Computes mean cross-entropy loss while ignoring one target index value.
     *
     * @param logits floating logits tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from the loss and mean denominator
     * @return shape {@code [1]} mean loss tensor
     */
    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, ignoreIndex, LossReduction.MEAN);
    }

    /**
     * Computes weighted cross-entropy loss while ignoring one target index value.
     *
     * @param logits floating logits tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from the loss and reduction weights
     * @param classWeights rank-1 floating tensor with one weight per class
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     */
    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, classWeights, Integer.valueOf(ignoreIndex), reduction);
    }

    /**
     * Computes cross-entropy loss while ignoring one target index value.
     *
     * @param logits floating logits tensor
     * @param targetIndices numeric integral target indices
     * @param classDimension class axis; negative axes are normalized
     * @param ignoreIndex target value excluded from the loss and reduction weights
     * @param reduction reduction mode; must be non-null
     * @return loss tensor with shape determined by {@code reduction}
     */
    public static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
        if (logits == null || targetIndices == null) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices inputs cannot be null");
        }
        if (reduction == null) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices reduction cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices requires numeric logits and numeric integral indices.");
        }
        int[] logitsShape = logits.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logitsShape.length);
        int[] expectedIndexShape = LossSupport.reduceShape(logitsShape, normalizedClassDimension);
        LossSupport.validateShape(targetIndices.getShape(), expectedIndexShape, "crossEntropyLossFromIndices targetIndices shape must equal logits shape without class axis.");
        return crossEntropyLossFromIndicesPrimitive(logits, targetIndices, normalizedClassDimension, reduction, ignoreIndex);
    }

    private static Tensor nllLossFromIndices(
            Tensor logProbs,
            Tensor targetIndices,
            int classDimension,
            Tensor classWeights,
            Integer ignoreIndexOrNull,
            LossReduction reduction
    ) {
        if (logProbs == null || targetIndices == null || classWeights == null) {
            throw new IllegalArgumentException("weighted nllLossFromIndices inputs cannot be null");
        }
        if (reduction == null) {
            throw new IllegalArgumentException("weighted nllLossFromIndices reduction cannot be null");
        }
        if (logProbs.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL || classWeights.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("weighted nllLossFromIndices requires numeric tensors and numeric integral indices.");
        }
        if (logProbs.getDataType() == DataType.INT32 || classWeights.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("weighted nllLossFromIndices requires floating logProbs and floating classWeights.");
        }
        if (logProbs.getDataType() != classWeights.getDataType()) {
            throw new IllegalArgumentException("classWeights dtype must match logProbs dtype.");
        }

        int[] logShape = logProbs.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logShape.length);
        int[] expectedIndexShape = LossSupport.reduceShape(logShape, normalizedClassDimension);
        LossSupport.validateShape(targetIndices.getShape(), expectedIndexShape, "weighted nllLossFromIndices targetIndices shape must equal logProbs shape without class axis.");
        LossSupport.validateClassWeightsShape(classWeights, logShape[normalizedClassDimension]);

        Tensor effectiveIndices = targetIndices;
        Tensor sampleMask = null;
        if (ignoreIndexOrNull != null) {
            sampleMask = LossSupport.buildIgnoreMask(targetIndices, ignoreIndexOrNull);
            effectiveIndices = LossSupport.buildSafeIndices(targetIndices, ignoreIndexOrNull);
        }

        Tensor gatheredLogProbs = logProbs.gather(effectiveIndices, normalizedClassDimension);
        int[] expandedWeightShape = new int[logShape.length];
        Arrays.fill(expandedWeightShape, 1);
        expandedWeightShape[normalizedClassDimension] = logShape[normalizedClassDimension];
        Tensor gatheredWeights = classWeights
                .reshape(expandedWeightShape)
                .expand(logShape)
                .takeAlongAxis(effectiveIndices.expandDims(normalizedClassDimension), normalizedClassDimension)
                .squeeze(normalizedClassDimension);
        Tensor perSampleLoss = gatheredLogProbs.neg().mul(gatheredWeights);
        Tensor reductionWeights = gatheredWeights;

        if (sampleMask != null) {
            Tensor zeroLoss = Tensor.zerosLike(perSampleLoss);
            perSampleLoss = Tensor.where(sampleMask, perSampleLoss, zeroLoss);
            reductionWeights = Tensor.where(sampleMask, reductionWeights, Tensor.zerosLike(reductionWeights));
        }

        return LossSupport.applyLossReduction(perSampleLoss, reductionWeights, reduction);
    }

    private static Tensor crossEntropyLossFromIndices(
            Tensor logits,
            Tensor targetIndices,
            int classDimension,
            Tensor classWeights,
            Integer ignoreIndexOrNull,
            LossReduction reduction
    ) {
        if (logits == null || targetIndices == null || classWeights == null) {
            throw new IllegalArgumentException("weighted crossEntropyLossFromIndices inputs cannot be null");
        }
        if (reduction == null) {
            throw new IllegalArgumentException("weighted crossEntropyLossFromIndices reduction cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL || classWeights.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("weighted crossEntropyLossFromIndices requires numeric logits, floating weights, and numeric integral indices.");
        }
        if (logits.getDataType() == DataType.INT32 || classWeights.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("weighted crossEntropyLossFromIndices requires floating logits and floating classWeights.");
        }
        if (logits.getDataType() != classWeights.getDataType()) {
            throw new IllegalArgumentException("classWeights dtype must match logits dtype.");
        }
        int[] logitsShape = logits.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logitsShape.length);
        int[] expectedIndexShape = LossSupport.reduceShape(logitsShape, normalizedClassDimension);
        LossSupport.validateShape(targetIndices.getShape(), expectedIndexShape, "weighted crossEntropyLossFromIndices targetIndices shape must equal logits shape without class axis.");
        LossSupport.validateClassWeightsShape(classWeights, logitsShape[normalizedClassDimension]);

        if (ignoreIndexOrNull == null) {
            return logits.logSoftmax(normalizedClassDimension).nllLossFromIndices(targetIndices, normalizedClassDimension, classWeights, reduction);
        }
        return logits.logSoftmax(normalizedClassDimension).nllLossFromIndices(targetIndices, normalizedClassDimension, ignoreIndexOrNull, classWeights, reduction);
    }

    private static Tensor crossEntropyLossFromIndicesPrimitive(
            Tensor logits,
            Tensor targetIndices,
            int normalizedClassDimension,
            LossReduction reduction,
            Integer ignoreIndexOrNull
    ) {
        DataType outputType = logits.getDataType();
        int[] logitsShape = logits.getShape();
        int[] reducedShape = LossSupport.reduceShape(logitsShape, normalizedClassDimension);
        int[] outputShape = reduction == LossReduction.NONE ? reducedShape : new int[]{1};
        Tensor out = TensorPrimitiveBuilder.nary(
                outputShape,
                LossSupport.asInputs(logits, targetIndices),
                new crossEntropyLossIndices(normalizedClassDimension, reduction, ignoreIndexOrNull),
                "crossEntropyLossFromIndices",
                outputType
        );
        out.setRequiresGrad(logits.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !logits.getRequiresGrad()) {
                return;
            }

            Tensor safeIndices = ignoreIndexOrNull == null ? targetIndices : LossSupport.buildSafeIndices(targetIndices, ignoreIndexOrNull);
            Tensor sampleScale = reductionScalePerSample(outGrad, logits, targetIndices, reducedShape, normalizedClassDimension, reduction, ignoreIndexOrNull);
            BackendIntentPropagator.preserve(sampleScale, out);
            Tensor grad = logits.softmax(normalizedClassDimension)
                    .mul(sampleScale.expandDims(normalizedClassDimension))
                    .sub(Tensor.zerosLike(logits).scatterAdd(safeIndices, sampleScale, normalizedClassDimension));
            BackendIntentPropagator.preserve(grad, out);
            LossSupport.accumulateGradient(logits, grad);
        });
        return out;
    }

    private static Tensor reductionScalePerSample(
            Tensor outGrad,
            Tensor logits,
            Tensor targetIndices,
            int[] reducedShape,
            int normalizedClassDimension,
            LossReduction reduction,
            Integer ignoreIndexOrNull
    ) {
        Tensor baseScale = switch (reduction) {
            case NONE -> outGrad;
            case SUM -> outGrad.expand(reducedShape);
            case MEAN -> {
                if (ignoreIndexOrNull == null) {
                    yield outGrad.mul(1.0 / LossSupport.sampleCount(logits.getShape(), normalizedClassDimension)).expand(reducedShape);
                }
                Tensor maskBase = outGrad.expand(reducedShape);
                Tensor validMask = LossSupport.buildIgnoreMask(targetIndices, ignoreIndexOrNull);
                Tensor validMaskNumeric = Tensor.where(validMask, Tensor.onesLike(maskBase), Tensor.zerosLike(maskBase));
                Tensor validCount = validMaskNumeric.sum();
                yield outGrad.div(validCount.clampMin(1.0)).expand(reducedShape);
            }
        };

        if (ignoreIndexOrNull == null) {
            return baseScale;
        }
        Tensor validMask = LossSupport.buildIgnoreMask(targetIndices, ignoreIndexOrNull);
        return Tensor.where(validMask, baseScale, Tensor.zerosLike(baseScale));
    }
}
