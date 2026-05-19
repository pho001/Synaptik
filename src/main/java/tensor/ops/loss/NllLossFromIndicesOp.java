package tensor.ops.loss;

import tensor.DataType;
import tensor.Tensor;
import tensor.layout.TensorLayoutTransform;
import tensor.loss.LossReduction;

import java.util.Arrays;

/**
 * Graph-building definition for index-target negative log-likelihood loss.
 */
public final class NllLossFromIndicesOp {
    private NllLossFromIndicesOp() {
    }

    public static Tensor build(Tensor logProbs, Tensor targetIndices, int classDimension) {
        return build(logProbs, targetIndices, classDimension, LossReduction.MEAN);
    }

    public static Tensor build(Tensor logProbs, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return build(logProbs, targetIndices, classDimension, classWeights, null, reduction);
    }

    public static Tensor build(Tensor logProbs, Tensor targetIndices, int classDimension, LossReduction reduction) {
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

    public static Tensor build(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return build(logProbs, targetIndices, classDimension, ignoreIndex, LossReduction.MEAN);
    }

    public static Tensor build(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return build(logProbs, targetIndices, classDimension, classWeights, Integer.valueOf(ignoreIndex), reduction);
    }

    public static Tensor build(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
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

    private static Tensor build(
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
        if (logProbs.getDataType() == DataType.INT32 || classWeights.getDataType() == DataType.INT32
                || logProbs.getDataType() == DataType.INT64 || classWeights.getDataType() == DataType.INT64) {
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
}
