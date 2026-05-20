package tensor.ops.loss;

import graph.compile.intent.BackendIntentPropagator;
import operations.loss.crossEntropyLossIndices;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.loss.LossReduction;

/**
 * Graph-building definition for index-target cross-entropy loss.
 */
public final class CrossEntropyLossFromIndicesOp {
    private CrossEntropyLossFromIndicesOp() {
    }

    public static Tensor build(Tensor logits, Tensor targetIndices, int classDimension) {
        return build(logits, targetIndices, classDimension, LossReduction.MEAN);
    }

    public static Tensor build(Tensor logits, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return build(logits, targetIndices, classDimension, classWeights, null, reduction);
    }

    public static Tensor build(Tensor logits, Tensor targetIndices, int classDimension, LossReduction reduction) {
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
        return primitive(logits, targetIndices, normalizedClassDimension, reduction, null);
    }

    public static Tensor build(Tensor logits, Tensor targetIndices, int classDimension, Tensor mask) {
        Tensor perSampleLoss = build(logits, targetIndices, classDimension, LossReduction.NONE);
        Tensor alignedMask = LossSupport.alignSampleMask(mask, perSampleLoss.getShapeUnsafe(), "crossEntropyLossFromIndices");
        Tensor maskedLoss = Tensor.where(alignedMask, perSampleLoss, Tensor.zerosLike(perSampleLoss));
        Tensor valid = Tensor.where(alignedMask, Tensor.onesLike(perSampleLoss), Tensor.zerosLike(perSampleLoss));
        return maskedLoss.sum().div(valid.sum().clampMin(1.0d));
    }

    public static Tensor build(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return build(logits, targetIndices, classDimension, ignoreIndex, LossReduction.MEAN);
    }

    public static Tensor build(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return build(logits, targetIndices, classDimension, classWeights, Integer.valueOf(ignoreIndex), reduction);
    }

    public static Tensor build(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
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
        return primitive(logits, targetIndices, normalizedClassDimension, reduction, ignoreIndex);
    }

    private static Tensor build(
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
        if (logits.getDataType() == DataType.INT32 || classWeights.getDataType() == DataType.INT32
                || logits.getDataType() == DataType.INT64 || classWeights.getDataType() == DataType.INT64) {
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

    private static Tensor primitive(
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
