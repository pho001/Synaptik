package tensor;

import operations.crossEntropyLoss;
import operations.nllLoss;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class TensorNaryOps {
    private TensorNaryOps() {}

    static List<Tensor> asInputs(Tensor first, Tensor... rest) {
        List<Tensor> inputs = new ArrayList<>(1 + rest.length);
        inputs.add(first);
        for (Tensor t : rest) {
            inputs.add(t);
        }
        return List.copyOf(inputs);
    }

    static Tensor nllLoss(Tensor logProbs, Tensor targets, int classDimension) {
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
        Tensor out = new Tensor(new int[]{1}, asInputs(logProbs, targets), new nllLoss(normalizedClassDimension), "nllLoss", outputType);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            double scale = outGrad.scalarAsDouble() / sampleCount(logShape, normalizedClassDimension);
            if (logProbs.getRequiresGrad()) {
                Tensor grad = targets.mul(-scale);
                accumulateGradient(logProbs, grad);
            }
            if (targets.getRequiresGrad()) {
                Tensor grad = logProbs.mul(-scale);
                accumulateGradient(targets, grad);
            }
        });
        return out;
    }

    static Tensor crossEntropyLoss(Tensor logits, Tensor targets, int classDimension) {
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
        Tensor out = new Tensor(new int[]{1}, asInputs(logits, targets), new crossEntropyLoss(normalizedClassDimension), "crossEntropyLoss", outputType);
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) return;

            double scale = outGrad.scalarAsDouble() / sampleCount(logitsShape, normalizedClassDimension);
            if (logits.getRequiresGrad()) {
                Tensor grad = logits.softmax(normalizedClassDimension).sub(targets).mul(scale);
                accumulateGradient(logits, grad);
            }
            if (targets.getRequiresGrad()) {
                Tensor grad = logits.logSoftmax(normalizedClassDimension).mul(-scale);
                accumulateGradient(targets, grad);
            }
        });
        return out;
    }

    static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, LossReduction.MEAN);
    }

    static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, classWeights, null, reduction);
    }

    static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, LossReduction reduction) {
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
        int[] expectedIndexShape = reduceShape(logShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "nllLossFromIndices targetIndices shape must equal logProbs shape without class axis.");
        Tensor perSampleLoss = logProbs.gather(targetIndices, normalizedClassDimension).neg();
        return applyLossReduction(perSampleLoss, null, reduction);
    }

    static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, ignoreIndex, LossReduction.MEAN);
    }

    static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return nllLossFromIndices(logProbs, targetIndices, classDimension, classWeights, Integer.valueOf(ignoreIndex), reduction);
    }

    static Tensor nllLossFromIndices(Tensor logProbs, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
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
        int[] expectedIndexShape = reduceShape(logShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "nllLossFromIndices targetIndices shape must equal logProbs shape without class axis.");

        Tensor validMask = buildIgnoreMask(targetIndices, ignoreIndex);
        Tensor safeIndices = buildSafeIndices(targetIndices, ignoreIndex);
        Tensor gathered = logProbs.gather(safeIndices, normalizedClassDimension);
        Tensor validMaskNumeric = Tensor.where(validMask, Tensor.onesLike(gathered), Tensor.zerosLike(gathered));
        Tensor maskedLoss = Tensor.where(validMask, gathered.neg(), Tensor.zerosLike(gathered));
        return applyLossReduction(maskedLoss, validMaskNumeric, reduction);
    }

    static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, LossReduction.MEAN);
    }

    static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, classWeights, null, reduction);
    }

    static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, LossReduction reduction) {
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
        int[] expectedIndexShape = reduceShape(logitsShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "crossEntropyLossFromIndices targetIndices shape must equal logits shape without class axis.");
        return logits.logSoftmax(normalizedClassDimension).nllLossFromIndices(targetIndices, normalizedClassDimension, reduction);
    }

    static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, ignoreIndex, LossReduction.MEAN);
    }

    static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction) {
        return crossEntropyLossFromIndices(logits, targetIndices, classDimension, classWeights, Integer.valueOf(ignoreIndex), reduction);
    }

    static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction) {
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
        int[] expectedIndexShape = reduceShape(logitsShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "crossEntropyLossFromIndices targetIndices shape must equal logits shape without class axis.");
        return logits.logSoftmax(normalizedClassDimension).nllLossFromIndices(targetIndices, normalizedClassDimension, ignoreIndex, reduction);
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
        int[] expectedIndexShape = reduceShape(logShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "weighted nllLossFromIndices targetIndices shape must equal logProbs shape without class axis.");
        validateClassWeightsShape(classWeights, logShape[normalizedClassDimension]);

        Tensor effectiveIndices = targetIndices;
        Tensor sampleMask = null;
        if (ignoreIndexOrNull != null) {
            sampleMask = buildIgnoreMask(targetIndices, ignoreIndexOrNull);
            effectiveIndices = buildSafeIndices(targetIndices, ignoreIndexOrNull);
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

        return applyLossReduction(perSampleLoss, reductionWeights, reduction);
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
        int[] expectedIndexShape = reduceShape(logitsShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "weighted crossEntropyLossFromIndices targetIndices shape must equal logits shape without class axis.");
        validateClassWeightsShape(classWeights, logitsShape[normalizedClassDimension]);

        if (ignoreIndexOrNull == null) {
            return logits.logSoftmax(normalizedClassDimension).nllLossFromIndices(targetIndices, normalizedClassDimension, classWeights, reduction);
        }
        return logits.logSoftmax(normalizedClassDimension).nllLossFromIndices(targetIndices, normalizedClassDimension, ignoreIndexOrNull, classWeights, reduction);
    }

    private static int sampleCount(int[] shape, int classDimension) {
        int count = 1;
        for (int i = 0; i < shape.length; i++) {
            if (i != classDimension) {
                count *= shape[i];
            }
        }
        return Math.max(1, count);
    }

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }

    private static Tensor applyLossReduction(Tensor perSampleLoss, Tensor reductionWeights, LossReduction reduction) {
        return switch (reduction) {
            case NONE -> perSampleLoss;
            case SUM -> perSampleLoss.sum();
            case MEAN -> {
                if (reductionWeights == null) {
                    yield perSampleLoss.mean();
                }
                Tensor validCount = reductionWeights.sum();
                Tensor totalLoss = perSampleLoss.sum();
                yield totalLoss.div(validCount.clampMin(1.0));
            }
        };
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static void validateShape(int[] actual, int[] expected, String message) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(message);
        }
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != expected[i]) {
                throw new IllegalArgumentException(message);
            }
        }
    }

    private static void validateClassWeightsShape(Tensor classWeights, int expectedClasses) {
        int[] shape = classWeights.getShape();
        if (shape.length != 1 || shape[0] != expectedClasses) {
            throw new IllegalArgumentException("classWeights must have shape [" + expectedClasses + "].");
        }
    }

    private static Tensor buildIgnoreMask(Tensor targetIndices, int ignoreIndex) {
        int size = targetIndices.getFlatDataSize();
        byte[] mask = new byte[size];
        for (int i = 0; i < size; i++) {
            long value = readIntegralIndex(targetIndices, i);
            mask[i] = value == ignoreIndex ? (byte) 0 : (byte) 1;
        }
        return new Tensor(mask, targetIndices.getShape().clone(), null, "index_valid_mask", DataType.BOOL);
    }

    private static Tensor buildSafeIndices(Tensor targetIndices, int ignoreIndex) {
        int size = targetIndices.getFlatDataSize();
        if (targetIndices.getDataType() == DataType.INT32) {
            int[] safe = new int[size];
            for (int i = 0; i < size; i++) {
                long value = readIntegralIndex(targetIndices, i);
                safe[i] = value == ignoreIndex ? 0 : (int) value;
            }
            return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", DataType.INT32);
        }
        double[] safe = new double[size];
        for (int i = 0; i < size; i++) {
            long value = readIntegralIndex(targetIndices, i);
            safe[i] = value == ignoreIndex ? 0.0 : (double) value;
        }
        return new Tensor(safe, targetIndices.getShape().clone(), null, "safe_indices", targetIndices.getDataType());
    }

    private static long readIntegralIndex(Tensor indices, int flatIndex) {
        double raw = indices.getByFlatIndex(flatIndex);
        if (!Double.isFinite(raw)) {
            throw new IllegalArgumentException("Index tensor contains non-finite value.");
        }
        long integral = Math.round(raw);
        if (Math.abs(raw - integral) > 1e-9) {
            throw new IllegalArgumentException("Index tensor contains non-integral value: " + raw);
        }
        return integral;
    }
}
