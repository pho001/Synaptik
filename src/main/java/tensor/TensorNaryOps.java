package tensor;

import operations.crossEntropyLoss;
import operations.nllLoss;

import java.util.ArrayList;
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

    static Tensor batchNorm(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            Tensor runningMean,
            Tensor runningVar,
            double epsilon,
            boolean training
    ) {
        throw new UnsupportedOperationException(
                "BatchNorm op is not implemented yet. " +
                "Use TensorNaryOps as extension point for multi-input operations."
        );
    }

    static Tensor nllLoss(Tensor logProbs, Tensor targets, int classDimension) {
        if (logProbs == null || targets == null) {
            throw new IllegalArgumentException("nllLoss inputs cannot be null");
        }
        if (logProbs.getDataType() == DataType.BOOL || targets.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("nllLoss requires numeric inputs.");
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
        if (logits.getDataType() == DataType.BOOL || targets.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("crossEntropyLoss requires numeric inputs.");
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
        if (logProbs == null || targetIndices == null) {
            throw new IllegalArgumentException("nllLossFromIndices inputs cannot be null");
        }
        if (logProbs.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("nllLossFromIndices requires numeric tensors and numeric integral indices.");
        }
        int[] logShape = logProbs.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logShape.length);
        int[] expectedIndexShape = reduceShape(logShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "nllLossFromIndices targetIndices shape must equal logProbs shape without class axis.");
        return logProbs.gather(targetIndices, normalizedClassDimension).neg().mean();
    }

    static Tensor crossEntropyLossFromIndices(Tensor logits, Tensor targetIndices, int classDimension) {
        if (logits == null || targetIndices == null) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices inputs cannot be null");
        }
        if (logits.getDataType() == DataType.BOOL || targetIndices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices requires numeric logits and numeric integral indices.");
        }
        int[] logitsShape = logits.getShape();
        int normalizedClassDimension = TensorLayoutTransform.normalizeAxis(classDimension, logitsShape.length);
        int[] expectedIndexShape = reduceShape(logitsShape, normalizedClassDimension);
        validateShape(targetIndices.getShape(), expectedIndexShape, "crossEntropyLossFromIndices targetIndices shape must equal logits shape without class axis.");
        return logits.logSoftmax(normalizedClassDimension).nllLossFromIndices(targetIndices, normalizedClassDimension);
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
}
