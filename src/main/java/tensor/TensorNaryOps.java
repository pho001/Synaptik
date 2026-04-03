package tensor;

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
}
