package tensor;

import operations.gather;
import operations.gatherGrad;

import java.util.List;

final class TensorIndexOps {
    private TensorIndexOps() {
    }

    static Tensor gather(Tensor input, Tensor indices, int dimension) {
        if (input == null || indices == null) {
            throw new IllegalArgumentException("gather inputs cannot be null");
        }
        if (indices.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("gather indices must be numeric integral values.");
        }
        int[] inputShape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, inputShape.length);
        int[] outputShape = reduceShape(inputShape, normalizedDimension);
        validateGatherIndicesShape(indices.getShape(), outputShape);

        Tensor out = new Tensor(outputShape, List.of(input, indices), new gather(normalizedDimension), "gather", input.getDataType());
        out.setRequiresGrad(input.getRequiresGrad());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor grad = new Tensor(input.getShape().clone(), List.of(indices, outGrad), new gatherGrad(normalizedDimension), "gather_grad", input.getDataType());
            accumulateGradient(input, grad);
        });
        return out;
    }

    private static void validateGatherIndicesShape(int[] indicesShape, int[] expectedShape) {
        if (indicesShape.length != expectedShape.length) {
            throw new IllegalArgumentException("gather indices shape must equal input shape without gathered axis.");
        }
        for (int i = 0; i < indicesShape.length; i++) {
            if (indicesShape[i] != expectedShape[i]) {
                throw new IllegalArgumentException("gather indices shape must equal input shape without gathered axis.");
            }
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

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
