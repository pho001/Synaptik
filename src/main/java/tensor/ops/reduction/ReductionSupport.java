package tensor.ops.reduction;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

final class ReductionSupport {
    private ReductionSupport() {
    }

    static int[] reduceShape(int[] shape, int normalizedDimension, boolean keepDims) {
        if (keepDims) {
            int[] newShape = shape.clone();
            newShape[normalizedDimension] = 1;
            return newShape;
        }
        int[] newShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != normalizedDimension) {
                newShape[j++] = shape[i];
            }
        }
        return newShape;
    }

    static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            TensorInternalAccess.setGradient(input, gradientDelta);
        } else {
            TensorInternalAccess.setGradient(input, input.getGradient().add(gradientDelta));
        }
    }

    static void requireFloatingInput(Tensor input, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(opName + " requires floating numeric input.");
        }
    }
}
