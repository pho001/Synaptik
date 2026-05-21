package tensor.ops.reduction;

import tensor.Tensor;

final class ReductionShapeRules {
    private ReductionShapeRules() {
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

    static void requireFloatingInput(Tensor input, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (!input.getDataType().isFloating()) {
            throw new IllegalArgumentException(opName + " requires floating numeric input.");
        }
    }
}
