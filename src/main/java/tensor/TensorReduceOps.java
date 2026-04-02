package tensor;

import operations.Operation;
import operations.sum;

import java.util.List;

final class TensorReduceOps {
    private TensorReduceOps() {}

    static Tensor sum(Tensor input, int dimension) {
        return sum(input, dimension, false);
    }

    static Tensor sum(Tensor input, int dimension, boolean keepDims) {
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        Operation op = new sum(normalizedDimension, keepDims);
        int[] newShape;
        if (keepDims) {
            newShape = shape.clone();
            newShape[normalizedDimension] = 1;
        } else {
            newShape = new int[shape.length - 1];
            for (int i = 0, j = 0; i < shape.length; i++) {
                if (i != normalizedDimension) {
                    newShape[j++] = shape[i];
                }
            }
        }
        Tensor out = new Tensor(newShape, List.of(input), op, "sum");
        out.setDataType(input.getDataType());
        return out;
    }

    static Tensor sumAll(Tensor input) {
        Operation op = new sum(-1);
        int[] newShape = new int[]{1};
        Tensor out = new Tensor(newShape, List.of(input), op, "sum");
        out.setDataType(input.getDataType());
        return out;
    }
}
