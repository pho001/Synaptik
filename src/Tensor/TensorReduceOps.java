package Tensor;

import Operations.Operation;
import Operations.sum;

import java.util.List;

final class TensorReduceOps {
    private TensorReduceOps() {}

    static Tensor sum(Tensor input, int dimension) {
        Operation op = new sum(dimension);
        int[] shape = input.getShape();
        int[] newShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != dimension) {
                newShape[j++] = shape[i];
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
