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
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor aligned = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            Tensor grad = aligned.expand(input.getShape());
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor sumAll(Tensor input) {
        Operation op = new sum(-1);
        int[] newShape = new int[]{1};
        Tensor out = new Tensor(newShape, List.of(input), op, "sum");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor grad = outGrad.expand(input.getShape());
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor mean(Tensor input, int dimension) {
        return mean(input, dimension, false);
    }

    static Tensor mean(Tensor input, int dimension, boolean keepDims) {
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        double divisor = input.getShape()[normalizedDimension];
        return sum(input, normalizedDimension, keepDims).mul(1.0 / divisor);
    }

    static Tensor meanAll(Tensor input) {
        double divisor = input.getFlatDataSize();
        return sumAll(input).mul(1.0 / divisor);
    }
}
