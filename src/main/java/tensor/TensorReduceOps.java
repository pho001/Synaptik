package tensor;

import operations.Operation;
import operations.reduceMax;
import operations.reduceMaxGrad;
import operations.reduceMin;
import operations.reduceMinGrad;
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

    static Tensor min(Tensor input, int dimension) {
        return min(input, dimension, false);
    }

    static Tensor min(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, false);
    }

    static Tensor minAll(Tensor input) {
        return reduceMinMaxAll(input, false);
    }

    static Tensor max(Tensor input, int dimension) {
        return max(input, dimension, false);
    }

    static Tensor max(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, true);
    }

    static Tensor maxAll(Tensor input) {
        return reduceMinMaxAll(input, true);
    }

    private static Tensor reduceMinMax(Tensor input, int dimension, boolean keepDims, boolean isMax) {
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        int[] newShape = reduceShape(shape, normalizedDimension, keepDims);
        Operation op = isMax ? new reduceMax(normalizedDimension, keepDims) : new reduceMin(normalizedDimension, keepDims);
        Tensor out = new Tensor(newShape, List.of(input), op, isMax ? "max_reduce" : "min_reduce");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;

            Tensor reducedForGrad = keepDims ? out : out.expandDims(normalizedDimension);
            Tensor outGradForGrad = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            Operation gradOp = isMax ? new reduceMaxGrad(normalizedDimension) : new reduceMinGrad(normalizedDimension);
            Tensor grad = new Tensor(
                    input.getShape().clone(),
                    List.of(input, reducedForGrad, outGradForGrad),
                    gradOp,
                    isMax ? "reduce_max_grad" : "reduce_min_grad"
            );
            grad.setDataType(outGrad.getDataType());
            accumulateGradient(input, grad);
        });
        return out;
    }

    private static Tensor reduceMinMaxAll(Tensor input, boolean isMax) {
        Tensor out = new Tensor(new int[]{1}, List.of(input), isMax ? new reduceMax(-1) : new reduceMin(-1), isMax ? "max_reduce" : "min_reduce");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;

            Operation gradOp = isMax ? new reduceMaxGrad(-1) : new reduceMinGrad(-1);
            Tensor grad = new Tensor(
                    input.getShape().clone(),
                    List.of(input, out, outGrad),
                    gradOp,
                    isMax ? "reduce_max_grad" : "reduce_min_grad"
            );
            grad.setDataType(outGrad.getDataType());
            accumulateGradient(input, grad);
        });
        return out;
    }

    private static int[] reduceShape(int[] shape, int normalizedDimension, boolean keepDims) {
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

    private static void accumulateGradient(Tensor input, Tensor gradientDelta) {
        if (input.getGradient() == null) {
            input.setGradient(gradientDelta);
        } else {
            input.setGradient(input.getGradient().add(gradientDelta));
        }
    }
}
