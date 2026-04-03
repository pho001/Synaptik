package tensor;

import operations.Operation;
import operations.mean;
import operations.reduceAll;
import operations.reduceAny;
import operations.reduceMax;
import operations.reduceMaxGrad;
import operations.reduceMin;
import operations.reduceMinGrad;
import operations.logSoftmax;
import operations.softmax;
import operations.sum;

import java.util.List;

final class TensorReduceOps {
    private TensorReduceOps() {}

    static Tensor sum(Tensor input, int dimension) {
        return sum(input, dimension, false);
    }

    static Tensor sum(Tensor input, int dimension, boolean keepDims) {
        requireFloatingInput(input, "sum");
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
        requireFloatingInput(input, "sum");
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
        if (input == null) {
            throw new IllegalArgumentException("mean input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("mean requires floating numeric input.");
        }
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor out = new Tensor(reduceShape(input.getShape(), normalizedDimension, keepDims), List.of(input), new mean(normalizedDimension, keepDims), "mean");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor aligned = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            Tensor grad = aligned.expand(input.getShape()).mul(1.0 / input.getShape()[normalizedDimension]);
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor meanAll(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("mean input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("mean requires floating numeric input.");
        }
        Tensor out = new Tensor(new int[]{1}, List.of(input), new mean(-1), "mean");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;
            Tensor grad = outGrad.expand(input.getShape()).mul(1.0 / input.getFlatDataSize());
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor softmax(Tensor input, int dimension) {
        if (input == null) {
            throw new IllegalArgumentException("softmax input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("softmax requires floating numeric input.");
        }
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor out = new Tensor(input.getShape().clone(), List.of(input), new softmax(normalizedDimension), "softmax");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;

            Tensor dot = outGrad.mul(out).sum(normalizedDimension, true);
            Tensor grad = out.mul(outGrad.sub(dot));
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor logSoftmax(Tensor input, int dimension) {
        if (input == null) {
            throw new IllegalArgumentException("logSoftmax input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException("logSoftmax requires floating numeric input.");
        }
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor out = new Tensor(input.getShape().clone(), List.of(input), new logSoftmax(normalizedDimension), "logSoftmax");
        out.setDataType(input.getDataType());
        out.setBackwardFunction(() -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) return;

            Tensor probs = out.exp();
            Tensor sumGrad = outGrad.sum(normalizedDimension, true);
            Tensor grad = outGrad.sub(probs.mul(sumGrad));
            if (input.getGradient() == null) input.setGradient(grad);
            else input.setGradient(input.getGradient().add(grad));
        });
        return out;
    }

    static Tensor min(Tensor input, int dimension) {
        requireFloatingInput(input, "min");
        return min(input, dimension, false);
    }

    static Tensor min(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, false);
    }

    static Tensor minAll(Tensor input) {
        requireFloatingInput(input, "min");
        return reduceMinMaxAll(input, false);
    }

    static Tensor max(Tensor input, int dimension) {
        requireFloatingInput(input, "max");
        return max(input, dimension, false);
    }

    static Tensor max(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, true);
    }

    static Tensor maxAll(Tensor input) {
        requireFloatingInput(input, "max");
        return reduceMinMaxAll(input, true);
    }

    static Tensor all(Tensor input, int dimension) {
        return all(input, dimension, false);
    }

    static Tensor all(Tensor input, int dimension, boolean keepDims) {
        return reduceBool(input, dimension, keepDims, true);
    }

    static Tensor allAll(Tensor input) {
        return reduceBoolAll(input, true);
    }

    static Tensor any(Tensor input, int dimension) {
        return any(input, dimension, false);
    }

    static Tensor any(Tensor input, int dimension, boolean keepDims) {
        return reduceBool(input, dimension, keepDims, false);
    }

    static Tensor anyAll(Tensor input) {
        return reduceBoolAll(input, false);
    }

    private static Tensor reduceMinMax(Tensor input, int dimension, boolean keepDims, boolean isMax) {
        requireFloatingInput(input, isMax ? "max" : "min");
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
        requireFloatingInput(input, isMax ? "max" : "min");
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

    private static void requireFloatingInput(Tensor input, String opName) {
        if (input == null) {
            throw new IllegalArgumentException(opName + " input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL || input.getDataType() == DataType.INT32) {
            throw new IllegalArgumentException(opName + " requires floating numeric input.");
        }
    }

    private static Tensor reduceBool(Tensor input, int dimension, boolean keepDims, boolean isAll) {
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException((isAll ? "all" : "any") + " requires BOOL input.");
        }
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        int[] newShape = reduceShape(shape, normalizedDimension, keepDims);
        Operation op = isAll ? new reduceAll(normalizedDimension, keepDims) : new reduceAny(normalizedDimension, keepDims);
        Tensor out = new Tensor(newShape, List.of(input), op, isAll ? "all_reduce" : "any_reduce", DataType.BOOL);
        out.setRequiresGrad(false);
        return out;
    }

    private static Tensor reduceBoolAll(Tensor input, boolean isAll) {
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException((isAll ? "all" : "any") + " requires BOOL input.");
        }
        Tensor out = new Tensor(new int[]{1}, List.of(input), isAll ? new reduceAll(-1) : new reduceAny(-1), isAll ? "all_reduce" : "any_reduce", DataType.BOOL);
        out.setRequiresGrad(false);
        return out;
    }
}
