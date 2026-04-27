package tensor.ops.reduction;

import graph.optimizer.intent.BackendIntentPropagator;
import operations.Operation;
import operations.reduction.logSoftmax;
import operations.reduction.mean;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.softmax;
import operations.reduction.sum;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

public final class TensorReduceOps {
    private TensorReduceOps() {
    }

    public static Tensor sum(Tensor input, int dimension) {
        return sum(input, dimension, false);
    }

    public static Tensor sum(Tensor input, int dimension, boolean keepDims) {
        ReductionSupport.requireFloatingInput(input, "sum");
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        Operation op = new sum(normalizedDimension, keepDims);
        int[] newShape = ReductionSupport.reduceShape(shape, normalizedDimension, keepDims);
        Tensor out = TensorPrimitiveBuilder.unary(input, newShape, op, "sum", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor aligned = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            ReductionSupport.accumulateGradient(input, aligned.expand(input.getShape()));
        });
        return out;
    }

    public static Tensor sumAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "sum");
        Tensor out = TensorPrimitiveBuilder.unary(input, new int[]{1}, new sum(-1), "sum", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            ReductionSupport.accumulateGradient(input, outGrad.expand(input.getShape()));
        });
        return out;
    }

    public static Tensor mean(Tensor input, int dimension) {
        return mean(input, dimension, false);
    }

    public static Tensor mean(Tensor input, int dimension, boolean keepDims) {
        ReductionSupport.requireFloatingInput(input, "mean");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                ReductionSupport.reduceShape(input.getShape(), normalizedDimension, keepDims),
                new mean(normalizedDimension, keepDims),
                "mean",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor aligned = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            Tensor grad = aligned.expand(input.getShape()).mul(1.0 / input.getShape()[normalizedDimension]);
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    public static Tensor meanAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "mean");
        Tensor out = TensorPrimitiveBuilder.unary(input, new int[]{1}, new mean(-1), "mean", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = outGrad.expand(input.getShape()).mul(1.0 / input.getFlatDataSize());
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    public static Tensor softmax(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "softmax");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                input.getShape().clone(),
                new softmax(normalizedDimension),
                "softmax",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor dot = outGrad.mul(out).sum(normalizedDimension, true);
            Tensor grad = out.mul(outGrad.sub(dot));
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    public static Tensor logSoftmax(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "logSoftmax");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                input.getShape().clone(),
                new logSoftmax(normalizedDimension),
                "logSoftmax",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor probs = out.exp();
            Tensor sumGrad = outGrad.sum(normalizedDimension, true);
            Tensor grad = outGrad.sub(probs.mul(sumGrad));
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    public static Tensor min(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "min");
        return min(input, dimension, false);
    }

    public static Tensor min(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, false);
    }

    public static Tensor minAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "min");
        return reduceMinMaxAll(input, false);
    }

    public static Tensor max(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "max");
        return max(input, dimension, false);
    }

    public static Tensor max(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, true);
    }

    public static Tensor maxAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "max");
        return reduceMinMaxAll(input, true);
    }

    public static Tensor all(Tensor input, int dimension) {
        return all(input, dimension, false);
    }

    public static Tensor all(Tensor input, int dimension, boolean keepDims) {
        return reduceBool(input, dimension, keepDims, true);
    }

    public static Tensor allAll(Tensor input) {
        return reduceBoolAll(input, true);
    }

    public static Tensor any(Tensor input, int dimension) {
        return any(input, dimension, false);
    }

    public static Tensor any(Tensor input, int dimension, boolean keepDims) {
        return reduceBool(input, dimension, keepDims, false);
    }

    public static Tensor anyAll(Tensor input) {
        return reduceBoolAll(input, false);
    }

    private static Tensor reduceMinMax(Tensor input, int dimension, boolean keepDims, boolean isMax) {
        ReductionSupport.requireFloatingInput(input, isMax ? "max" : "min");
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        int[] newShape = ReductionSupport.reduceShape(shape, normalizedDimension, keepDims);
        Operation op = isMax ? new reduceMax(normalizedDimension, keepDims) : new reduceMin(normalizedDimension, keepDims);
        Tensor out = TensorPrimitiveBuilder.unary(input, newShape, op,
                isMax ? "max_reduce" : "min_reduce", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Tensor reducedForGrad = keepDims ? out : out.expandDims(normalizedDimension);
            Tensor outGradForGrad = keepDims ? outGrad : outGrad.expandDims(normalizedDimension);
            Operation gradOp = isMax ? new operations.reduction.reduceMaxGrad(normalizedDimension) : new operations.reduction.reduceMinGrad(normalizedDimension);
            Tensor grad = TensorPrimitiveBuilder.ternary(
                    input,
                    reducedForGrad,
                    outGradForGrad,
                    input.getShape().clone(),
                    gradOp,
                    isMax ? "reduce_max_grad" : "reduce_min_grad",
                    outGrad.getDataType()
            );
            BackendIntentPropagator.preserve(grad, out);
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    private static Tensor reduceMinMaxAll(Tensor input, boolean isMax) {
        ReductionSupport.requireFloatingInput(input, isMax ? "max" : "min");
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                new int[]{1},
                isMax ? new reduceMax(-1) : new reduceMin(-1),
                isMax ? "max_reduce" : "min_reduce",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }

            Operation gradOp = isMax ? new operations.reduction.reduceMaxGrad(-1) : new operations.reduction.reduceMinGrad(-1);
            Tensor grad = TensorPrimitiveBuilder.ternary(
                    input,
                    out,
                    outGrad,
                    input.getShape().clone(),
                    gradOp,
                    isMax ? "reduce_max_grad" : "reduce_min_grad",
                    outGrad.getDataType()
            );
            BackendIntentPropagator.preserve(grad, out);
            ReductionSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    private static Tensor reduceBool(Tensor input, int dimension, boolean keepDims, boolean isAll) {
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException((isAll ? "all" : "any") + " requires BOOL input.");
        }
        int[] shape = input.getShape();
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, shape.length);
        int[] newShape = ReductionSupport.reduceShape(shape, normalizedDimension, keepDims);
        Operation op = isAll ? new reduceAll(normalizedDimension, keepDims) : new reduceAny(normalizedDimension, keepDims);
        return TensorPrimitiveBuilder.unaryNoGrad(input, newShape, op, isAll ? "all_reduce" : "any_reduce", DataType.BOOL);
    }

    private static Tensor reduceBoolAll(Tensor input, boolean isAll) {
        if (input.getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException((isAll ? "all" : "any") + " requires BOOL input.");
        }
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                new int[]{1},
                isAll ? new reduceAll(-1) : new reduceAny(-1),
                isAll ? "all_reduce" : "any_reduce",
                DataType.BOOL
        );
    }
}
