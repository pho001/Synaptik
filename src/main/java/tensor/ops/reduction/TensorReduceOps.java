package tensor.ops.reduction;

import graph.optimizer.intent.BackendIntentPropagator;
import operations.Operation;
import operations.reduction.cumSum;
import operations.reduction.mean;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.reduceProd;
import operations.reduction.argMax;
import operations.reduction.sum;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

/**
 * Reductions and axis-wise normalization-style tensor operations.
 *
 * <p>Floating reductions are differentiable and require floating numeric input.
 * Boolean reductions require {@link DataType#BOOL} input and do not propagate
 * gradients. Methods build graph tensors and do not mutate input storage.</p>
 */
public final class TensorReduceOps {
    private TensorReduceOps() {
    }

    /**
     * Sums along one dimension and removes that dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @return sum tensor with reduced shape
     * @throws IllegalArgumentException if input dtype is not floating or the axis is invalid
     */
    public static Tensor sum(Tensor input, int dimension) {
        return sum(input, dimension, false);
    }

    /**
     * Sums along one dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to retain the reduced axis with size 1
     * @return sum tensor
     * @throws IllegalArgumentException if input dtype is not floating or the axis is invalid
     */
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

    /**
     * Sums all elements into a shape {@code [1]} tensor.
     *
     * @param input floating tensor; must be non-null
     * @return scalar-like tensor containing the total sum
     * @throws IllegalArgumentException if input dtype is not floating
     */
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

    /**
     * Averages along one dimension and removes that dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @return mean tensor with reduced shape
     * @throws IllegalArgumentException if input dtype is not floating or the axis is invalid
     */
    public static Tensor mean(Tensor input, int dimension) {
        return mean(input, dimension, false);
    }

    /**
     * Averages along one dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to retain the reduced axis with size 1
     * @return mean tensor
     * @throws IllegalArgumentException if input dtype is not floating or the axis is invalid
     */
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

    /**
     * Averages all elements into a shape {@code [1]} tensor.
     *
     * @param input floating tensor; must be non-null
     * @return scalar-like tensor containing the mean
     * @throws IllegalArgumentException if input dtype is not floating
     */
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

    public static Tensor prod(Tensor input, int dimension) {
        return prod(input, dimension, false);
    }

    public static Tensor prod(Tensor input, int dimension, boolean keepDims) {
        ReductionSupport.requireFloatingInput(input, "prod");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                ReductionSupport.reduceShape(input.getShape(), normalizedDimension, keepDims),
                new reduceProd(normalizedDimension, keepDims),
                "prod_reduce",
                input.getDataType()
        );
    }

    public static Tensor prodAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "prod");
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                new int[]{1},
                new reduceProd(-1),
                "prod_reduce",
                input.getDataType()
        );
    }

    public static Tensor argMax(Tensor input, int dimension) {
        return argMax(input, dimension, false);
    }

    public static Tensor argMax(Tensor input, int dimension, boolean keepDims) {
        if (input == null) {
            throw new IllegalArgumentException("argMax input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("argMax requires numeric input.");
        }
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                ReductionSupport.reduceShape(input.getShape(), normalizedDimension, keepDims),
                new argMax(normalizedDimension, keepDims),
                "argmax",
                DataType.INT32
        );
    }

    public static Tensor cumSum(Tensor input, int axis) {
        return cumSum(input, axis, false, false);
    }

    public static Tensor cumSum(Tensor input, int axis, boolean exclusive, boolean reverse) {
        if (input == null) {
            throw new IllegalArgumentException("cumSum input cannot be null");
        }
        if (input.getDataType() == DataType.BOOL) {
            throw new IllegalArgumentException("cumSum requires floating or INT32 input.");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, input.getShape().length);
        return TensorPrimitiveBuilder.unaryNoGrad(
                input,
                input.getShape(),
                new cumSum(normalizedAxis, exclusive, reverse),
                "cumsum",
                input.getDataType()
        );
    }

    /**
     * Applies softmax along one dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis over which probabilities sum to 1; negative axes are normalized
     * @return shape-preserving softmax tensor
     * @throws IllegalArgumentException if input dtype is not floating or the axis is invalid
     */
    public static Tensor softmax(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "softmax");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor max = input.max(normalizedDimension, true);
        Tensor shifted = input.sub(max);
        Tensor exp = shifted.exp();
        Tensor denominator = exp.sum(normalizedDimension, true);
        Tensor out = exp.div(denominator);
        out.setLabel("softmax");
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

    /**
     * Applies log-softmax along one dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis over which log probabilities are normalized; negative axes are normalized
     * @return shape-preserving log-softmax tensor
     * @throws IllegalArgumentException if input dtype is not floating or the axis is invalid
     */
    public static Tensor logSoftmax(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "logSoftmax");
        int normalizedDimension = TensorLayoutTransform.normalizeAxis(dimension, input.getShape().length);
        Tensor max = input.max(normalizedDimension, true);
        Tensor shifted = input.sub(max);
        Tensor exp = shifted.exp();
        Tensor denominator = exp.sum(normalizedDimension, true);
        Tensor out = shifted.sub(denominator.log());
        out.setLabel("logSoftmax");
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

    /**
     * Finds the minimum along one dimension and removes that dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced minimum tensor
     */
    public static Tensor min(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "min");
        return min(input, dimension, false);
    }

    /**
     * Finds the minimum along one dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to retain the reduced axis with size 1
     * @return reduced minimum tensor
     */
    public static Tensor min(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, false);
    }

    /**
     * Finds the minimum over all elements.
     *
     * @param input floating tensor; must be non-null
     * @return shape {@code [1]} tensor containing the minimum
     */
    public static Tensor minAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "min");
        return reduceMinMaxAll(input, false);
    }

    /**
     * Finds the maximum along one dimension and removes that dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @return reduced maximum tensor
     */
    public static Tensor max(Tensor input, int dimension) {
        ReductionSupport.requireFloatingInput(input, "max");
        return max(input, dimension, false);
    }

    /**
     * Finds the maximum along one dimension.
     *
     * @param input floating tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to retain the reduced axis with size 1
     * @return reduced maximum tensor
     */
    public static Tensor max(Tensor input, int dimension, boolean keepDims) {
        return reduceMinMax(input, dimension, keepDims, true);
    }

    /**
     * Finds the maximum over all elements.
     *
     * @param input floating tensor; must be non-null
     * @return shape {@code [1]} tensor containing the maximum
     */
    public static Tensor maxAll(Tensor input) {
        ReductionSupport.requireFloatingInput(input, "max");
        return reduceMinMaxAll(input, true);
    }

    /**
     * Computes logical AND over one dimension and removes that dimension.
     *
     * @param input BOOL tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @return boolean tensor containing per-slice all results
     * @throws IllegalArgumentException if input is not BOOL or the axis is invalid
     */
    public static Tensor all(Tensor input, int dimension) {
        return all(input, dimension, false);
    }

    /**
     * Computes logical AND over one dimension.
     *
     * @param input BOOL tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to retain the reduced axis with size 1
     * @return boolean tensor containing per-slice all results
     * @throws IllegalArgumentException if input is not BOOL or the axis is invalid
     */
    public static Tensor all(Tensor input, int dimension, boolean keepDims) {
        return reduceBool(input, dimension, keepDims, true);
    }

    /**
     * Computes logical AND over all elements.
     *
     * @param input BOOL tensor; must be non-null
     * @return shape {@code [1]} BOOL tensor
     * @throws IllegalArgumentException if input is not BOOL
     */
    public static Tensor allAll(Tensor input) {
        return reduceBoolAll(input, true);
    }

    /**
     * Computes logical OR over one dimension and removes that dimension.
     *
     * @param input BOOL tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @return boolean tensor containing per-slice any results
     * @throws IllegalArgumentException if input is not BOOL or the axis is invalid
     */
    public static Tensor any(Tensor input, int dimension) {
        return any(input, dimension, false);
    }

    /**
     * Computes logical OR over one dimension.
     *
     * @param input BOOL tensor; must be non-null
     * @param dimension axis to reduce; negative axes are normalized
     * @param keepDims true to retain the reduced axis with size 1
     * @return boolean tensor containing per-slice any results
     * @throws IllegalArgumentException if input is not BOOL or the axis is invalid
     */
    public static Tensor any(Tensor input, int dimension, boolean keepDims) {
        return reduceBool(input, dimension, keepDims, false);
    }

    /**
     * Computes logical OR over all elements.
     *
     * @param input BOOL tensor; must be non-null
     * @return shape {@code [1]} BOOL tensor
     * @throws IllegalArgumentException if input is not BOOL
     */
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
