package tensor.ops.layout;

import operations.layout.unfoldAxis;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.layout.TensorLayoutTransform;

/**
 * Graph-building definition for materialized 1-D axis unfold.
 */
public final class UnfoldAxisOp {
    private UnfoldAxisOp() {
    }

    public static Tensor build(Tensor input, int axis, int size, int step) {
        if (input == null) {
            throw new IllegalArgumentException("unfold input cannot be null");
        }
        int[] inputShape = input.getShapeUnsafe();
        int rank = inputShape.length;
        if (rank == 0) {
            throw new IllegalArgumentException("unfold requires rank >= 1.");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, rank);
        if (size <= 0) {
            throw new IllegalArgumentException("unfold size must be positive.");
        }
        if (step <= 0) {
            throw new IllegalArgumentException("unfold step must be positive.");
        }
        int dim = inputShape[normalizedAxis];
        if (size > dim) {
            throw new IllegalArgumentException("unfold size cannot exceed the selected axis length.");
        }
        int windows = ((dim - size) / step) + 1;
        int[] outShape = new int[rank + 1];
        System.arraycopy(inputShape, 0, outShape, 0, rank);
        outShape[normalizedAxis] = windows;
        outShape[rank] = size;

        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                outShape,
                new unfoldAxis(normalizedAxis, size, step),
                "unfold",
                input.getDataType()
        );
        if (!isFloating(input.getDataType())) {
            out.setRequiresGrad(false);
            return out;
        }
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = Tensor.zerosLike(input);
            for (int offset = 0; offset < size; offset++) {
                Tensor offsetGrad = outGrad.select(rank, offset);
                for (int window = 0; window < windows; window++) {
                    int inputIndex = window * step + offset;
                    Tensor windowGrad = offsetGrad.sliceAxis(normalizedAxis, window, window + 1);
                    int[] before = new int[rank];
                    int[] after = new int[rank];
                    before[normalizedAxis] = inputIndex;
                    after[normalizedAxis] = dim - inputIndex - 1;
                    grad = grad.add(windowGrad.pad(before, after, 0.0d));
                }
            }
            context.accumulate(input, grad);
        });
        return out;
    }

    private static boolean isFloating(DataType dataType) {
        return dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32 || dataType == DataType.BFLOAT16;
    }
}
