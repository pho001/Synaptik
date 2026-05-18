package tensor.ops.layout;

import operations.layout.concat;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

import java.util.List;

/**
 * Graph-building definition for {@code concat}.
 */
public final class ConcatOp {
    private ConcatOp() {
    }

    public static Tensor build(int axis, List<Tensor> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException("concat requires at least one input tensor.");
        }
        Tensor first = inputs.getFirst();
        if (first == null) {
            throw new IllegalArgumentException("concat inputs cannot contain null tensors.");
        }
        int rank = first.getShapeUnsafe().length;
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, rank);
        int[] outShape = first.getShape();
        int concatSize = 0;
        for (Tensor input : inputs) {
            if (input == null) {
                throw new IllegalArgumentException("concat inputs cannot contain null tensors.");
            }
            if (input.getDataType() != first.getDataType()) {
                throw new IllegalArgumentException("concat inputs must have matching dtypes.");
            }
            int[] shape = input.getShapeUnsafe();
            if (shape.length != rank) {
                throw new IllegalArgumentException("concat inputs must have matching ranks.");
            }
            for (int d = 0; d < rank; d++) {
                if (d != normalizedAxis && shape[d] != outShape[d]) {
                    throw new IllegalArgumentException("concat input shapes must match outside the concat axis.");
                }
            }
            concatSize += shape[normalizedAxis];
        }
        outShape[normalizedAxis] = concatSize;
        List<Tensor> copiedInputs = List.copyOf(inputs);
        Tensor out = TensorPrimitiveBuilder.nary(outShape, copiedInputs, new concat(normalizedAxis), "concat", first.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }
            int offset = 0;
            for (Tensor input : copiedInputs) {
                int axisSize = input.getShapeUnsafe()[normalizedAxis];
                if (input.getRequiresGrad()) {
                    int[] starts = new int[rank];
                    int[] ends = out.getShape();
                    starts[normalizedAxis] = offset;
                    ends[normalizedAxis] = offset + axisSize;
                    Tensor grad = outGrad.slice(starts, ends, LayoutSupport.allAxes(rank), LayoutSupport.ones(rank));
                    LayoutSupport.accumulateGradient(input, grad);
                }
                offset += axisSize;
            }
        });
        return out;
    }
}
