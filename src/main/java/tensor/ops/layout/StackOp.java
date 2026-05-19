package tensor.ops.layout;

import tensor.Tensor;
import tensor.layout.TensorLayoutTransform;

import java.util.ArrayList;
import java.util.List;

/**
 * Graph-building definition for {@code stack}.
 */
public final class StackOp {
    private StackOp() {
    }

    public static Tensor build(int axis, List<Tensor> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException("stack requires at least one input tensor.");
        }
        Tensor first = inputs.getFirst();
        if (first == null) {
            throw new IllegalArgumentException("stack inputs cannot contain null tensors.");
        }
        int rank = first.getShapeUnsafe().length;
        int normalizedAxis = TensorLayoutTransform.normalizeInsertAxis(axis, rank);
        int[] expectedShape = first.getShapeUnsafe();
        for (Tensor input : inputs) {
            if (input == null) {
                throw new IllegalArgumentException("stack inputs cannot contain null tensors.");
            }
            if (input.getDataType() != first.getDataType()) {
                throw new IllegalArgumentException("stack inputs must have matching dtypes.");
            }
            int[] shape = input.getShapeUnsafe();
            if (shape.length != expectedShape.length) {
                throw new IllegalArgumentException("stack inputs must have matching ranks.");
            }
            for (int d = 0; d < shape.length; d++) {
                if (shape[d] != expectedShape[d]) {
                    throw new IllegalArgumentException("stack inputs must have identical shapes.");
                }
            }
        }
        List<Tensor> expanded = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            expanded.add(input.expandDims(normalizedAxis));
        }
        return ConcatOp.build(normalizedAxis, expanded);
    }
}
