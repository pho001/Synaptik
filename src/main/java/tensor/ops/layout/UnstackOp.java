package tensor.ops.layout;

import tensor.Tensor;
import tensor.layout.TensorLayoutTransform;

/**
 * Graph-building definition for {@code unstack}.
 */
public final class UnstackOp {
    private UnstackOp() {
    }

    public static Tensor[] build(Tensor input, int axis) {
        if (input == null) {
            throw new IllegalArgumentException("unstack input cannot be null");
        }
        int normalizedAxis = TensorLayoutTransform.normalizeAxis(axis, input.getShapeUnsafe().length);
        int count = input.getShapeUnsafe()[normalizedAxis];
        Tensor[] out = new Tensor[count];
        for (int i = 0; i < count; i++) {
            out[i] = input.select(normalizedAxis, i);
        }
        return out;
    }
}
