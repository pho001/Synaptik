package tensor.ops.layout;

import operations.Operation;
import operations.layout.expandDims;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code expandDims}.
 */
public final class ExpandDimsOp {
    private ExpandDimsOp() {
    }

    public static Tensor build(Tensor input, int axis) {
        int rank = input.getShape().length;
        int normalizedAxis = TensorLayoutTransform.normalizeInsertAxis(axis, rank);
        int[] inShape = input.getShape();
        int[] inStrides = input.getStridesUnsafe();
        int[] outShape = new int[rank + 1];
        int[] outStrides = new int[rank + 1];
        for (int i = 0, j = 0; i < outShape.length; i++) {
            if (i == normalizedAxis) {
                outShape[i] = 1;
                outStrides[i] = LayoutSupport.insertedAxisStride(inShape, inStrides, normalizedAxis);
            } else {
                outShape[i] = inShape[j];
                outStrides[i] = inStrides[j];
                j++;
            }
        }
        Operation op = new expandDims(normalizedAxis);
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                outShape,
                outStrides,
                input.getStorageOffsetUnsafe(),
                op,
                "expandDims",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            LayoutSupport.accumulateGradient(input, outGrad.squeeze(normalizedAxis));
        });
        return out;
    }
}
