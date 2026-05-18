package tensor.ops.layout;

import operations.Operation;
import operations.layout.permute;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for axis-reordering {@code permute}.
 */
public final class PermuteOp {
    private PermuteOp() {
    }

    public static Tensor build(Tensor input, int[] axes) {
        int rank = input.getShape().length;
        int[] normalizedAxes = TensorLayoutTransform.normalizeAxes(rank, axes);
        int[] inShape = input.getShape();
        int[] inStrides = input.getStrides();
        int[] outShape = new int[rank];
        int[] outStrides = new int[rank];
        for (int i = 0; i < rank; i++) {
            outShape[i] = inShape[normalizedAxes[i]];
            outStrides[i] = inStrides[normalizedAxes[i]];
        }

        Operation op = new permute(normalizedAxes);
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                outShape,
                outStrides,
                input.getStorageOffsetUnsafe(),
                op,
                "permute",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            int[] inverse = TensorLayoutTransform.inverseAxes(normalizedAxes);
            LayoutSupport.accumulateGradient(input, outGrad.permute(inverse));
        });
        return out;
    }
}
