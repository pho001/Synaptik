package tensor.ops.layout;

import operations.layout.slice;
import operations.layout.sliceScatterAdd;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.ops.layout.LayoutSupport.SliceSpec;

/**
 * Graph-building definition for strided {@code slice} views.
 */
public final class SliceOp {
    private SliceOp() {
    }

    public static Tensor build(Tensor input, int[] starts, int[] ends, int[] axes, int[] steps) {
        if (input == null) {
            throw new IllegalArgumentException("slice input cannot be null");
        }
        SliceSpec spec = LayoutSupport.normalizeSlice(input.getShapeUnsafe(), starts, ends, axes, steps);
        int[] inputStrides = input.getStridesUnsafe();
        int[] outStrides = inputStrides.clone();
        int storageOffset = input.getStorageOffsetUnsafe();
        for (int i = 0; i < spec.axes().length; i++) {
            int axis = spec.axes()[i];
            storageOffset += spec.starts()[i] * inputStrides[axis];
            outStrides[axis] = inputStrides[axis] * spec.steps()[i];
        }
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                spec.outputShape(),
                outStrides,
                storageOffset,
                new slice(spec.starts(), spec.ends(), spec.axes(), spec.steps(), spec.outputShape()),
                "slice",
                input.getDataType()
        );
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad;
            if (LayoutSupport.allOnes(spec.steps())) {
                int rank = input.getShapeUnsafe().length;
                int[] before = new int[rank];
                int[] after = new int[rank];
                int[] inputShape = input.getShapeUnsafe();
                for (int i = 0; i < spec.axes().length; i++) {
                    int axis = spec.axes()[i];
                    before[axis] = spec.starts()[i];
                    after[axis] = inputShape[axis] - spec.starts()[i] - spec.outputShape()[axis];
                }
                grad = outGrad.pad(before, after, 0.0d);
            } else {
                grad = TensorPrimitiveBuilder.unaryNoGrad(
                        outGrad,
                        input.getShape(),
                        new sliceScatterAdd(spec.starts(), spec.axes(), spec.steps(), input.getShape()),
                        "slice_scatter_add",
                        input.getDataType()
                );
            }
            LayoutSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
