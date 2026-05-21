package tensor.ops.layout;

import operations.layout.pad;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code pad}.
 */
public final class PadOp {
    private PadOp() {
    }

    public static Tensor build(Tensor input, int[] before, int[] after, double constantValue) {
        if (input == null) {
            throw new IllegalArgumentException("pad input cannot be null");
        }
        int rank = input.getShapeUnsafe().length;
        int[] normalizedBefore = LayoutGeometryRules.normalizePads(before, rank, "before");
        int[] normalizedAfter = LayoutGeometryRules.normalizePads(after, rank, "after");
        int[] outShape = input.getShape();
        for (int d = 0; d < rank; d++) {
            outShape[d] = Math.addExact(Math.addExact(outShape[d], normalizedBefore[d]), normalizedAfter[d]);
        }
        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                outShape,
                new pad(normalizedBefore, normalizedAfter, constantValue),
                "pad",
                input.getDataType()
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            int[] starts = normalizedBefore.clone();
            int[] ends = new int[rank];
            int[] axes = LayoutGeometryRules.allAxes(rank);
            int[] steps = LayoutGeometryRules.ones(rank);
            int[] inputShape = input.getShapeUnsafe();
            for (int d = 0; d < rank; d++) {
                ends[d] = starts[d] + inputShape[d];
            }
            context.accumulate(input, outGrad.slice(starts, ends, axes, steps));
        });
        return out;
    }
}
