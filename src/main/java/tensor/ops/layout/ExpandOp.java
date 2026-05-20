package tensor.ops.layout;

import operations.Operation;
import operations.layout.expand;
import tensor.Tensor;
import tensor.TensorBroadcastOps;
import tensor.TensorInternalAccess;
import tensor.layout.TensorLayoutTransform;
import tensor.internal.TensorPrimitiveBuilder;

/**
 * Graph-building definition for broadcast view {@code expand}.
 */
public final class ExpandOp {
    private ExpandOp() {
    }

    public static Tensor build(Tensor input, int[] requestedShape) {
        int[] targetShape = TensorLayoutTransform.inferExpandShape(input.getShape(), requestedShape);
        int[] targetStrides = LayoutSupport.buildExpandedStrides(input.getShapeUnsafe(), input.getStridesUnsafe(), targetShape);
        Operation op = new expand(targetShape);
        Tensor out = TensorPrimitiveBuilder.unaryView(
                input,
                targetShape,
                targetStrides,
                input.getStorageOffsetUnsafe(),
                op,
                "expand",
                input.getDataType()
        );
        TensorInternalAccess.setGradientRule(out, context -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            context.accumulate(input, TensorBroadcastOps.sumToShape(outGrad, input.getShape()));
        });
        return out;
    }
}
