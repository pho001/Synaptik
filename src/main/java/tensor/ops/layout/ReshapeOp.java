package tensor.ops.layout;

import operations.Operation;
import operations.layout.reshape;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorLayoutTransform;
import tensor.TensorMetadata;
import tensor.TensorPrimitiveBuilder;

/**
 * Graph-building definition for {@code reshape}.
 *
 * <p>The descriptor remains an immutable primitive under {@code operations};
 * this class owns public shape inference, view/materialization choice, and
 * backward graph wiring.</p>
 */
public final class ReshapeOp {
    private ReshapeOp() {
    }

    /**
     * Reshapes a tensor without changing logical element order.
     *
     * @param input tensor to reshape; must be non-null
     * @param requestedShape target shape; product must match input flat size
     * @return tensor with the requested shape
     */
    public static Tensor build(Tensor input, int[] requestedShape) {
        int[] newShape = TensorLayoutTransform.inferReshape(input.getShape(), requestedShape);
        Operation op = new reshape(newShape);
        Tensor out = input.isContiguous()
                ? TensorPrimitiveBuilder.unaryView(
                        input,
                        newShape,
                        TensorMetadata.computeStrides(newShape),
                        input.getStorageOffsetUnsafe(),
                        op,
                        "reshape",
                        input.getDataType()
                )
                : TensorPrimitiveBuilder.unary(input, newShape, op, "reshape", input.getDataType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            LayoutSupport.accumulateGradient(input, outGrad.reshape(input.getShape()));
        });
        return out;
    }
}
