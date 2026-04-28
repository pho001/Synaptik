package operations.nn.pool;
import operations.Operation;

import tensor.options.Pool2dOptions;

/**
 * Describes NCHW 2-D maximum pooling backward computation for the input gradient.
 *
 * <p>The descriptor stores the original rank-4 input shape so kernels can
 * reconstruct the gradient tensor from pooled-output gradients.</p>
 */
public final class maxPool2dBackwardInput implements Operation {
    private final Pool2dOptions options;
    private final int[] inputShape;

    /**
     * Creates a pooling backward descriptor.
     *
     * @param options non-null pooling window and stride options from the forward pass
     * @param inputShape rank-4 shape of the original input tensor
     * @throws IllegalArgumentException if options are null or the shape is not
     *        rank 4
     */
    public maxPool2dBackwardInput(Pool2dOptions options, int[] inputShape) {
        if (options == null) {
            throw new IllegalArgumentException("options cannot be null");
        }
        if (inputShape == null || inputShape.length != 4) {
            throw new IllegalArgumentException("inputShape must be rank-4.");
        }
        this.options = options;
        this.inputShape = inputShape.clone();
    }

    /**
     * Returns the pooling options from the paired forward operation.
     *
     * @return non-null kernel, stride, padding, and count settings
     */
    public Pool2dOptions getOptions() {
        return options;
    }

    /**
     * Returns a defensive copy of the original input shape.
     *
     * @return rank-4 NCHW input shape
     */
    public int[] getInputShape() {
        return inputShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.MAX_POOL2D_BACKWARD_INPUT;
    }

    @Override
    public String getExpression() {
        return "maxPool2dBackwardInput";
    }
}
