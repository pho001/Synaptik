package operations.nn.conv;
import operations.Operation;

import tensor.options.Conv2dOptions;

/**
 * Describes GEMM-lowered NCHW 2-D convolution backward computation for the input gradient.
 *
 * <p>The descriptor stores the rank-4 output shape of the requested gradient so
 * kernels can reconstruct the full tensor even when convolution parameters
 * make the shape ambiguous.</p>
 */
public final class conv2dBackwardInputGemm implements Operation {
    private final Conv2dOptions options;
    private final int[] inputShape;

    /**
     * Creates a convolution backward descriptor.
     *
     * @param options non-null convolution shape options from the forward pass
     * @param inputShape rank-4 shape of the requested input gradient
     * @throws IllegalArgumentException if options are null or the shape is not
     *        rank 4
     */
    public conv2dBackwardInputGemm(Conv2dOptions options, int[] inputShape) {
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
     * Returns the convolution options from the paired forward operation.
     *
     * @return non-null stride, padding, dilation, and group settings
     */
    public Conv2dOptions getOptions() {
        return options;
    }

    /**
     * Returns a defensive copy of the requested input gradient shape.
     *
     * @return rank-4 NCHW shape
     */
    public int[] getInputShape() {
        return inputShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.CONV2D_BACKWARD_INPUT_GEMM;
    }

    @Override
    public String getExpression() {
        return "conv2d_backward_input_gemm";
    }
}
