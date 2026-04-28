package operations.nn.conv;
import operations.Operation;

import tensor.options.Conv2dOptions;

/**
 * Describes an NCHW 2-D convolution using the direct execution form.
 *
 * <p>Input and weight tensors are expected to be rank-4 by the tensor/backend
 * contract. {@link Conv2dOptions} supplies stride, padding, dilation, and group
 * semantics; optional bias is recorded by {@link #hasBias()}.</p>
 */
public final class conv2d implements Operation {
    private final Conv2dOptions options;
    private final boolean hasBias;

    /**
     * Creates a convolution descriptor.
     *
     * @param options non-null convolution shape options
     * @param hasBias whether a bias input is expected
     * @throws IllegalArgumentException if {@code options} is {@code null}
     */
    public conv2d(Conv2dOptions options, boolean hasBias) {
        if (options == null) {
            throw new IllegalArgumentException("options cannot be null");
        }
        this.options = options;
        this.hasBias = hasBias;
    }

    /**
     * Returns the convolution options.
     *
     * @return non-null stride, padding, dilation, and group settings
     */
    public Conv2dOptions getOptions() {
        return options;
    }

    /**
     * Indicates whether this convolution includes bias addition.
     *
     * @return {@code true} when a bias input is expected
     */
    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public OpType opType() {
        return OpType.CONV2D;
    }

    @Override
    public String getExpression() {
        return hasBias ? "conv2d+bias" : "conv2d";
    }
}
