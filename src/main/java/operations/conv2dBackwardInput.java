package operations;

import tensor.Conv2dOptions;

public final class conv2dBackwardInput implements Operation {
    private final Conv2dOptions options;
    private final int[] inputShape;

    public conv2dBackwardInput(Conv2dOptions options, int[] inputShape) {
        if (options == null) {
            throw new IllegalArgumentException("options cannot be null");
        }
        if (inputShape == null || inputShape.length != 4) {
            throw new IllegalArgumentException("inputShape must be rank-4.");
        }
        this.options = options;
        this.inputShape = inputShape.clone();
    }

    public Conv2dOptions getOptions() {
        return options;
    }

    public int[] getInputShape() {
        return inputShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.CONV2D_BACKWARD_INPUT;
    }

    @Override
    public String getExpression() {
        return "conv2d_backward_input";
    }
}
