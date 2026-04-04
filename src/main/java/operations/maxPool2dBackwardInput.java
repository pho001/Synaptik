package operations;

import tensor.Pool2dOptions;

public final class maxPool2dBackwardInput implements Operation {
    private final Pool2dOptions options;
    private final int[] inputShape;

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

    public Pool2dOptions getOptions() {
        return options;
    }

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
