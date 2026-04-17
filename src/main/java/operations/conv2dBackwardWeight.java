package operations;

import tensor.options.Conv2dOptions;

public final class conv2dBackwardWeight implements Operation {
    private final Conv2dOptions options;
    private final int[] weightShape;

    public conv2dBackwardWeight(Conv2dOptions options, int[] weightShape) {
        if (options == null) {
            throw new IllegalArgumentException("options cannot be null");
        }
        if (weightShape == null || weightShape.length != 4) {
            throw new IllegalArgumentException("weightShape must be rank-4.");
        }
        this.options = options;
        this.weightShape = weightShape.clone();
    }

    public Conv2dOptions getOptions() {
        return options;
    }

    public int[] getWeightShape() {
        return weightShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.CONV2D_BACKWARD_WEIGHT;
    }

    @Override
    public String getExpression() {
        return "conv2d_backward_weight";
    }
}
