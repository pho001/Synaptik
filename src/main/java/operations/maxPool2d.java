package operations;

import tensor.Pool2dOptions;

public final class maxPool2d implements Operation {
    private final Pool2dOptions options;

    public maxPool2d(Pool2dOptions options) {
        if (options == null) {
            throw new IllegalArgumentException("options cannot be null");
        }
        this.options = options;
    }

    public Pool2dOptions getOptions() {
        return options;
    }

    @Override
    public OpType opType() {
        return OpType.MAX_POOL2D;
    }

    @Override
    public String getExpression() {
        return "maxPool2d";
    }
}
