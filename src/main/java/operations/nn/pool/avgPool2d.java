package operations.nn.pool;
import operations.Operation;

import tensor.options.Pool2dOptions;

public final class avgPool2d implements Operation {
    private final Pool2dOptions options;

    public avgPool2d(Pool2dOptions options) {
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
        return OpType.AVG_POOL2D;
    }

    @Override
    public String getExpression() {
        return "avgPool2d";
    }
}
