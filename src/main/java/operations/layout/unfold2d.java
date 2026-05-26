package operations.layout;

import operations.Operation;
import tensor.options.Window2dOptions;

/**
 * Materializes NCHW 2-D sliding windows into im2col layout.
 */
public final class unfold2d implements Operation {
    private final Window2dOptions options;

    public unfold2d(Window2dOptions options) {
        if (options == null) {
            throw new IllegalArgumentException("unfold2d options cannot be null");
        }
        this.options = options;
    }

    public Window2dOptions getOptions() {
        return options;
    }

    @Override
    public OpType opType() {
        return OpType.UNFOLD2D;
    }

    @Override
    public String getExpression() {
        return "unfold2d(" + options + ")";
    }
}
