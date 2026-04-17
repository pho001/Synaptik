package operations;

import tensor.options.Conv2dOptions;

public final class conv2d implements Operation {
    private final Conv2dOptions options;
    private final boolean hasBias;

    public conv2d(Conv2dOptions options, boolean hasBias) {
        if (options == null) {
            throw new IllegalArgumentException("options cannot be null");
        }
        this.options = options;
        this.hasBias = hasBias;
    }

    public Conv2dOptions getOptions() {
        return options;
    }

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
