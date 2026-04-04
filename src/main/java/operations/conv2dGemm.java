package operations;

import tensor.Conv2dOptions;

public final class conv2dGemm implements Operation {
    private final Conv2dOptions options;
    private final boolean hasBias;

    public conv2dGemm(Conv2dOptions options, boolean hasBias) {
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
        return OpType.CONV2D_GEMM;
    }

    @Override
    public String getExpression() {
        return hasBias ? "conv2dGemm+bias" : "conv2dGemm";
    }
}
