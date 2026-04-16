package operations;

public final class scaledDotProductAttentionBackward implements Operation {
    public enum OutputKind {
        QUERY,
        KEY,
        VALUE
    }

    private final OutputKind outputKind;

    public scaledDotProductAttentionBackward(OutputKind outputKind) {
        if (outputKind == null) {
            throw new IllegalArgumentException("scaledDotProductAttentionBackward outputKind cannot be null");
        }
        this.outputKind = outputKind;
    }

    @Override
    public OpType opType() {
        return OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD;
    }

    @Override
    public String getExpression() {
        return "scaledDotProductAttentionBackward[" + outputKind.name().toLowerCase() + "]";
    }

    public OutputKind getOutputKind() {
        return outputKind;
    }
}
