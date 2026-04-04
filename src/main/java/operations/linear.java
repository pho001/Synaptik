package operations;

public final class linear implements Operation {
    private final boolean hasBias;

    public linear(boolean hasBias) {
        this.hasBias = hasBias;
    }

    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public OpType opType() {
        return OpType.LINEAR;
    }

    @Override
    public String getExpression() {
        return hasBias ? "linear+bias" : "linear";
    }
}
