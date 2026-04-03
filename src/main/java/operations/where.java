package operations;

public final class where implements Operation {
    @Override
    public OpType opType() {
        return OpType.WHERE;
    }

    @Override
    public String getExpression() {
        return "where";
    }
}
