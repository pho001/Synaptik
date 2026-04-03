package operations;

public final class crossEntropyLoss implements Operation {
    private final int classDimension;

    public crossEntropyLoss(int classDimension) {
        this.classDimension = classDimension;
    }

    @Override
    public OpType opType() {
        return OpType.CROSS_ENTROPY_LOSS;
    }

    @Override
    public String getExpression() {
        return "crossEntropyLoss";
    }

    public int getClassDimension() {
        return classDimension;
    }
}
