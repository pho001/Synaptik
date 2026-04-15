package operations;

public final class crossEntropyLossIndicesGrad implements Operation {
    private final int classDimension;

    public crossEntropyLossIndicesGrad(int classDimension) {
        this.classDimension = classDimension;
    }

    @Override
    public OpType opType() {
        return OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD;
    }

    @Override
    public String getExpression() {
        return "crossEntropyLossFromIndicesGrad";
    }

    public int getClassDimension() {
        return classDimension;
    }
}
