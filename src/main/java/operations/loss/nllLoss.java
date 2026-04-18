package operations.loss;
import operations.Operation;

public final class nllLoss implements Operation {
    private final int classDimension;

    public nllLoss(int classDimension) {
        this.classDimension = classDimension;
    }

    @Override
    public OpType opType() {
        return OpType.NLL_LOSS;
    }

    @Override
    public String getExpression() {
        return "nllLoss";
    }

    public int getClassDimension() {
        return classDimension;
    }
}
