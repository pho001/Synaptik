package operations.linalg;
import operations.Operation;

public final class scaledDotProductAttentionWeights implements Operation {
    @Override
    public OpType opType() {
        return OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS;
    }

    @Override
    public String getExpression() {
        return "scaledDotProductAttentionWeights";
    }
}
