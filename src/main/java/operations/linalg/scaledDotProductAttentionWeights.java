package operations.linalg;
import operations.Operation;

/**
 * Computes or exposes the attention probability weights from scaled
 * dot-product attention.
 *
 * <p>This descriptor is used when graph execution needs the softmax-normalized
 * attention weights as a distinct value, typically for backward computation or
 * diagnostics.</p>
 */
public final class scaledDotProductAttentionWeights implements Operation {
    @Override
    public OpType opType() {
        return OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.SPECIAL;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.SPECIAL;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.EXPENSIVE;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.NONE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "scaledDotProductAttentionWeights";
    }
}
