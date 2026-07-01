package operations.linalg;

import operations.Operation;

/**
 * Matrix multiplication descriptor.
 *
 * <p>Rank-2 inputs follow standard {@code [m,k] x [k,n] -> [m,n]} semantics;
 * higher-rank inputs are interpreted by the tensor front end with broadcasted
 * leading batch dimensions. The result dtype is resolved by backend matmul
 * execution.</p>
 */
public final class matmul implements Operation {
    @Override
    public OpType opType() {
        return OpType.MATMUL;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.LINEAR_ALGEBRA;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.LINEAR_ALGEBRA;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.EXPENSIVE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "matmul";
    }
}
