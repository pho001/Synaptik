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
    public String getExpression() {
        return "matmul";
    }
}
