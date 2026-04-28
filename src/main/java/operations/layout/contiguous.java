package operations.layout;

import operations.Operation;

/**
 * Requests a contiguous physical layout for the input tensor.
 *
 * <p>The logical shape and dtype are unchanged; execution may materialize data
 * when the input is a strided or otherwise non-contiguous view.</p>
 */
public final class contiguous implements Operation {
    @Override
    public OpType opType() {
        return OpType.CONTIGUOUS;
    }

    @Override
    public String getExpression() {
        return "contiguous";
    }
}
