package operations.layout;

import operations.Operation;

/**
 * Concatenates tensors along one static axis.
 */
public final class concat implements Operation {
    private final int axis;

    public concat(int axis) {
        this.axis = axis;
    }

    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.CONCAT;
    }

    @Override
    public String getExpression() {
        return "concat(axis=" + axis + ")";
    }
}
