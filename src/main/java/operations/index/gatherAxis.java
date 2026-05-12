package operations.index;

import operations.Operation;

/**
 * ONNX-style gather: inserts the index tensor shape at the gathered axis.
 */
public final class gatherAxis implements Operation {
    private final int axis;

    public gatherAxis(int axis) {
        this.axis = axis;
    }

    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.GATHER_AXIS;
    }

    @Override
    public String getExpression() {
        return "gatherAxis(axis=" + axis + ")";
    }
}
