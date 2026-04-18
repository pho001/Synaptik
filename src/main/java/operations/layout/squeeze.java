package operations.layout;

import operations.Operation;

public final class squeeze implements Operation {
    private final int axis;

    public squeeze(int axis) {
        this.axis = axis;
    }

    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.SQUEEZE;
    }

    @Override
    public String getExpression() {
        return "squeeze(" + axis + ")";
    }
}
