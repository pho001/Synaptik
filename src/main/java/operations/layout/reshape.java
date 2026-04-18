package operations.layout;

import operations.Operation;

import java.util.Arrays;

public final class reshape implements Operation {
    private final int[] targetShape;

    public reshape(int[] targetShape) {
        this.targetShape = targetShape == null ? new int[0] : targetShape.clone();
    }

    public int[] getTargetShape() {
        return targetShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.RESHAPE;
    }

    @Override
    public String getExpression() {
        return "reshape" + Arrays.toString(targetShape);
    }
}
