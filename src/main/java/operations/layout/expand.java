package operations.layout;
import operations.Operation;

import java.util.Arrays;

public final class expand implements Operation {
    private final int[] targetShape;

    public expand(int[] targetShape) {
        this.targetShape = targetShape == null ? null : targetShape.clone();
    }

    public int[] getTargetShape() {
        return targetShape == null ? null : targetShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.EXPAND;
    }

    @Override
    public String getExpression() {
        return "expand(" + Arrays.toString(targetShape) + ")";
    }
}
