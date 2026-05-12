package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Repeats a tensor along each axis.
 */
public final class tile implements Operation {
    private final int[] repeats;

    public tile(int[] repeats) {
        this.repeats = repeats == null ? new int[0] : repeats.clone();
    }

    @Override
    public OpType opType() {
        return OpType.TILE;
    }

    @Override
    public String getExpression() {
        return "tile(repeats=" + Arrays.toString(repeats) + ")";
    }

    public int[] getRepeats() {
        return repeats.clone();
    }
}
