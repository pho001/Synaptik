package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Reorders tensor dimensions according to an axis permutation.
 *
 * <p>The descriptor stores a defensive copy of the supplied shape metadata so
 * callers cannot mutate graph semantics after construction.</p>
 */
public final class permute implements Operation {
    private final int[] axes;

    /**
     * Creates a layout descriptor.
     *
     * @param axes permutation of input axes
     */
    public permute(int[] axes) {
        this.axes = axes == null ? new int[0] : axes.clone();
    }

    /**
     * Returns a defensive copy of the stored layout metadata.
     *
     * @return permutation of input axes
     */
    public int[] getAxes() {
        return axes.clone();
    }

    @Override
    public OpType opType() {
        return OpType.PERMUTE;
    }

    @Override
    public String getExpression() {
        return "permute" + Arrays.toString(axes);
    }
}
