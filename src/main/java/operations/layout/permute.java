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
    public OpArityClass arityClass() {
        return OpArityClass.LAYOUT;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.LAYOUT;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.TRIVIAL;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.NONE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.SHAPE_VIEW;
    }

    @Override
    public String getExpression() {
        return "permute" + Arrays.toString(axes);
    }
}
