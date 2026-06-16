package operations.layout;
import operations.Operation;

import java.util.Arrays;

/**
 * Broadcasts a tensor view to a target shape.
 *
 * <p>The descriptor stores a defensive copy of the supplied shape metadata so
 * callers cannot mutate graph semantics after construction.</p>
 */
public final class expand implements Operation {
    private final int[] targetShape;

    /**
     * Creates a layout descriptor.
     *
     * @param targetShape target broadcast shape, or {@code null} when deferred to the tensor front end
     */
    public expand(int[] targetShape) {
        this.targetShape = targetShape == null ? null : targetShape.clone();
    }

    /**
     * Returns a defensive copy of the stored layout metadata.
     *
     * @return target broadcast shape, or {@code null} when deferred to the tensor front end
     */
    public int[] getTargetShape() {
        return targetShape == null ? null : targetShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.EXPAND;
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
        return "expand(" + Arrays.toString(targetShape) + ")";
    }
}
