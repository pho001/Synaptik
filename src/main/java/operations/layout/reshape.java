package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Changes tensor shape without changing logical element order.
 *
 * <p>The descriptor stores a defensive copy of the supplied shape metadata so
 * callers cannot mutate graph semantics after construction.</p>
 */
public final class reshape implements Operation {
    private final int[] targetShape;

    /**
     * Creates a layout descriptor.
     *
     * @param targetShape target shape. An empty array represents scalar shape in this descriptor.
     */
    public reshape(int[] targetShape) {
        this.targetShape = targetShape == null ? new int[0] : targetShape.clone();
    }

    /**
     * Returns a defensive copy of the stored layout metadata.
     *
     * @return target shape. An empty array represents scalar shape in this descriptor.
     */
    public int[] getTargetShape() {
        return targetShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.RESHAPE;
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
    public OpResultKind resultKind() {
        return OpResultKind.SHAPE_VIEW;
    }

    @Override
    public String getExpression() {
        return "reshape" + Arrays.toString(targetShape);
    }
}
