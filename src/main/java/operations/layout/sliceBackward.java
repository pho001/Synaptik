package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Scatters a slice output gradient back into the original input shape.
 *
 * <p>The operation creates a tensor with {@code inputShape}, initializes it to
 * zero, and adds update values at coordinates described by starts/axes/steps.
 * It is the backend-neutral backward counterpart of {@link slice}.</p>
 */
public final class sliceBackward implements Operation {
    private final int[] starts;
    private final int[] axes;
    private final int[] steps;
    private final int[] inputShape;

    public sliceBackward(int[] starts, int[] axes, int[] steps, int[] inputShape) {
        this.starts = copy(starts);
        this.axes = copy(axes);
        this.steps = copy(steps);
        this.inputShape = copy(inputShape);
    }

    public int[] getStarts() {
        return starts.clone();
    }

    public int[] getAxes() {
        return axes.clone();
    }

    public int[] getSteps() {
        return steps.clone();
    }

    public int[] getInputShape() {
        return inputShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.SLICE_BACKWARD;
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
        return OpComputationalCost.MEDIUM;
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
        return "sliceBackward(starts=" + Arrays.toString(starts)
                + ",axes=" + Arrays.toString(axes)
                + ",steps=" + Arrays.toString(steps)
                + ",inputShape=" + Arrays.toString(inputShape) + ")";
    }

    private static int[] copy(int[] values) {
        return values == null ? new int[0] : values.clone();
    }
}
