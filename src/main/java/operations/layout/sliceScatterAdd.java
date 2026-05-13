package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Functional sparse write into a sliced region.
 *
 * <p>The operation creates a tensor with {@code inputShape}, initializes it to
 * zero, and adds update values at coordinates described by starts/axes/steps.
 * It is the general indexed-write counterpart of {@link slice}.</p>
 */
public final class sliceScatterAdd implements Operation {
    private final int[] starts;
    private final int[] axes;
    private final int[] steps;
    private final int[] inputShape;

    public sliceScatterAdd(int[] starts, int[] axes, int[] steps, int[] inputShape) {
        this.starts = starts == null ? new int[0] : starts.clone();
        this.axes = axes == null ? new int[0] : axes.clone();
        this.steps = steps == null ? new int[0] : steps.clone();
        this.inputShape = inputShape == null ? new int[0] : inputShape.clone();
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
        return OpType.SLICE_SCATTER_ADD;
    }

    @Override
    public String getExpression() {
        return "sliceScatterAdd(starts=" + Arrays.toString(starts)
                + ",axes=" + Arrays.toString(axes)
                + ",steps=" + Arrays.toString(steps)
                + ",inputShape=" + Arrays.toString(inputShape) + ")";
    }
}
