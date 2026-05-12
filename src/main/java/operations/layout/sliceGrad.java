package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Scatters a slice output gradient back into the original input shape.
 */
public final class sliceGrad implements Operation {
    private final int[] starts;
    private final int[] axes;
    private final int[] steps;
    private final int[] inputShape;

    public sliceGrad(int[] starts, int[] axes, int[] steps, int[] inputShape) {
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
        return OpType.SLICE_GRAD;
    }

    @Override
    public String getExpression() {
        return "sliceGrad(starts=" + Arrays.toString(starts)
                + ",axes=" + Arrays.toString(axes)
                + ",steps=" + Arrays.toString(steps) + ")";
    }

    private static int[] copy(int[] values) {
        return values == null ? new int[0] : values.clone();
    }
}
