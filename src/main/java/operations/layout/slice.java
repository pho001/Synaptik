package operations.layout;

import operations.Operation;

import java.util.Arrays;

/**
 * Selects a strided static slice from a tensor while preserving rank.
 */
public final class slice implements Operation {
    private final int[] starts;
    private final int[] ends;
    private final int[] axes;
    private final int[] steps;
    private final int[] outputShape;

    public slice(int[] starts, int[] ends, int[] axes, int[] steps, int[] outputShape) {
        this.starts = copy(starts);
        this.ends = copy(ends);
        this.axes = copy(axes);
        this.steps = copy(steps);
        this.outputShape = copy(outputShape);
    }

    public int[] getStarts() {
        return starts.clone();
    }

    public int[] getEnds() {
        return ends.clone();
    }

    public int[] getAxes() {
        return axes.clone();
    }

    public int[] getSteps() {
        return steps.clone();
    }

    public int[] getOutputShape() {
        return outputShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.SLICE;
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
        return "slice(starts=" + Arrays.toString(starts)
                + ",ends=" + Arrays.toString(ends)
                + ",axes=" + Arrays.toString(axes)
                + ",steps=" + Arrays.toString(steps) + ")";
    }

    private static int[] copy(int[] values) {
        return values == null ? new int[0] : values.clone();
    }
}
