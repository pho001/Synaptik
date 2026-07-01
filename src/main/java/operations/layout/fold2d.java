package operations.layout;

import operations.Operation;
import tensor.options.Window2dOptions;

import java.util.Arrays;

/**
 * Accumulates im2col 2-D sliding-window columns into an NCHW tensor.
 */
public final class fold2d implements Operation {
    private final int[] outputShape;
    private final Window2dOptions options;

    public fold2d(int[] outputShape, Window2dOptions options) {
        if (outputShape == null) {
            throw new IllegalArgumentException("fold2d outputShape cannot be null");
        }
        if (options == null) {
            throw new IllegalArgumentException("fold2d options cannot be null");
        }
        this.outputShape = outputShape.clone();
        this.options = options;
    }

    public int[] getOutputShape() {
        return outputShape.clone();
    }

    public Window2dOptions getOptions() {
        return options;
    }

    @Override
    public OpType opType() {
        return OpType.FOLD2D;
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
    public OpResultKind resultKind() {
        return OpResultKind.SHAPE_VIEW;
    }

    @Override
    public String getExpression() {
        return "fold2d(outputShape=" + Arrays.toString(outputShape) + ", options=" + options + ")";
    }
}
