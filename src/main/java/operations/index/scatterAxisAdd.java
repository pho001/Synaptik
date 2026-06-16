package operations.index;

import operations.Operation;

/**
 * Functional inverse of ONNX-style gather along an axis.
 *
 * <p>The operation copies a base tensor and adds updates into positions selected
 * by a rank-changing index tensor. It is a general indexed-write primitive, not
 * a gradient-only operation.</p>
 */
public final class scatterAxisAdd implements Operation {
    private final int axis;

    public scatterAxisAdd(int axis) {
        this.axis = axis;
    }

    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.SCATTER_AXIS_ADD;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.SPECIAL;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.SPECIAL;
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
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "scatterAxisAdd(axis=" + axis + ")";
    }
}
