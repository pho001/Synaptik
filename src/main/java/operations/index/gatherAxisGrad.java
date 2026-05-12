package operations.index;

import operations.Operation;

import java.util.Arrays;

/**
 * Accumulates an ONNX-style gather gradient back into the source tensor shape.
 */
public final class gatherAxisGrad implements Operation {
    private final int axis;
    private final int[] dataShape;

    public gatherAxisGrad(int axis, int[] dataShape) {
        this.axis = axis;
        this.dataShape = dataShape == null ? new int[0] : dataShape.clone();
    }

    public int getAxis() {
        return axis;
    }

    public int[] getDataShape() {
        return dataShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.GATHER_AXIS_GRAD;
    }

    @Override
    public String getExpression() {
        return "gatherAxisGrad(axis=" + axis + ",dataShape=" + Arrays.toString(dataShape) + ")";
    }
}
