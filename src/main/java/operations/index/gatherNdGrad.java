package operations.index;

import operations.Operation;

import java.util.Arrays;

/**
 * Accumulates an ONNX GatherND gradient back into the source tensor shape.
 */
public final class gatherNdGrad implements Operation {
    private final int batchDims;
    private final int[] dataShape;

    public gatherNdGrad(int batchDims, int[] dataShape) {
        if (batchDims < 0) {
            throw new IllegalArgumentException("gatherNdGrad batchDims must be non-negative.");
        }
        this.batchDims = batchDims;
        this.dataShape = dataShape == null ? new int[0] : dataShape.clone();
    }

    public int getBatchDims() {
        return batchDims;
    }

    public int[] getDataShape() {
        return dataShape.clone();
    }

    @Override
    public OpType opType() {
        return OpType.GATHER_ND_GRAD;
    }

    @Override
    public String getExpression() {
        return "gatherNdGrad(batchDims=" + batchDims + ",dataShape=" + Arrays.toString(dataShape) + ")";
    }
}
