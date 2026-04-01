package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class expandDims implements Operation {
    private final int axis;

    public expandDims(int axis) {
        this.axis = axis;
    }

    public int getAxis() {
        return axis;
    }

    @Override
    public OpType opType() {
        return OpType.EXPAND_DIMS;
    }


    @Override
    public String getExpression() {
        return "expandDims(" + axis + ")";
    }
}

