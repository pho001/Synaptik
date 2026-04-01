package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public class permute implements Operation {
    private final int[] axes;

    public permute(int[] axes) {
        this.axes = axes == null ? new int[0] : axes.clone();
    }

    public int[] getAxes() {
        return axes.clone();
    }

    @Override
    public OpType opType() {
        return OpType.PERMUTE;
    }


    @Override
    public String getExpression() {
        return "permute" + Arrays.toString(axes);
    }
}

