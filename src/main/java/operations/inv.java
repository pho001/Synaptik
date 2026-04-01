package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class inv implements Operation {

    @Override
    public OpType opType() {
        return OpType.INV;
    }

    @Override
    public boolean isCheap() {
        return false;
    }

    @Override
    public String getExpression() {
        return "inv";
    }
}
