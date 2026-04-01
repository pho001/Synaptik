package operations;

import backend.ComputeBackend;
import tensor.Tensor;

import java.util.List;

public class matmul implements Operation {
    @Override
    public OpType opType() {
        return OpType.MATMUL;
    }


    @Override
    public String getExpression() {
        return "matmul";
    }
}
