package operations;

import backend.ComputeBackend;
import tensor.Tensor;
import tensor.TensorRemap;

import java.util.List;


public class contiguous implements Operation{
    @Override
    public OpType opType() {
        return OpType.CONTIGUOUS;
    }



    @Override
    public String getExpression() {
        return "log";
    }


}
