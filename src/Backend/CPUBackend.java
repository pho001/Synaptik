package Backend;

import Tensor.Tensor;
import Operations.Operation;

import java.util.List;

public class CPUBackend implements BackendExecutor{


    @Override
    public void execute(Operation op, List<Tensor> inputs,Tensor node) {
        op.apply(inputs,node);
    }

    @Override
    public void backward(Operation op, List<Tensor> inputs,Tensor node) {
        op.gradient(inputs,node);
    }


}
