package Backend;

import Tensor.Tensor;
import Operations.Operation;

import java.util.List;

public interface BackendExecutor {

    void execute(Operation op, List<Tensor> inputs, Tensor node);
    void backward(Operation op, List<Tensor> inputs, Tensor node);

}


