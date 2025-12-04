package Backend;

import Tensor.Tensor;
import Operations.Operation;

import java.util.List;

public class OpenClBackend implements BackendExecutor {
    @Override
    public void execute(Operation op, List<Tensor> inputs, Tensor node) {

    }

    @Override
    public void backward(Operation op, List<Tensor> inputs, Tensor node) {

    }
}
