package Backend;

import Backend.kernels.cpu.CpuKernel;
import Backend.registry.CpuKernelRegistry;
import Tensor.Tensor;
import Operations.Operation;

import java.util.List;

public class CPUBackend implements BackendExecutor{


    @Override
    public void execute(Operation op, List<Tensor> inputs,Tensor node) {
        if (op == null) {
            return;
        }
        CpuKernel kernel = CpuKernelRegistry.resolve(op.opType());
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }
        kernel.forward(op, inputs, node);
    }

    @Override
    public void backward(Operation op, List<Tensor> inputs,Tensor node) {
        op.gradient(inputs,node);
    }


}
