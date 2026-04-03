package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogicalAndKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        BoolKernelSupport.runBinary(Operation.OpType.LOGICAL_AND, inputs.get(0).getBoolData(), inputs.get(1).getBoolData(), node.getBoolData(), context.broadcastPlan());
    }
}
